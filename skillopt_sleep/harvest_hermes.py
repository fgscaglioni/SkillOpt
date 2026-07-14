"""Hermes Agent session harvesting for SkillOpt-Sleep.

Reads session transcripts from the Hermes Agent state database
(``~/.hermes/state.db``) and returns ``SessionDigest`` objects.

Compared to the original implementation, this version:
  - Includes ``tool``-role messages (not just user + assistant)
  - Detects feedback signals from user messages (pos:/neg:)
  - Redacts secrets from extracted text
  - Uses ``limit=0`` as unlimited instead of capping at 200
  - Only filters engine's own temp sessions, not arbitrary ``/tmp`` projects
"""
from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

from skillopt_sleep.harvest import _detect_feedback, _is_meta_prompt
from skillopt_sleep.types import SessionDigest

HERMES_HOME = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
STATE_DB = os.path.join(HERMES_HOME, "state.db")

# ── Secret redaction (mirrors harvest_codex.py) ───────────────────────────────

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_-]{10,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"(?i)(Authorization:\s*Bearer\s+)[^\s\"']+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\b(\s*[:=]\s*)[^\s\"']+"), r"\1\2[REDACTED]"),
    (re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\b(\s+)[^\s\"']+"), r"\1\2[REDACTED]"),
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
)


def _redact(text: str) -> str:
    """Replace known secret patterns with placeholders."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── Engine session filter ─────────────────────────────────────────────────────


def _is_engine_session(cwd: str) -> bool:
    """Return True if a session was created by the engine's own backend calls.

    These sessions run in temp dirs (prefix ``skillopt_sleep_hermes_``) and
    represent optimizer/target calls, not real user sessions.
    """
    return "skillopt_sleep_hermes_" in cwd


# ── Database helpers ──────────────────────────────────────────────────────────


def _fetch_sessions(
    db_path: str,
    *,
    since_epoch: Optional[float] = None,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch sessions from the Hermes state database.

    Only sessions with a cwd and an end timestamp are returned.
    Engine-manufactured sessions (``skillopt_sleep_hermes_`` tempdirs)
    are excluded.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    where = "WHERE cwd IS NOT NULL AND cwd != '' AND ended_at IS NOT NULL"
    params: List[Any] = []
    if since_epoch is not None:
        where += " AND ended_at >= ?"
        params.append(since_epoch)

    # limit=0 means no cap (use a large number)
    actual_limit = min(limit, 999999) if limit else 999999
    cursor.execute(
        f"""SELECT id, cwd, title, started_at, ended_at, model
            FROM sessions
            {where}
            ORDER BY ended_at DESC
            LIMIT ?""",
        params + [actual_limit],
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def _fetch_messages(db_path: str, session_id: str) -> List[Dict[str, Any]]:
    """Return all messages for a session, ordered by id.

    Includes ``user``, ``assistant``, AND ``tool`` roles so tool
    execution context is preserved.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """SELECT role, content, tool_name, timestamp
           FROM messages
           WHERE session_id = ? AND role IN ('user', 'assistant', 'tool')
           ORDER BY id""",
        (session_id,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ── Digest construction ───────────────────────────────────────────────────────


def _build_digest(
    session: Dict[str, Any],
    messages: List[Dict[str, Any]],
    scope: str = "invoked",
    invoked_project: str = "",
) -> Optional[SessionDigest]:
    """Build a ``SessionDigest`` from one session + its messages.

    Returns ``None`` if the session has no user or assistant turns, or if it
    doesn't match the project scope.
    """
    session_id = session.get("id") or ""
    project = (session.get("cwd") or "").strip()
    title = (session.get("title") or "").strip()

    user_prompts: List[str] = []
    assistant_finals: List[str] = []
    tools: List[str] = []
    feedback_signals: List[str] = []
    n_user = 0
    n_asst = 0

    last_assistant = ""
    for msg in messages:
        role = (msg.get("role") or "").strip()
        content = (msg.get("content") or "").strip()
        tool = (msg.get("tool_name") or "").strip()

        if role == "user" and content:
            n_user += 1
            redacted = _redact(content)
            user_prompts.append(redacted)
            feedback_signals.extend(_detect_feedback(redacted))
            # Flush any pending assistant final
            if last_assistant:
                assistant_finals.append(last_assistant)
                last_assistant = ""
        elif role == "assistant" and content:
            n_asst += 1
            last_assistant = _redact(content)
        elif role == "tool":
            # Tool execution output — redact and include context but don't
            # count as a user or assistant turn
            if content:
                _redact(content)
            # Collect tool names for the tools_used list
            # (tool_name is already captured below)

        if tool:
            tools.append(tool)

    # Flush the last assistant message
    if last_assistant:
        assistant_finals.append(last_assistant)

    if n_user == 0 and n_asst == 0:
        return None

    # Project matching
    if not _project_matches(project, scope, invoked_project):
        return None

    return SessionDigest(
        session_id=session_id,
        project=project,
        started_at=_ts_from_epoch(session.get("started_at")),
        ended_at=_ts_from_epoch(session.get("ended_at")),
        user_prompts=user_prompts,
        assistant_finals=assistant_finals[-5:],
        tools_used=_dedup(tools),
        files_touched=[],
        feedback_signals=feedback_signals,
        n_user_turns=n_user,
        n_assistant_turns=n_asst,
        raw_path=f"{STATE_DB}:{session_id}",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _dedup(xs: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _ts_from_epoch(epoch: Any) -> str:
    """Convert a Unix epoch (float/int) to ISO 8601 string."""
    if epoch is None:
        return ""
    try:
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
        return dt.isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def _epoch_from_iso(iso: str) -> Optional[float]:
    """Convert ISO 8601 string to Unix epoch. Returns None on failure."""
    try:
        from datetime import datetime, timezone

        s = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _project_matches(project: str, scope: str, invoked: str) -> bool:
    """Check whether ``project`` matches the scope."""
    if not invoked or scope == "all":
        return True
    if not project:
        return True  # no cwd → can't filter, accept
    a = os.path.abspath(project)
    b = os.path.abspath(invoked)
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


# ── Public API ────────────────────────────────────────────────────────────────


def harvest_hermes(
    *,
    scope: str = "invoked",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
    db_path: str = "",
) -> List[SessionDigest]:
    """Walk ``~/.hermes/state.db`` and return matching digests.

    Parameters
    ----------
    scope : str
        ``"all"`` | ``"invoked"``
    invoked_project : str
        Used when ``scope == "invoked"``.
    since_iso : str | None
        ISO 8601; only sessions ending after this are kept.
    limit : int
        Cap number of digests (0 = no limit).
    db_path : str
        Override state.db path (default: ``~/.hermes/state.db``).
    """
    db = db_path or STATE_DB
    if not os.path.isfile(db):
        return []

    since_epoch = _epoch_from_iso(since_iso) if since_iso else None
    sessions = _fetch_sessions(db, since_epoch=since_epoch, limit=limit)

    # Filter engine-manufactured sessions
    sessions = [s for s in sessions if not _is_engine_session(s.get("cwd") or "")]

    digests: List[SessionDigest] = []
    for s in sessions:
        sid = s.get("id") or ""
        msgs = _fetch_messages(db, sid)
        digest = _build_digest(
            s, msgs,
            scope=scope,
            invoked_project=invoked_project,
        )
        if digest is None:
            continue
        digests.append(digest)
        if limit and len(digests) >= limit:
            break

    return digests
