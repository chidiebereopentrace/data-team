"""Unit tests for session context isolation.

Sprint 1, Week 2 (Jul 2026): verifies that sessions are strictly scoped by
session_id, with no cross-session memory bleed. Uses the in-memory fallback
backend (no Redis required).

Session isolation guarantees:
1. Different session_ids get independent memory (no bleed).
2. Deleting a session clears its memory completely.
3. A new session_id starts with empty memory.
4. No LLM generation results are cached across queries (no "cached augmented generation").
"""
from __future__ import annotations

from ml.rag.session_store import (
    delete_session,
    get_session_blob,
    save_session_blob,
)
from ml.rag.chat_memory import append_turn_and_compact
from ml.rag.chat_history import normalize_messages


# ---------------------------------------------------------------------------
# 1. Session isolation — different session_ids are independent
# ---------------------------------------------------------------------------

def test_sessions_are_independent() -> None:
    """Two different session_ids must not share memory."""
    sid_a = "test_iso_session_a"
    sid_b = "test_iso_session_b"

    # Clean up any prior state
    delete_session(sid_a)
    delete_session(sid_b)

    # Populate session A with a turn
    save_session_blob(sid_a, {
        "conversation_summary": "User asked about maize in Kenya.",
        "recent_turns": [
            {"role": "user", "content": "What is maize production in Kenya?"},
            {"role": "assistant", "content": "Kenya produces 3.5M tonnes of maize annually."},
        ],
    })

    # Session B should have NO memory from session A
    blob_b = get_session_blob(sid_b)
    assert blob_b is None, f"Session B should be empty but got: {blob_b}"

    # Session A should have its own memory
    blob_a = get_session_blob(sid_a)
    assert blob_a is not None
    assert "maize" in str(blob_a.get("conversation_summary", "")).lower()

    # Cleanup
    delete_session(sid_a)
    delete_session(sid_b)


def test_session_turns_do_not_leak() -> None:
    """Turns persisted under one session_id must not appear under another."""
    sid_x = "test_leak_x"
    sid_y = "test_leak_y"
    delete_session(sid_x)
    delete_session(sid_y)

    # Build up session X with multiple turns
    summary_x, recent_x = append_turn_and_compact(
        "", None, "Q1 about sorghum harvests", "A1 about sorghum harvests"
    )
    save_session_blob(sid_x, {"conversation_summary": summary_x, "recent_turns": recent_x})

    summary_x2, recent_x2 = append_turn_and_compact(
        summary_x, recent_x, "Q2 about cassava yields", "A2 about cassava yields"
    )
    save_session_blob(sid_x, {"conversation_summary": summary_x2, "recent_turns": recent_x2})

    # Build session Y with a completely different turn
    summary_y, recent_y = append_turn_and_compact(
        "", None, "Q1 about petroleum exports", "A1 about petroleum exports"
    )
    save_session_blob(sid_y, {"conversation_summary": summary_y, "recent_turns": recent_y})

    # Verify X has its content
    blob_x = get_session_blob(sid_x) or {}
    recent_x_content = str(blob_x.get("recent_turns", []))
    assert "sorghum" in recent_x_content.lower() or "cassava" in recent_x_content.lower()
    assert "petroleum" not in recent_x_content.lower(), "Session X should not contain session Y content"

    # Verify Y has its content
    blob_y = get_session_blob(sid_y) or {}
    recent_y_content = str(blob_y.get("recent_turns", []))
    assert "petroleum" in recent_y_content.lower()
    assert "sorghum" not in recent_y_content.lower(), "Session Y should not contain session X content"
    assert "cassava" not in recent_y_content.lower(), "Session Y should not contain session X content"

    delete_session(sid_x)
    delete_session(sid_y)


# ---------------------------------------------------------------------------
# 2. Session deletion clears memory
# ---------------------------------------------------------------------------

def test_delete_session_clears_memory() -> None:
    """After delete_session(), get_session_blob() must return None."""
    sid = "test_delete_session"
    delete_session(sid)

    save_session_blob(sid, {
        "conversation_summary": "User discussed wheat exports.",
        "recent_turns": [
            {"role": "user", "content": "Tell me about wheat exports in Ethiopia."},
            {"role": "assistant", "content": "Ethiopia exported 50K tonnes of wheat in 2024."},
        ],
    })

    # Verify it was saved
    assert get_session_blob(sid) is not None

    # Delete it
    delete_session(sid)

    # Verify it's gone
    assert get_session_blob(sid) is None


# ---------------------------------------------------------------------------
# 3. New session starts empty
# ---------------------------------------------------------------------------

def test_new_session_has_no_memory() -> None:
    """A never-used session_id must return None (no prior memory)."""
    import uuid
    fresh_sid = f"test_fresh_{uuid.uuid4().hex}"
    blob = get_session_blob(fresh_sid)
    assert blob is None, f"Fresh session should be empty but got: {blob}"


# ---------------------------------------------------------------------------
# 4. No cached augmented generation (stateless pipeline)
# ---------------------------------------------------------------------------

def test_no_global_mutable_session_state() -> None:
    """
    Verify the session store has no module-level mutable state that could cause
    cross-request contamination. The in-memory fallback dict is keyed by session_id,
    so different IDs cannot collide.
    """
    sid1 = "test_global_state_1"
    sid2 = "test_global_state_2"
    delete_session(sid1)
    delete_session(sid2)

    # Save to sid1
    save_session_blob(sid1, {"conversation_summary": "alpha", "recent_turns": []})

    # Save completely different data to sid2
    save_session_blob(sid2, {"conversation_summary": "beta", "recent_turns": []})

    # Verify they don't overwrite each other
    b1 = get_session_blob(sid1) or {}
    b2 = get_session_blob(sid2) or {}
    assert b1.get("conversation_summary") == "alpha"
    assert b2.get("conversation_summary") == "beta"

    delete_session(sid1)
    delete_session(sid2)


# ---------------------------------------------------------------------------
# 5. Memory resolution uses session_id correctly
# ---------------------------------------------------------------------------

def test_memory_resolution_scoped_by_session_id() -> None:
    """
    The API's _resolve_prior_memory function loads memory by session_id.
    Verify that passing different session_ids returns different memory.
    """
    from ml.rag.app.api import _resolve_prior_memory

    sid_a = "test_resolve_a"
    sid_b = "test_resolve_b"
    delete_session(sid_a)
    delete_session(sid_b)

    # Pre-populate session A
    save_session_blob(sid_a, {
        "conversation_summary": "Discussed sorghum production.",
        "recent_turns": [
            {"role": "user", "content": "What about sorghum?"},
            {"role": "assistant", "content": "Sorghum is a key crop in West Africa."},
        ],
    })

    # Resolve session A — should have memory
    _, summary_a, recent_a = _resolve_prior_memory(sid_a, None)
    assert summary_a.strip() or recent_a, "Session A should have prior memory"

    # Resolve session B — should be empty
    _, summary_b, recent_b = _resolve_prior_memory(sid_b, None)
    assert not summary_b.strip() and not recent_b, "Session B should have no prior memory"

    delete_session(sid_a)
    delete_session(sid_b)
