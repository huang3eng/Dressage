"""Tests for SessionManager._cleanup_expired_locked behavior.

Primary coverage target: a session whose request_lock is held must NOT be
evicted by the idle-timeout sweep. Previously _cleanup_expired_locked
evicted purely by wall-clock, so a long-running chat_completions (slow
sglang worker, large batch) could exceed session_timeout, get its session
removed mid-flight, and then record_step would silently no-op — losing the
trajectory step without any caller-visible signal.
"""

from __future__ import annotations

import asyncio
import logging
import time

from dressage.proxy.session_manager import SessionManager


def test_cleanup_evicts_idle_session_past_timeout():
    """Baseline: idle past timeout, no lock held → evicted."""
    manager = SessionManager(session_timeout=0.01)
    session, _ = manager.get_or_create_session("sess-1", messages=[])
    # Simulate idle: backdate last_active.
    session.last_active = time.time() - 1.0
    assert manager.active_count() == 0  # active_count runs cleanup first


def test_cleanup_defers_session_with_held_request_lock():
    """The fix: a session past timeout but with request_lock held is NOT
    evicted. Its last_active is touched so it gets a fresh idle window
    after the lock is released."""
    manager = SessionManager(session_timeout=0.01)
    session, _ = manager.get_or_create_session("sess-busy", messages=[])
    # Backdate so it's past the timeout.
    session.last_active = time.time() - 1.0

    async def run():
        async with session.request_lock:
            # While the lock is held, even though we're past timeout,
            # cleanup must defer.
            assert manager.active_count() == 1, (
                "session with held request_lock should not be evicted"
            )
            # last_active was touched, deferring the eviction.
            assert session.last_active >= time.time() - 0.001

    asyncio.run(run())

    # After the lock is released, cleanup follows the normal path: the
    # touched last_active is now ~now, so it stays alive until the next
    # full idle window elapses.
    assert manager.active_count() == 1
    time.sleep(0.02)
    assert manager.active_count() == 0


def test_cleanup_does_not_defer_when_only_idle_no_lock():
    """Negative control: session past timeout without lock IS evicted."""
    manager = SessionManager(session_timeout=0.01)
    session, _ = manager.get_or_create_session("sess-idle", messages=[])
    # Touch request_lock so the asyncio.Lock binds to *some* loop, then
    # release immediately — locked() returns False afterwards.

    async def briefly_acquire():
        async with session.request_lock:
            pass

    asyncio.run(briefly_acquire())
    session.last_active = time.time() - 1.0
    assert manager.active_count() == 0


def test_record_step_warns_when_session_missing(caplog):
    """If a future race ever bypasses the cleanup guard, record_step now
    surfaces the dropped step as a WARNING instead of silently returning
    (previously was a silent ``return``)."""
    manager = SessionManager()

    with caplog.at_level(logging.WARNING, logger="dressage.proxy.session_manager"):
        manager.record_step(
            session_id="ghost-session",
            turn_id="t1",
            request_messages=[],
            normalized_request_messages=[],
            prompt_token_ids=[],
            prompt_token_logprobs=[],
            snapshot_token_ids=[],
            response_token_ids=[],
            response_logprobs=[],
            all_token_ids=[],
            all_logprobs=[],
            input_token_texts=[],
            output_token_texts=[],
            messages=[],
            raw_response_text="",
        )

    assert any(
        "ghost-session" in r.message and "stale-cleanup" in r.message
        for r in caplog.records
    )


def test_cleanup_still_evicts_finalized_sessions_past_timeout():
    """The lock-held guard applies only to active sessions, not the
    finalized id set (which doesn't carry a lock)."""
    manager = SessionManager(session_timeout=0.01)
    session, _ = manager.get_or_create_session("sess-done", messages=[])
    manager.finalize_session("sess-done")
    # Backdate the finalized record so cleanup considers it expired.
    manager._finalized_session_ids["sess-done"] = time.time() - 1.0
    manager.active_count()  # triggers cleanup
    assert "sess-done" not in manager._finalized_session_ids
