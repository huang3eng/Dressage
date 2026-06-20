"""Rollout-side lifecycle helpers for paddock sessions."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_TERMINATE_TASKS: set[asyncio.Task] = set()


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def exception_summary(exc: BaseException) -> str:
    return " ".join(str(exc).splitlines()) or type(exc).__name__


def terminate_timeout_sec() -> float:
    value = os.environ.get("DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC", "30")
    try:
        timeout = float(value)
    except ValueError:
        logger.warning(
            "invalid DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC=%r; falling back to 30",
            value,
        )
        return 30.0
    if timeout <= 0:
        logger.warning(
            "invalid DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC=%r; falling back to 30",
            value,
        )
        return 30.0
    return timeout


async def terminate_paddock_best_effort(
    paddock: Any,
    *,
    session_id: str,
    env_args: dict[str, Any],
) -> None:
    timeout = terminate_timeout_sec()
    release_task = asyncio.create_task(
        _call_paddock_terminate(paddock, session_id, env_args)
    )
    try:
        await asyncio.wait_for(asyncio.shield(release_task), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "timed out waiting for sandbox release RPC for session_id=%s "
            "after %.3fs; release continues in background",
            session_id,
            timeout,
        )
        _TERMINATE_TASKS.add(release_task)
        release_task.add_done_callback(_discard_terminate_task)
    except Exception as exc:
        logger.warning(
            "failed to terminate sandbox for session_id=%s: %s",
            session_id,
            exception_summary(exc),
        )


def schedule_terminate_paddock(
    paddock: Any,
    *,
    session_id: str,
    env_args: dict[str, Any],
) -> None:
    task = asyncio.create_task(
        terminate_paddock_best_effort(
            paddock,
            session_id=session_id,
            env_args=dict(env_args),
        )
    )
    _TERMINATE_TASKS.add(task)
    task.add_done_callback(_discard_terminate_task)


async def drain_terminate_tasks() -> None:
    while _TERMINATE_TASKS:
        tasks = tuple(_TERMINATE_TASKS)
        await asyncio.gather(*tasks, return_exceptions=True)


async def _call_paddock_terminate(
    paddock: Any,
    session_id: str,
    env_args: dict[str, Any],
) -> Any:
    terminate = paddock.terminate
    if inspect.iscoroutinefunction(terminate):
        return await terminate(session_id, env_args)
    return await maybe_await(await asyncio.to_thread(terminate, session_id, env_args))


def _discard_terminate_task(task: asyncio.Task) -> None:
    _TERMINATE_TASKS.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning("background terminate task failed: %s", exception_summary(exc))
