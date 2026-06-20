"""Shared runtime glue for Dressage generate hooks."""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

from dressage.config import proxy_url
from dressage.proxy.proxy_client import ProxyClient

logger = logging.getLogger(__name__)

_PADDOCK = None
_PROXY_CLIENT: ProxyClient | None = None

_PADDOCK_ENV_ARG_KEYS = (
    "sandbox_timeout_sec",
    "sandbox_image",
    "sandbox_cmd",
    "sandbox_extra_params",
)


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def get_proxy_client() -> ProxyClient:
    global _PROXY_CLIENT
    if _PROXY_CLIENT is None:
        _PROXY_CLIENT = ProxyClient(proxy_url())
    return _PROXY_CLIENT


def get_paddock_from_env(*, allow_whitebox_mode: bool) -> Any:
    global _PADDOCK
    if _PADDOCK is not None:
        return _PADDOCK

    paddock_class_path = os.environ.get("DRESSAGE_PADDOCK_CLASS")
    paddock_mode = (os.environ.get("DRESSAGE_PADDOCK_MODE") or "blackbox").strip().lower()
    if not paddock_class_path and not allow_whitebox_mode and paddock_mode == "whitebox":
        raise ValueError(
            "blackbox_dispatch does not support whitebox mode; set "
            "DRESSAGE_PADDOCK_MODE=blackbox for this rollout hook, or use "
            "the Paddock API for whitebox tool execution"
        )

    from dressage.paddock import factory as paddock_factory

    _PADDOCK = paddock_factory.create_paddock_from_env()
    if paddock_class_path:
        logger.info("initialized paddock class override: %s", paddock_class_path)
    else:
        logger.info("initialized paddock from mode/provider env: %s", type(_PADDOCK).__name__)
    return _PADDOCK


def paddock_env_args_from_metadata(
    metadata: dict[str, Any],
    *,
    extra_env_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env_args = {key: metadata[key] for key in _PADDOCK_ENV_ARG_KEYS if key in metadata}
    if extra_env_args:
        env_args.update(extra_env_args)
    return env_args
