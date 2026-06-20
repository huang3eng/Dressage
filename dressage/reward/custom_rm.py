"""Slime custom reward model entrypoint backed by Dressage reward registry."""

from __future__ import annotations

import asyncio
from typing import Any

from dressage.reward import call_reward_fn, load_reward_modules

_MODULES_LOADED = False


def _reward_name(sample: Any) -> str:
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        return str(metadata.get("reward_fn") or "default")
    return "default"


async def _score_one(args: Any, sample: Any, **kwargs: Any) -> float:
    global _MODULES_LOADED
    if not _MODULES_LOADED:
        load_reward_modules()
        _MODULES_LOADED = True
    return await call_reward_fn(_reward_name(sample), sample, args=args, **kwargs)


async def custom_rm(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    """Reward function loaded through Slime ``--custom-rm-path``.

    Slime calls this after custom generation when ``sample.reward`` is still
    unset. This function intentionally returns rewards to Slime instead of
    writing to proxy state.
    """
    if isinstance(sample_or_samples, list):
        return await asyncio.gather(
            *[_score_one(args, sample, **kwargs) for sample in sample_or_samples]
        )
    return await _score_one(args, sample_or_samples, **kwargs)
