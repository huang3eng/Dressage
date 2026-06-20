from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from dressage import reward as reward_package  # noqa: F401
from dressage.reward.custom_rm import custom_rm
from dressage.reward.registry import register_reward


@dataclass
class SampleLike:
    response: str = ""
    label: str | None = None
    metadata: dict = field(default_factory=dict)
    reward: float | None = None


@pytest.fixture(autouse=True)
def _restore_rewards():
    from dressage.reward import helpers  # noqa: F401
    from dressage.reward.registry import _LOADED_REWARD_MODULES, _REWARD_REGISTRY

    saved = dict(_REWARD_REGISTRY)
    loaded = set(_LOADED_REWARD_MODULES)
    yield
    _REWARD_REGISTRY.clear()
    _REWARD_REGISTRY.update(saved)
    _LOADED_REWARD_MODULES.clear()
    _LOADED_REWARD_MODULES.update(loaded)


def test_custom_rm_builtin_single_sample():
    sample = SampleLike(response="the answer is 4", label="4")
    reward = asyncio.run(custom_rm(None, sample))
    assert reward == 1.0


def test_custom_rm_batch_constant_rewards():
    samples = [
        SampleLike(metadata={"reward_fn": "constant", "constant_reward": 0.25}),
        SampleLike(metadata={"reward_fn": "constant", "constant_reward": 0.75}),
    ]
    rewards = asyncio.run(custom_rm(None, samples))
    assert rewards == [0.25, 0.75]


def test_custom_rm_supports_async_reward_function():
    @register_reward("async_score")
    async def async_score(sample, *, args=None, **kwargs):
        del sample, args, kwargs
        return 0.5

    sample = SampleLike(metadata={"reward_fn": "async_score"})
    assert asyncio.run(custom_rm(None, sample)) == 0.5
