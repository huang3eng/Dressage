"""Tests for the reward function registry."""

from __future__ import annotations

import pytest

from dressage.reward.registry import (
    _REWARD_REGISTRY,
    get_reward_context,
    get_reward_fn,
    list_reward_fns,
    load_reward_modules,
    register_reward,
    set_reward_context,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    saved_reg = dict(_REWARD_REGISTRY)
    yield
    _REWARD_REGISTRY.clear()
    _REWARD_REGISTRY.update(saved_reg)


class TestRegisterReward:
    def test_register_and_retrieve(self):
        @register_reward("test_fn_1")
        def my_fn(sample, *, args=None, **kwargs):
            del sample, args, kwargs
            return 1.0

        fn = get_reward_fn("test_fn_1")
        assert fn is my_fn
        assert fn(None) == 1.0

    def test_duplicate_registration_raises(self):
        @register_reward("dup_fn")
        def fn1(sample, *, args=None, **kwargs):
            del sample, args, kwargs
            return 0.0

        with pytest.raises(ValueError, match="already registered"):
            @register_reward("dup_fn")
            def fn2(sample, *, args=None, **kwargs):
                del sample, args, kwargs
                return 1.0

    def test_missing_fn_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_reward_fn("nonexistent_function_xyz")

    def test_list_reward_fns(self):
        @register_reward("list_test_fn")
        def fn(sample, *, args=None, **kwargs):
            del sample, args, kwargs
            return 0.0

        names = list_reward_fns()
        assert "list_test_fn" in names


class TestRewardContext:
    def test_set_and_get(self):
        set_reward_context(paddock="mock_paddock", proxy="mock_proxy")
        ctx = get_reward_context()
        assert ctx["paddock"] == "mock_paddock"
        assert ctx["proxy"] == "mock_proxy"

    def test_update_context(self):
        set_reward_context(key1="val1")
        set_reward_context(key2="val2")
        ctx = get_reward_context()
        assert ctx["key1"] == "val1"
        assert ctx["key2"] == "val2"


class TestBuiltinRewards:
    def test_builtin_rewards_registered(self):
        import dressage.reward  # noqa: F401 — triggers helper import

        expected = {
            "constant",
            "contains_label",
            "default",
            "exact_match",
            "metadata_score",
        }
        assert expected.issubset(set(list_reward_fns()))


class TestLoadRewardModules:
    def test_load_reward_modules_imports_custom_rewards(self, tmp_path, monkeypatch):
        module_path = tmp_path / "custom_rewards.py"
        module_path.write_text(
            "from dressage.reward import register_reward\n"
            "@register_reward('tmp_reward')\n"
            "def tmp_reward(sample, *, args=None, **kwargs):\n"
            "    return 0.42\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        load_reward_modules("custom_rewards")

        assert get_reward_fn("tmp_reward")(None) == 0.42
