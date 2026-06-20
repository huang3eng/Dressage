"""Slime API surface drift detector.

Asserts that every slime symbol dressage relies on still exists with a
compatible shape.  All checks are AST-only — slime / megatron / ray are
not imported, so this test runs in a lightweight CPU environment.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SLIME_ROOT = REPO_ROOT / "slime"
SLIME_PKG = SLIME_ROOT / "slime"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _module_tree(rel: str) -> ast.Module:
    return ast.parse((SLIME_PKG / rel).read_text())


def _top_level_funcs(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _top_level_classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _class_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for item in cls.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return item
    return None


def _param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = func.args
    return {a.arg for a in args.args} | {a.arg for a in args.kwonlyargs}


# ---------------------------------------------------------------------------
# Slime symbols dressage depends on
# ---------------------------------------------------------------------------


def test_convert_samples_to_train_data_accepts_custom():
    """RolloutManager._convert_samples_to_train_data must check
    custom_convert_samples_to_train_data_func so dressage can override it."""
    tree = _module_tree("ray/rollout.py")
    classes = _top_level_classes(tree)
    assert "RolloutManager" in classes
    method = _class_method(classes["RolloutManager"], "_convert_samples_to_train_data")
    assert method is not None, "slime RolloutManager._convert_samples_to_train_data missing"
    body_src = ast.unparse(method)
    assert "custom_convert_samples_to_train_data" in body_src, (
        "slime _convert_samples_to_train_data no longer checks "
        "custom_convert_samples_to_train_data_func"
    )


def test_log_rollout_data_accepts_custom():
    """_log_rollout_data must check custom_rollout_log_function_path."""
    tree = _module_tree("ray/rollout.py")
    funcs = _top_level_funcs(tree)
    assert "_log_rollout_data" in funcs, "slime _log_rollout_data missing"
    body_src = ast.unparse(funcs["_log_rollout_data"])
    assert "custom_rollout_log_function_path" in body_src, (
        "slime _log_rollout_data no longer checks custom_rollout_log_function_path"
    )


def test_rollout_mask_sums_in_convert_samples():
    """_convert_samples_to_train_data must produce rollout_mask_sums."""
    tree = _module_tree("ray/rollout.py")
    classes = _top_level_classes(tree)
    method = _class_method(classes["RolloutManager"], "_convert_samples_to_train_data")
    body_src = ast.unparse(method)
    assert "rollout_ids" in body_src
    assert "rollout_mask_sums" in body_src


def test_rollout_mask_sums_in_split_train_data():
    """_split_train_data_by_dp must include rollout_mask_sums in per-rank data."""
    tree = _module_tree("ray/rollout.py")
    classes = _top_level_classes(tree)
    method = _class_method(classes["RolloutManager"], "_split_train_data_by_dp")
    assert method is not None
    body_src = ast.unparse(method)
    assert "rollout_ids" in body_src
    assert "rollout_mask_sums" in body_src


def test_parse_args_accepts_add_custom_arguments_kwarg():
    funcs = _top_level_funcs(_module_tree("utils/arguments.py"))
    assert "parse_args" in funcs
    assert "add_custom_arguments" in _param_names(funcs["parse_args"])


def test_create_placement_groups_still_exists():
    funcs = _top_level_funcs(_module_tree("ray/placement_group.py"))
    for name in ("create_placement_groups", "create_rollout_manager", "create_training_models"):
        assert name in funcs, f"slime {name} missing"


def test_get_sum_of_sample_mean_accepts_sample_denoms():
    """get_sum_of_sample_mean must accept sample_denoms kwarg for custom denoms."""
    funcs = _top_level_funcs(_module_tree("backends/megatron_utils/cp_utils.py"))
    assert "get_sum_of_sample_mean" in funcs
    assert "sample_denoms" in _param_names(funcs["get_sum_of_sample_mean"])


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="slime loss.py uses match/case (Python 3.10+ syntax)",
)
def test_policy_loss_function_exists():
    """slime.backends.megatron_utils.loss.policy_loss_function must exist."""
    funcs = _top_level_funcs(_module_tree("backends/megatron_utils/loss.py"))
    assert "policy_loss_function" in funcs


# ---------------------------------------------------------------------------
# Slime submodule version sanity (informational; non-blocking).
# ---------------------------------------------------------------------------


def test_slime_submodule_head_is_pinned():
    out = subprocess.check_output(
        ["git", "-C", str(SLIME_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    assert re.fullmatch(r"[0-9a-f]{40}", out), f"unexpected slime HEAD: {out!r}"
