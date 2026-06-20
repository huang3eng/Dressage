"""Structural / wiring checks for blackbox_dispatch.generate's multi-segment path.

Multi-segment is always on: every trajectory goes through
expand_segments_to_samples (single-segment trajectories produce exactly
one sample). These tests verify the wiring is correct.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DISPATCH_FILE = REPO_ROOT / "dressage" / "rollout" / "generate" / "blackbox_dispatch.py"


def _source() -> str:
    return DISPATCH_FILE.read_text()


def test_imports_multi_segment_module():
    text = _source()
    assert "from dressage.rollout import multi_segment" in text, (
        "blackbox_dispatch must import dressage.rollout.multi_segment"
    )


def test_generate_calls_expand_segments_to_samples():
    text = _source()
    assert "multi_segment.expand_segments_to_samples(" in text, (
        "generate() must call multi_segment.expand_segments_to_samples"
    )


def test_log_template_uses_original_sample():
    """expand_segments_to_samples returns a list, so artifact logging
    must receive the original sample as template, not the list."""
    text = _source()
    assert "log_template" in text, (
        "blackbox_dispatch must thread a log_template through artifact logging"
    )


def test_expand_call_passes_required_args():
    """ast-level check: the call to expand_segments_to_samples in
    blackbox_dispatch passes the required kwargs (args, agent_response,
    session_id, instance_id)."""
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "expand_segments_to_samples"
        ):
            kwargs = {kw.arg for kw in node.keywords}
            for required in ("args", "agent_response", "session_id", "instance_id"):
                assert required in kwargs, (
                    f"expand_segments_to_samples call missing required kwarg: {required}"
                )
            return
    raise AssertionError("expand_segments_to_samples call not found in blackbox_dispatch")


def test_expand_is_inside_generate():
    """Make sure expand_segments_to_samples is in the generate() coroutine."""
    tree = ast.parse(_source())
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "generate"
        ):
            src = ast.unparse(node)
            assert "expand_segments_to_samples" in src
            return
    raise AssertionError("blackbox_dispatch.generate function not found")
