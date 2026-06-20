"""ALFWorld reward function for whitebox agents.

Pure binary signal aligned with Embodied-Planner-R1 (arXiv 2506.23127):
- sample.metadata["task_success"] == True  → 1.0
- otherwise                                → 0.0

The paper explicitly avoids format/length shaping to prevent reward hacking.
Metadata-driven: the agent stamps task_success during rollout, so this
function never re-executes the environment.
"""

from __future__ import annotations

from typing import Any

from dressage.reward import register_reward


@register_reward("alfworld")
def alfworld(sample: Any, *, args: Any = None, **kwargs: Any) -> float:
    metadata = getattr(sample, "metadata", None) or {}
    return 1.0 if metadata.get("task_success") else 0.0
