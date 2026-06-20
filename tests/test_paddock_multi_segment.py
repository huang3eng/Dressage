"""Tests for the paddock-multi-segment pipeline.

Tests cover:
  - mark_aborted_no_grad / compute_multi_segment_metrics (multi_segment.py)
  - reward_post_process anchor broadcast
  - log_helpers.compute_trajectory_mean_raw_reward
  - log_rollout.log_rollout_data partition alignment

No slime/megatron/ray dependencies — all slime types are faked.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fake Sample (mirrors slime.utils.types.Sample fields used by dressage)
# ---------------------------------------------------------------------------

@dataclass
class FakeSample:
    class Status:
        COMPLETED = "completed"
        TRUNCATED = "truncated"

    index: int = 0
    tokens: list[int] = field(default_factory=list)
    response_length: int = 0
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    reward: float | None = None
    response: str = ""
    status: str = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)
    train_metadata: dict[str, Any] | None = None
    session_id: str | None = None
    group_index: int | None = None
    rollout_id: int | None = None
    remove_sample: bool = False
    label: str | None = None


def _make_sample(
    index: int,
    instance_id: str,
    parent_traj_id: str,
    segment_index: int = 0,
    loss_mask: list[int] | None = None,
    reward: float | None = None,
    group_index: int = 0,
    remove_sample: bool = False,
) -> FakeSample:
    mask = loss_mask or [1, 1, 1]
    return FakeSample(
        index=index,
        tokens=list(range(10 + len(mask))),
        response_length=len(mask),
        loss_mask=mask,
        rollout_log_probs=[-0.5] * len(mask),
        reward=reward,
        metadata={
            "instance_id": instance_id,
            "parent_traj_id": parent_traj_id,
            "segment_index": segment_index,
            "session_id": f"session-{index}",
        },
        group_index=group_index,
        remove_sample=remove_sample,
    )


def _make_args(**overrides):
    defaults = {
        "advantage_estimator": "grpo",
        "rewards_normalization": True,
        "grpo_std_normalization": False,
        "n_samples_per_prompt": 2,
        "global_batch_size": 4,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ========================================================================
# Tests for reward_post_process (full function)
# ========================================================================


class TestRewardPostProcess:
    """Test the complete reward_post_process with multi-segment."""

    def test_anchor_broadcast(self):
        from dressage.training.reward_post_process import reward_post_process

        # Trajectory with 2 segments: seg0 (reward=0.0), seg1 (reward=1.0, anchor)
        samples = [
            _make_sample(0, "A", "traj-1", segment_index=0, reward=0.0, group_index=0),
            _make_sample(1, "A", "traj-1", segment_index=1, reward=1.0, group_index=0),
        ]
        args = _make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            n_samples_per_prompt=1,
            global_batch_size=2,
        )
        raw_rewards, rewards = reward_post_process(args, samples)

        # raw_rewards NOT broadcast: [0.0, 1.0]
        assert raw_rewards == [0.0, 1.0]
        # rewards (advantage) broadcast from anchor to all segments
        assert rewards[0] == rewards[1]




# ========================================================================
# Tests for log_rollout helpers
# ========================================================================


class TestLogRolloutHelpers:

    def test_partition_alignment_with_raw_reward(self):
        from dressage.training.log_helpers import compute_trajectory_mean_raw_reward

        # Simulate: 4 samples total, this rank has partition [1, 3]
        # raw_reward is broadcast (full): [0.0, 1.0, 0.0, 0.5]
        # metadata is sliced (per-rank): 2 items for partition [1, 3]
        metadata = [
            {"parent_traj_id": "traj-A", "segment_index": 0},
            {"parent_traj_id": "traj-B", "segment_index": 0},
        ]
        raw_reward_full = [0.0, 1.0, 0.0, 0.5]
        partition = [1, 3]

        raw_reward = [raw_reward_full[j] for j in partition]
        parent_traj_ids = [m["parent_traj_id"] for m in metadata]
        segment_indices = [m["segment_index"] for m in metadata]

        result = compute_trajectory_mean_raw_reward(
            parent_traj_ids, raw_reward, segment_indices
        )
        # traj-A anchor raw_reward=1.0, traj-B anchor raw_reward=0.5
        # mean = (1.0 + 0.5) / 2 = 0.75
        assert result == pytest.approx(0.75)


# ========================================================================
# Tests for multi_segment.py
# ========================================================================


class TestMarkAbortedNoGrad:
    def test_sets_required_fields(self):
        from dressage.rollout.multi_segment import mark_aborted_no_grad

        sample = FakeSample(index=5, session_id="old-session", metadata={})
        mark_aborted_no_grad(sample, session_id="s1", instance_id="i1")

        assert sample.remove_sample is True
        assert sample.metadata["parent_traj_id"] == "s1"
        assert sample.metadata["instance_id"] == "i1"
        assert sample.metadata["last_failed_session_id"] == "s1"
        assert sample.session_id is None
        assert "session_id" not in sample.metadata

    def test_no_session_id(self):
        from dressage.rollout.multi_segment import mark_aborted_no_grad

        sample = FakeSample(index=5, metadata={})
        mark_aborted_no_grad(sample, session_id=None, instance_id="i1")

        assert sample.metadata["parent_traj_id"] == "i1"
        assert sample.metadata["instance_id"] == "i1"

    def test_preserves_existing_parent_traj_id(self):
        from dressage.rollout.multi_segment import mark_aborted_no_grad

        sample = FakeSample(index=5, metadata={"parent_traj_id": "existing"})
        mark_aborted_no_grad(sample, session_id="s1", instance_id="i1")

        assert sample.metadata["parent_traj_id"] == "existing"


class TestComputeMultiSegmentMetrics:
    def test_basic(self):
        from dressage.rollout.multi_segment import compute_multi_segment_metrics

        samples = [
            _make_sample(0, "A", "traj-1", segment_index=0),
            _make_sample(1, "A", "traj-1", segment_index=1),
            _make_sample(2, "B", "traj-2", segment_index=0),
        ]
        metrics = compute_multi_segment_metrics(samples)

        assert metrics["rollout/num_trajectories"] == 2.0
        assert metrics["rollout/num_segments"] == 3.0
        assert metrics["rollout/segments_per_trajectory_mean"] == 1.5
        assert metrics["rollout/segments_per_trajectory_max"] == 2.0

    def test_dead_excluded(self):
        from dressage.rollout.multi_segment import compute_multi_segment_metrics

        samples = [
            _make_sample(0, "A", "traj-1", segment_index=0, remove_sample=True),
        ]
        metrics = compute_multi_segment_metrics(samples)
        assert metrics == {}


# ========================================================================
# Tests for log_helpers
# ========================================================================


class TestComputeTrajectoryMeanRawReward:
    def test_single_segment_trajectories(self):
        from dressage.training.log_helpers import compute_trajectory_mean_raw_reward

        result = compute_trajectory_mean_raw_reward(
            parent_traj_ids=["t1", "t2", "t3"],
            raw_rewards=[1.0, 0.0, 1.0],
            segment_indices=[0, 0, 0],
        )
        assert result == pytest.approx(2.0 / 3)

    def test_multi_segment_uses_anchor(self):
        from dressage.training.log_helpers import compute_trajectory_mean_raw_reward

        # traj-1 has 2 segments: seg0 raw_reward=0.0, seg1 (anchor) raw_reward=1.0
        # traj-2 has 1 segment: raw_reward=0.5
        result = compute_trajectory_mean_raw_reward(
            parent_traj_ids=["t1", "t1", "t2"],
            raw_rewards=[0.0, 1.0, 0.5],
            segment_indices=[0, 1, 0],
        )
        # anchor of t1 = seg1 = 1.0, anchor of t2 = seg0 = 0.5
        # mean = (1.0 + 0.5) / 2 = 0.75
        assert result == pytest.approx(0.75)

    def test_empty_returns_none(self):
        from dressage.training.log_helpers import compute_trajectory_mean_raw_reward

        result = compute_trajectory_mean_raw_reward([], [], [])
        assert result is None

