from __future__ import annotations

import json
from pathlib import Path

from dressage.sandbox.local.bwrap.slot import SlotConfig


def test_reset_runtime_dirs_can_archive_active_dirs_by_session_id(tmp_path):
    config = SlotConfig(
        slot_id=2,
        port=31002,
        bind_host="0.0.0.0",
        advertise_host="127.0.0.1",
        base_dir=tmp_path,
    )
    config.ensure_dirs()
    (config.home_dir / "home.txt").write_text("home")
    (config.work_dir / "work.txt").write_text("work")
    (config.runtime_dir / "runtime.txt").write_text("runtime")
    (config.tmp_dir / "tmp.txt").write_text("tmp")

    archive = config.reset_runtime_dirs(
        preserve_artifacts=True,
        session_id="bbs-session-001",
        lease_id="lease-1",
        generation=7,
        reason="test-release",
    )

    assert archive == config.archive_dir / "bbs-session-001"
    assert (archive / "home" / "home.txt").read_text() == "home"
    assert (archive / "work" / "work.txt").read_text() == "work"
    assert (archive / "runtime" / "runtime.txt").read_text() == "runtime"
    assert (archive / "tmp" / "tmp.txt").read_text() == "tmp"
    assert not any(config.home_dir.iterdir())
    assert not any(config.work_dir.iterdir())
    metadata = json.loads((archive / "metadata.json").read_text())
    assert metadata["session_id"] == "bbs-session-001"
    assert metadata["bound_session_id"] == "bbs-session-001"
    assert metadata["generation"] == 7
    assert metadata["lease_id"] == "lease-1"


def test_archive_collision_falls_back_without_using_generation_by_default(tmp_path):
    config = SlotConfig(
        slot_id=0,
        port=31000,
        bind_host="0.0.0.0",
        advertise_host="127.0.0.1",
        base_dir=tmp_path,
    )
    config.ensure_dirs()
    (config.home_dir / "first.txt").write_text("first")
    first = config.reset_runtime_dirs(
        preserve_artifacts=True,
        session_id="same-session",
        generation=1,
    )
    (config.home_dir / "second.txt").write_text("second")
    second = config.reset_runtime_dirs(
        preserve_artifacts=True,
        session_id="same-session",
        generation=2,
    )

    assert first == config.archive_dir / "same-session"
    assert second is not None
    assert second.name.startswith("same-session-collision-")
    assert "gen" not in first.name
    assert (first / "home" / "first.txt").exists()
    assert (second / "home" / "second.txt").exists()
