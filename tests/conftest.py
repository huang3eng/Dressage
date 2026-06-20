"""Shared pytest cleanup hooks."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEST_OUTPUT_DIRS = ("infra", "log")


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001
    del session, exitstatus
    for dirname in _TEST_OUTPUT_DIRS:
        _remove_untracked_test_output_dir(_REPO_ROOT / dirname)


def _remove_untracked_test_output_dir(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.parent != _REPO_ROOT or path.name not in _TEST_OUTPUT_DIRS:
        return
    if _has_tracked_files(path):
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _has_tracked_files(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", str(path.relative_to(_REPO_ROOT))],
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, ValueError):
        return True
    return bool(result.stdout.strip())
