from __future__ import annotations

from blackbox_server.adapters.base import BackendAdapter
from blackbox_server.adapters.openclaw import OpenClawAdapter
from blackbox_server.adapters.opencode import OpencodeAdapter
from blackbox_server.core.errors import ApiError


IMPLEMENTED_BACKENDS = ["opencode", "openclaw"]
KNOWN_BACKENDS = ["opencode", "openclaw", "claude_code"]


def create_adapter(blackbox_type: str) -> BackendAdapter:
    if blackbox_type == "opencode":
        return OpencodeAdapter()
    if blackbox_type == "openclaw":
        return OpenClawAdapter()
    if blackbox_type == "claude_code":
        raise ApiError(
            status_code=501,
            error="not_implemented",
            message="Backend claude_code is reserved but not implemented in phase 1.",
            details={"blackbox_type": blackbox_type},
        )
    raise ApiError(
        status_code=400,
        error="request_error",
        message=f"Unsupported blackbox_type: {blackbox_type}",
        details={"blackbox_type": blackbox_type},
    )
