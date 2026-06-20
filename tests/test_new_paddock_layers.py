from __future__ import annotations

import asyncio
from typing import Any

import httpx

from dressage.paddock.blackbox.paddock import BlackboxAgentPaddock
from dressage.paddock.whitebox.paddock import WhiteboxToolPaddock
from dressage.sandbox.types import CommandResult, SandboxEndpoint, SandboxLease, SandboxSpec


class FakeProvider:
    name = "local_bwrap"

    def __init__(self) -> None:
        self.created: list[SandboxSpec] = []
        self.terminated: list[SandboxLease] = []
        self.commands: list[tuple[SandboxLease, str | list[str], dict[str, Any]]] = []
        self.files: dict[str, str] = {}

    async def create(self, spec: SandboxSpec) -> SandboxLease:
        self.created.append(spec)
        paddock_mode = spec.metadata.get("paddock_mode")
        lease = SandboxLease(
            trajectory_id=spec.trajectory_id,
            provider=self.name,
            sandbox_id=f"lease-{spec.trajectory_id}",
            capabilities=(
                {"command", "file", "public_url"}
                if paddock_mode == "blackbox"
                else {"command", "file"}
            ),
            metadata={"node_ip": "10.0.0.12", "port": 31000},
        )
        if paddock_mode == "blackbox":
            lease.endpoints["blackbox"] = SandboxEndpoint(
                url="http://sandbox.test",
                headers={"x-test": "1"},
            )
        return lease

    async def terminate(self, lease):
        self.terminated.append(lease)
        return {"terminated": True}

    async def get_public_url(self, lease, *, port, service_name=None):
        return lease.endpoints[service_name or "blackbox"]

    async def run_command(self, lease, command, **kwargs):
        self.commands.append((lease, command, kwargs))
        return CommandResult(cmd=command, stdout="ran\n", stderr="", returncode=0)

    async def read_file(self, lease, path, *, encoding="utf-8", max_bytes=None):
        value = self.files.get(path, "")
        return value if max_bytes is None else value[:max_bytes]

    async def write_file(self, lease, path, content, *, encoding="utf-8", append=False):
        text = content.decode(encoding or "utf-8") if isinstance(content, bytes) else str(content)
        self.files[path] = self.files.get(path, "") + text if append else text
        return {"path": path, "bytes": len(text)}


def test_blackbox_agent_paddock_uses_provider_and_blackbox_client():
    asyncio.run(_run_blackbox_agent_paddock_uses_provider_and_blackbox_client())


async def _run_blackbox_agent_paddock_uses_provider_and_blackbox_client():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/rollout/register":
            assert request.headers["x-test"] == "1"
            body = request.content.decode()
            assert "bound_instance_id" in body
            assert "runtime_root" not in body
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/v1/sessions/traj-1/messages":
            return httpx.Response(200, json={"response": "done"})
        if request.url.path == "/v1/sessions/traj-1/execute_cmd":
            return httpx.Response(200, json={"returncode": 0, "stdout": "Python\n"})
        raise AssertionError(f"unexpected path {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = FakeProvider()
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        blackbox_client=None,
        wait_health=False,
    )
    # Replace the internally created client with one backed by MockTransport.
    from dressage.paddock.blackbox.client import BlackboxServerClient

    paddock._client = BlackboxServerClient(client=client)

    state = await paddock.init("traj-1")
    assert state.sandbox_url == "http://sandbox.test"
    assert provider.created[0].services[0].name == "blackbox"
    assert provider.created[0].metadata == {"paddock_mode": "blackbox"}

    assert await paddock.register_agent(state, instance_id="inst", session_id="traj-1") == {"ok": True}
    assert (await paddock.call_agent(state, session_id="traj-1", messages=[]))["response"] == "done"
    assert (await paddock.execute_cmd(state, session_id="traj-1", cmd="python -V"))["returncode"] == 0

    await client.aclose()


def test_whitebox_tool_paddock_maps_tools_to_provider():
    asyncio.run(_run_whitebox_tool_paddock_maps_tools_to_provider())


async def _run_whitebox_tool_paddock_maps_tools_to_provider():
    provider = FakeProvider()
    paddock = WhiteboxToolPaddock(provider=provider)
    await paddock.init("traj-1")

    assert provider.created[0].services == ()
    assert provider.created[0].metadata == {"paddock_mode": "whitebox"}

    text, meta = await paddock.tool_call("traj-1", "shell.exec", {"cmd": "echo hi"})
    assert text == "ran\n"
    assert meta["returncode"] == 0

    await paddock.tool_call("traj-1", "file.write", {"path": "/workspace/a.txt", "content": "abc"})
    text, meta = await paddock.tool_call("traj-1", "file.read", {"path": "/workspace/a.txt"})
    assert text == "abc"
    assert meta["chars"] == 3
