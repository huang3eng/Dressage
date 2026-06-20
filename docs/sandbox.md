# Sandbox Backends

**Pluggable Isolation for Safe Agent Execution**

[← Back to Main README](../README.md) · [Overview](#-overview) · [Provider Interface](#-provider-interface) · [Bubblewrap](#️-local-bubblewrap-local_bwrap) · [E2B](#️-e2b-e2b) · [Runtime Matrix](#-runtime-matrix) · [Artifact Archiving](#️-session-artifact-archiving)

## 📖 Overview

Sandbox backends provide **isolated execution environments** for agent rollouts. The same rollout code can run locally (bubblewrap) or remotely (E2B) by swapping a single environment variable. The sandbox layer handles **where** code runs, while the [Paddock](./paddock.md) handles **what** to do.

This separation is fundamental to Dressage's design philosophy: agent interaction semantics and execution isolation are orthogonal concerns. You can run the same blackbox agent on local bubblewrap during development and E2B cloud sandboxes for scaling — without changing a single line of agent or rollout code.

```text
Paddock (what-to-do)
    │
    ▼
SandboxProvider (where-it-runs)
    ├── local_bwrap   → Ray-managed bubblewrap pool
    │                   Unprivileged namespace isolation
    │                   Supervisor health monitoring
    │                   Automatic slot recovery
    │
    └── e2b           → E2B cloud sandboxes
                        Elastic, API-driven provisioning
                        Template-based configuration
                        No local infrastructure needed
```

## ✨ Key Features

- **Pluggable Architecture** — Switch providers via `DRESSAGE_SANDBOX_PROVIDER` environment variable. No code changes needed — the factory pattern handles all wiring. The paddock receives a `SandboxProvider` and calls `create/terminate`, agnostic to the backend.
- **Local Pool Management** — Ray-managed bubblewrap slot pools with automatic lifecycle management. Slots are pre-created at startup, health-checked by a supervisor actor, and automatically recovered on failure. Supports concurrent rollout across multiple workers.
- **Cloud Native** — E2B provider launches sandboxes from pre-built templates via API. Scales elastically without managing local infrastructure. Template-based configuration allows pre-installing tools, dependencies, and BlackboxServer.
- **Two Pool Modes** — `blackbox` mode provisions full BlackboxServer endpoints (for blackbox agent rollouts). `command_only` mode provides bare shell/file access (for whitebox tool execution). The factory validates mode compatibility at startup.
- **Safety Checks** — Mismatched paddock/pool configurations raise clear errors before creating any sandbox lease. For example, a blackbox paddock with a `command_only` pool is rejected immediately with an actionable error message.

## 🧱 Provider Interface

All sandbox providers implement the `SandboxProvider` interface. The contract is provider-neutral: paddocks describe the sandbox they need with a `SandboxSpec`, receive a `SandboxLease`, and then use endpoint, command, and file capabilities exposed by the provider.

```python
class SandboxProvider:
    name: str

    async def create(self, spec: SandboxSpec) -> SandboxLease:
        """Create a sandbox lease for one trajectory."""
        ...

    async def terminate(self, lease: SandboxLease | str) -> dict:
        """Terminate a sandbox lease."""
        ...

    async def get_public_url(
        self,
        lease: SandboxLease,
        *,
        port: int,
        service_name: str | None = None,
    ) -> SandboxEndpoint:
        """Return a public endpoint for a service/port in the sandbox."""
        ...

    async def run_command(
        self,
        lease: SandboxLease,
        command: str | list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        stdin: str | bytes | None = None,
    ) -> CommandResult:
        """Execute a command in the sandbox."""
        ...

    async def read_file(
        self,
        lease: SandboxLease,
        path: str,
        *,
        encoding: str | None = "utf-8",
        max_bytes: int | None = None,
    ) -> str | bytes:
        """Read a file from the sandbox."""
        ...

    async def write_file(
        self,
        lease: SandboxLease,
        path: str,
        content: str | bytes,
        *,
        encoding: str | None = "utf-8",
        append: bool = False,
    ) -> dict:
        """Write a file in the sandbox."""
        ...
```

Created via factory — the factory reads environment variables and returns the correct provider:

```python
from dressage.sandbox.factory import create_sandbox_provider_from_env
from dressage.sandbox.types import SandboxServiceSpec, SandboxSpec

provider = create_sandbox_provider_from_env()
lease = await provider.create(
    SandboxSpec(
        trajectory_id="traj-001",
        services=(SandboxServiceSpec(name="blackbox", port=31000),),
    )
)
# ... use the sandbox ...
await provider.terminate(lease)
```

### Core Data Types

The provider API is built around these provider-neutral data types:

 | Type | Description |
 | :------ | :------------ | 
 | `SandboxSpec` | Creation request containing `trajectory_id`, optional `env_type` / `env_args`, requested services, timeout, metadata, and environment variables. |
 | `SandboxLease` | Provider-neutral lease returned by `create()`. Contains provider name, sandbox ID, service endpoints, capabilities, metadata, and raw provider state. |
 | `SandboxEndpoint` | Public endpoint URL plus optional provider-specific forwarding headers. |
 | `CommandResult` | Result of `run_command()`, including stdout, stderr, return code, timeout status, and metadata. |

## 🖥️ Local Bubblewrap (`local_bwrap`)

Ray-managed pool of [bubblewrap](https://github.com/containers/bubblewrap)-isolated slots. Each slot is an independent filesystem namespace with restricted capabilities — unprivileged sandboxing that works without root access (on supported kernels). This is the recommended provider for local development and testing.

### Architecture

The bubblewrap pool is managed by a trio of Ray actors that handle provisioning, health checking, and cleanup:

```text
Ray Head
    │
    ├── BwrapManager (named Ray actor)
    │       │  Manages the pool of sandbox slots
    │       │  Handles provider create/terminate requests
    │       │  Tracks slot assignments per trajectory
    │       │
    │       ├── slot_0: BwrapSlot
    │       │       ├── bwrap namespace (isolated filesystem)
    │       │       ├── BlackboxServer process (blackbox mode)
    │       │       └── work directory + home + tmp
    │       ├── slot_1: BwrapSlot
    │       ├── slot_2: BwrapSlot
    │       └── ...
    │
    └── BwrapSupervisor (background actor)
            │  Monitors slot health on configurable interval
            │  Detects crashed BlackboxServer processes
            │  Restarts failed slots automatically
            └── Reports health metrics for observability
```

### Pool Modes

The pool operates in one of two modes, determined by `DRESSAGE_LOCAL_BWRAP_POOL_MODE`:

 | Mode | What's Inside Each Slot | Use Case | 
 | :----- | :------------------------ | :--------- | 
 | `blackbox` | Full BlackboxServer process with public HTTP endpoint. The server manages the backend agent (opencode/openclaw), in-process LLM proxy, and session lifecycle. Health-checked by supervisor. | Blackbox agent rollouts — the paddock communicates with the agent entirely via HTTP. | 
 | `command_only` | Bare slot directory with shell/file access functions. No HTTP endpoint, no agent process. Commands execute directly in the bubblewrap namespace. | Whitebox tool execution — the paddock uses `tool_call` to run shell commands and file operations. | 

### Slot Lifecycle

```text
1️⃣  BwrapManager.start_pool()
    └── Creates N slots, each with its own bwrap namespace
    └── In blackbox mode: starts BlackboxServer per slot
    └── Supervisor begins health monitoring

2️⃣  provider.create(SandboxSpec)
    └── Manager assigns an available slot to the trajectory
    └── Returns SandboxLease with service endpoints and capabilities

3️⃣  (rollout uses the sandbox)

4️⃣  provider.terminate(lease)
    └── Manager resets the slot (cleans filesystem, restarts server if needed)
    └── Slot becomes available for next trajectory

5️⃣  BwrapManager.stop_pool()
    └── Stops all BlackboxServer processes
    └── Tears down bwrap namespaces
    └── Supervisor stops
```

<details>
<summary><b> Configuration</b></summary>
<br>

```bash
# Provider selection
DRESSAGE_SANDBOX_PROVIDER=local_bwrap

# Pool mode (must match paddock mode)
DRESSAGE_LOCAL_BWRAP_POOL_MODE=blackbox      # for blackbox agents
DRESSAGE_LOCAL_BWRAP_POOL_MODE=command_only  # for whitebox tools

# Manager naming (for Ray actor discovery)
DRESSAGE_LOCAL_BWRAP_MANAGER_NAME=dressage_local_bwrap_manager

# Advanced note: local_bwrap always uses the Ray-backed pool.
# DRESSAGE_LOCAL_BWRAP_BACKEND is valid only when unset or set to ray_pool.
```

</details>

### CLI Commands

Manage the bubblewrap pool cluster with these CLI entry points:

 | Command | Description | 
 | :-------- | :------------ | 
 | `dressage-local-bwrap-start` | Start the bwrap pool cluster. Creates the Ray actors, provisions slots, starts BlackboxServer processes (in blackbox mode), and begins health monitoring. | 
 | `dressage-local-bwrap-status` | Check pool status: total slots, available/busy/failed counts, per-slot health, supervisor state. Useful for debugging rollout issues. | 
 | `dressage-local-bwrap-stop` | Stop all slots, tear down bwrap namespaces, destroy Ray actors. Clean shutdown of the entire pool. | 
 | `dressage-local-blackbox-start` / `dressage-blackbox-start` | Start the local BlackboxServer cluster management path. |
 | `dressage-local-blackbox-status` / `dressage-blackbox-status` | Check local blackbox cluster health. |
 | `dressage-local-blackbox-stop` / `dressage-blackbox-stop` | Stop local blackbox cluster resources. |

<details>
<summary><b> Key Modules</b></summary>
<br>

 | Module | Role | 
 | :------- | :----- | 
 | `local/bwrap/manager.py` | Ray actor managing the slot pool. Handles create/terminate leases, tracks assignments, provisions slots at startup. |
 | `local/bwrap/slot.py` | Individual slot representation. Encapsulates bwrap namespace, BlackboxServer process, and filesystem layout. | 
 | `local/bwrap/runner.py` | Slot runner — starts/stops BlackboxServer, executes commands, manages slot lifecycle. | 
 | `local/bwrap/supervisor.py` | Health monitoring actor. Periodically checks slot health via HTTP health checks, restarts failed slots. | 
 | `scripts/` | CLI entry points: `start_local_bwrap.py`, `local_bwrap_status.py`, `stop_local_bwrap.py`. | 

</details>

## ☁️ E2B (`e2b`)

Remote sandboxes from [E2B](https://e2b.dev). Provides elastic, cloud-native isolation without managing local infrastructure. Sandboxes are launched from pre-built **templates** that define the base image, installed tools, and any startup behavior needed by the recipe.

### How It Works

```text
Rollout
    │
    ├── create(SandboxSpec)
    │       ├── E2B API → launch sandbox from template
    │       ├── Wait for sandbox to become ready
    │       ├── Optionally run metadata/env_args sandbox_cmd
    │       ├── Resolve requested SandboxSpec.services with get_host(port)
    │       └── Return SandboxLease with endpoint/capability info
    │
    ├── (rollout uses the sandbox — same as local)
    │
    └── terminate(lease)
            ├── E2B API → terminate sandbox
            └── Resources freed immediately
```

### Template Requirements

E2B sandboxes are created from templates/images selected by `sample.metadata["sandbox_image"]` or `DRESSAGE_SANDBOX_DEFAULT_IMAGE`:

 | Paddock Mode | Template Must Include | Why | 
 | :------------- | :--------------------- | :---- | 
 | Blackbox | A reachable BlackboxServer on the requested port, either started by the template or by `metadata["sandbox_cmd"]` / `env_args["sandbox_cmd"]` after creation | The paddock communicates with the agent via HTTP and verifies the endpoint with a health check |
 | Whitebox | No special services | Shell/file access is provided by E2B's built-in sandbox API — no server needed | 

> [!NOTE]
> `E2BSandboxProvider.create()` uses the requested `SandboxSpec.services` and the E2B SDK's `get_host(port)` to expose endpoints. It does not read services from the template metadata.

<details>
<summary><b> Configuration</b></summary>
<br>

```bash
DRESSAGE_SANDBOX_PROVIDER=e2b

# E2B API key (required)
DRESSAGE_E2B_API_KEY=your-api-key-here

# Template/image selection
DRESSAGE_SANDBOX_DEFAULT_IMAGE=dressage-blackbox
DRESSAGE_E2B_BLACKBOX_PORT=31000
DRESSAGE_E2B_TIMEOUT_SEC=3600

# Blackbox template: start BlackboxServer itself, or provide sandbox_cmd to start it
# Whitebox template: bare environment with shell access
```

</details>

### When to Use E2B

- **No local GPU infrastructure** — sandboxes run in E2B's cloud
- **Elastic scaling** — spin up/down sandboxes on demand
- **Quick start** — no bubblewrap installation or kernel configuration needed
- **Template versioning** — pin sandbox environments for reproducibility

## 📊 Runtime Matrix

This matrix shows all valid combinations of paddock mode, sandbox provider, and pool mode:

 | Paddock Mode | Provider | Pool Mode | Services Exposed | Notes | 
 | :---: | :---: | :---: | :--- | :--- | 
 | blackbox | `local_bwrap` | `blackbox` | BlackboxServer HTTP endpoint | Full local development setup | 
 | blackbox | `e2b` | — | BlackboxServer in E2B template | Cloud-native scaling | 
 | whitebox | `local_bwrap` | `command_only` | Shell commands + file I/O | Local tool execution | 
 | whitebox | `e2b` | — | Shell commands + file I/O via E2B API | Cloud tool execution | 

> [!CAUTION]
> **Mismatch detection**: The factory validates that paddock mode and pool mode are compatible at startup. Using a blackbox paddock with a `command_only` pool (or vice versa) raises a clear `ValueError` before any sandbox lease is created. This prevents silent failures where the agent would start but fail on the first tool call.

## 🗄️ Session Artifact Archiving

For blackbox sandbox slots, Dressage supports **session artifact archiving** for forensic debugging. When a session completes or fails, the slot's filesystem contents (`home`, `work`, `runtime`, `tmp` directories) can be preserved with configurable TTL and max-per-slot limits.

```bash
# Enable session artifact preservation
DRESSAGE_BLACKBOX_PRESERVE_SESSION_ARTIFACTS=1
DRESSAGE_BLACKBOX_SESSION_ARCHIVE_DIRS=home,work,runtime,tmp
DRESSAGE_BLACKBOX_SESSION_ARCHIVE_MAX_PER_SLOT=20
DRESSAGE_BLACKBOX_SESSION_ARCHIVE_TTL_SEC=86400
```

### What Gets Archived

 | Directory | Contents | 
 | :---------- | :--------- | 
 | `home/` | Agent's home directory — config files, caches | 
 | `work/` | Working directory — cloned repos, modified files, build artifacts | 
 | `runtime/` | BlackboxServer runtime files — logs, session data | 
 | `tmp/` | Temporary files created during execution | 

> [!TIP]
> Artifact archiving is invaluable for debugging failed rollouts. You can inspect the exact filesystem state the agent left behind — including modified source files, build outputs, error logs, and partial work. Archived artifacts are stored with timestamps and can be cleaned up based on TTL policy.

## 📁 Package Structure

```text
dressage/sandbox/
├── provider.py                      # SandboxProvider abstract interface
├── factory.py                       # create_sandbox_provider_from_env()
├── types.py                         # SandboxSpec, SandboxLease, endpoints, command results
├── local/
│   └── bwrap/
│       ├── manager.py               # Ray-managed slot pool actor
│       ├── slot.py                  # Individual slot representation
│       ├── runner.py                # Slot runner (start/stop/execute)
│       ├── provider.py              # BwrapSandboxProvider
│       └── supervisor.py            # Health monitoring actor
├── remote/
│   └── e2b/
│       └── provider.py              # E2BSandboxProvider
└── scripts/
    ├── start_local_bwrap.py         # CLI: dressage-local-bwrap-start
    ├── local_bwrap_status.py        # CLI: dressage-local-bwrap-status
    ├── stop_local_bwrap.py          # CLI: dressage-local-bwrap-stop
    ├── start_blackbox_cluster.py    # BlackboxServer cluster management
    ├── blackbox_cluster_status.py   # Cluster health check
    └── stop_blackbox_cluster.py     # Cluster teardown
```

## 🔗 Integration Points

 | Component | Relationship | 
 | :---------- | :------------ | 
 | [Paddock](./paddock.md) | Paddock uses provider `create()` / `terminate()` for lease management. The paddock is the only consumer of sandbox providers. |
 | [BlackboxServer](./blackbox-server.md) | In `blackbox` pool mode, each slot runs a BlackboxServer process. The paddock communicates with the agent through this server. | 
 | [Rollout](./rollout.md) | Rollout hooks trigger sandbox acquisition indirectly via paddock `init/terminate`. | 
 | [Training](./training.md) | Sandbox artifacts (preserved sessions) provide debugging data for training failures. | 

---

[← Paddock](./paddock.md) · [Back to Main README](../README.md) · [Next: BlackboxServer →](./blackbox-server.md)
