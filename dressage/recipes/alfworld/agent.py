"""ALFWorld whitebox agent — standalone generate function.

Follows the same pattern as ``whitebox_loop.generate``: standalone
``async def generate(args, sample, sampling_params, evaluation)``
with session management, drain, and segment expansion reusing helpers
from shared rollout modules.

Wiring:
  - ``--custom-generate-function-path dressage.recipes.alfworld.agent.generate``
  - ``--dressage-multi-segment``
  - Proxy: ``--trajectory-build-mode concat``
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any

from dressage.config import proxy_local_url
from dressage.proxy.proxy_client import ProxyClient
from dressage.rollout import multi_segment
from dressage.rollout.artifacts.samples import (
    instance_id as _instance_id,
    set_status as _set_status,
)
from dressage.rollout.artifacts.writer import DEFAULT_WRITER

logger = logging.getLogger(__name__)

_PROXY_CLIENT: ProxyClient | None = None


def _get_proxy_client() -> ProxyClient:
    global _PROXY_CLIENT
    if _PROXY_CLIENT is None:
        _PROXY_CLIENT = ProxyClient(os.environ.get("DRESSAGE_PROXY_URL") or proxy_local_url())
    return _PROXY_CLIENT


def _session_id(sample: Any) -> str:
    session_id = getattr(sample, "session_id", None)
    if session_id is None:
        session_id = str(uuid.uuid4())
    session_id = str(session_id)
    if not session_id.startswith("alf-"):
        session_id = f"alf-{session_id}"
    sample.session_id = session_id
    return session_id


# ── Prompts ─────────────────────────────────────────────────────────────


ALFWORLD_SYSTEM_PROMPT = (
    "You are acting in ALFWorld TextWorld. Each turn, choose exactly one "
    "command from the admissible commands provided in the latest tool "
    "response. Reason briefly inside <think></think> if useful, then call "
    "the `env_step` tool with that exact command. Follow ALFWorld TextWorld "
    "command style such as `go to dresser 1`, `take mug 1 from cabinet 3`, "
    "`use desklamp 1`. Do not output a final natural-language answer."
)

ALFWORLD_INITIAL_USER_TEMPLATE = """### Task
{task_text}

### Initial Observation
{observation}

### Admissible Commands
{admissible_commands}

### Output Format
<think>
[Your brief reasoning about the next command.]
</think>
<tool_call>
{{"name": "env_step", "arguments": {{"command": "[one admissible command]"}}}}
</tool_call>
"""

ALFWORLD_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "env_step",
            "description": (
                "Execute one ALFWorld TextWorld command and return the next "
                "official observation plus the new admissible commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "A single ALFWorld TextWorld command such as "
                            "`go to dresser 1`, `open cabinet 3`, "
                            "`take mug 1 from cabinet 3`, `use desklamp 1`. "
                            "Must exactly match one currently admissible "
                            "command."
                        ),
                    }
                },
                "required": ["command"],
            },
        },
    }
]


# ── Config ──────────────────────────────────────────────────────────────


def _load_config() -> dict[str, Any]:
    return {
        "max_steps": int(os.environ.get("ALFWORLD_MAX_STEPS", "50")),
        "max_episode_steps": int(os.environ.get("ALFWORLD_MAX_EPISODE_STEPS", "50")),
    }


# ── Helpers ─────────────────────────────────────────────────────────────


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _parse_json_loose(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    if len(raw) > 8192 or "__" in raw:
        return None
    try:
        import ast
        return ast.literal_eval(raw)
    except Exception:
        return None


def _extract_structured_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    tcs = message.get("tool_calls")
    if not isinstance(tcs, list):
        return []
    return [tc for tc in tcs if isinstance(tc, dict)]


def _extract_assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if choices:
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            return msg
    return {"role": "assistant", "content": ""}


def _extract_assistant_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def _extract_finish_reason(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return "stop"
    return choices[0].get("finish_reason") or "stop"


def _first_command(
    content: str,
    structured: list[dict[str, Any]],
) -> tuple[str | None, str | None, str | None]:
    for tc in structured:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name")
        if name != "env_step":
            return tc.get("id"), None, f"expected env_step, got {name!r}"
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _parse_json_loose(args)
        if not isinstance(args, dict):
            return tc.get("id"), None, "arguments not a JSON object"
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return tc.get("id"), None, "missing command argument"
        return tc.get("id"), command.strip(), None

    blocks = _TOOL_CALL_BLOCK_RE.findall(content or "")
    if not blocks:
        return None, None, "missing <tool_call> block"
    for raw in blocks:
        obj = _parse_json_loose(raw)
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if name != "env_step":
            return None, None, f"expected env_step, got {name!r}"
        args = obj.get("arguments")
        if isinstance(args, str):
            args = _parse_json_loose(args)
        if not isinstance(args, dict):
            return None, None, "arguments not a JSON object"
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return None, None, "missing command argument"
        return None, command.strip(), None
    return None, None, "could not parse <tool_call> JSON"


def _make_env(game_file: str, max_episode_steps: int):
    import textworld
    import textworld.gym
    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        description=True,
        inventory=True,
    )
    env_id = textworld.gym.register_game(
        game_file,
        request_infos=request_infos,
        max_episode_steps=max_episode_steps,
    )
    return textworld.gym.make(env_id)


def _format_admissible(commands: list[str]) -> str:
    if not commands:
        return "None"
    return "\n".join(f"- {cmd}" for cmd in commands if cmd != "help")


def _extract_task_text(observation: str, fallback: str) -> str:
    marker = "Your task is to:"
    if observation and marker in observation:
        task = observation.split(marker, 1)[1].strip()
        task = task.split("\n", 1)[0].strip()
        return f"{marker} {task}"
    fallback = (fallback or "").strip()
    if fallback:
        return f"{marker} {fallback}"
    return f"{marker} Unknown."


def _format_tool_response(observation: str, admissible: list[str]) -> str:
    obs = (observation or "").strip()[:2000]
    return (
        f"### Observation\n{obs}\n\n"
        f"### Admissible Commands\n{_format_admissible(admissible)}"
    )


def _format_invalid_response(
    previous_obs: str,
    admissible: list[str],
    reason: str,
) -> str:
    return (
        "Invalid tool call. Call `env_step` with JSON arguments like "
        '{"command": "<one admissible command>"}. '
        f"Reason: {reason}\n\n"
        "Environment state did not change.\n\n"
        f"{_format_tool_response(previous_obs, admissible)}"
    )


# ── generate (slime entry point) ─────────────────────────────────────────


async def generate(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    if evaluation:
        raise ValueError("alfworld_agent does not support evaluation mode")

    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata

    session_id = _session_id(sample)
    instance_id = _instance_id(sample)
    metadata["session_id"] = session_id
    metadata["instance_id"] = instance_id

    config = _load_config()
    game_file = metadata.get("game_file")

    prompt_fallback = ""
    prompt = getattr(sample, "prompt", None)
    if isinstance(prompt, str):
        prompt_fallback = prompt
    elif isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                prompt_fallback = str(msg.get("content", ""))
                break

    if not game_file:
        logger.error("alfworld_agent: no game_file in sample.metadata")
        metadata["task_success"] = False
        metadata["num_steps"] = 0
        metadata["invalid_action_count"] = 0
        _set_status(sample, "ABORTED")
        return sample

    final_response = ""

    try:
        proxy_client = _get_proxy_client()
        temperature = sampling_params.get("temperature", 1.0)
        max_tokens = int(
            sampling_params.get("max_new_tokens")
            or getattr(args, "rollout_max_response_len", 1024)
        )

        env = _make_env(game_file, config["max_episode_steps"])
        full_response_parts: list[str] = []
        task_success = False
        invalid_action_count = 0
        executed_steps = 0

        try:
            obs, infos = env.reset()
            task_text = _extract_task_text(obs, prompt_fallback)
            admissible = list(infos.get("admissible_commands", []) or [])

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ALFWORLD_INITIAL_USER_TEMPLATE.format(
                        task_text=task_text,
                        observation=(obs or "").strip()[:2000],
                        admissible_commands=_format_admissible(admissible),
                    ),
                },
            ]

            for step_idx in range(config["max_steps"]):
                response = await proxy_client.chat_completions(
                    {
                        "model": "proxy-model",
                        "messages": messages,
                        "tools": ALFWORLD_TOOL_SCHEMAS,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    },
                    session_id=session_id,
                    instance_id=instance_id,
                    turn_id=f"alf-step{step_idx}",
                )

                assistant_msg = _extract_assistant_message(response)
                assistant_content = _extract_assistant_content(response)
                full_response_parts.append(assistant_content)
                messages.append(assistant_msg)

                finish = _extract_finish_reason(response)
                structured = _extract_structured_tool_calls(response)
                tc_id, command, reason = _first_command(assistant_content, structured)
                tool_call_id = tc_id or f"call_{uuid.uuid4().hex[:12]}"

                if command is None:
                    invalid_action_count += 1
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": _format_invalid_response(obs, admissible, reason or "unknown"),
                    })
                    if finish == "length":
                        break
                    continue

                if command not in admissible:
                    invalid_action_count += 1
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": _format_invalid_response(
                            obs, admissible, f"command {command!r} is not in admissible commands"
                        ),
                    })
                    if finish == "length":
                        break
                    continue

                obs, _reward, done, infos = env.step(command)
                executed_steps += 1
                admissible = list(infos.get("admissible_commands", []) or [])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": _format_tool_response(obs, admissible),
                })

                if infos.get("won", False):
                    task_success = True
                    break
                if done:
                    break
                if finish == "length":
                    break

        finally:
            try:
                env.close()
            except Exception:
                pass

        metadata["task_success"] = task_success
        metadata["num_steps"] = executed_steps
        metadata["invalid_action_count"] = invalid_action_count
        final_response = "".join(full_response_parts)

        # ── drain trajectory & expand segments ──
        await proxy_client.finalize_session(
            session_id,
            instance_id=instance_id,
            label=getattr(sample, "label", None),
        )
        trajectory_payload = await proxy_client.read_trajectory(
            trajectory_id=session_id,
            instance_id=instance_id,
            drain=True,
        )
        try:
            await DEFAULT_WRITER.write_session_payload(
                trajectory_payload,
                session_id=session_id,
                instance_id=instance_id,
            )
        except Exception:
            logger.warning("failed to write trajectory payload log session=%s", session_id, exc_info=True)

        segments = trajectory_payload.get("data") or []
        if not segments:
            logger.warning("no segments returned session=%s", session_id)
            multi_segment.mark_aborted_no_grad(sample, session_id=session_id, instance_id=instance_id)
            _set_status(sample, "ABORTED")
            return sample

        result = multi_segment.expand_segments_to_samples(
            sample,
            segments,
            args=args,
            agent_response=final_response,
            session_id=session_id,
            instance_id=instance_id,
        )
        return result

    except Exception as exc:
        logger.warning(
            "alfworld rollout failed session=%s: %s",
            session_id,
            " ".join(str(exc).splitlines()) or type(exc).__name__,
        )
        metadata["alfworld_error"] = str(exc)
        multi_segment.mark_aborted_no_grad(
            sample,
            session_id=session_id,
            instance_id=instance_id,
        )
        _set_status(sample, "ABORTED")
        return sample
