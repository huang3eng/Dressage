"""HotpotQA whitebox agent — standalone generate function.

Follows the same pattern as ``whitebox_loop.generate``: a standalone
``async def generate(args, sample, sampling_params, evaluation)`` that
drives the Dressage proxy, drains trajectory segments, and returns
training-ready Samples.  No base class needed — session management,
drain, and segment expansion reuse shared rollout helpers.

Append-only ``messages`` list across the whole trajectory so the proxy
classifies every step as ``append_only_continuation`` and emits ONE segment
per trajectory (not one segment per step).

Wiring:
  - ``--custom-generate-function-path dressage.recipes.hotpotqa.agent.generate``
  - ``--dressage-multi-segment``
  - Proxy: ``--trajectory-build-mode concat``
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

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
    if not session_id.startswith("hpqa-"):
        session_id = f"hpqa-{session_id}"
    sample.session_id = session_id
    return session_id


# ── Local FAISS+BGE retrieval ────────────────────────────────────────────

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DEFAULT_CORPUS_DIR = os.environ.get(
    "HOTPOTQA_CORPUS_DIR", "/path/to/hotpotqa/corpus"
)
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "HOTPOTQA_EMBEDDING_MODEL", "/path/to/models/bge-large-en-v1.5"
)
DEFAULT_EMBEDDING_DEVICE = os.environ.get("HOTPOTQA_EMBEDDING_DEVICE", "cpu")
DEFAULT_TOPK = int(os.environ.get("HOTPOTQA_TOPK", "5"))


def _resolve_device(requested: str) -> str:
    dev = (requested or "cpu").strip().lower()
    if dev == "cpu":
        return "cpu"
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "HOTPOTQA_EMBEDDING_DEVICE=%r but CUDA unavailable; falling back to cpu", requested
            )
            return "cpu"
        if ":" in dev:
            try:
                idx = int(dev.split(":")[-1])
                if idx >= torch.cuda.device_count():
                    logger.warning("cuda:%d invalid (device_count=%d); using cpu", idx, torch.cuda.device_count())
                    return "cpu"
            except ValueError:
                pass
    return dev


class HotpotQALocalSearch:
    _lock = threading.RLock()
    _shared_key: Optional[str] = None
    _shared_index: Optional["faiss.Index"] = None
    _shared_corpus: Optional[list] = None
    _shared_model: Optional["SentenceTransformer"] = None

    def __init__(
        self,
        corpus_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_device: Optional[str] = None,
        topk: Optional[int] = None,
    ) -> None:
        self.corpus_dir = Path(corpus_dir or DEFAULT_CORPUS_DIR)
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.embedding_device = _resolve_device(embedding_device or DEFAULT_EMBEDDING_DEVICE)
        self.topk = topk if topk is not None else DEFAULT_TOPK
        self._index: Optional["faiss.Index"] = None
        self._corpus: list[str] = []
        self._model: Optional["SentenceTransformer"] = None
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        cache_key = f"{self.corpus_dir}|{self.embedding_device}|{self.embedding_model}"
        with self.__class__._lock:
            if (
                self.__class__._shared_key != cache_key
                or self.__class__._shared_index is None
                or self.__class__._shared_corpus is None
                or self.__class__._shared_model is None
            ):
                index_path = self.corpus_dir / "index.bin"
                corpus_path = self.corpus_dir / "hpqa_corpus.jsonl"
                if not index_path.exists():
                    raise FileNotFoundError(f"FAISS index not found: {index_path}")
                if not corpus_path.exists():
                    raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

                logger.info("Loading FAISS index from %s", index_path)
                index = faiss.read_index(str(index_path))
                logger.info("Loading corpus from %s", corpus_path)
                corpus: list[str] = []
                with corpus_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        title = str(rec.get("title", ""))
                        text = str(rec.get("text", ""))
                        corpus.append(f"{title}\n{text}".strip())

                logger.info("Loading BGE model=%s device=%s", self.embedding_model, self.embedding_device)
                model = SentenceTransformer(self.embedding_model, device=self.embedding_device)
                self.__class__._shared_key = cache_key
                self.__class__._shared_index = index
                self.__class__._shared_corpus = corpus
                self.__class__._shared_model = model

            self._index = self.__class__._shared_index
            self._corpus = self.__class__._shared_corpus or []
            self._model = self.__class__._shared_model

    def _encode_queries(self, queries: list[str]) -> "np.ndarray":
        prefixed = [QUERY_INSTRUCTION + q for q in queries]
        with self.__class__._lock:
            assert self._model is not None
            out = self._model.encode(prefixed, normalize_embeddings=True)
        arr = np.asarray(out, dtype=np.float32)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr

    def _format_results(self, ids: "np.ndarray") -> str:
        parts: list[str] = []
        for rank, idx in enumerate(ids):
            idx = int(idx)
            if idx < 0 or idx >= len(self._corpus):
                continue
            entry = self._corpus[idx]
            lines = entry.split("\n", 1)
            title = lines[0] if lines else ""
            text = lines[1] if len(lines) > 1 else ""
            parts.append(f"Doc {rank + 1}(Title: {title}) {text}")
        return "\n".join(parts)

    def execute(self, query: str) -> str:
        try:
            embeddings = self._encode_queries([query])
            assert self._index is not None
            _, ids = self._index.search(embeddings, self.topk)
            return self._format_results(ids[0])
        except Exception as e:
            logger.warning("Local search failed for query=%r: %s", query[:50], e)
            return ""


_SEARCH_TOOL: Optional[HotpotQALocalSearch] = None
_SEARCH_TOOL_LOCK = threading.Lock()


def _get_search_tool() -> HotpotQALocalSearch:
    global _SEARCH_TOOL
    if _SEARCH_TOOL is None:
        with _SEARCH_TOOL_LOCK:
            if _SEARCH_TOOL is None:
                _SEARCH_TOOL = HotpotQALocalSearch()
    return _SEARCH_TOOL


# ── Prompts ──────────────────────────────────────────────────────────────


HOTPOTQA_SYSTEM_PROMPT = (
    "You are a research agent. Your goal is to answer the User Query using "
    "Wikipedia search evidence.\n\n"
    "Each turn, briefly reason inside <analysis>...</analysis>, then either "
    "(a) issue one or more `search` tool calls in parallel for new evidence, "
    "or (b) emit the final short answer inside <answer>...</answer> tags. "
    "Never repeat a query that already appears in the conversation. Once you "
    "can answer from accumulated passages, stop searching and emit <answer>."
)

HOTPOTQA_INITIAL_USER_TEMPLATE = """### User Query
{user_query}

### Seed Evidence
{seed_evidence}

### Instructions
- Reason briefly inside `<analysis>...</analysis>`.
- Call the `search` tool one or more times in parallel when new evidence is needed.
- When you can answer from prior passages, output `<answer>short answer</answer>` (no extra prose, no further tool calls).

### Output Format
<analysis>
[Your analysis...]
</analysis>
<tool_call>
{{"name": "search", "arguments": {{"query": "..."}}}}
</tool_call>
"""

HOTPOTQA_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search Wikipedia for passages relevant to the user question. "
                "Use natural-language or keyword queries; must differ from "
                "any prior query in the conversation when possible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A single search query (natural language or "
                            "keywords). Must differ from prior queries when "
                            "seeking new evidence."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# ── Config ───────────────────────────────────────────────────────────────


def _load_config() -> dict[str, Any]:
    return {
        "max_steps": int(os.environ.get("HOTPOTQA_MAX_STEPS", "5")),
        "max_parallel_calls": int(os.environ.get("HOTPOTQA_MAX_PARALLEL_CALLS", "4")),
        "force_first_search": os.environ.get("HOTPOTQA_FORCE_FIRST_SEARCH", "1").lower()
        in {"1", "true", "yes"},
        "passage_max_chars": int(os.environ.get("HOTPOTQA_PASSAGE_MAX_CHARS", "1200")),
    }


# ── Helpers ──────────────────────────────────────────────────────────────


_QUESTION_MARKER_RE = re.compile(r"Question:\s*(.+)\Z", re.DOTALL)


def _extract_user_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return "\n".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
    return str(prompt) if prompt is not None else ""


def _extract_question(prompt_text: str) -> str:
    text = prompt_text.strip()
    m = _QUESTION_MARKER_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


async def _do_search(query: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_search_tool().execute, query)


def _truncate_passage(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


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


def _extract_query_from_call(name: Any, arguments: Any) -> tuple[str | None, str | None]:
    if name != "search":
        return None, f"unknown tool {name!r}"
    if isinstance(arguments, str):
        arguments = _parse_json_loose(arguments)
    if not isinstance(arguments, dict):
        return None, "arguments not a JSON object"
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return None, "missing query"
    return query.strip(), None


def _collect_calls(
    content: str,
    structured: list[dict[str, Any]],
    *,
    max_parallel: int,
) -> list[tuple[str | None, str | None, str]]:
    out: list[tuple[str | None, str | None, str]] = []
    for tc in structured:
        if len(out) >= max_parallel:
            break
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        query, err = _extract_query_from_call(fn.get("name"), fn.get("arguments"))
        out.append((tc.get("id"), query, err or ""))
    if out:
        return out
    for raw in _TOOL_CALL_BLOCK_RE.findall(content or ""):
        if len(out) >= max_parallel:
            break
        obj = _parse_json_loose(raw)
        if not isinstance(obj, dict):
            out.append((None, None, "could not parse <tool_call> JSON"))
            continue
        query, err = _extract_query_from_call(obj.get("name"), obj.get("arguments"))
        out.append((None, query, err or ""))
    return out


def _extract_answer(content: str) -> str | None:
    matches = _ANSWER_BLOCK_RE.findall(content or "")
    if not matches:
        return None
    return matches[-1].strip() or None


# ── generate (slime entry point) ─────────────────────────────────────────


async def generate(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    if evaluation:
        raise ValueError("hotpotqa_agent does not support evaluation mode")

    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata

    session_id = _session_id(sample)
    instance_id = _instance_id(sample)
    metadata["session_id"] = session_id
    metadata["instance_id"] = instance_id

    config = _load_config()
    final_response = ""

    try:
        proxy_client = _get_proxy_client()
        base_user_query = _extract_question(_extract_user_prompt(sample.prompt))

        seed_evidence = "None"
        if config["force_first_search"]:
            seed_text = await _do_search(base_user_query)
            seed_evidence = (
                f"[seed query: {base_user_query}]\n"
                f"{_truncate_passage(seed_text, config['passage_max_chars'])}"
                if seed_text
                else "None (initial search returned no passages)"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": HOTPOTQA_INITIAL_USER_TEMPLATE.format(
                    user_query=base_user_query,
                    seed_evidence=seed_evidence,
                ),
            },
        ]

        full_response_parts: list[str] = []
        valid_search_count = 0
        temperature = sampling_params.get("temperature", 1.0)
        max_tokens = int(
            sampling_params.get("max_new_tokens")
            or getattr(args, "rollout_max_response_len", 1024)
        )

        for step_idx in range(config["max_steps"]):
            response = await proxy_client.chat_completions(
                {
                    "model": "proxy-model",
                    "messages": messages,
                    "tools": HOTPOTQA_TOOL_SCHEMAS,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
                session_id=session_id,
                instance_id=instance_id,
                turn_id=f"hpqa-step{step_idx}",
                )

            assistant_msg = _extract_assistant_message(response)
            assistant_content = _extract_assistant_content(response)
            full_response_parts.append(assistant_content)
            messages.append(assistant_msg)

            if _extract_answer(assistant_content) is not None:
                break
            if _extract_finish_reason(response) == "length":
                break

            calls = _collect_calls(
                assistant_content,
                _extract_structured_tool_calls(response),
                max_parallel=config["max_parallel_calls"],
            )
            if not calls:
                break

            tasks = [
                _do_search(query) if query is not None else asyncio.sleep(0, result="")
                for _, query, _ in calls
            ]
            results = await asyncio.gather(*tasks)

            for (tc_id, query, err), text in zip(calls, results):
                if query is None:
                    tool_content = (
                        f"Tool call could not be parsed: {err}. "
                        'Use {"name": "search", "arguments": {"query": "..."}}.'
                    )
                else:
                    truncated = _truncate_passage(text, config["passage_max_chars"])
                    tool_content = truncated or "(no passages returned)"
                    if text:
                        valid_search_count += 1

                messages.append({
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": tc_id or f"call_{uuid.uuid4().hex[:12]}",
                })

        metadata["valid_search_count"] = valid_search_count
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
            "hotpotqa rollout failed session=%s: %s",
            session_id,
            " ".join(str(exc).splitlines()) or type(exc).__name__,
        )
        metadata["hotpotqa_error"] = str(exc)
        multi_segment.mark_aborted_no_grad(
            sample,
            session_id=session_id,
            instance_id=instance_id,
        )
        _set_status(sample, "ABORTED")
        return sample
