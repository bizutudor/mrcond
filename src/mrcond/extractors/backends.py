"""LLM backends: Ollama, any OpenAI-compatible server, and deterministic replay.

* :class:`OllamaBackend` - ``POST /api/chat`` with ``format`` set to the JSON
  schema (Ollama >= 0.5 structured outputs).
* :class:`OpenAICompatBackend` - ``POST /v1/chat/completions``; works with the
  llama.cpp server (``llama-server``), llama-cpp-python's server, vLLM and LM
  Studio. ``schema_mode`` selects how the JSON schema is passed:
  ``json_schema`` (OpenAI style, llama.cpp server, vLLM, LM Studio),
  ``json_object`` (llama-cpp-python style ``{"type": "json_object", "schema": ...}``)
  or ``none`` (prompt-only; output is parsed leniently).
* :class:`ReplayBackend` - returns recorded outputs keyed by
  ``(doc_id, run)``; used by tests and CI so results are reproducible without a
  model. :class:`RecordingBackend` wraps a live backend and records outputs.

Recorded files store only model outputs and a hash of the prompt - never the
document text.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

__all__ = [
    "GenerationRequest",
    "LLMBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "RecordingBackend",
    "ReplayBackend",
    "ReplayMiss",
    "backend_from_env",
    "prompt_hash",
]


@dataclass
class GenerationRequest:
    system: str
    user: str
    schema: dict
    doc_id: str
    run: int = 0
    temperature: float = 0.0
    seed: int | None = 0
    max_tokens: int = 1200
    meta: dict = field(default_factory=dict)


class LLMBackend(Protocol):
    name: str

    def generate(self, req: GenerationRequest) -> str: ...


def prompt_hash(req: GenerationRequest) -> str:
    h = hashlib.sha256()
    for part in (req.system, req.user, json.dumps(req.schema, sort_keys=True), str(req.temperature), str(req.seed)):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class OllamaBackend:
    def __init__(
        self, model: str, host: str = "http://localhost:11434", timeout: float = 600.0, num_ctx: int = 8192
    ) -> None:
        self.model, self.host, self.timeout, self.num_ctx = model, host.rstrip("/"), timeout, num_ctx
        self.name = f"ollama:{model}"

    def generate(self, req: GenerationRequest) -> str:
        options: dict[str, Any] = {
            "temperature": req.temperature,
            "num_predict": req.max_tokens,
            "num_ctx": self.num_ctx,
        }
        if req.seed is not None:
            options["seed"] = req.seed
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
            "format": req.schema,
            "stream": False,
            "options": options,
        }
        r = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAICompatBackend:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        api_key: str | None = None,
        schema_mode: str = "json_schema",
        timeout: float = 600.0,
    ) -> None:
        if schema_mode not in ("json_schema", "json_object", "none"):
            raise ValueError("schema_mode must be json_schema, json_object or none")
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.schema_mode, self.timeout = schema_mode, timeout
        self.name = f"openai-compat:{model}"

    def generate(self, req: GenerationRequest) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.seed is not None:
            payload["seed"] = req.seed
        if self.schema_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "mrcond_extraction", "schema": req.schema, "strict": True},
            }
        elif self.schema_mode == "json_object":
            payload["response_format"] = {"type": "json_object", "schema": req.schema}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class ReplayMiss(KeyError):
    pass


class ReplayBackend:
    """Replays recorded outputs from a JSONL file (one record per (doc_id, run))."""

    def __init__(self, path: str | Path, strict_prompt: bool = False) -> None:
        self.path = Path(path)
        self.strict_prompt = strict_prompt
        self.records: dict[tuple[str, int], dict] = {}
        self.model_tag = "replay"
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.records[(rec["doc_id"], int(rec.get("run", 0)))] = rec
                self.model_tag = rec.get("model", self.model_tag)
        self.name = f"replay:{self.model_tag}"

    def generate(self, req: GenerationRequest) -> str:
        rec = self.records.get((req.doc_id, req.run))
        if rec is None:
            raise ReplayMiss(f"no recorded output for doc_id={req.doc_id!r} run={req.run} in {self.path}")
        if self.strict_prompt and rec.get("prompt_hash") != prompt_hash(req):
            raise ReplayMiss(f"prompt changed since recording for {req.doc_id!r}")
        return rec["content"]


class RecordingBackend:
    """Wraps a live backend and appends every output to a replay JSONL file."""

    def __init__(self, inner: LLMBackend, path: str | Path, model_tag: str | None = None) -> None:
        self.inner, self.path = inner, Path(path)
        self.model_tag = model_tag or inner.name
        self.name = inner.name

    def generate(self, req: GenerationRequest) -> str:
        t0 = time.time()
        content = self.inner.generate(req)
        rec = {
            "doc_id": req.doc_id,
            "run": req.run,
            "model": self.model_tag,
            "prompt_hash": prompt_hash(req),
            "temperature": req.temperature,
            "seed": req.seed,
            "seconds": round(time.time() - t0, 2),
            "content": content,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return content


def backend_from_env(env: dict[str, str] | None = None) -> LLMBackend:
    """Build a backend from MRCOND_LLM_* environment variables.

    MRCOND_LLM_BACKEND   ollama | openai | replay (default: replay)
    MRCOND_LLM_MODEL     model name (e.g. qwen2.5:7b-instruct for Ollama)
    MRCOND_LLM_BASE_URL  OpenAI-compatible base URL (default http://localhost:8080/v1)
    MRCOND_LLM_HOST      Ollama host (default http://localhost:11434)
    MRCOND_LLM_SCHEMA_MODE json_schema | json_object | none
    MRCOND_LLM_API_KEY   optional bearer token
    MRCOND_LLM_REPLAY    replay JSONL path
    """
    env = dict(os.environ if env is None else env)
    kind = env.get("MRCOND_LLM_BACKEND", "replay")
    model = env.get("MRCOND_LLM_MODEL", "local-model")
    if kind == "ollama":
        return OllamaBackend(model, env.get("MRCOND_LLM_HOST", "http://localhost:11434"))
    if kind == "openai":
        return OpenAICompatBackend(
            model,
            env.get("MRCOND_LLM_BASE_URL", "http://localhost:8080/v1"),
            env.get("MRCOND_LLM_API_KEY"),
            env.get("MRCOND_LLM_SCHEMA_MODE", "json_schema"),
        )
    if kind == "replay":
        path = env.get("MRCOND_LLM_REPLAY")
        if not path:
            raise ValueError("MRCOND_LLM_REPLAY must point to a replay .jsonl file")
        return ReplayBackend(path)
    raise ValueError(f"unknown MRCOND_LLM_BACKEND: {kind}")
