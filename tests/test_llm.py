"""LLM extractor with the deterministic replay backend, recording, and HTTP payloads."""

import json
import os
from pathlib import Path

import httpx
import pytest

from mrcond.benchmark.metrics import compare_docs
from mrcond.corpus import load_annotation, load_manifest, load_text
from mrcond.extractors.backends import (
    GenerationRequest,
    OllamaBackend,
    OpenAICompatBackend,
    RecordingBackend,
    ReplayBackend,
    ReplayMiss,
    backend_from_env,
)
from mrcond.extractors.llm import OUTPUT_SCHEMA, LLMExtractor, build_context, parse_json_lenient
from mrcond.pipeline import run_extraction
from mrcond.schema import FieldStatus

FIX = Path(__file__).resolve().parent / "fixtures" / "replay_fda_t4a.jsonl"


@pytest.fixture()
def t4a():
    return load_text(load_manifest().get("fda-guidance-2023-t4a"))


def test_replay_raw_vs_verified(t4a):
    ex = LLMExtractor(ReplayBackend(FIX), "fixture")
    raw, _ = run_extraction(t4a, ex, verify=False)
    ver, out = run_extraction(t4a, ex, verify=True)
    gold = load_annotation("fda-guidance-2023-t4a")
    raw_score, ver_score = compare_docs(gold, raw, t4a), compare_docs(gold, ver, t4a)
    # the fixture contains a hallucinated head-SAR quote: unsupported before, zero after verification
    assert raw_score.n_unsupported >= 1
    assert ver_score.n_unsupported == 0
    s3 = next(cs for cs in ver.condition_sets if cs.b0_tesla.normalized_value == [3.0])
    assert s3.head_sar_max_w_per_kg.status == FieldStatus.NEEDS_REVIEW
    assert s3.max_sfg_t_per_m.status == FieldStatus.NEEDS_REVIEW  # 2500 "T/m" from a gauss/cm quote
    assert s3.b1rms_max_uT.status == FieldStatus.EXTRACTED and s3.b1rms_max_uT.normalized_value == 1.7
    assert out.meta["verification"]["n_rejected"] >= 2
    assert ver.status_value().value == "MR Conditional"


def test_replay_miss_raises(t4a, tmp_path):
    be = ReplayBackend(tmp_path / "empty.jsonl")
    with pytest.raises(ReplayMiss):
        LLMExtractor(be).extract(t4a)


def test_recording_backend_writes_replayable_file(t4a, tmp_path):
    class Fake:
        name = "fake"

        def generate(self, req):
            return json.dumps(
                {"mr_status": "MR Conditional", "mr_status_quote": "MR Conditional", "condition_sets": []}
            )

    path = tmp_path / "rec.jsonl"
    LLMExtractor(RecordingBackend(Fake(), path, "fake-model")).extract(t4a)
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["doc_id"] == t4a.doc_id and rec["model"] == "fake-model" and "prompt_hash" in rec
    assert "Star" not in json.dumps({k: v for k, v in rec.items() if k != "content"})  # no document text stored
    replayed = LLMExtractor(ReplayBackend(path)).extract(t4a)
    assert replayed.status_candidates[0].fv.normalized_value == "MR Conditional"


def test_http_payloads(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        seen["url"], seen["json"] = url, json
        if url.endswith("/api/chat"):
            return httpx.Response(200, json={"message": {"content": "{}"}}, request=httpx.Request("POST", url))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "{}"}}]}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    req = GenerationRequest("sys", "user", OUTPUT_SCHEMA, "d", 0, 0.0, 7, 100)
    OllamaBackend("qwen2.5:7b-instruct").generate(req)
    assert seen["url"].endswith("/api/chat") and seen["json"]["format"] == OUTPUT_SCHEMA
    assert seen["json"]["options"]["seed"] == 7 and seen["json"]["stream"] is False
    OpenAICompatBackend("m", "http://localhost:8080/v1").generate(req)
    assert seen["json"]["response_format"]["type"] == "json_schema"
    OpenAICompatBackend("m", "http://localhost:8080/v1", schema_mode="json_object").generate(req)
    assert seen["json"]["response_format"] == {"type": "json_object", "schema": OUTPUT_SCHEMA}


def test_backend_from_env(tmp_path):
    assert isinstance(backend_from_env({"MRCOND_LLM_BACKEND": "ollama", "MRCOND_LLM_MODEL": "x"}), OllamaBackend)
    be = backend_from_env({"MRCOND_LLM_BACKEND": "openai", "MRCOND_LLM_BASE_URL": "http://h:1/v1"})
    assert isinstance(be, OpenAICompatBackend) and be.base_url == "http://h:1/v1"
    with pytest.raises(ValueError):
        backend_from_env({"MRCOND_LLM_BACKEND": "replay"})


def test_context_and_lenient_json(t4a):
    ctx = build_context(t4a, 6000)
    assert ctx.startswith("[[page 30]]") and "B1+rms" in ctx
    assert parse_json_lenient('noise {"a": 1} trailing') == {"a": 1}
    assert parse_json_lenient("not json") == {}


def test_field_strength_range_comes_from_the_quote():
    from mrcond.extractors.llm import _b0_comparator

    assert _b0_comparator("Static magnetic field of 3-Tesla or less", [3.0]) == "le"
    assert _b0_comparator("Static Magnetic Field Strength (B0) 1.5T and 3T", [1.5, 3.0]) == "in"
    assert _b0_comparator("", [1.5]) == "in"


def _quotes(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("quote") and isinstance(v, str):
                yield v
            else:
                yield from _quotes(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _quotes(v)


def test_committed_recordings_hold_only_short_model_output():
    """Recordings must not become a way to redistribute labeling text: model output and prompt hash only."""
    from mrcond.corpus import repo_root

    allowed = {"doc_id", "run", "model", "prompt_hash", "temperature", "seed", "seconds", "content", "quotes_capped"}
    for path in sorted((repo_root() / "data" / "replay").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            assert set(rec) <= allowed, (path.name, set(rec) - allowed)
            assert len(rec["content"]) <= 4000, (path.name, rec["doc_id"])
            for q in _quotes(parse_json_lenient(rec["content"])):  # capped by scripts/cap_replay_quotes.py
                assert len(q.split()) <= 25, (path.name, rec["doc_id"], q)


@pytest.mark.live_llm
def test_live_llm_smoke(t4a):
    be = backend_from_env()
    label, out = run_extraction(t4a, LLMExtractor(be, os.environ.get("MRCOND_LLM_MODEL")), verify=True)
    assert out.meta["parse_ok"]
