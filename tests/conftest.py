from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_collection_modifyitems(config, items):
    skip_net = pytest.mark.skip(reason="network test; set MRCOND_NETWORK_TESTS=1 to run")
    skip_llm = pytest.mark.skip(reason="live LLM test; set MRCOND_LIVE_LLM=1 (and MRCOND_LLM_* variables) to run")
    for item in items:
        if "network" in item.keywords and os.environ.get("MRCOND_NETWORK_TESTS") != "1":
            item.add_marker(skip_net)
        if "live_llm" in item.keywords and os.environ.get("MRCOND_LIVE_LLM") != "1":
            item.add_marker(skip_llm)


@pytest.fixture()
def data_copy(tmp_path, monkeypatch) -> Path:
    """A writable copy of the committed data directory (without fetched documents)."""
    dst = tmp_path / "data"
    src = ROOT / "data"
    for sub in ("corpus", "annotations", "public", "benchmark", "replay"):
        if (src / sub).exists():
            shutil.copytree(src / sub, dst / sub, ignore=shutil.ignore_patterns("verified"))
    (dst / "annotations" / "verified").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MRCOND_DATA", str(dst))
    return dst
