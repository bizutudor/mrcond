"""JSON Schema export, GUDID gap analysis (offline), CLI."""

import csv
import json

import jsonschema
import pytest

from mrcond.cli import main
from mrcond.corpus import load_annotation
from mrcond.gudid import SNAPSHOT_2026_10_15, run_gap_analysis
from mrcond.schema import export_json_schemas


def test_json_schema_export_validates_annotations(tmp_path):
    paths = {p.name: p for p in export_json_schemas(tmp_path)}
    assert set(paths) == {"label_document.schema.json", "check_request.schema.json", "check_result.schema.json"}
    schema = json.loads(paths["label_document.schema.json"].read_text())
    ann = json.loads(load_annotation("fda-guidance-2023-t4a").model_dump_json())
    jsonschema.validate(ann, schema)
    ann["condition_sets"][0]["wb_sar_max_w_per_kg"]["status"] = "PROBABLY"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(ann, schema)


def test_committed_schemas_are_current(tmp_path):
    from mrcond.corpus import repo_root

    for p in export_json_schemas(tmp_path):
        committed = repo_root() / "schemas" / p.name
        assert committed.exists() and json.loads(committed.read_text()) == json.loads(p.read_text()), p.name


def test_gudid_offline_snapshot(tmp_path):
    out = run_gap_analysis(tmp_path, offline=True, by_year=False)
    assert out["total"] == 1141848 == sum(SNAPSHOT_2026_10_15["counts"].values())
    assert out["percent"]["Labeling does not contain MRI Safety Information"] == pytest.approx(78.99, abs=0.01)
    assert out["percent"]["MR Conditional"] == pytest.approx(16.09, abs=0.01)
    assert out["percent"]["MR Safe"] == pytest.approx(2.73, abs=0.01)
    assert out["percent"]["MR Unsafe"] == pytest.approx(2.2, abs=0.01)
    rows = list(csv.DictReader((tmp_path / "gudid_mri_safety_counts.csv").open()))
    assert [r["mri_safety"] for r in rows][1] == "MR Conditional"


@pytest.mark.network
def test_gudid_live_query(tmp_path):
    from mrcond.gudid import count_by_mri_safety

    counts, meta = count_by_mri_safety()
    assert sum(counts.values()) > 1_000_000 and "MR Conditional" in counts


def test_cli_scenarios_and_extract(capsys, data_copy):
    assert main(["scenarios"]) == 0
    assert '"unsafe_pass": 0' in capsys.readouterr().out
    path = data_copy / "public" / "fda-guidance-2023-t3.json"
    assert main(["extract", str(path), "--extractor", "rule+verifier"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mr_status"]["normalized_value"] == "MR Conditional"
    assert out["condition_sets"][0]["max_sfg_t_per_m"]["normalized_value"] == 30.0
