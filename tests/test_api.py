"""REST API tests (FastAPI TestClient) on a temporary copy of the data directory."""

import json
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

DOC = "fda-guidance-2023-t4a"


@pytest.fixture()
def client(data_copy):
    from mrcond.api import create_app

    return TestClient(create_app())


def test_health_and_fields(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "not a medical device" in h["disclaimer"]
    names = [f["name"] for f in client.get("/api/fields").json()]
    assert "b1rms_max_uT" in names and len(names) == 20


def test_documents_and_text_availability(client):
    docs = {d["doc_id"]: d for d in client.get("/api/documents").json()}
    assert len(docs) >= 15
    assert docs[DOC]["text_available"] is True  # public-domain FDA text ships with the repository
    assert docs[DOC]["annotation_status"] == "draft-unverified"
    pages = client.get(f"/api/documents/{DOC}/pages").json()
    assert pages["pages"][0]["number"] == 30
    # manufacturer documents are not redistributed: no text until `mrcond fetch`
    assert docs["gore-tigris-mri-html"]["text_available"] is False
    assert client.get("/api/documents/gore-tigris-mri-html/pages").status_code == 404
    with_syn = client.get("/api/documents", params={"include_synthetic": True}).json()
    assert any(d["source"] == "synthetic" for d in with_syn)


def test_annotation_and_extractions(client):
    ann = client.get(f"/api/documents/{DOC}/annotation").json()
    assert ann["provenance"]["verification"] == "draft-unverified"
    lab = client.get(f"/api/documents/{DOC}/extractions/rule+verifier").json()
    assert len(lab["condition_sets"]) == 2
    assert client.get(f"/api/documents/{DOC}/extractions/nonsense").status_code == 400
    assert client.get(f"/api/documents/{DOC}/extractions/llm+verifier").status_code == 400  # not configured


def test_verify_accept_edit_reject_writes_verified_annotation(client, data_copy):
    decisions = [
        {"set_id": "cs2", "field": "b1rms_max_uT", "action": "accept"},
        {
            "set_id": "cs1",
            "field": "wb_sar_max_w_per_kg",
            "action": "edit",
            "value": 2,
            "unit": "W/kg",
            "quote": "Whole-Body SAR ≤ 2 W/kg",
            "page": 30,
        },
        {"set_id": "cs1", "field": "receive_coil", "action": "reject"},
    ]
    r = client.post(f"/api/documents/{DOC}/verify", json={"reviewer": "Test Reviewer", "decisions": decisions})
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["provenance"]["verification"] == "human-verified"
    assert saved["provenance"]["reviewer"] == "Test Reviewer" and saved["provenance"]["reviewed_at"]
    path = data_copy / "annotations" / "verified" / f"{DOC}.json"
    on_disk = json.loads(path.read_text())
    cs1 = on_disk["condition_sets"][0]
    assert cs1["wb_sar_max_w_per_kg"]["evidence"][0]["page"] == 30
    assert cs1["receive_coil"].get("status", "NOT_STATED") == "NOT_STATED"
    assert (data_copy / "annotations" / "verified" / "audit.jsonl").exists()
    docs = {d["doc_id"]: d for d in client.get("/api/documents").json()}
    assert docs[DOC]["annotation_status"] == "human-verified"
    implants = {i["implant_id"]: i for i in client.get("/api/implants").json()}
    assert implants[DOC]["verification"] == "human-verified"


def test_verify_rejects_non_verbatim_quotes_and_missing_reviewer(client):
    bad = [
        {
            "set_id": "cs1",
            "field": "wb_sar_max_w_per_kg",
            "action": "edit",
            "value": 4,
            "unit": "W/kg",
            "quote": "Whole-Body SAR ≤ 4 W/kg",
        }
    ]
    r = client.post(f"/api/documents/{DOC}/verify", json={"reviewer": "Test Reviewer", "decisions": bad})
    assert r.status_code == 422 and "not found verbatim" in r.text
    wrong_unit = [
        {
            "set_id": "cs1",
            "field": "b1rms_max_uT",
            "action": "edit",
            "value": 2,
            "unit": "W/kg",
            "quote": "Whole-Body SAR ≤ 2 W/kg",
        }
    ]
    assert (
        client.post(
            f"/api/documents/{DOC}/verify", json={"reviewer": "Test Reviewer", "decisions": wrong_unit}
        ).status_code
        == 422
    )
    assert client.post(f"/api/documents/{DOC}/verify", json={"reviewer": "", "decisions": []}).status_code == 422


def test_flagged_fields_need_a_decision(client, monkeypatch):
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "replay_fda_t4a.jsonl"
    monkeypatch.setenv("MRCOND_LLM_BACKEND", "replay")
    monkeypatch.setenv("MRCOND_LLM_REPLAY", str(fixture))
    lab = client.get(f"/api/documents/{DOC}/extractions/llm+verifier").json()
    flagged = [
        (cs["set_id"], f)
        for cs in lab["condition_sets"]
        for f, v in cs.items()
        if isinstance(v, dict) and v.get("status") == "NEEDS_REVIEW"
    ]
    assert ("cs2", "head_sar_max_w_per_kg") in flagged and ("cs2", "max_sfg_t_per_m") in flagged
    base = {"reviewer": "Test Reviewer", "base": "extraction:llm+verifier"}
    r = client.post(f"/api/documents/{DOC}/verify", json={**base, "decisions": []})
    assert r.status_code == 422 and "need a decision" in r.text
    decisions = [{"set_id": s, "field": f, "action": "reject"} for s, f in flagged]
    decisions = [d for d in decisions if d["field"] != "max_sfg_t_per_m"]
    decisions.append(
        {
            "set_id": "cs2",
            "field": "max_sfg_t_per_m",
            "action": "edit",
            "value": 25,
            "unit": "T/m",
            "quote": "Maximum Spatial Field Gradient 25 T/m (2500 gauss/cm)",
        }
    )
    if lab["mr_status"]["status"] == "NEEDS_REVIEW":
        decisions.append({"set_id": "_doc", "field": "mr_status", "action": "reject"})
    r = client.post(f"/api/documents/{DOC}/verify", json={**base, "decisions": decisions})
    assert r.status_code == 200, r.text
    cs2 = r.json()["condition_sets"][1]
    assert cs2["max_sfg_t_per_m"]["normalized_value"] == 25.0 and cs2["max_sfg_t_per_m"]["status"] == "EXTRACTED"
    assert cs2["head_sar_max_w_per_kg"]["status"] == "NOT_STATED"


def test_check_endpoint(client):
    scanner = client.get("/api/scanner-profiles").json()[1]
    body = {
        "implant_ids": ["demo:IMP-PASSIVE-STAR", DOC],
        "scanner": scanner,
        "protocol": {
            "operating_mode": "normal",
            "transmit_coil": "body",
            "rf_excitation": "CP",
            "sequences": [{"name": "t2", "b1rms_uT": 1.5, "wb_sar_w_per_kg": 1.0, "head_sar_w_per_kg": 1.0}],
            "total_active_scan_time_min": 20,
            "sfg_at_implant_t_per_m": 10,
            "device_configuration_confirmed": True,
        },
    }
    res = client.post("/api/check", json=body).json()
    assert res["outcome"] == "NEEDS_REVIEW"  # the FDA example annotation is still a draft
    assert "sign-off" not in res["outcome_text"] and "not a medical device" in res["disclaimer"]
    assert client.post("/api/check", json={**body, "implant_ids": ["nope"]}).status_code == 404


def test_ui_routes_do_not_shadow_api(client):
    assert client.get("/api/does-not-exist").status_code == 404
