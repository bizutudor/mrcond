"""REST API (FastAPI) for the review workflow and the checker.

Local research tool: bind to 127.0.0.1; there is no authentication.

Endpoints (all JSON, prefix /api):

GET  /health                                  version and disclaimer
GET  /fields                                  field specifications (labels, units, kinds)
GET  /documents                               corpus documents with text/annotation status
GET  /documents/{doc_id}                      document metadata
GET  /documents/{doc_id}/pages                page-indexed text (only if available locally)
GET  /documents/{doc_id}/annotation           current annotation (human-verified, else draft)
GET  /documents/{doc_id}/extractions/{name}   run an extractor: rule | rule+verifier | llm | llm+verifier
POST /documents/{doc_id}/verify               per-field human decisions -> verified annotation JSON
GET  /implants                                labels usable by the checker
GET  /implants/{implant_id}                   one label
GET  /scanner-profiles                        generic scanner presets
POST /check                                   run the checker
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .checker import check
from .corpus import data_dir, load_annotation, load_manifest, load_text, verified_path
from .ingest import DocumentText
from .merge import values_equal
from .pipeline import get_extractor, run_extraction
from .profiles import PRESETS
from .schema import (
    DISCLAIMER,
    FIELD_NAMES,
    FIELD_SPECS,
    CheckResult,
    ConditionSet,
    Evidence,
    FieldStatus,
    FieldValue,
    LabelDocument,
    MRStatus,
    ProtocolPlan,
    ScannerProfile,
)
from .textnorm import find_all, word_count
from .units import normalize_unit, to_canonical

EXTRACTORS = ("rule", "rule+verifier", "llm", "llm+verifier")


# ----------------------------------------------------------------------------------------------
# request models
# ----------------------------------------------------------------------------------------------


class FieldDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    set_id: str = Field(description="Condition set id, or '_doc' for the MR status.")
    field: str = Field(description="Field name, or 'mr_status'.")
    action: Literal["accept", "edit", "reject"]
    value: float | str | list[float] | list[str] | bool | None = None
    unit: str | None = None
    normalized_value: float | str | list[float] | list[str] | bool | None = None
    comparator: Literal["le", "ge", "eq", "in", "any"] | None = None
    quote: str | None = Field(default=None, description="New verbatim evidence (required for edits).")
    page: int | None = None
    comment: str | None = None


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str = Field(min_length=2)
    base: str = Field(default="annotation", description="'annotation' or 'extraction:<extractor>'")
    decisions: list[FieldDecision] = Field(default_factory=list)
    notes: str | None = None


class CheckApiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implant_ids: list[str] = Field(default_factory=list)
    implants: list[LabelDocument] = Field(default_factory=list)
    scanner: ScannerProfile
    protocol: ProtocolPlan


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------


def _synthetic_docs() -> dict[str, tuple[DocumentText, LabelDocument]]:
    from .benchmark.runner import TRACK_S_DIR
    from .benchmark.synth import load_corpus

    d = TRACK_S_DIR()
    if not (d / "index.json").exists():
        return {}
    return {doc.doc_id: (doc, gold) for doc, gold, _ in load_corpus(d)}


@lru_cache(maxsize=1)
def _synthetic_cached(root: str) -> dict[str, tuple[DocumentText, LabelDocument]]:  # noqa: ARG001
    return _synthetic_docs()


def _synthetic() -> dict[str, tuple[DocumentText, LabelDocument]]:
    return _synthetic_cached(str(data_dir()))


def _demo_implants() -> dict[str, LabelDocument]:
    from .benchmark.scenarios import build_implants

    keep = ["IMP-PASSIVE-STAR", "IMP-STAR-ACTIVE", "IMP-LEGACY-3T", "IMP-UNSAFE", "IMP-SAFE"]
    imps = build_implants()
    out = {}
    for k in keep:
        lab = imps[k].model_copy(deep=True)
        lab.device_name = f"{lab.device_name} [synthetic demo]"
        out[f"demo:{k}"] = lab
    return out


def _doc_text(doc_id: str) -> DocumentText | None:
    man = load_manifest()
    ids = {e.doc_id for e in man.entries}
    if doc_id in ids:
        return load_text(man.get(doc_id))
    syn = _synthetic()
    if doc_id in syn:
        return syn[doc_id][0]
    return None


def _annotation(doc_id: str) -> LabelDocument | None:
    ann = load_annotation(doc_id)
    if ann is not None:
        return ann
    syn = _synthetic()
    if doc_id in syn:
        return syn[doc_id][1]
    return None


def _count_fields(label: LabelDocument | None) -> dict[str, int]:
    out = {"EXTRACTED": 0, "NEEDS_REVIEW": 0}
    if label is None:
        return out
    for cs in label.condition_sets:
        for _, fv in cs.iter_fields():
            if fv.status.value in out:
                out[fv.status.value] += 1
    return out


def _locate(doc: DocumentText, quote: str, page: int | None) -> Evidence:
    if not quote or not quote.strip():
        raise HTTPException(422, "an edit needs a verbatim quote from the document as evidence")
    if word_count(quote) > 25:
        raise HTTPException(422, "evidence quotes are limited to 25 words")
    pages = [p for p in doc.pages if page is None or p.number == page] or doc.pages
    for p in pages:
        spans = find_all(p.text, quote)
        if spans:
            s, e = spans[0]
            return Evidence(
                doc_id=doc.doc_id, page=p.number, char_start=s, char_end=e, quote=p.text[s:e], verified=True
            )
    raise HTTPException(422, f"quote not found verbatim in the document text: {quote!r}")


_ALLOWED_COMPARATORS = {"upper": {"le", "eq", "any"}, "lower": {"ge", "eq", "any"}, "info": {"le", "ge", "eq", "any"}}
B0_MAX_TESLA = 25.0


def _finite(x, what: str) -> float:
    """A finite number from a reviewer's input (bools, text and NaN/infinity are rejected with HTTP 422)."""
    if isinstance(x, bool):
        raise HTTPException(422, f"{what} must be a number")
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"{what} must be a number, got {x!r}") from exc
    if not math.isfinite(v):
        raise HTTPException(422, f"{what} must be a finite number")
    return v


def _parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    raise HTTPException(422, f"expected true or false, got {x!r}")


def _validate_value(name: str, fv: FieldValue) -> None:
    """Reject values a person could not have meant (NaN, out-of-range, unknown tokens) before they become
    human-verified: the checker relies on human-verified values."""
    if name == "mr_status" or fv.status != FieldStatus.EXTRACTED:
        return
    spec = FIELD_SPECS[name]
    label = spec.label.lower()
    if spec.kind == "numeric":
        if fv.comparator not in _ALLOWED_COMPARATORS[spec.direction or "info"]:
            raise HTTPException(
                422,
                f"{spec.label}: comparator {fv.comparator!r} does not fit a "
                f"{'maximum' if spec.direction == 'upper' else 'minimum'} limit",
            )
        if fv.comparator == "any":
            return
        v = _finite(fv.normalized_value, label)
        if spec.plausible and not (spec.plausible[0] <= v <= spec.plausible[1]):
            raise HTTPException(
                422,
                f"{spec.label} {v:g} {spec.canonical_unit} is outside the plausible range "
                f"{spec.plausible[0]:g}-{spec.plausible[1]:g} {spec.canonical_unit}",
            )
    elif spec.kind == "b0":
        vals = fv.normalized_value if isinstance(fv.normalized_value, list) else [fv.normalized_value]
        if not vals:
            raise HTTPException(422, "enter at least one field strength (tesla)")
        for x in vals:
            v = _finite(x, "field strength")
            if not 0 < v <= B0_MAX_TESLA:
                raise HTTPException(422, f"field strength {v:g} T is not plausible")
    elif spec.kind == "categorical":
        toks = fv.normalized_value if isinstance(fv.normalized_value, list) else [fv.normalized_value]
        unknown = [t for t in toks if t not in spec.vocabulary]
        if not toks or unknown:
            raise HTTPException(422, f"{spec.label}: use the options {list(spec.vocabulary)} (got {unknown or toks})")
        if name in ("operating_mode", "scanner_type", "b0_orientation") and len(toks) != 1:
            raise HTTPException(422, f"{spec.label}: choose exactly one option")
    elif spec.kind == "text":
        if not str(fv.normalized_value or "").strip():
            raise HTTPException(422, f"{spec.label}: describe the condition")


def _apply_edit(doc: DocumentText, name: str, d: FieldDecision) -> FieldValue:
    ev = _locate(doc, d.quote or "", d.page)
    if name == "mr_status":
        val = str(d.value or d.normalized_value or "")
        if val not in {s.value for s in MRStatus}:
            raise HTTPException(422, f"mr_status must be one of {[s.value for s in MRStatus]}")
        return FieldValue(status=FieldStatus.EXTRACTED, value=val, normalized_value=val, comparator="eq", evidence=[ev])
    spec = FIELD_SPECS[name]
    norm = d.normalized_value
    nunit = spec.canonical_unit
    comparator = d.comparator
    value = d.value
    if spec.kind == "numeric":
        if d.comparator == "any":
            norm = None
        else:
            info = normalize_unit(d.unit) if d.unit else None
            if d.unit and info is None:
                raise HTTPException(422, f"unrecognised unit {d.unit!r}")
            if info is not None and info.dimension != spec.dimension:
                raise HTTPException(422, f"unit {d.unit} is not a {spec.dimension} unit")
            if d.value is not None:
                value = _finite(d.value, "value")
                computed, nunit = to_canonical(value, d.unit or spec.canonical_unit)  # type: ignore[arg-type]
                if norm is not None and not values_equal(_finite(norm, "normalized value"), computed):
                    raise HTTPException(422, f"normalized value {norm!r} does not match {d.value!r} {d.unit or ''}")
                norm = computed
            elif norm is not None:
                norm = _finite(norm, "normalized value")
            else:
                raise HTTPException(422, "enter the value as stated in the labeling")
        comparator = comparator or {"upper": "le", "lower": "ge", "info": "eq"}[spec.direction or "info"]
    elif spec.kind == "b0":
        vals = norm if norm is not None else d.value
        vals = vals if isinstance(vals, list) else [vals]
        norm = [_finite(v, "field strength") for v in vals]
        comparator = comparator or "in"
        if comparator not in ("in", "le"):
            raise HTTPException(422, "field strengths are a list ('in') or a maximum ('le')")
        nunit = "T"
    elif spec.kind == "bool":
        norm = _parse_bool(d.value if norm is None else norm)
        value = norm
        comparator = "eq"
    else:
        norm = norm if norm is not None else d.value
        comparator = comparator or ("eq" if spec.kind == "text" else "in")
    fv = FieldValue(
        status=FieldStatus.EXTRACTED,
        value=value,
        unit=d.unit,
        normalized_value=norm,
        normalized_unit=nunit,
        comparator=comparator,
        evidence=[ev],
        review_reasons=[f"edited by reviewer: {d.comment}"] if d.comment else [],
    )
    _validate_value(name, fv)
    return fv


def _json_safe(x):
    """Validation errors echo the rejected input; NaN/infinity cannot be rendered as JSON (HTTP 500)."""
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def _ui_dist() -> Path | None:
    env = os.environ.get("MRCOND_UI_DIST")
    candidates = [Path(env)] if env else []
    here = Path(__file__).resolve()
    candidates += [p / "web" / "dist" for p in here.parents]
    for c in candidates:
        if (c / "index.html").exists():
            return c
    return None


# ----------------------------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="MRCond API", version=__version__, description="Research prototype — not a medical device. " + DISCLAIMER
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    extraction_cache: dict[tuple[str, str], LabelDocument] = {}

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # non-finite, negative or absurd inputs are rejected (422), never checked
        return JSONResponse(status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))})

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "disclaimer": DISCLAIMER, "data_dir": str(data_dir())}

    @app.get("/api/fields")
    def fields() -> list[dict]:
        return [
            {
                "name": s.name,
                "label": s.label,
                "kind": s.kind,
                "canonical_unit": s.canonical_unit,
                "direction": s.direction,
                "evaluated": s.evaluated,
                "vocabulary": list(s.vocabulary),
            }
            for s in FIELD_SPECS.values()
        ]

    @app.get("/api/documents")
    def documents(include_synthetic: bool = False) -> list[dict]:
        out = []
        for e in load_manifest().entries:
            verified = verified_path(e.doc_id).exists()
            ann = load_annotation(e.doc_id)
            text = load_text(e)
            out.append(
                {
                    "doc_id": e.doc_id,
                    "title": e.title,
                    "manufacturer": e.manufacturer,
                    "device_name": e.device_name,
                    "device_type": e.device_type,
                    "source": "track_r",
                    "url": e.url,
                    "retrieved": e.retrieved,
                    "text_available": text is not None,
                    "redistributable": e.redistributable,
                    "annotation_status": "human-verified"
                    if verified
                    else (ann.provenance.verification if ann else (e.annotation_status or "none")),
                    "field_counts": _count_fields(ann),
                }
            )
        if include_synthetic:
            for doc_id, (doc, gold) in list(_synthetic().items())[100:110]:
                out.append(
                    {
                        "doc_id": doc_id,
                        "title": f"Synthetic labeling ({doc.meta.get('style')})",
                        "manufacturer": None,
                        "device_name": gold.device_name,
                        "device_type": gold.device_type,
                        "source": "synthetic",
                        "url": None,
                        "retrieved": None,
                        "text_available": True,
                        "redistributable": True,
                        "annotation_status": "synthetic-gold",
                        "field_counts": _count_fields(gold),
                    }
                )
        return out

    @app.get("/api/documents/{doc_id}")
    def document(doc_id: str) -> dict:
        lst = [d for d in documents(include_synthetic=True) if d["doc_id"] == doc_id]
        if not lst:
            raise HTTPException(404, "unknown document")
        return lst[0]

    @app.get("/api/documents/{doc_id}/pages")
    def pages(doc_id: str) -> dict:
        doc = _doc_text(doc_id)
        if doc is None:
            raise HTTPException(
                404, "text not available locally - run `mrcond fetch` (documents are not redistributed)"
            )
        return {
            "doc_id": doc_id,
            "text_extractor": doc.text_extractor,
            "pages": [{"number": p.number, "text": p.text} for p in doc.pages],
        }

    @app.get("/api/documents/{doc_id}/annotation", response_model=LabelDocument)
    def annotation(doc_id: str) -> LabelDocument:
        ann = _annotation(doc_id)
        if ann is None:
            raise HTTPException(404, "no annotation for this document")
        return ann

    @app.get("/api/documents/{doc_id}/extractions/{extractor}", response_model=LabelDocument)
    def extraction(doc_id: str, extractor: str) -> LabelDocument:
        if extractor not in EXTRACTORS:
            raise HTTPException(400, f"extractor must be one of {EXTRACTORS}")
        key = (doc_id, extractor)
        if key in extraction_cache:
            return extraction_cache[key]
        doc = _doc_text(doc_id)
        if doc is None:
            raise HTTPException(404, "text not available locally - run `mrcond fetch`")
        try:
            ex, verify = get_extractor(extractor)
        except ValueError as exc:
            raise HTTPException(400, f"LLM backend not configured (set MRCOND_LLM_* variables): {exc}") from exc
        try:
            label, _ = run_extraction(doc, ex, verify=verify)
        except KeyError as exc:  # replay backend without a recording for this document
            raise HTTPException(404, f"no recorded LLM output for this document: {exc}") from exc
        extraction_cache[key] = label
        return label

    @app.post("/api/documents/{doc_id}/verify", response_model=LabelDocument)
    def verify(doc_id: str, req: VerifyRequest) -> LabelDocument:
        doc = _doc_text(doc_id)
        if doc is None:
            raise HTTPException(404, "text not available locally - run `mrcond fetch`")
        if req.base == "annotation":
            base = _annotation(doc_id)
        elif req.base.startswith("extraction:"):
            base = extraction(doc_id, req.base.split(":", 1)[1])
        else:
            raise HTTPException(422, "base must be 'annotation' or 'extraction:<name>'")
        if base is None:
            raise HTTPException(404, "nothing to verify: no annotation for this document")
        label = base.model_copy(deep=True)
        sets = {cs.set_id: cs for cs in label.condition_sets}
        decided: set[tuple[str, str]] = set()
        log = []
        for d in req.decisions:
            if d.field == "mr_status":
                target = label.mr_status
            else:
                if d.field not in FIELD_NAMES:
                    raise HTTPException(422, f"unknown field {d.field}")
                if d.set_id not in sets:
                    if d.action != "edit":
                        raise HTTPException(422, f"unknown condition set {d.set_id}")
                    sets[d.set_id] = ConditionSet(set_id=d.set_id)
                    label.condition_sets.append(sets[d.set_id])
                target = sets[d.set_id].field(d.field)
            if d.action == "accept":
                if target.status == FieldStatus.NOT_STATED:
                    raise HTTPException(422, f"{d.set_id}.{d.field}: nothing to accept - add a value instead")
                if target.status == FieldStatus.NEEDS_REVIEW:
                    if target.value is None or not target.evidence:
                        raise HTTPException(
                            422,
                            f"{d.set_id}.{d.field}: cannot accept a flagged value without "
                            "value and evidence - edit it instead",
                        )
                    ev = target.evidence[0]
                    _locate(doc, ev.quote, ev.page)  # evidence must still be verbatim
                    target = target.model_copy(
                        update={"status": FieldStatus.EXTRACTED, "review_reasons": ["accepted by reviewer after flag"]}
                    )
                try:
                    _validate_value(d.field, target)
                except HTTPException as exc:
                    raise HTTPException(422, f"{d.set_id}.{d.field}: {exc.detail} - edit it instead") from exc
                new = target
            elif d.action == "reject":
                new = FieldValue(status=FieldStatus.NOT_STATED, review_reasons=["rejected by reviewer"])
            else:
                new = _apply_edit(doc, d.field, d)
            if d.field == "mr_status":
                label.mr_status = new
            else:
                sets[d.set_id].set_field(d.field, new)
            decided.add((d.set_id, d.field))
            log.append(d.model_dump(exclude_none=True))
        undecided = [
            f"{cs.set_id}.{n}"
            for cs in label.condition_sets
            for n, fv in cs.iter_fields()
            if fv.status == FieldStatus.NEEDS_REVIEW and (cs.set_id, n) not in decided
        ]
        if label.mr_status.status == FieldStatus.NEEDS_REVIEW and ("_doc", "mr_status") not in decided:
            undecided.append("_doc.mr_status")
        if undecided:
            raise HTTPException(422, f"fields flagged NEEDS_REVIEW need a decision: {undecided}")
        for cs in label.condition_sets:  # values kept from the base must be valid too
            for n, fv in cs.iter_fields():
                try:
                    _validate_value(n, fv)
                except HTTPException as exc:
                    raise HTTPException(422, f"{cs.set_id}.{n}: {exc.detail} - edit or reject it") from exc
        now = datetime.now(UTC)
        label.provenance = label.provenance.model_copy(
            update={
                "extractor": f"human-verified (base: {req.base}; originally {base.provenance.extractor})",
                "verification": "human-verified",
                "reviewer": req.reviewer,
                "reviewed_at": now,
                "notes": [
                    *base.provenance.notes,
                    f"{len(req.decisions)} field decisions by {req.reviewer}",
                    *([req.notes] if req.notes else []),
                ],
            }
        )
        p = verified_path(doc_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(label.model_dump_json(indent=1, exclude_defaults=True) + "\n", encoding="utf-8")
        with (p.parent / "audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "reviewer": req.reviewer,
                        "at": now.isoformat(),
                        "base": req.base,
                        "decisions": log,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        extraction_cache.clear()
        return label

    def _implants() -> dict[str, tuple[LabelDocument, str]]:
        out: dict[str, tuple[LabelDocument, str]] = {}
        for e in load_manifest().entries:
            ann = load_annotation(e.doc_id)
            if ann is not None:
                src = (
                    "human-verified annotation"
                    if ann.provenance.verification == "human-verified"
                    else "draft annotation (not verified)"
                )
                out[e.doc_id] = (ann, src)
        for k, lab in _demo_implants().items():
            out[k] = (lab, "synthetic demo implant")
        return out

    @app.get("/api/implants")
    def implants() -> list[dict]:
        return [
            {
                "implant_id": k,
                "device_name": lab.device_name,
                "manufacturer": lab.manufacturer,
                "mr_status": lab.status_value().value,
                "verification": lab.provenance.verification,
                "source": src,
                "n_condition_sets": len(lab.condition_sets),
            }
            for k, (lab, src) in _implants().items()
        ]

    @app.get("/api/implants/{implant_id}", response_model=LabelDocument)
    def implant(implant_id: str) -> LabelDocument:
        imps = _implants()
        if implant_id not in imps:
            raise HTTPException(404, "unknown implant")
        return imps[implant_id][0]

    @app.get("/api/scanner-profiles", response_model=list[ScannerProfile])
    def scanner_profiles() -> list[ScannerProfile]:
        return list(PRESETS.values())

    @app.post("/api/check", response_model=CheckResult)
    def run_check(req: CheckApiRequest) -> CheckResult:
        imps = _implants()
        labels = list(req.implants)
        for i in req.implant_ids:
            if i not in imps:
                raise HTTPException(404, f"unknown implant {i}")
            lab = imps[i][0].model_copy(deep=True)
            lab.doc_id = i
            labels.append(lab)
        return check(labels, req.scanner, req.protocol)

    dist = _ui_dist()
    if dist is not None:
        if (dist / "assets").exists():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "unknown API route")
            target = (dist / path).resolve()
            if path and target.is_file() and dist.resolve() in target.parents:
                return FileResponse(target)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
