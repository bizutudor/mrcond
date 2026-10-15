"""GUDID gap analysis via the openFDA device UDI endpoint.

Counts implantable device identifiers (GMDN ``implantable: true``) by the
``mri_safety`` attribute, overall and by GUDID publish year, and writes CSV
files plus a figure under ``reports/``.

GUDID stores only the MRI safety *category*; the conditions themselves live in
manufacturer labeling (PDF/web pages) - the gap MRCond addresses.

Data licence: openFDA data are released under CC0 (https://open.fda.gov/license/);
GMDN terms are licensed separately by the GMDN Agency - only counts are written
here, no GMDN term text. openFDA disclaimer: "Do not rely on openFDA to make
decisions regarding medical care."
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

__all__ = ["SNAPSHOT_2026_10_15", "count_by_mri_safety", "count_by_year", "run_gap_analysis", "QUERY_URL"]

BASE = "https://api.fda.gov/device/udi.json"
QUERY_URL = f"{BASE}?search=gmdn_terms.implantable:true&count=mri_safety.exact"
CATEGORIES = ["Labeling does not contain MRI Safety Information", "MR Conditional", "MR Safe", "MR Unsafe"]
SHORT = {
    "Labeling does not contain MRI Safety Information": "No MRI safety information in labeling",
    "MR Conditional": "MR Conditional",
    "MR Safe": "MR Safe",
    "MR Unsafe": "MR Unsafe",
}

# Query run on 15 Oct 2026 (openFDA dataset last_updated 2026-10-01):
SNAPSHOT_2026_10_15 = {
    "query": QUERY_URL,
    "retrieved": "2026-10-15",
    "openfda_last_updated": "2026-10-01",
    "total": 1141848,
    "counts": {
        "Labeling does not contain MRI Safety Information": 901918,
        "MR Conditional": 183726,
        "MR Safe": 31129,
        "MR Unsafe": 25075,
    },
}


def _get(url: str, params: dict | None = None, attempts: int = 4) -> dict:
    import time

    import httpx

    for i in range(attempts):
        try:
            r = httpx.get(url, params=params, timeout=60.0, headers={"User-Agent": "MRCond/0.1 (gap analysis)"})
            if r.status_code == 404:  # openFDA returns 404 when a search matches nothing
                return {"results": [], "meta": {}}
            if r.status_code in (429, 500, 502, 503, 504) and i < attempts - 1:
                time.sleep(2**i)
                continue
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.TimeoutException):
            if i == attempts - 1:
                raise
            time.sleep(2**i)
    raise RuntimeError("unreachable")  # pragma: no cover


def count_by_mri_safety(search: str = "gmdn_terms.implantable:true") -> tuple[dict[str, int], dict]:
    data = _get(BASE, {"search": search, "count": "mri_safety.exact"})
    return {r["term"]: int(r["count"]) for r in data.get("results", [])}, data.get("meta", {})


def count_by_year(first: int = 2013, last: int | None = None) -> dict[int, dict[str, int]]:
    last = last or date.today().year
    out: dict[int, dict[str, int]] = {}
    for y in range(first, last + 1):
        counts, _ = count_by_mri_safety(f"gmdn_terms.implantable:true AND publish_date:[{y}-01-01 TO {y}-12-31]")
        if counts:
            out[y] = counts
    return out


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def _figure(counts: dict[str, int], by_year: dict[int, dict[str, int]], path: Path, snapshot_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    series = {
        "Labeling does not contain MRI Safety Information": "#2a78d6",
        "MR Conditional": "#eb6834",
        "MR Safe": "#1baf7a",
        "MR Unsafe": "#eda100",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": axis,
            "axes.labelcolor": ink2,
            "xtick.color": muted,
            "ytick.color": muted,
        }
    )
    ncols = 2 if by_year else 1
    fig, axes = plt.subplots(1, ncols, figsize=(12.5 if by_year else 7.5, 4.6), facecolor=surface)
    axes = axes if ncols == 2 else [axes]
    total = sum(counts.values())

    ax = axes[0]
    ax.set_facecolor(surface)
    cats = [c for c in CATEGORIES if c in counts][::-1]
    vals = [_pct(counts[c], total) for c in cats]
    bars = ax.barh([SHORT[c] for c in cats], vals, color="#2a78d6", height=0.56)
    for b, c, v in zip(bars, cats, vals):
        ax.text(
            b.get_width() + 1.0,
            b.get_y() + b.get_height() / 2,
            f"{v:.1f}%  ({counts[c]:,})",
            va="center",
            color=ink,
            fontsize=9.5,
        )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of implantable device identifiers (%)")
    ax.set_title(
        f"GUDID implantable devices by MRI safety label\n(n = {total:,}; openFDA, {snapshot_label})",
        loc="left",
        color=ink,
        fontsize=11,
    )
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", colors=ink2, length=0)

    if by_year:
        ax = axes[1]
        ax.set_facecolor(surface)
        years = sorted(by_year)
        ends: list[tuple[float, str, str]] = []
        for cat in CATEGORIES:
            ys = []
            for y in years:
                t = sum(by_year[y].values())
                ys.append(_pct(by_year[y].get(cat, 0), t))
            ax.plot(years, ys, color=series[cat], linewidth=2, marker="o", markersize=4, label=SHORT[cat])
            ends.append((ys[-1], f"{ys[-1]:.0f}%", cat))
        # end-of-line labels, pushed apart so they never overlap (min 4.5 points of the y axis)
        ends.sort()
        placed: list[float] = []
        for v, txt, _cat in ends:
            y = max(v, placed[-1] + 4.5) if placed else v
            placed.append(y)
            ax.text(years[-1] + 0.35, y, txt, va="center", color=ink2, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_xlim(years[0] - 0.5, years[-1] + 1.4)
        ax.set_ylabel("Share of DIs published that year (%)")
        partial = " (last year partial)" if years[-1] == date.today().year else ""
        ax.set_title(f"By GUDID publish year{partial}", loc="left", color=ink, fontsize=11)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.02, 0.47), labelcolor=ink2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def run_gap_analysis(out_dir: str | Path = "reports", offline: bool = False, by_year: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if offline:
        counts, meta = (
            dict(SNAPSHOT_2026_10_15["counts"]),
            {"last_updated": SNAPSHOT_2026_10_15["openfda_last_updated"]},
        )
        retrieved = SNAPSHOT_2026_10_15["retrieved"]
        years: dict[int, dict[str, int]] = {}
        yp = out / "gudid_mri_safety_by_publish_year.csv"
        if by_year and yp.exists():  # reuse a previous live run
            with yp.open() as f:
                for row in csv.DictReader(f):
                    years.setdefault(int(row["publish_year"]), {})[row["mri_safety"]] = int(row["count"])
    else:
        counts, meta = count_by_mri_safety()
        retrieved = date.today().isoformat()
        years = count_by_year() if by_year else {}
    total = sum(counts.values())
    with (out / "gudid_mri_safety_counts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["mri_safety", "count", "percent", "total_implantable_dis", "retrieved", "openfda_last_updated", "query"]
        )
        for c in CATEGORIES:
            w.writerow(
                [
                    c,
                    counts.get(c, 0),
                    _pct(counts.get(c, 0), total),
                    total,
                    retrieved,
                    meta.get("last_updated", ""),
                    QUERY_URL,
                ]
            )
    if years:
        with (out / "gudid_mri_safety_by_publish_year.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["publish_year", "mri_safety", "count", "share_percent"])
            for y in sorted(years):
                t = sum(years[y].values())
                for c in CATEGORIES:
                    w.writerow([y, c, years[y].get(c, 0), _pct(years[y].get(c, 0), t)])
    snap = {
        "retrieved": retrieved,
        "openfda_last_updated": meta.get("last_updated"),
        "query": QUERY_URL,
        "total": total,
        "counts": counts,
        "by_publish_year": {str(k): v for k, v in sorted(years.items())},
        "license": "openFDA data: CC0 (https://open.fda.gov/license/). GMDN terms: separate GMDN Agency licence "
        "(no term text included).",
        "matches_2026_10_15_snapshot": counts == SNAPSHOT_2026_10_15["counts"],
    }
    (out / "gudid_mri_safety_snapshot.json").write_text(json.dumps(snap, indent=1), encoding="utf-8")
    try:
        _figure(counts, years, out / "gudid_mri_safety.png", f"retrieved {retrieved}")
        snap["figure"] = str(out / "gudid_mri_safety.png")
    except ImportError:  # matplotlib is optional ("reports" extra)
        snap["figure"] = None
    snap["percent"] = {c: _pct(counts.get(c, 0), total) for c in CATEGORIES}
    return snap
