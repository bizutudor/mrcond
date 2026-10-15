"""Capture README screenshots of the web UI with Playwright (Chromium).

Prerequisites: a running server with the built UI (`npm --prefix web run build && mrcond serve`)
and the Python `playwright` package with a Chromium build. If Chromium is installed
elsewhere, set PLAYWRIGHT_BROWSERS_PATH or MRCOND_CHROMIUM (path to the executable).

    python scripts/screenshots.py --base-url http://127.0.0.1:8000 --out docs/screenshots

Only public-domain (FDA guidance example) labeling and synthetic demo implants are
shown, so the images contain no manufacturer text.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="docs/screenshots")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    launch: dict = {}
    if os.environ.get("MRCOND_CHROMIUM"):
        launch["executable_path"] = os.environ["MRCOND_CHROMIUM"]
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1500, "height": 980}, device_scale_factor=1)

        # 1. Review page: FDA Table 4a example with highlighted evidence, B1+rms selected.
        page.goto(f"{args.base_url}/#/review?doc=fda-guidance-2023-t4a")
        page.wait_for_selector("mark.ev")
        page.get_by_role("tab", name="Whole-body MR examinations, 3 T").click()
        page.locator(".field-head", has_text="Max B1+rms").click()
        page.wait_for_timeout(600)
        page.locator(".fieldspane .settabs").evaluate("t => t.scrollIntoView({block: 'start'})")
        page.wait_for_timeout(200)
        page.screenshot(path=str(out / "review.png"))

        # 2. Checker page: three implants, generic 3 T preset, mixed per-condition results.
        page.goto(f"{args.base_url}/#/check")
        page.wait_for_selector(".implant-pick")
        for name in [
            "Star Active Implant System, whole-body exams",
            "Passive implant (FDA Table 3 style)",
            "Legacy passive implant",
        ]:
            page.locator(".implant-pick", has_text=name).first.locator("input").check()
        # show the first selected implant at the top of the scrollable list
        page.locator(".implant-list").evaluate(
            "ul => { const li = ul.querySelector('input:checked').closest('li'); ul.scrollTop = li.offsetTop - ul.offsetTop; }"
        )
        page.get_by_label("Spatial field gradient at the implant").fill("12")
        page.get_by_role("button", name="Compare with labeling").click()
        page.wait_for_selector(".outcome")
        page.set_viewport_size({"width": 1500, "height": 1250})
        page.wait_for_timeout(400)
        page.screenshot(path=str(out / "checker.png"))

        # 3. Narrow viewport check (layout must not overflow on phones).
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        mobile.goto(f"{args.base_url}/#/review?doc=fda-guidance-2023-t3")
        mobile.wait_for_selector("mark.ev")
        mobile.screenshot(path=str(out / "review-mobile.png"))
        browser.close()
    for f in sorted(out.glob("*.png")):
        print(f)


if __name__ == "__main__":
    main()
