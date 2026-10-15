# Validation record

QA review of MRCond 0.1.0, 15 October 2026 (a separate verification pass after the build). This records **software verification** of a research
prototype. It is not clinical validation: MRCond is not a medical device, and the Track R annotations are drafts
that a person still has to verify.

## What was verified

| Area | How | Result |
|---|---|---|
| Checker outcomes never more permissive than the labeling (no "safe" wording; missing or non-checkable conditions → needs review; several implants → worst wins) | `tests/test_checker_properties.py`: 6,000 seeded random label/exam combinations (flagged, missing and malformed values, alternative condition sets, up to 3 implants, values at, within float noise of, and beyond every limit) compared with an independent oracle written from the labeling semantics | 0 disagreements (2,250 no conflict, 2,315 conflict, 1,435 needs review). The original code gave 18 unsafe "No conflict found" on the same cases |
| Hand-written scenarios | `mrcond scenarios` | 73 scenarios, 0 mismatches, 0 unsafe PASS |
| API and UI input handling | regression tests, live API probe, Playwright check of the checker form | NaN, infinite, negative and absurd values are rejected (HTTP 422) or flagged in the form; stale results are cleared |
| Verifier (quote-or-abstain) | adversarial cases in `tests/test_safety_regressions.py` | rejects numbers taken from another quantity, "no limit"/"any coil" claims without such wording, negated status, effective slew rates, categorical values quoted from another field's sentence |
| Evidence offsets | every quote compared with the page text at its offsets | 223 Track R and 1,769 Track S evidence quotes match exactly; all ≤ 25 words |
| Units and boundaries | unit tests and regression tests | 1 T/m = 100 G/cm, 1 T/m/s = 1 mT/m/ms, "3,000 G/cm" = 30 T/m, "1,5 T" = 1.5 T, G/cm/s is a slew rate, limits inclusive with 1e-9 tolerance |
| Benchmark | metric code review; Track S spot check of 10 test documents against their gold; build log | gold matches the text in all 10; dev (0000-0099) and test (0100-0199) splits are disjoint; the rule extractor was tuned on dev documents only and frozen when the test split was first scored; post-hoc changes are listed in the README |
| LLM recordings | prompt hash of every record recomputed with the current code | 122/122 records match prompt v2 and the current context building |
| No redistribution | `.gitignore`; longest run of words copied from manufacturer text in committed files | `data/raw/` and `data/text/` are not committed; recorded quotes capped at 25 words (longest copied run 40 → 25 words) with identical metrics; only FDA (public domain) text is included; no ASTM text |
| External facts in the README | openFDA query re-run; FDA MR guidance, Rose et al., ESR survey, ECR 2026 abstract and FDA CDS guidance pages opened | openFDA counts unchanged (1,141,848 identifiers); citations corrected where needed (CDS guidance now dated January 2026, ECR abstract cited) |
| Clean install | new Python 3.11 venv, `npm ci` in a clean copy of `web/` | 146 passed, 3 skipped (network/live-LLM tests); vitest 25 passed; typecheck and build pass |

## Defects found and fixed

| Severity | Defect | Fix | Test |
|---|---|---|---|
| High | A stated B1+rms, whole-body or head SAR limit that was not a number (e.g. null after a NaN edit) skipped the comparison: "No conflict found" | unreadable limits → needs review (`checker._rf_checks`) | `test_unreadable_rf_limit_is_never_no_conflict` |
| High | NaN, negative and absurd exam values were accepted: NaN SAR, −10 min, −1 T/m gave "No conflict found" through the API | bounds and `allow_inf_nan=False` on the request models, readable 422 errors, defensive checks in the checker | `test_invalid_inputs_are_rejected`, `test_api_rejects_nan_and_negative_values`, `test_checker_defends_against_unvalidated_inputs` |
| Medium | Exam values were assumed when not entered (nucleus 1H, cylindrical scanner, horizontal B0) | defaults removed: not entered → insufficient information | `test_exam_values_are_never_assumed` |
| Medium | Verifier accepted hallucinated values: a SAR of 3 W/kg from "For 3T: … SAR ≤ 2 W/kg", "no limit" and "any coil" without such wording, "not MR Safe" as MR Safe | stricter support rules (see README safety rule 2) | `test_verifier_*` |
| Medium | UI: an invalid B0 kept the previous value, invalid sessions became 1, removing a sequence showed another sequence's values, results stayed visible after inputs changed | validated inputs, stable row keys, results cleared on change and on errors | Playwright check; vitest `numberInputError` |
| Low | Merge was order-dependent for an unreadable limit next to "no limit"; unrecognised operating modes were dropped | → needs review | `test_numeric_merge_never_drops_an_unreadable_limit` |
| Low | "Not configured as required" was treated as missing input; MR Safe records with conditions passed | → conflict; → needs review | `test_device_configuration_not_set_is_a_conflict`, `test_mr_safe_with_conditions_needs_review` |
| Low | "5,000 G/cm/s" was read as a spatial field gradient of 50 T/m; LLM coil values containing "all"/"small" became "any" | G/cm/s slew unit; word-bounded matching | `test_slew_rate_in_gauss_per_cm_per_second_is_not_a_gradient`, `test_llm_coil_normalisation_does_not_read_all_as_any` |
| Low | Reviewer edits were not validated (NaN, out-of-range, "false" read as true, unknown tokens) | validation of edited, accepted and kept values | `test_reviewer_edits_are_validated`, `test_reviewer_bool_text_is_parsed` |
| Metric | The less-conservative count ignored a more permissive MR status and field strengths inside a gold "X T or less" range | both counted (adds 1–3 errors to five rows, disclosed in the README) | `test_more_permissive_mr_status_is_a_less_conservative_error`, `test_field_strength_inside_a_gold_range_is_checked_against_its_limits` |

## How to reproduce

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,reports]" ruff==0.17.1
ruff check . && ruff format --check .
pytest                                    # 146 passed, 3 skipped without network / live LLM
mrcond scenarios                          # 73 scenarios, 0 mismatches, 0 unsafe PASS
mrcond fetch                              # Track R documents (downloaded, never committed)
python scripts/benchmark_report.py        # all tables from data/replay -> reports/ and README
python scripts/cap_replay_quotes.py --check
cd web && npm ci && npm run typecheck && npm test && npm run build
```

`tests/test_safety_regressions.py::test_readme_results_match_the_benchmark_json` fails if the README tables and
`reports/benchmark_results.json` disagree; `test_recordings_match_the_current_prompt` fails if a change to the
prompt or text handling makes the recorded LLM outputs stale.

## Known limitations

* Track R scores use AI-drafted annotations; three long spinal-cord-stimulator manuals are not annotated.
* The local LLM results cover two small quantised models, one prompt version, 21 real and 30 synthetic documents;
  the 3B recording was interrupted once (server out of memory) and resumed with identical settings.
* The prompt v2 example uses `"mr_status": "Not stated"`, which both models copy; MR-status accuracy of the LLM rows
  reflects that.
* `POST /api/check` accepts label records inline without authenticating their provenance (local tool, no accounts).

## Open items for a human MR safety expert

1. Verify the 21 Track R draft annotations field by field in the review page (and annotate the three remaining
   manuals); only then can Track R be called gold.
2. Confirm or change the checker's interpretation choices: condition sets of one implant are alternatives (the best
   applicable set is used); nominal field strengths match within ±0.05 T; an unlisted parameter is no condition
   (FDA 2023), which may not hold for older labels that are silent on RF excitation or transmit coil; a wait time
   only matters for more than one session; head SAR is always required when the labeling states it, even next to a
   B1+rms fallback.
3. Decide how extraction should treat documents that mention several MR statuses (currently MR Conditional wins over
   MR Unsafe) and field strengths that only appear in scoped statements (they create their own condition set).
4. Decide whether saving a draft should require an explicit decision for every value: today untouched values become
   "human-verified" under the reviewer's name.
5. Review the per-axis slew-rate convention: labels that state an effective (vector) slew rate are flagged, not
   converted.
