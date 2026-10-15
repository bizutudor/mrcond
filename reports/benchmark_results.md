# MRCond benchmark results

Generated: 2026-10-15T17:07:34+00:00 (MRCond 0.1.0, prompt v2)

## Setup

```json
{
 "models": {
  "qwen2.5-1.5b-instruct-q4_k_m": "Qwen2.5-1.5B-Instruct, GGUF Q4_K_M (Qwen/Qwen2.5-1.5B-Instruct-GGUF on Hugging Face)",
  "qwen2.5-3b-instruct-q4_k_m": "Qwen2.5-3B-Instruct, GGUF Q4_K_M (Qwen/Qwen2.5-3B-Instruct-GGUF on Hugging Face)"
 },
 "runtime": "llama-cpp-python 0.3.36 OpenAI-compatible server, n_ctx 8192, 1 generation thread, 2 batch threads",
 "hardware": "2 vCPU (Intel Xeon 2.1 GHz, AVX2), 8 GB RAM, no GPU; about 1.5-2 min per document",
 "decoding": "run 0: temperature 0 (greedy), seed 0; runs 1-2 (consistency, 1.5B only, first 10 Track R documents): temperature 0.7, seeds 1 and 2; max_tokens 1000; JSON schema passed as response_format json_object",
 "context": "keyword-retrieved statements, at most 6000 characters, page markers kept",
 "llm_documents": "Track R: all annotated documents; Track S: the first 30 test documents (synth-20261015-0100..0129)",
 "notes": "The 3B server was OOM-killed once (after 26 of 51 documents) and restarted with identical settings (plus use_mlock false); the remaining documents were recorded afterwards."
}
```

## Track S

Track S (synthetic, exact ground truth), test split

Documents: 100

| Extractor | Docs | Precision | Recall | F1 | MR status acc. | Unsupported values | Less-conservative errors (rate) | Abstention | Selective acc. | Run-to-run consistency |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 100 | 96.6 % | 98.6 % | 97.6 % | 100.0 % | 0/856 | 6 (0.6 %) | 0.0 % | 96.3 % | deterministic |
| rule + verifier | 100 | 100.0 % | 94.1 % | 96.9 % | 100.0 % | 0/798 | 3 (0.3 %) | 7.9 % | 99.7 % | deterministic |
| rule (LLM subset, 30 docs) | 30 | 98.6 % | 99.6 % | 99.1 % | 100.0 % | 0/238 | 0 (0.0 %) | 0.0 % | 98.6 % | deterministic |
| rule + verifier (LLM subset, 30 docs) | 30 | 100.0 % | 98.2 % | 99.1 % | 100.0 % | 0/230 | 0 (0.0 %) | 2.9 % | 100.0 % | deterministic |
| LLM, qwen2.5-1.5b-instruct-q4_k_m | 30 | 78.9 % | 40.9 % | 53.8 % | 0.0 % | 12/97 | 121 (44.2 %) | 0.0 % | 37.7 % | — |
| LLM + verifier, qwen2.5-1.5b-instruct-q4_k_m | 30 | 96.5 % | 30.7 % | 46.5 % | 0.0 % | 0/59 | 118 (43.1 %) | 14.8 % | 33.1 % | — |
| LLM, qwen2.5-3b-instruct-q4_k_m | 30 | 93.9 % | 73.0 % | 82.1 % | 20.0 % | 32/152 | 46 (16.8 %) | 0.0 % | 70.7 % | — |
| LLM + verifier, qwen2.5-3b-instruct-q4_k_m | 30 | 98.7 % | 56.2 % | 71.6 % | 10.0 % | 0/110 | 43 (15.7 %) | 15.2 % | 64.2 % | — |

Less-conservative errors by kind:

* rule: SAR wrongly marked as B1+rms fallback 3, missed constraint 3
* rule + verifier: missed constraint 3
* LLM, qwen2.5-1.5b-instruct-q4_k_m: higher limit 5, missed constraint 113, more permissive MR status 3
* LLM + verifier, qwen2.5-1.5b-instruct-q4_k_m: missed constraint 118
* LLM, qwen2.5-3b-instruct-q4_k_m: extra field strength 2, higher limit 1, missed constraint 41, more permissive MR status 2
* LLM + verifier, qwen2.5-3b-instruct-q4_k_m: extra field strength 2, missed constraint 41

* Documents for `rule (LLM subset, 30 docs)`, `rule+verifier (LLM subset, 30 docs)`, `llm:qwen2.5-1.5b-instruct-q4_k_m`, `llm+verifier:qwen2.5-1.5b-instruct-q4_k_m`, `llm:qwen2.5-3b-instruct-q4_k_m`, `llm+verifier:qwen2.5-3b-instruct-q4_k_m`: synth-20261015-0100, synth-20261015-0101, synth-20261015-0102, synth-20261015-0103, synth-20261015-0104, synth-20261015-0105, synth-20261015-0106, synth-20261015-0107, synth-20261015-0108, synth-20261015-0109, synth-20261015-0110, synth-20261015-0111, synth-20261015-0112, synth-20261015-0113, synth-20261015-0114, synth-20261015-0115, synth-20261015-0116, synth-20261015-0117, synth-20261015-0118, synth-20261015-0119, synth-20261015-0120, synth-20261015-0121, synth-20261015-0122, synth-20261015-0123, synth-20261015-0124, synth-20261015-0125, synth-20261015-0126, synth-20261015-0127, synth-20261015-0128, synth-20261015-0129

## Track R

Track R (real public labeling) - on DRAFT annotations pending human verification

Documents: 21

| Extractor | Docs | Precision | Recall | F1 | MR status acc. | Unsupported values | Less-conservative errors (rate) | Abstention | Selective acc. | Run-to-run consistency |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 21 | 92.6 % | 85.2 % | 88.8 % | 85.7 % | 0/193 | 12 (4.3 %) | 0.0 % | 82.6 % | deterministic |
| rule + verifier | 21 | 93.2 % | 83.8 % | 88.3 % | 85.7 % | 0/188 | 12 (4.3 %) | 2.1 % | 82.9 % | deterministic |
| LLM, qwen2.5-1.5b-instruct-q4_k_m | 21 | 75.2 % | 29.5 % | 42.4 % | 33.3 % | 21/69 | 141 (50.7 %) | 0.0 % | 27.4 % | 34.3 % (10 docs × 3 runs) |
| LLM + verifier, qwen2.5-1.5b-instruct-q4_k_m | 21 | 97.9 % | 16.9 % | 28.8 % | 28.6 % | 0/32 | 135 (48.6 %) | 17.4 % | 19.0 % | 26.0 % (10 docs × 3 runs) |
| LLM, qwen2.5-3b-instruct-q4_k_m | 21 | 91.0 % | 54.7 % | 68.3 % | 23.8 % | 26/108 | 83 (29.9 %) | 0.0 % | 52.6 % | — |
| LLM + verifier, qwen2.5-3b-instruct-q4_k_m | 21 | 98.2 % | 40.3 % | 57.1 % | 14.3 % | 0/73 | 78 (28.1 %) | 15.4 % | 46.3 % | — |

Less-conservative errors by kind:

* rule: extra field strength 1, higher limit 3, missed constraint 7, more permissive MR status 1
* rule + verifier: extra field strength 1, higher limit 3, missed constraint 7, more permissive MR status 1
* LLM, qwen2.5-1.5b-instruct-q4_k_m: higher limit 6, missed constraint 135
* LLM + verifier, qwen2.5-1.5b-instruct-q4_k_m: missed constraint 135
* LLM, qwen2.5-3b-instruct-q4_k_m: extra field strength 1, higher limit 4, missed constraint 78
* LLM + verifier, qwen2.5-3b-instruct-q4_k_m: higher limit 1, missed constraint 77


