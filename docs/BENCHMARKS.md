# Synthetic benchmark results

## Scope

These are reproducible regression diagnostics, not independent real-world accuracy estimates. The fixtures use hand-authored templates and invented identities. Some templates informed the rules; there is no held-out accuracy claim or confidence interval.

Each profile runs 544 synthetic conversations twice: with and without known values. The suite includes 120 written-form, 120 ASR-style, 40 negative controls, 240 language probes (ten languages, written and ASR), and 24 mixed-language probes. That is **2,176 document evaluations across two detector profiles**. Repeated templates and the two knowledge configurations are not independent samples.

Both reports record seed, corpus and source hashes, dependency versions, detector revisions, status counts, per-entity metrics, output hashes, and runtime. All model evaluation runs locally; no private conversations are used.

## Main results

Rows below use the 120-document written and ASR suites with known values. Full breakdowns, including runs without known values and each language, are in [core.json](benchmarks/core.json) and [gliner.json](benchmarks/gliner.json).

| Profile | Input | Exact typed recall | Full-span coverage miss rate | Whole direct surface retained | Passed with retained surface | Median ms/doc |
|---|---|---:|---:|---:|---:|---:|
| core | written | 74.0% | 35.8% | 0/120 | 0/120 | 5.9 |
| core | asr | 60.6% | 92.5% | 51/120 | 0/120 | 6.4 |
| gliner | written | 70.8% | 19.2% | 0/120 | 0/120 | 209.3 |
| gliner | asr | 59.4% | 39.2% | 3/120 | 0/120 | 207.0 |

## How to read the metrics

- **Exact typed recall** requires both the gold span boundaries and entity label to match. Partial and untyped matching are reported separately in JSON.
- **Full-span coverage miss rate** is the fraction of conversations with an annotated direct identifier not covered in full by the union of transformed spans. The JSON field retains the name `conversation_leakage_rate`. It is conservative: even a retained article or space inside a gold span counts as uncovered. It is not measured re-identification probability.
- **Whole direct surface retained** checks whether any complete annotated direct identifier appears verbatim, case-insensitively, in candidate output. This catches exact surface survival but can miss partial names, variants, and combinations of contextual details.
- **Passed with retained surface** counts those cases assigned `passed`. Zero in a row does not prove zero unknown PII. Review and blocked outcomes are explicitly separate.
- **Negative controls** contain no annotated identifiers. Their character redaction fraction measures loss of ordinary text on this fixture set.

## Utility and gates

- **core:** 0.00% of negative-control characters removed; 0 passed documents retained a whole annotated direct surface across all buckets/configurations.
- **gliner:** 5.49% of negative-control characters removed; 0 passed documents retained a whole annotated direct surface across all buckets/configurations.

Long uncased text and unvalidated languages/scripts require review. That withholding is a workflow safeguard, not successful multilingual anonymization. The GLiNER profile can increase coverage and over-redaction simultaneously. Use the full per-language reports when choosing a profile; do not extrapolate from English.

## Reproduce

```bash
uv sync --locked --extra ml --extra fast
uv run --locked --extra ml --extra fast python -m benchmarks.run --profile core \
  --output docs/benchmarks/core.json
HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
uv run --locked --extra ml --extra fast python -m benchmarks.run --profile gliner \
  --device cpu --output docs/benchmarks/gliner.json
uv run --locked python scripts/render_benchmarks.py
```

The first model run downloads pinned weights and tokenizer/backbone files. Afterward, add `HF_HUB_OFFLINE=1` to require local assets. Published runs use Python 3.12.11 on Darwin arm64, with four PyTorch CPU threads and the optional native string matcher enabled. Runtime numbers are hardware- and load-dependent. Cross-platform bit-identical floating-point inference is not promised.

The runner also compares repeated output on one fixed sample, excluding timing fields. This is a repeatability smoke check, not a multi-seed robustness experiment. The checked-in reports retain all bucket results rather than filtering misses.

Core and model source hash: `e545f4915e6691ed759281271f649000b9035d8fba59cccec774e4af715eaeb3`.

Corpus hash: `ad09e8d8d207898538e234ce412aa3d80c6501eb752e2f3db5ad6dc5c5403ece`.
