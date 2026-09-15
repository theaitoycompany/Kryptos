# Release validation

## Production boundary

Kryptos is a local de-identification and review tool. Version 0.1.1 is **not qualified
for unattended publication of anonymized records**. `passed` means the configured
automated checks passed; it does not mean all identifiers were removed.

The independent evaluation below found complete annotated identifiers in some
`passed` outputs. Human review and a representative, independently labelled corpus
from the intended deployment remain necessary. No real child records were used in
this release evaluation. No custom model was trained or tuned on these test rows.

## Engineering checks

- 96 Python tests, including malformed input, offset validation, strict detector
  failures, identity scope, private report output, and actual ffmpeg waveform checks.
- Six CLI self-test cases with 27 assertions, plus installation of the built wheel
  in a new environment with network access disabled for package installation.
- 15 browser tests across Chromium, Firefox, and WebKit: desktop/mobile processing,
  review withholding, clearing, invalid input recovery, corrupt package rejection,
  and stalled engine handling. The processing test asserts no post-load requests
  and no use of local or session storage.
- Ruff checks and formatting; Prettier checks; release inventory and credential scan.
- Python dependency audit: 44 installed third-party distributions, zero known
  vulnerabilities on 2026-09-15. The local Kryptos package is reviewed through source
  and tests; it is not in PyPI's advisory inventory. npm audit found zero advisories
  across the complete demo dependency tree. An advisory scan cannot detect unknown flaws.
- CI covers Python 3.11–3.13 and all three browser engines. See the exact commit's
  [Actions results](https://github.com/theaitoycompany/Kryptos/actions).

The [synthetic regression suite](BENCHMARKS.md) has 2,176 evaluations. Its templates
partly informed the rules, so those results cannot stand in for an independent test.

## Independent public dataset

Source: Gretel AI's [public synthetic PII test split](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1),
Apache-2.0, revision `e06eb1499ca8d54470f085021cd8e54f9efac7fd`.
The [dataset card](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1/blob/e06eb1499ca8d54470f085021cd8e54f9efac7fd/README.md)
describes generated domain documents, rather than real child conversations.

The test takes 200 of 5,000 rows using `random.Random(20260915).sample`, without
filtering by length, domain, label, or outcome. Both profiles use the same rows and
receive **no known-value answer key**. All 874 annotations align exactly to one
surface occurrence; no rows or annotations were dropped. Model pretraining overlap
with this public dataset is unknown, so this is not a proven contamination-free test.

| Measurement | Core | GLiNER |
|---|---:|---:|
| Documents evaluated | 200 | 200 |
| Fully covered annotations, all types | 506/874 | 661/874 |
| Fully covered direct annotations | 461/793 | 598/793 |
| Candidate documents retaining a complete direct surface | 140/200 | 110/200 |
| `passed` documents retaining a complete direct surface | 10/200 | 4/200 |
| `passed` documents with any uncovered direct span | 32/200 | 14/200 |
| Passed / review / blocked | 59 / 0 / 141 | 71 / 3 / 126 |

Complete span coverage requires every annotated character to be transformed. Whole
surface retention checks whether an entire annotated value remains verbatim,
case-insensitively. Neither measures every possible identifying fragment or context.
Direct metrics exclude city, company name, country, generic date/date-time, postcode,
state, and time; all-type counts still include them. Those categories can be retained
or generalized by the contextual policy. This benchmark does not measure utility on
negative controls, real-world accuracy, or re-identification probability.

Machine-readable results include every measured entity type, aggregate status counts,
selected row indices, hashes, and detector revisions:
[core](benchmarks/gretel-core.json), [GLiNER](benchmarks/gretel-gliner.json).
They contain no transcript text or source document IDs.

## Reproduce the independent evaluation

```bash
uv sync --locked --extra ml --extra fast --extra benchmark
umask 077
mkdir -p .cache/public-evaluation
curl --fail --location \
  https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1/resolve/e06eb1499ca8d54470f085021cd8e54f9efac7fd/data/test-00000-of-00001.parquet \
  --output .cache/public-evaluation/gretel-test.parquet
uv run --locked --extra ml --extra fast --extra benchmark \
  python scripts/evaluate_public.py --profile core \
  --data .cache/public-evaluation/gretel-test.parquet \
  --output docs/benchmarks/gretel-core.json
HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
uv run --locked --extra ml --extra fast --extra benchmark \
  python scripts/evaluate_public.py --profile gliner \
  --data .cache/public-evaluation/gretel-test.parquet \
  --output docs/benchmarks/gretel-gliner.json
```

The script verifies the Parquet file's pinned SHA-256 before reading it. Published
runs use the same CPU model configuration and environment as the synthetic regression
report, plus `pyarrow==25.0.1`. Add `HF_HUB_OFFLINE=1` after caching model assets.
Dataset text stays local and outside source control. Aggregate metrics are the only
published dataset-derived output.
