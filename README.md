# Kryptos

**Local transcript de-identification, with role-aware pseudonyms and explicit review gates.**

[Try the browser demo](https://huggingface.co/spaces/the-ai-toy-company/Kryptos) ·
[Benchmarks](docs/BENCHMARKS.md) · [Configuration](docs/USAGE.md) · [Apache-2.0](LICENSE)

Kryptos detects names, contact details, and contextual identifiers in conversation
transcripts. It replaces detected spans, preserves speaker turns, and checks the
result for possible residual identifiers. The Python core has no runtime dependencies
and makes no network requests. Optional models run locally after downloading weights.

This is a de-identification tool, not a guarantee of anonymity. Synthetic benchmarks
show misses and over-redaction, especially on ASR and multilingual text. Review output
before sharing. A sanitized transcript does not anonymize a recording or a voice.

## Quick start

Python 3.11 or later:

```bash
python -m pip install "git+https://github.com/theaitoycompany/Kryptos.git@v0.1.0"
```

```python
from kryptos import Pipeline

result = Pipeline().process_text(
    "My name is Aisha. You can reach me at parent@example.com.",
    known_values={"CHILD_NAME": ["Aisha"]},
)

if result.status == "passed":
    print(result.sanitized_text)
else:
    print("Local review required:", result.status)
```

```text
My name is <CHILD_01>. You can reach me at <EMAIL_01>.
```

The distribution is named `kryptos-pii`; the Python import and command are `kryptos`.
Install from the GitHub tag or its release wheel. This project is not published on PyPI.

### Command line

From a cloned repository:

```bash
python -m pip install .
kryptos run examples/transcript.txt --known examples/known.json --turns
kryptos run examples/transcript.txt --known examples/known.json --output result.json
kryptos detectors
kryptos selftest
```

| Status | Exit code | Meaning |
|---|---:|---|
| `passed` | 0 | Enabled checks found no blocking or review findings. Identifiers can still be missed. |
| `review` | 10 | Manual review required. CLI text is withheld unless `--allow-review` is explicit. |
| `blocked` | 20 | A blocking check fired. The CLI and demo withhold candidate text. |
| Error | 2 | Processing failed; a required detector may be unavailable. |

The Python API returns candidate text for local inspection, including blocked results.
Check `result.status` before using it. Reports omit source document IDs, original
speaker labels, matched text, and QA snippets. Output files are created with owner-only
permissions on POSIX systems. Source text and unknown residual PII still require care.

## Optional models

```bash
python -m pip install '.[ml]'
kryptos run examples/transcript.txt --known examples/known.json --enable gliner_child
```

The GLiNER profile adds schema-conditioned entity detection to the same core pipeline.
Its weights and revision are pinned; enabled detectors must load and run successfully.
Weights are downloaded from Hugging Face on first use. Public weights need no token.
The browser demo uses the dependency-free core and does not load these models.

See [model setup](docs/MODELS.md) for the exact revision, offline use, the generic
token-classification adapter, and compatibility limits. Model licenses are separate
from this repository's license. No custom-trained weights are included.

## Reproduce

```bash
git clone https://github.com/theaitoycompany/Kryptos.git
cd Kryptos
git checkout v0.1.0
uv sync --locked
uv run --locked python -m unittest discover -s tests -q
uv run --locked kryptos selftest
uv run --locked ruff check src tests benchmarks scripts
```

`uv.lock` pins the Python environment, `package-lock.json` pins browser tooling,
and model configurations pin weight revisions. [Benchmark commands and measured
limits](docs/BENCHMARKS.md) cover the core and GLiNER profiles on the same synthetic
fixtures. No production data is needed or included.

### Run your own browser demo

```bash
python scripts/build_space.py
python -m http.server 8765 --bind 127.0.0.1 --directory .build/space
```

Open `http://127.0.0.1:8765`. You can also duplicate the public Hugging Face Space.
The app runs this same Python core in a browser worker with Pyodide 0.28.3. Text is
neither uploaded by the app nor saved in browser storage. The browser downloads
static application and runtime assets; the worker verifies the core archive hash.

## How it works

1. Parse turns and normalize character offsets.
2. Run known-value, structured-identifier, and contextual detectors on the original text.
3. Merge spans, apply entity policy, and assess combinations of contextual details.
4. Allocate placeholders and transform the text once.
5. Rescan the output and assign `passed`, `review`, or `blocked`.

The output rescan reuses the configured detectors; it is not an independent accuracy
validation. Long text without capitalization, non-Latin scripts, and explicitly
non-English transcripts require review. These checks reduce automatic release; they
do not establish detection quality for those inputs.

Supported inputs include text, speaker-labelled text, JSON turns, Whisper-style
segments, JSONL, SRT, and VTT. Word timings can produce an audio redaction plan;
audio files are not processed by the library. See [usage](docs/USAGE.md).

## Development

```bash
uv sync --locked
uv run --locked python -m build
uv run --locked python scripts/check_distribution.py
uv run --locked python scripts/check_release.py
npm ci --ignore-scripts
npx playwright install chromium
npm test
```

CI covers Python 3.11–3.13, synthetic regressions, package installation outside the
checkout, code formatting, and browser behavior. Browser tests check that processing
sends no network requests after the engine loads.

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md)
