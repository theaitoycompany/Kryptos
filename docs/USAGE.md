# Usage

## Input and offsets

```python
from kryptos import Pipeline

pipeline = Pipeline()
result = pipeline.process_text(
    'CHILD: My name is Aisha.\nPARENT: Hello Aisha.',
    known_values={"CHILD_NAME": ["Aisha"]},
)
```

For JSON, use `{"turns": [{"speaker": "CHILD", "text": "Hello"}]}`. Whisper-style
`segments` and optional word timings are also accepted. Offsets refer to NFC-normalized,
trimmed turn text joined by newlines, excluding speaker labels. They are not byte
offsets in the original input. `Pipeline.process` requires a `Document` with aligned
turns; `process_text` and `process_file` create them for you.

The API mutates its `Document` metadata, roles, and spans while processing. Keep
the original in your own restricted storage if your workflow needs it. Do not use
filenames or raw person IDs as published identifiers; reports drop source IDs.

## Configuration

```python
from kryptos import Config, Pipeline

config = Config.load(overrides={
    "known_values": {"CHILD_NAME": ["Aisha"], "SCHOOL_NAME": ["Greenfield School"]},
    "surrogates": {"scope": "per_conversation"},
})
pipeline = Pipeline(config)
```

Policies are in `src/kryptos/resources/default.json`; entity labels and actions are
in `taxonomy.json`. `Config.load("policy.json")` deep-merges objects; lists, including
`detectors`, are replaced in full. A custom relative `taxonomy_path` resolves relative
to the policy file, or the working directory when supplied as an override. Defaults
and the first-name lexicon are installed inside the package.

Known values support exact, fuzzy, and phonetic matching. Supply only values appropriate
to the document. Phonetic similarity can over-match; short or unusual names can still
be missed. Policy thresholds are heuristics, not calibrated probabilities.

`--set KEY=VALUE` overrides a policy field. Disabling QA, rescanning, or strict detector
failures forces review. Other custom policies can weaken detection; evaluate them on
your own reviewed inputs. Enabling a model is explicit (`--enable gliner_child`).

## Pseudonyms and mappings

The default placeholders reset per conversation. The first detected child becomes
`<CHILD_01>`. Matching known aliases can share a placeholder; this is heuristic entity
linking, not proof that every mention refers to the same person.

`per_turn` resets identity scope for each turn. For keyed, reproducible identifiers,
set `surrogates.style` to `hmac`, provide at least 32 bytes through `KRYPTOS_HMAC_KEY`,
and pass an explicit opaque `doc_id`. The key and document ID define the scope. Different
document IDs produce different pseudonyms. Never embed keys in source or config files.

`surrogates.keep_mapping=true` exposes a sensitive mapping only through
`result.pseudonyms`; it is never serialized by `to_dict` or the CLI. The optional
`surrogate` style uses names from the included lexicon and requires the same review
as other output. Placeholders are the default.

## Review and reports

`result.sanitized_text` is a candidate, even when a detector fails to recognize
identifiers. The API intentionally retains candidates for a local review workflow.
The CLI and demo withhold blocked text and require an explicit option to show review
text. Do not use a passing regex or a `passed` status as proof that all PII is gone.

Span reports include entity labels, offsets, scores, actions, replacements, and
detector provenance. They omit original matched strings and speaker names. Detected
quasi-identifiers may be generalized or retained by policy. Exact timing information
and residual text can remain sensitive. Secure outputs according to your use case.

The library makes no database requests, uploads, or remote inference calls. Optional
model loading downloads weights. `kryptos offsets` explicitly prints unsanitized
text for local annotation; never use it in shared logs.

## Audio plans

When word timings are present, `result.audio_plan` identifies candidate intervals
to mute. `kryptos audio-plan result.json --input recording.wav --output redacted.wav`
prints an ffmpeg command; it does not execute it. Inspect both the command and timing
coverage. Misaligned ASR timings can leave spoken PII. Muting content does not remove
voice identity, background clues, or metadata.
