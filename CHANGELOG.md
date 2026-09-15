# Changelog

## 0.1.1 — 2026-09-15

- Reject malformed transcript segments, non-finite timings, invalid annotations,
  and inconsistent document offsets instead of dropping input or reporting success.
- Parse bracketed speaker timestamps and short WebVTT timestamps; preserve numeric
  subtitle speech and align Unicode word tokens after normalization.
- Keep per-turn pseudonyms and speaker labels in separate identity scopes.
- Remove source IDs from evaluation reports and write them atomically with private permissions.
- Correct beep gating and preserve audio outside selected intervals. Incomplete word
  timing coverage now requires review.
- Reject unavailable required detectors, reversed detector offsets, invalid weights,
  unknown identifier labels, and invalid explicit speaker roles. Input placeholders require review.
- Report browser engine download timeouts and processing errors, with recovery tests.
  Test Chromium, Firefox, and WebKit alongside the Python and real ffmpeg checks.
- Exclude generated build metadata from benchmark source hashes. Publish an
  independent public test that documents residual identifiers in some passed outputs.

## 0.1.0 — 2026-09-15

Initial standalone open-source release: installable Python package, CLI, local browser
demo, optional pinned GLiNER detector, synthetic diagnostics, and CI.

The extraction includes stricter detector failures, report minimization, private CLI
output files, package-contained resources, explicit review states, and corrected
cross-turn coverage and evaluation behavior. No internal audit application or data is included.
