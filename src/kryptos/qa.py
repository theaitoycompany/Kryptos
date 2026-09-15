"""Output rescan and the fail-closed release gate.

The pipeline's own detectors already fired once.  Running them again on the
*output* is not redundant: it is the only check that the transformation
actually did what the spans said it would, and it catches the case where a
name occurs twice and only one mention was found the first time.

The gate is fail-closed by design.  Token-level F1 is the wrong release
metric for children's data - one transcript out of a hundred that still
carries a full name and a school is an unacceptable system even at 99% F1.
That is what ``conversation_leakage`` measures, and why a single blocking
finding blocks the whole document.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from .config import Config
from .detectors.base import Detector, run_detectors
from .io_formats import build_document
from .normalize import fuzzy_match, normalize
from .types import Document, QAFinding, Span, Turn

PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]{1,96}>")
DIGIT_RUN_RE = re.compile(r"\d[\d\s.-]{3,}\d")
CAP_TOKEN_RE = re.compile(r"(?-i:\b[A-Z][a-z]{2,}\b)")

STATUS_PASSED = "passed"
STATUS_REVIEW = "review"
STATUS_BLOCKED = "blocked"


def run(
    sanitized_turns: list[dict[str, Any]],
    sanitized_text: str,
    doc: Document,
    spans: list[Span],
    detectors: list[Detector],
    config: Config,
    quasi_report: Any | None = None,
) -> tuple[list[QAFinding], str]:
    findings: list[QAFinding] = []
    if not config.get("qa.enabled", True):
        return [QAFinding("qa_disabled", "warn", "Quality checks are disabled")], STATUS_REVIEW
    if not config.get("qa.rescan", True):
        findings.append(QAFinding("rescan_disabled", "warn", "Output rescan is disabled"))
    if not config.get("runtime.strict_detectors", True):
        findings.append(
            QAFinding(
                "detector_failures_allowed",
                "warn",
                "Detector failures are allowed by this configuration",
            )
        )
    if PLACEHOLDER_RE.search(doc.text):
        findings.append(
            QAFinding(
                "input_placeholder_review",
                "warn",
                "Input contains placeholder-like text that requires review",
            )
        )
    if doc.language not in {"en", "eng", "en-US", "en-GB"} or any(
        char.isalpha() and ord(char) > 127 and "LATIN" not in unicodedata.name(char, "")
        for char in doc.text
    ):
        findings.append(
            QAFinding("language_review", "warn", "This language or script requires manual review")
        )

    letters = [char for char in doc.text if char.isalpha()]
    if len(letters) >= 40 and all(not char.isupper() for char in letters):
        findings.append(
            QAFinding(
                "uncased_transcript_review",
                "warn",
                "Long text without capitalization requires review; ASR can obscure names",
            )
        )

    masked, placeholder_spans = _mask_placeholders(sanitized_text)

    findings.extend(_check_known_values(sanitized_text, doc, config))
    if config.get("qa.rescan", True):
        findings.extend(
            _rescan(sanitized_turns, sanitized_text, placeholder_spans, detectors, config, doc)
        )
    findings.extend(_check_digit_runs(masked, config))
    findings.extend(_check_unreplaced(spans))
    findings.extend(_check_asr_confidence(doc, spans, config))
    if config.get("audio.enabled", True) and doc.has_timings():
        incomplete = 0
        for span in spans:
            if span.action == "keep":
                continue
            covered = set()
            for turn in doc.turns:
                for word in turn.words:
                    if word.start is not None and word.end is not None and word.end > word.start:
                        covered.update(
                            range(max(span.start, word.char_start), min(span.end, word.char_end))
                        )
            if any(
                not doc.text[i].isspace() and i not in covered for i in range(span.start, span.end)
            ):
                incomplete += 1
        if incomplete:
            findings.append(
                QAFinding(
                    "audio_alignment_review",
                    "warn",
                    f"{incomplete} transformed span(s) lack complete word timing coverage",
                )
            )
    if config.get("qa.capitalized_token_audit", True):
        findings.extend(_check_capitalised(masked, config))
    if quasi_report is not None:
        findings.extend(_check_quasi(quasi_report, config))

    _apply_severity_policy(findings, config)
    status = decide(findings, config)
    return findings, status


def _apply_severity_policy(findings: list[QAFinding], config: Config) -> None:
    """qa.block_on / qa.warn_on decide which checks can actually block."""
    block_on = set(config.get("qa.block_on", []) or [])
    warn_on = set(config.get("qa.warn_on", []) or [])
    for f in findings:
        if f.check in warn_on and f.severity == "fail":
            f.severity = "warn"
        elif block_on and f.severity == "fail" and f.check not in block_on:
            f.severity = "warn"


def decide(findings: Sequence[QAFinding], config: Config) -> str:
    fail_closed = bool(config.get("qa.fail_closed", True))
    blocking = {f.check for f in findings if f.severity == "fail"}
    if blocking:
        return STATUS_BLOCKED if fail_closed else STATUS_REVIEW
    if any(f.severity == "warn" for f in findings):
        return STATUS_REVIEW
    return STATUS_PASSED


# ---------------------------------------------------------------------------


def _mask_placeholders(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Blank out our own placeholders so they are not re-detected as PII."""
    spans: list[tuple[int, int]] = []
    out = list(text)
    for m in PLACEHOLDER_RE.finditer(text):
        spans.append((m.start(), m.end()))
        for i in range(m.start(), m.end()):
            out[i] = " "
    return "".join(out), spans


def _check_known_values(sanitized: str, doc: Document, config: Config) -> list[QAFinding]:
    """Nothing the caller told us is private may survive, in any spelling."""
    known: dict[str, Any] = {}
    for source in (config.known_values or {}, doc.meta.get("known_values") or {}):
        for k, v in source.items():
            known.setdefault(k, []).extend(v if isinstance(v, list) else [v])

    findings: list[QAFinding] = []
    tokens = [
        (m.start(), m.end(), normalize(m.group(0))) for m in re.finditer(r"[\w'’-]+", sanitized)
    ]
    index = None
    if config.get("qa.indexed_known_values", False):
        index = {}
        for i, token in enumerate(tokens):
            index.setdefault(token[2], []).append(i)
    candidate_cache = {}
    for entity, values in known.items():
        if entity.startswith("_"):
            continue
        if isinstance(values, (str, int, float)):
            values = [values]
        for value in values or []:
            surface = str(value.get("value") if isinstance(value, dict) else value).strip()
            if not surface:
                continue
            parts = [normalize(p) for p in re.findall(r"[\w'’-]+", surface)]
            parts = [p for p in parts if p]
            if not parts:
                continue
            starts = range(len(tokens) - len(parts) + 1)
            if index is not None:
                key = (parts[0], entity.endswith("_NAME"))
                if key not in candidate_cache:
                    candidate_cache[key] = sorted(
                        i
                        for word, positions in index.items()
                        if fuzzy_match(
                            parts[0],
                            word,
                            0.9,
                            phonetic=key[1] and min(len(parts[0]), len(word)) >= 4,
                        )[0]
                        for i in positions
                    )
                starts = (i for i in candidate_cache[key] if i + len(parts) <= len(tokens))
            for i in starts:
                window = tokens[i : i + len(parts)]
                # Short function words can share a phonetic key with a name
                # (Ava/of, Aria/or). They are not sufficient evidence of a leak.
                if all(
                    fuzzy_match(
                        p,
                        w[2],
                        0.9,
                        phonetic=entity.endswith("_NAME") and min(len(p), len(w[2])) >= 4,
                    )[0]
                    for p, w in zip(parts, window)
                ):
                    findings.append(
                        QAFinding(
                            check="known_value_leak",
                            severity="fail",
                            message=f"known {entity} survived sanitisation",
                            detail={
                                "entity": entity,
                                "at": window[0][0],
                                "end": window[-1][1],
                                "context": sanitized[
                                    max(0, window[0][0] - 40) : window[-1][1] + 40
                                ],
                            },
                        )
                    )
    return findings


def _rescan(
    sanitized_turns: list[dict[str, Any]],
    sanitized_text: str,
    placeholder_spans: list[tuple[int, int]],
    detectors: list[Detector],
    config: Config,
    original: Document,
) -> list[QAFinding]:
    """Run the ensemble again on the output."""
    turns = [
        Turn(speaker=t["speaker"], text=t["text"], role=t.get("role", "unknown"))
        for t in sanitized_turns
    ] or [Turn(speaker="S", text=sanitized_text)]
    doc2 = build_document(
        turns, doc_id=(original.doc_id or "") + ":rescan", language=original.language
    )
    doc2.meta["speaker_roles"] = original.meta.get("speaker_roles", {})
    dets = run_detectors(detectors, doc2, parallel=config.get("runtime.parallel_detectors", False))

    findings: list[QAFinding] = []
    max_direct = int(config.get("qa.max_residual_direct", 0))
    residual_direct = 0
    for d in dets:
        if _inside(d.start, d.end, placeholder_spans):
            continue
        if d.meta.get("known_value"):
            continue
        policy = config.taxonomy.policy(d.entity)
        if d.score < config.entity_threshold(d.entity):
            continue
        snippet = doc2.text[max(0, d.start - 40) : d.end + 40]
        if policy.tier == "direct":
            residual_direct += 1
            findings.append(
                QAFinding(
                    check="direct_entity_residual",
                    severity="fail" if residual_direct > max_direct else "warn",
                    message=f"possible {d.entity} remains after sanitisation",
                    detail={
                        "entity": d.entity,
                        "text": doc2.text[d.start : d.end],
                        "detector": d.detector,
                        "score": round(d.score, 3),
                        "context": snippet,
                    },
                )
            )
        elif policy.tier in ("quasi", "sensitive"):
            findings.append(
                QAFinding(
                    check="quasi_residual",
                    severity="warn",
                    message=f"quasi-identifier {d.entity} remains",
                    detail={
                        "entity": d.entity,
                        "text": doc2.text[d.start : d.end],
                        "detector": d.detector,
                        "score": round(d.score, 3),
                    },
                )
            )
    return findings


def _inside(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if s <= start and end <= e:
            return True
    return False


def _check_digit_runs(masked: str, config: Config) -> list[QAFinding]:
    n = int(config.get("qa.digit_run_length", 6))
    findings: list[QAFinding] = []
    for m in DIGIT_RUN_RE.finditer(masked):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) >= n:
            findings.append(
                QAFinding(
                    check="digit_run_residual",
                    severity="fail",
                    message=f"a {len(digits)}-digit run survived sanitisation",
                    detail={"text": m.group(0).strip(), "at": m.start(), "end": m.end()},
                )
            )
    return findings


def _check_unreplaced(spans: list[Span]) -> list[QAFinding]:
    bad = [s for s in spans if s.action != "keep" and not s.replacement]
    if not bad:
        return []
    return [
        QAFinding(
            check="unreplaced_span",
            severity="fail",
            message=f"{len(bad)} span(s) were never given a replacement",
            detail={"entities": sorted({s.entity for s in bad})},
        )
    ]


def _check_asr_confidence(doc: Document, spans: list[Span], config: Config) -> list[QAFinding]:
    """Low ASR confidence around PII means the detectors were reading noise."""
    threshold = float(config.get("qa.min_asr_confidence", 0.5))
    confs: list[float] = []
    for turn in doc.turns:
        for w in turn.words:
            if w.conf is None:
                continue
            for s in spans:
                if w.char_start < s.end and w.char_end > s.start:
                    confs.append(w.conf)
                    break
    if not confs:
        return []
    low = [c for c in confs if c < threshold]
    if not low:
        return []
    return [
        QAFinding(
            check="low_asr_confidence",
            severity="warn",
            message=f"{len(low)} of {len(confs)} words inside detected PII are below "
            f"ASR confidence {threshold:.2f}; recall in these regions is unreliable",
            detail={
                "low": len(low),
                "total": len(confs),
                "mean": round(sum(confs) / len(confs), 3),
            },
        )
    ]


def _check_capitalised(masked: str, config: Config) -> list[QAFinding]:
    from .detectors.context_rules import STOP_CAPS

    suspects: list[str] = []
    for m in CAP_TOKEN_RE.finditer(masked):
        tok = m.group(0)
        prev = masked[: m.start()].rstrip()
        if (not prev) or prev[-1] in ".!?\n":
            continue
        if normalize(tok) in STOP_CAPS:
            continue
        suspects.append(tok)
    if not suspects:
        return []
    uniq = sorted(set(suspects))
    return [
        QAFinding(
            check="capitalized_token_audit",
            severity="warn",
            message=f"{len(uniq)} capitalised token(s) remain and were not classified as PII",
            detail={"tokens": uniq[:25]},
        )
    ]


def _check_quasi(report: Any, config: Config) -> list[QAFinding]:
    score = getattr(report, "risk_score", 0.0)
    if score < 1.0:
        return []
    severity = "warn" if score < 3.0 else "fail"
    return [
        QAFinding(
            check="quasi_combination_risk",
            severity=severity,
            message=f"quasi-identifier combination risk score {score:.2f}",
            detail=report.to_dict() if hasattr(report, "to_dict") else {},
        )
    ]
