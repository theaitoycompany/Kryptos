"""End-to-end orchestration.

raw transcript
    -> parse + metadata strip
    -> speaker role inference
    -> detector ensemble over the original text
    -> span fusion with provenance
    -> quasi-identifier risk pass
    -> pseudonymise / generalise / redact
    -> output rescan
    -> fail-closed QA gate
    -> sanitized transcript (+ audio redaction plan)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from . import audio as audio_mod
from . import qa as qa_mod
from . import quasi as quasi_mod
from .config import Config
from .detectors.base import Detector, build_detectors, run_detectors
from .fusion import fuse
from .io_formats import load_file, load_text, validate_document
from .roles import assign_roles
from .surrogates import SurrogateAllocator
from .transform import transform
from .types import Detection, Document, Result, Span

log = logging.getLogger("kryptos.pipeline")

# metadata keys that are themselves identifiers and must never ride along
_UNSAFE_META = re.compile(
    r"(file|path|name|user|account|device|serial|imei|mac|ip|email|phone|owner|"
    r"author|creator|location|gps|lat|lon|session_user|upload|url|token|key)",
    re.I,
)
_SAFE_META_KEYS = {
    "language",
    "lang",
    "duration",
    "sample_rate",
    "model",
    "task",
    "known_values",
    "speaker_roles",
    "source_basename_removed",
}


class Pipeline:
    def __init__(
        self,
        config: Config | None = None,
        detectors: list[Detector] | None = None,
        only: list[str] | None = None,
    ):
        self.config = config or Config.load()
        self.detectors = (
            detectors if detectors is not None else build_detectors(self.config, only=only)
        )
        self.available = [d for d in self.detectors if d.available]
        if self.config.get("runtime.strict_detectors", True) and len(self.available) != len(
            self.detectors
        ):
            raise RuntimeError("Required detector is unavailable")
        if not self.available:
            raise RuntimeError("no detectors are available - refusing to run")

    # ------------------------------------------------------------------
    def process_text(
        self,
        text: str,
        doc_id: str = "",
        fmt: str = "auto",
        known_values: dict[str, Any] | None = None,
        speaker_roles: dict[str, str] | None = None,
    ) -> Result:
        doc = load_text(text, doc_id=doc_id, fmt=fmt)
        return self.process(doc, known_values=known_values, speaker_roles=speaker_roles)

    def process_file(
        self,
        path: str,
        fmt: str = "auto",
        known_values: dict[str, Any] | None = None,
        speaker_roles: dict[str, str] | None = None,
    ) -> Result:
        doc = load_file(path, fmt=fmt)
        return self.process(doc, known_values=known_values, speaker_roles=speaker_roles)

    # ------------------------------------------------------------------
    def process(
        self,
        doc: Document,
        known_values: dict[str, Any] | None = None,
        speaker_roles: dict[str, str] | None = None,
    ) -> Result:
        t0 = time.time()
        validate_document(doc)
        strip_metadata(doc)
        if known_values:
            merged = dict(doc.meta.get("known_values") or {})
            for k, v in known_values.items():
                cur = merged.get(k) or []
                if isinstance(cur, (str, int, float)):
                    cur = [cur]
                merged[k] = list(cur) + (v if isinstance(v, list) else [v])
            doc.meta["known_values"] = merged

        all_known = _merged_known(self.config, doc)
        # config-level and per-document known values, in one place, for the
        # speaker-label allocator as well as the detectors
        doc.meta["_participants"] = all_known
        role_overrides = dict(doc.meta.get("speaker_roles") or {})
        role_overrides.update(speaker_roles or {})
        assign_roles(doc, known_values=all_known, overrides=role_overrides)

        detections: list[Detection] = run_detectors(
            self.available, doc, parallel=self.config.get("runtime.parallel_detectors", False)
        )
        t_detect = time.time()

        spans: list[Span] = fuse(detections, doc, self.config)
        quasi_report = quasi_mod.apply(doc, spans, self.config)

        allocator = SurrogateAllocator(self.config, doc)
        sanitized_text, sanitized_turns, speaker_map = transform(doc, spans, allocator, self.config)

        audio_plan = audio_mod.plan(doc, spans, self.config)

        findings, status = qa_mod.run(
            sanitized_turns, sanitized_text, doc, spans, self.available, self.config, quasi_report
        )

        result = Result(
            doc_id="",
            sanitized_text=sanitized_text,
            sanitized_turns=sanitized_turns,
            spans=spans,
            detections=detections,
            pseudonyms=(
                allocator.mapping() if self.config.get("surrogates.keep_mapping", False) else {}
            ),
            qa=findings,
            status=status,
            audio_plan=audio_plan,
        )
        result.stats = self._stats(
            doc,
            detections,
            spans,
            quasi_report,
            allocator,
            speaker_map,
            audio_plan,
            findings,
            timings={"detect_s": round(t_detect - t0, 3), "total_s": round(time.time() - t0, 3)},
        )
        _redact_report(result, self.config)
        return result

    # ------------------------------------------------------------------
    def _stats(
        self,
        doc,
        detections,
        spans,
        quasi_report,
        allocator,
        speaker_map,
        audio_plan,
        findings,
        timings,
    ) -> dict[str, Any]:
        by_entity: dict[str, int] = {}
        by_action: dict[str, int] = {}
        for s in spans:
            by_entity[s.entity] = by_entity.get(s.entity, 0) + 1
            by_action[s.action] = by_action.get(s.action, 0) + 1

        contribution: dict[str, dict[str, int]] = {}
        for s in spans:
            for name in s.detectors():
                node = contribution.setdefault(name, {"spans": 0, "sole_source": 0})
                node["spans"] += 1
            if len(s.detectors()) == 1:
                contribution[s.detectors()[0]]["sole_source"] += 1

        direct = sum(1 for s in spans if s.tier == "direct")
        chars = len(doc.text)
        redacted_chars = sum(s.length for s in spans if s.action != "keep")

        return {
            "detectors": [d.describe() for d in self.detectors],
            "n_detections": len(detections),
            "n_spans": len(spans),
            "spans_by_entity": dict(sorted(by_entity.items(), key=lambda kv: -kv[1])),
            "spans_by_action": by_action,
            "direct_spans": direct,
            "detector_contribution": contribution,
            "speaker_roles": {speaker_map[turn.index]: turn.role for turn in doc.turns},
            "pseudonym_pools": allocator.inventory(),
            "quasi": quasi_report.to_dict(),
            "audio": audio_mod.coverage(doc, audio_plan) if audio_plan else {},
            "characters": chars,
            "redacted_characters": redacted_chars,
            "redacted_fraction": round(redacted_chars / chars, 4) if chars else 0.0,
            "turns": len(doc.turns),
            "qa_counts": {
                sev: sum(1 for f in findings if f.severity == sev)
                for sev in ("fail", "warn", "info")
            },
            "timings": timings,
        }


# ---------------------------------------------------------------------------


def strip_metadata(doc: Document) -> Document:
    """Drop anything in the transcript metadata that could identify anyone."""
    removed: list[str] = []
    clean: dict[str, Any] = {}
    for key, value in (doc.meta or {}).items():
        if key in _SAFE_META_KEYS:
            clean[key] = value
        elif _UNSAFE_META.search(key):
            removed.append(key)
        elif isinstance(value, (int, float, bool)):
            clean[key] = value
        else:
            removed.append(key)
    if removed:
        clean["_removed_metadata_keys"] = sorted(removed)
    doc.meta = clean
    return doc


def _merged_known(config: Config, doc: Document) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for source in (config.known_values or {}, doc.meta.get("known_values") or {}):
        for k, v in source.items():
            if k.startswith("_"):
                continue
            if k not in config.taxonomy.entities:
                raise ValueError("Known identifiers require canonical entity labels")
            vals = v if isinstance(v, list) else [v]
            out.setdefault(k, []).extend(
                str(x.get("value") if isinstance(x, dict) else x) for x in vals
            )
    return out


def _redact_report(result: Result, config: Config) -> None:
    """The report itself must not become a copy of the PII."""
    for s in result.spans:
        if s.action == "keep":
            s.replacement = ""
        s.text = ""
        s.canonical_hint = ""
        s.speaker = ""
        s.sources = [
            {
                key: value
                for key, value in src.items()
                if key in {"detector", "entity", "score", "start", "end", "channel"}
            }
            for src in s.sources
        ]
    for d in result.detections:
        d.text = ""
        d.raw_label = ""
        d.meta = {
            key: value
            for key, value in d.meta.items()
            if key in {"channel", "known_value", "partial"}
            and isinstance(value, (str, bool, int, float))
        }
    for f in result.qa:
        f.detail = {
            key: value
            for key, value in f.detail.items()
            if key not in {"context", "text", "tokens", "uniqueness_claims"}
        }
    for interval in result.audio_plan:
        interval.pop("speaker", None)
    result.stats.get("quasi", {}).pop("uniqueness_claims", None)


def process_text(text: str, config: Config | None = None, **kwargs) -> Result:
    return Pipeline(config).process_text(text, **kwargs)
