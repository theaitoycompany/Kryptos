"""Core data structures shared by every stage of the pipeline.

Everything is plain-stdlib dataclasses so the core pipeline runs with no
third-party dependencies.  Offsets are always *character* offsets into the
document's flat ``text`` field, which keeps span fusion, audio alignment and
QA rescanning on a single coordinate system.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# risk tiers / actions
# --------------------------------------------------------------------------

TIER_DIRECT = "direct"  # identifies on its own -> never survives
TIER_QUASI = "quasi"  # identifies in combination -> generalize
TIER_SENSITIVE = "sensitive"  # not identifying, but harmful to retain

ACTION_PSEUDONYMIZE = "pseudonymize"
ACTION_GENERALIZE = "generalize"
ACTION_REDACT = "redact"
ACTION_KEEP = "keep"


@dataclass
class Word:
    """One ASR token with its audio timing, used for audio redaction plans."""

    text: str
    start: float | None = None
    end: float | None = None
    conf: float | None = None
    # character offsets into Document.text, filled in by the transcript loader
    char_start: int = -1
    char_end: int = -1


@dataclass
class Turn:
    """A single speaker turn."""

    speaker: str
    text: str
    start: float | None = None
    end: float | None = None
    words: list[Word] = field(default_factory=list)
    # character offsets of ``text`` inside Document.text
    char_start: int = 0
    char_end: int = 0
    # role assigned by the pipeline: child / parent / clinician / unknown
    role: str = "unknown"
    index: int = 0


@dataclass
class Detection:
    """One detector's claim that ``text[start:end]`` is PII."""

    start: int
    end: int
    entity: str  # canonical entity label (see taxonomy)
    score: float
    detector: str  # which detector produced it
    raw_label: str = ""  # the detector's own label, pre-mapping
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Detection) -> bool:
        return self.start < other.end and other.start < self.end

    def key(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.entity)


@dataclass
class Span:
    """A fused, policy-resolved span: what actually gets transformed."""

    start: int
    end: int
    entity: str
    tier: str
    score: float
    text: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    action: str = ACTION_PSEUDONYMIZE
    replacement: str = ""
    # the real-world value this span refers to, when a detector knew it
    # (a known-value match); used to link ASR variants onto one pseudonym
    canonical_hint: str = ""
    # populated by the quasi-identifier pass
    risk: float = 0.0
    risk_reasons: list[str] = field(default_factory=list)
    # populated by the audio aligner
    audio_start: float | None = None
    audio_end: float | None = None
    speaker: str = ""
    turn_index: int = -1

    @property
    def length(self) -> int:
        return self.end - self.start

    def detectors(self) -> list[str]:
        return sorted({s.get("detector", "?") for s in self.sources})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Document:
    """A whole conversation, plus everything downstream stages attach to it."""

    text: str
    turns: list[Turn] = field(default_factory=list)
    doc_id: str = ""
    language: str = "en"
    meta: dict[str, Any] = field(default_factory=dict)
    # cached turn start offsets for turn_at(); rebuilt when turns change
    _turn_starts: list[int] | None = field(default=None, repr=False, compare=False)
    _turn_fingerprint: tuple[int, int, int] | None = field(default=None, repr=False, compare=False)

    def word_at(self, char_index: int) -> Word | None:
        for turn in self.turns:
            if turn.char_start <= char_index < turn.char_end:
                for w in turn.words:
                    if w.char_start <= char_index < w.char_end:
                        return w
        return None

    def turn_at(self, char_index: int) -> Turn | None:
        # Turns are in document order, so a binary search answers this in
        # O(log n).  A linear scan here is O(turns) per span, which is the
        # difference between linear and quadratic cost on long transcripts.
        starts = self._turn_starts
        fingerprint = (
            len(self.turns),
            self.turns[0].char_start if self.turns else -1,
            self.turns[-1].char_end if self.turns else -1,
        )
        if starts is None or self._turn_fingerprint != fingerprint:
            starts = [t.char_start for t in self.turns]
            self._turn_starts = starts
            self._turn_fingerprint = fingerprint
        i = bisect.bisect_right(starts, char_index) - 1
        # ``char_end`` is inclusive here, so two turns can both contain an
        # index; the caller's contract is the *first* such turn in document
        # order, which is what the linear scan this replaced returned.
        while i > 0 and self.turns[i - 1].char_end >= char_index:
            i -= 1
        if 0 <= i < len(self.turns):
            turn = self.turns[i]
            if turn.char_start <= char_index <= turn.char_end:
                return turn
        # turns were mutated in place behind the cache: fall back to the scan
        for turn in self.turns:
            if turn.char_start <= char_index <= turn.char_end:
                return turn
        return None

    def has_timings(self) -> bool:
        return any(w.start is not None for t in self.turns for w in t.words)


@dataclass
class QAFinding:
    check: str
    severity: str  # "fail" | "warn" | "info"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """The end-to-end output of the pipeline."""

    doc_id: str
    sanitized_text: str
    sanitized_turns: list[dict[str, Any]] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    pseudonyms: dict[str, str] = field(default_factory=dict)
    qa: list[QAFinding] = field(default_factory=list)
    status: str = "unknown"  # passed | review | blocked
    stats: dict[str, Any] = field(default_factory=dict)
    audio_plan: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_detections: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "doc_id": self.doc_id,
            "status": self.status,
            "sanitized_text": self.sanitized_text,
            "sanitized_turns": self.sanitized_turns,
            "spans": [s.to_dict() for s in self.spans],
            "qa": [dataclasses.asdict(f) for f in self.qa],
            "stats": self.stats,
            "audio_plan": self.audio_plan,
        }
        if include_detections:
            out["detections"] = [dataclasses.asdict(d) for d in self.detections]
        return out

    def to_json(self, include_detections: bool = True, indent: int = 2) -> str:
        return json.dumps(self.to_dict(include_detections), indent=indent, ensure_ascii=False)
