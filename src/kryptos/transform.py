"""Apply resolved spans to the document.

Replacement happens once, right-to-left, on the original text, so offsets stay
valid throughout.  Turn boundaries are respected: a span is clamped to the
turn it starts in, which keeps the per-turn reconstruction exact.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .config import Config
from .normalize import normalize
from .surrogates import SurrogateAllocator
from .types import ACTION_KEEP, ACTION_PSEUDONYMIZE, Document, Span

SPEAKER_ROLE_ENTITY = {
    "child": "CHILD_NAME",
    "parent": "PARENT_NAME",
    "sibling": "SIBLING_NAME",
    "clinician": "THERAPIST_NAME",
    "teacher": "TEACHER_NAME",
    "unknown": "SPEAKER",
}


def transform(
    doc: Document, spans: list[Span], allocator: SurrogateAllocator, config: Config
) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
    """Return (sanitized_text, sanitized_turns, speaker_label_map)."""
    spans[:] = _clamp_to_turns(doc, spans)
    # Speaker labels are allocated first, and through the same allocator, for
    # two reasons: they must not collide with span pseudonyms (two pools both
    # emitting <PARENT_01> is a real bug), and seeding the pool with the
    # participant's known name makes their later mentions link to the same id.
    speaker_map = _speaker_labels(doc, allocator, config, spans)

    # allocate in document order so <CHILD_01> is the first child mentioned
    for span in sorted(spans, key=lambda s: s.start):
        span.replacement = allocator.allocate(span)

    # Bucket spans by the turn that contains them in one pass.  Scanning the
    # whole span list once per turn is O(turns x spans), which is the dominant
    # cost on long transcripts.
    positions = {id(t): i for i, t in enumerate(doc.turns)}
    buckets: dict[int, list[Span]] = {}
    for span in spans:
        turn = doc.turn_at(span.start)
        if turn is not None:
            buckets.setdefault(positions[id(turn)], []).append(span)

    turns_out: list[dict[str, Any]] = []
    for index, turn in enumerate(doc.turns):
        local = [
            s
            for s in buckets.get(index, ())
            if s.start >= turn.char_start
            and s.end <= turn.char_end
            and s.action != ACTION_KEEP
            and s.replacement != s.text
        ]
        text = _apply(
            turn.text,
            [(s.start - turn.char_start, s.end - turn.char_start, s.replacement) for s in local],
        )
        turns_out.append(
            {
                "speaker": speaker_map.get(turn.speaker, turn.speaker),
                "role": turn.role,
                "text": text,
                "start": turn.start,
                "end": turn.end,
                "index": turn.index,
                "redactions": len(local),
            }
        )

    sanitized = "\n".join(t["text"] for t in turns_out)
    return sanitized, turns_out, speaker_map


def _apply(text: str, edits: list[tuple[int, int, str]]) -> str:
    for start, end, replacement in sorted(edits, key=lambda e: -e[0]):
        if 0 <= start < end <= len(text):
            text = text[:start] + replacement + text[end:]
    return text


def _clamp_to_turns(doc: Document, spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for s in spans:
        turn = doc.turn_at(s.start)
        if turn is None:
            out.append(s)
            continue
        if s.end <= turn.char_end:
            out.append(s)
            continue
        # Preserve every portion of a detection spanning multiple turns.
        for part in doc.turns:
            start, end = max(s.start, part.char_start), min(s.end, part.char_end)
            if start < end:
                out.append(
                    replace(
                        s,
                        start=start,
                        end=end,
                        text=doc.text[start:end],
                        speaker=part.speaker,
                        turn_index=part.index,
                    )
                )
    return out


def _self_introductions(doc: Document, spans: list[Span]) -> dict[str, str]:
    """speaker label -> the name they gave for themselves.

    "CHILD: my name is Aisha" tells us the speaker and the name are the same
    referent.  Without this they get two pseudonyms and the transcript reads
    as if a third person were present.
    """
    out: dict[str, str] = {}
    roles = doc.meta.get("speaker_roles", {})
    for span in sorted(spans, key=lambda s: s.start):
        if not span.speaker or span.speaker in out:
            continue
        role = roles.get(span.speaker, "unknown")
        if span.entity != SPEAKER_ROLE_ENTITY.get(role):
            continue
        if any(src.get("raw_label") == "introduction" for src in span.sources):
            out[span.speaker] = span.canonical_hint or span.text
    return out


def _speaker_labels(
    doc: Document, allocator: SurrogateAllocator, config: Config, spans: list[Span]
) -> dict[str, str]:
    """Speaker labels are themselves identifiers ('AISHA:', 'MUM:')."""
    roles = doc.meta.get("speaker_roles", {})
    known = doc.meta.get("_participants") or doc.meta.get("known_values") or {}
    introduced = _self_introductions(doc, spans)
    mapping: dict[str, str] = {}
    for turn in doc.turns:
        if turn.speaker in mapping:
            continue
        role = roles.get(turn.speaker, turn.role or "unknown")
        entity = SPEAKER_ROLE_ENTITY.get(role, "SPEAKER")
        key = _speaker_key(turn.speaker, entity, known)
        if key.startswith("speaker:") and turn.speaker in introduced:
            key = normalize(introduced[turn.speaker])
        synthetic = Span(
            start=-1,
            end=-1,
            entity=entity,
            tier="direct",
            score=1.0,
            text=key,
            action=ACTION_PSEUDONYMIZE,
            speaker=turn.speaker,
            turn_index=turn.index,
        )
        mapping[turn.speaker] = allocator.pseudonym(synthetic)
    return mapping


def _speaker_key(label: str, entity: str, known: dict[str, Any]) -> str:
    """The referent behind a speaker label, as specifically as we can name it.

    A label that is already a name ("AISHA") identifies the referent directly.
    A generic label ("SPEAKER_01", "PARENT") is resolved to the known
    participant for that role when there is exactly one, so the speaker and
    their in-text mentions share a pseudonym instead of splitting in two.
    """
    from .normalize import normalize

    generic = re.match(r"^(speaker|spk|s|participant|voice)[\s_-]*\d*$", label, re.I)
    role_word = normalize(label) in _GENERIC_SPEAKER_WORDS
    if not generic and not role_word:
        return normalize(label)
    candidates = known.get(entity) or []
    if isinstance(candidates, (str, int, float)):
        candidates = [candidates]
    values = [str(c.get("value") if isinstance(c, dict) else c) for c in candidates]
    if len(values) == 1 and values[0].strip():
        return normalize(values[0])
    return "speaker:" + normalize(label)


_GENERIC_SPEAKER_WORDS = {
    "child",
    "kid",
    "boy",
    "girl",
    "son",
    "daughter",
    "student",
    "pupil",
    "minor",
    "parent",
    "mum",
    "mom",
    "mother",
    "mummy",
    "dad",
    "father",
    "daddy",
    "adult",
    "caregiver",
    "carer",
    "guardian",
    "doctor",
    "clinician",
    "therapist",
    "nurse",
    "interviewer",
    "researcher",
    "assessor",
    "teacher",
    "sibling",
    "brother",
    "sister",
    "unknown",
    "speaker unknown",
    "speaker",
}
