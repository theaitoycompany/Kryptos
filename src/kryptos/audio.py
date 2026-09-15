"""Audio redaction planning.

The sanitized transcript says nothing about the recording.  If the audio must
leave the restricted zone as well, two independent things have to happen:

1. spoken PII is removed from the waveform - that is what this module plans,
   using the word timings the ASR already produced
2. the *voice* is anonymised - a separate problem this module does not solve
   and must not be mistaken for; see docs/AUDIO.md

Emitting a plan rather than editing audio is deliberate: the plan is
reviewable, and the edit is a single deterministic ffmpeg invocation.
"""

from __future__ import annotations

from typing import Any

from .config import Config
from .types import ACTION_KEEP, Document, Span


def plan(doc: Document, spans: list[Span], config: Config) -> list[dict[str, Any]]:
    """Map character spans onto audio intervals via word timings."""
    if not config.get("audio.enabled", True) or not doc.has_timings():
        return []
    pad = float(config.get("audio.pad_seconds", 0.12))
    mode = config.get("audio.mode", "mute")

    intervals: list[dict[str, Any]] = []
    for span in spans:
        if span.action == ACTION_KEEP:
            continue
        words = _words_in(doc, span)
        times = [(w.start, w.end) for w in words if w.start is not None and w.end is not None]
        if not times:
            continue
        start = max(0.0, min(t[0] for t in times) - pad)
        end = max(t[1] for t in times) + pad
        span.audio_start, span.audio_end = start, end
        intervals.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "entity": span.entity,
                "action": span.action,
                "mode": mode,
                "speaker": span.speaker,
                "turn_index": span.turn_index,
                "words": len(words),
            }
        )
    return _merge(intervals)


def _words_in(doc: Document, span: Span):
    out = []
    for turn in doc.turns:
        if turn.char_end < span.start or turn.char_start > span.end:
            continue
        for w in turn.words:
            if w.char_start < 0:
                continue
            if w.char_start < span.end and w.char_end > span.start:
                out.append(w)
    return out


def _merge(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not intervals:
        return []
    intervals.sort(key=lambda i: i["start"])
    out = [intervals[0]]
    for cur in intervals[1:]:
        prev = out[-1]
        if cur["start"] <= prev["end"]:
            prev["end"] = max(prev["end"], cur["end"])
            if cur["entity"] not in prev["entity"]:
                prev["entity"] = "{}+{}".format(prev["entity"], cur["entity"])
        else:
            out.append(cur)
    return out


def ffmpeg_command(
    input_path: str, output_path: str, intervals: list[dict[str, Any]], mode: str = "mute"
) -> str:
    """A single deterministic ffmpeg call that applies the whole plan."""
    if not intervals:
        return f"cp {_q(input_path)} {_q(output_path)}"
    if mode == "beep":
        # mute the region and mix a 1 kHz tone over it
        expr = "+".join("between(t,{:.3f},{:.3f})".format(i["start"], i["end"]) for i in intervals)
        return (
            f"ffmpeg -y -i {_q(input_path)} -f lavfi -i sine=frequency=1000:sample_rate=48000 "
            f"-filter_complex \"[0:a]volume=0:enable='{expr}'[main];"
            f"[1:a]volume=0.25:enable='{expr}'[beep];[main][beep]amix=inputs=2:duration=first[a]\" "
            f'-map "[a]" {_q(output_path)}'
        )
    expr = "+".join("between(t,{:.3f},{:.3f})".format(i["start"], i["end"]) for i in intervals)
    return f"ffmpeg -y -i {_q(input_path)} -af \"volume=0:enable='{expr}'\" {_q(output_path)}"


def _q(path: str) -> str:
    return "'{}'".format(path.replace("'", "'\\''"))


def coverage(doc: Document, intervals: list[dict[str, Any]]) -> dict[str, Any]:
    """How much of the recording the plan silences - a utility guard-rail."""
    total = 0.0
    for turn in doc.turns:
        if turn.start is not None and turn.end is not None:
            total += max(0.0, turn.end - turn.start)
    redacted = sum(i["end"] - i["start"] for i in intervals)
    return {
        "speech_seconds": round(total, 2),
        "redacted_seconds": round(redacted, 2),
        "redacted_fraction": round(redacted / total, 4) if total else 0.0,
        "intervals": len(intervals),
    }
