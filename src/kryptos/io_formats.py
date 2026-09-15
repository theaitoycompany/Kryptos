"""Transcript loading and serialisation.

Supported inputs (auto-detected):
  * plain text, no speaker labels
  * speaker-labelled lines:  ``CHILD: ...`` / ``[00:01:12] SPEAKER_01: ...``
  * JSON with ``turns`` (this package's own format)
  * Whisper / WhisperX / faster-whisper JSON with ``segments`` (+ word timings)
  * pyannote-style diarisation JSON merged with segments
  * JSONL, one segment object per line
  * SRT and WebVTT

Document.text holds only the *spoken* text, never speaker labels, so no
detector ever wastes a span on the label - labels are pseudonymised
separately.  Every turn and word carries character offsets into that flat
string, which is what makes audio redaction possible later.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import unicodedata
from typing import Any

from .types import Document, Turn, Word

TURN_SEP = "\n"

_SPEAKER_LINE = re.compile(
    r"""^\s*
        (?:\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)\]?\s*[-–]?\s*)?
        (?P<speaker>[A-Za-z][A-Za-z0-9 _.'\-]{0,40}?)
        \s*[:>]\s+
        (?P<text>\S.*)$""",
    re.VERBOSE,
)
_TIMECODE = re.compile(r"^\s*\[?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\]?\s*$")
_CUE_TIMESTAMP = r"(?:\d{2,}:)?\d{2}:\d{2}[.,]\d{3}"
_SRT_TIME = re.compile(rf"^({_CUE_TIMESTAMP})\s+-->\s+({_CUE_TIMESTAMP})(?:\s+.*)?$", re.M)


def _ts_to_seconds(ts: str | None) -> float | None:
    if not ts:
        return None
    parts = ts.replace(",", ".").split(":")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    sec = 0.0
    for v in vals:
        sec = sec * 60 + v
    return sec


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def build_document(
    turns: list[Turn], doc_id: str = "", language: str = "en", meta: dict[str, Any] | None = None
) -> Document:
    """Flatten turns into a single text buffer, assigning char offsets.

    Text is normalised to NFC first.  In decomposed (NFD) input a name is a
    base letter followed by a separate combining mark, and a span boundary that
    lands between them both mangles the output and leaves part of the name
    behind: "Ekstro\u0308m" masks as "<PERSON>\u0308m".  Normalising here, before
    any offset is assigned, keeps every downstream offset consistent.
    """
    chunks: list[str] = []
    cursor = 0
    clean: list[Turn] = []
    for i, turn in enumerate(turns):
        if not isinstance(turn.text, str):
            raise ValueError("Turn text must be a string")
        text = unicodedata.normalize("NFC", turn.text).strip()
        if not text:
            if turn.words:
                raise ValueError("Word tokens require turn text")
            continue
        turn.text = text
        turn.index = len(clean)
        turn.char_start = cursor
        turn.char_end = cursor + len(text)
        _assign_word_offsets(turn)
        chunks.append(text)
        cursor = turn.char_end + len(TURN_SEP)
        clean.append(turn)
    return Document(
        text=TURN_SEP.join(chunks), turns=clean, doc_id=doc_id, language=language, meta=meta or {}
    )


def _assign_word_offsets(turn: Turn) -> None:
    """Align supplied word tokens onto the turn's text by scanning forward."""
    if not turn.words:
        return
    cursor = 0
    text = turn.text
    folded, positions = [], []
    for index, char in enumerate(text):
        part = char.lower()
        folded.append(part)
        positions.extend([index] * len(part))
    low = "".join(folded)
    for w in turn.words:
        w.char_start = w.char_end = -1
        token = unicodedata.normalize("NFC", w.text).strip()
        if not token:
            continue
        token = token.lower()
        idx = low.find(token, cursor)
        if idx < 0:
            # ASR word lists sometimes carry punctuation the text lacks
            stripped = re.sub(r"[^\w']+", "", token)
            idx = low.find(stripped, cursor) if stripped else -1
            if idx < 0:
                continue
            token = stripped
        w.char_start = turn.char_start + positions[idx]
        w.char_end = turn.char_start + positions[idx + len(token) - 1] + 1
        cursor = idx + len(token)


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------


def parse_plain_text(text: str, doc_id: str = "") -> Document:
    """Speaker-labelled or bare plain text."""
    turns: list[Turn] = []
    pending_speaker: str | None = None
    pending_ts: float | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            body = " ".join(x.strip() for x in buffer if x.strip())
            if body:
                turns.append(
                    Turn(speaker=pending_speaker or "SPEAKER_UNKNOWN", text=body, start=pending_ts)
                )
        buffer.clear()

    for line in text.splitlines():
        if not line.strip():
            continue
        if _TIMECODE.match(line):
            continue
        m = _SPEAKER_LINE.match(line)
        if m and not _looks_like_sentence_colon(m):
            flush()
            pending_speaker = m.group("speaker").strip()
            pending_ts = _ts_to_seconds(m.group("ts"))
            buffer.append(m.group("text"))
        else:
            buffer.append(line)
    flush()
    if not turns:
        turns = [Turn(speaker="SPEAKER_UNKNOWN", text=text.strip())]
    return build_document(turns, doc_id=doc_id)


def _looks_like_sentence_colon(m: re.Match) -> bool:
    """Reject 'I said: hello' style lines being read as speaker labels."""
    spk = m.group("speaker").strip()
    if m.group("ts"):
        return False
    words = spk.split()
    if len(words) > 4:
        return True
    # a real label is usually upper-case, Title Case, or SPEAKER_xx
    if spk.isupper() or re.match(r"^[A-Z][\w'\-]*(\s+[A-Z][\w'\-]*)*$", spk):
        return False
    if re.match(r"^(speaker|spk|s)[ _-]?\d+$", spk, re.I):
        return False
    return True


def parse_json(obj: Any, doc_id: str = "") -> Document:
    if isinstance(obj, list):
        obj = {"segments": obj}
    if not isinstance(obj, dict):
        raise ValueError("unsupported JSON transcript structure")

    doc_id = obj.get("doc_id") or obj.get("id") or doc_id
    language = obj.get("language") or obj.get("lang") or "en"
    raw_turns = []
    for key in ("turns", "segments", "utterances", "results"):
        if key in obj:
            if not isinstance(obj[key], list):
                raise ValueError("Transcript segments must be a list")
            if raw_turns and obj[key]:
                raise ValueError("Ambiguous transcript segment fields")
            raw_turns = raw_turns or obj[key]
    if "text" in obj and not isinstance(obj["text"], str):
        raise ValueError("Transcript text must be a string")
    if not any(key in obj for key in ("text", "turns", "segments", "utterances", "results")):
        raise ValueError("No supported transcript field")
    if not raw_turns and isinstance(obj.get("text"), str):
        doc = parse_plain_text(obj["text"], doc_id=doc_id)
        doc.language = language
        doc.meta = {key: obj[key] for key in ("known_values", "speaker_roles") if key in obj}
        return doc

    turns: list[Turn] = []
    for seg in raw_turns:
        if not isinstance(seg, dict):
            raise ValueError("Each transcript segment must be an object")
        text = next((seg[key] for key in ("text", "transcript", "utterance") if key in seg), None)
        if not isinstance(text, str):
            raise ValueError("Segment text must be a string")
        speaker = (
            seg.get("speaker")
            or seg.get("speaker_id")
            or seg.get("spk")
            or seg.get("role")
            or "SPEAKER_UNKNOWN"
        )
        words = []
        raw_words = seg.get("words", seg.get("tokens", []))
        if not isinstance(raw_words, list):
            raise ValueError("Word tokens must be a list")
        for w in raw_words:
            if isinstance(w, dict):
                word_text = next((w[key] for key in ("word", "text", "token") if key in w), None)
                if not isinstance(word_text, str):
                    raise ValueError("Word text must be a string")
                words.append(
                    Word(
                        text=word_text,
                        start=_num(w.get("start", w.get("start_time"))),
                        end=_num(w.get("end", w.get("end_time"))),
                        conf=_num(w.get("score", w.get("confidence", w.get("probability")))),
                    )
                )
            elif isinstance(w, (list, tuple)) and w and isinstance(w[0], str):
                words.append(
                    Word(
                        text=str(w[0]),
                        start=_num(w[1]) if len(w) > 1 else None,
                        end=_num(w[2]) if len(w) > 2 else None,
                        conf=_num(w[3]) if len(w) > 3 else None,
                    )
                )
            else:
                raise ValueError("Unsupported word token")
        turns.append(
            Turn(
                speaker=str(speaker),
                text=str(text),
                start=_num(seg.get("start")),
                end=_num(seg.get("end")),
                words=words,
            )
        )
    meta = {
        k: v
        for k, v in obj.items()
        if k not in {"turns", "segments", "utterances", "results", "text"}
    }
    return build_document(turns, doc_id=str(doc_id or ""), language=str(language), meta=meta)


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        number = float(v)
    except (TypeError, ValueError):
        raise ValueError("Invalid numeric transcript field") from None
    if isinstance(v, bool) or not math.isfinite(number) or number < 0:
        raise ValueError("Invalid numeric transcript field")
    return number


def parse_jsonl(text: str, doc_id: str = "") -> Document:
    segs = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            segs.append(json.loads(line))
    return parse_json({"segments": segs}, doc_id=doc_id)


def parse_srt_vtt(text: str, doc_id: str = "") -> Document:
    """Parse subtitle cues, including short WebVTT timestamps and numeric speech."""
    turns: list[Turn] = []
    text = text.lstrip("\ufeff").replace("\r\n", "\n")
    if text.startswith("WEBVTT"):
        text = text.partition("\n")[2]
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        if not lines:
            continue
        if re.match(r"^(?:NOTE(?:\s|$)|STYLE$|REGION$)", lines[0]):
            continue
        cue = next((i for i in range(min(2, len(lines))) if _SRT_TIME.fullmatch(lines[i])), None)
        if cue is None:
            raise ValueError("Invalid subtitle cue")
        match = _SRT_TIME.fullmatch(lines[cue])
        start, end = (_subtitle_seconds(value) for value in match.groups())
        if end <= start:
            raise ValueError("Invalid subtitle timing interval")
        body = " ".join(lines[cue + 1 :]).strip()
        speaker = "SPEAKER_UNKNOWN"
        voice = re.match(r"^\s*(?:<v\s+([^>]+)>|([A-Z][A-Za-z0-9 _'\-]{0,30})\s*:)\s*(.*)$", body)
        if voice:
            speaker = (voice.group(1) or voice.group(2) or speaker).strip()
            body = voice.group(3)
        body = re.sub(r"</?(?:v|c|i|b|u|ruby|rt|lang)(?:[ .][^>]*)?>", "", body)
        body = re.sub(r"<" + _CUE_TIMESTAMP + r">", "", body)
        body = html.unescape(body).strip()
        if body:
            turns.append(Turn(speaker=speaker, text=body, start=start, end=end))
    return build_document(turns, doc_id=doc_id)


def _subtitle_seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    hours, minutes, seconds = (0, *map(float, parts)) if len(parts) == 2 else map(float, parts)
    if minutes >= 60 or seconds >= 60:
        raise ValueError("Invalid subtitle timestamp")
    return hours * 3600 + minutes * 60 + seconds


def validate_document(doc: Document) -> None:
    """Reject inconsistent coordinates before any detector or transformation runs."""
    if not isinstance(doc.text, str) or not isinstance(doc.language, str):
        raise ValueError("Document text and language must be strings")
    if doc.text != TURN_SEP.join(turn.text for turn in doc.turns):
        raise ValueError("Document requires aligned turns; use process_text to build them")
    cursor = 0
    for index, turn in enumerate(doc.turns):
        if (turn.char_start, turn.char_end, turn.index) != (cursor, cursor + len(turn.text), index):
            raise ValueError("Document requires aligned turns; use process_text to build them")
        if not isinstance(turn.speaker, str):
            raise ValueError("Speaker labels must be strings")
        _validate_interval(turn.start, turn.end)
        for word in turn.words:
            _validate_interval(word.start, word.end)
            if word.conf is not None and not 0 <= _num(word.conf) <= 1:
                raise ValueError("Word confidence must be between zero and one")
            if (word.char_start, word.char_end) != (-1, -1) and not (
                turn.char_start <= word.char_start < word.char_end <= turn.char_end
            ):
                raise ValueError("Invalid word offsets")
        cursor = turn.char_end + len(TURN_SEP)
    doc._turn_starts = None
    doc._turn_fingerprint = None


def _validate_interval(start: float | None, end: float | None) -> None:
    start, end = _num(start), _num(end)
    if start is not None and end is not None and end < start:
        raise ValueError("Invalid transcript timing interval")


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def load_text(text: str, doc_id: str = "", fmt: str = "auto") -> Document:
    stripped = text.lstrip()
    if fmt == "auto":
        if _SPEAKER_LINE.match(
            stripped.splitlines()[0] if stripped else ""
        ) and stripped.startswith("["):
            fmt = "text"
        elif stripped.startswith(("{", "[")):
            fmt = "json"
        elif stripped.upper().startswith("WEBVTT") or _SRT_TIME.search(text[:2000] or ""):
            fmt = "srt"
        elif "\n" in text and all(
            line.strip().startswith("{") for line in text.strip().splitlines()[:3] if line.strip()
        ):
            fmt = "jsonl"
        else:
            fmt = "text"
    if fmt == "json":
        try:
            return parse_json(json.loads(text), doc_id=doc_id)
        except json.JSONDecodeError:
            return parse_jsonl(text, doc_id=doc_id)
    if fmt == "jsonl":
        return parse_jsonl(text, doc_id=doc_id)
    if fmt in ("srt", "vtt"):
        return parse_srt_vtt(text, doc_id=doc_id)
    if fmt != "text":
        raise ValueError("Unsupported transcript format")
    return parse_plain_text(text, doc_id=doc_id)


def load_file(path: str, fmt: str = "auto") -> Document:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    doc_id = ""
    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        fmt = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".ndjson": "jsonl",
            ".srt": "srt",
            ".vtt": "vtt",
        }.get(ext, "auto")
    doc = load_text(text, doc_id=doc_id, fmt=fmt)
    # never let the source filename become a re-identification vector
    doc.meta.setdefault("source_basename_removed", True)
    return doc


def render_turns(turns: list[dict[str, Any]]) -> str:
    return "\n".join("{}: {}".format(t["speaker"], t["text"]) for t in turns)
