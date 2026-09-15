"""Known-value detector (deny list).

If the system already knows the participants - the child's name, the school,
the account number - detecting them with a probabilistic model is a waste and
a risk.  These are matched deterministically at confidence 1.0, through three
channels (exact / edit-distance / phonetic) so that ASR corruption of exactly
those tokens does not become a privacy failure.

Known values come from config (``known_values``) or per-document metadata
(``doc.meta['known_values']``), so a caller can pass what the enrolment
record already holds for this conversation.
"""

from __future__ import annotations

import re
from typing import Any

from ..normalize import despoken_email, fuzzy_match, normalize, phonetic_key
from ..types import Detection, Document
from .base import Detector, register


def _channel_score(channels: set[str], val: _Value) -> tuple[str, float]:
    """Weakest channel used decides the confidence."""
    if "phonetic" in channels:
        channel, score = "phonetic", 0.9
    elif "merged" in channels:
        channel, score = "merged", 0.92
    elif "fuzzy" in channels:
        channel, score = "fuzzy", 0.95
    else:
        channel, score = "exact", 1.0
    if val.partial:
        score = min(score, 0.97)
    return channel, score


_TOKEN_RE = re.compile(r"[\w'’-]+", re.UNICODE)

# ASR and speakers use these interchangeably; collapse them to one token so
# "Ms Rupa" still matches a transcript that says "Miss Roopa".
_HONORIFICS = {
    "mr": "TITLE",
    "mister": "TITLE",
    "mrs": "TITLE",
    "ms": "TITLE",
    "miss": "TITLE",
    "sir": "TITLE",
    "madam": "TITLE",
    "maam": "TITLE",
    "dr": "TITLE_DR",
    "doctor": "TITLE_DR",
    "prof": "TITLE_DR",
    "professor": "TITLE_DR",
    "nurse": "TITLE_DR",
    "coach": "TITLE_C",
    "teacher": "TITLE_T",
    "auntie": "TITLE_A",
    "aunty": "TITLE_A",
    "aunt": "TITLE_A",
    "uncle": "TITLE_U",
}


def _canon_token(tok: str) -> str:
    return _HONORIFICS.get(tok, tok)


_NAME_ENTITIES = ("_NAME", "WORKPLACE", "SCHOOL", "DAYCARE", "TEAM", "CLUB")

# Components too generic to be identifiers on their own.  Without this,
# "Greenfield Primary School" would make the bare word "school" a known value
# and every mention of school in the transcript would be redacted.
_GENERIC_COMPONENTS = set(
    """
school schools primary secondary academy college nursery preschool kindergarten daycare
montessori grammar elementary institute university club team squad fc united rovers
hospital clinic centre center surgery practice park street road avenue lane drive close
court crescent way place terrace gardens house flat apartment the and of for
""".split()
)


class _Value:
    __slots__ = ("surface", "entity", "tokens", "keys", "n", "partial")

    def __init__(self, surface: str, entity: str, partial: bool = False):
        self.surface = surface
        self.entity = entity
        self.partial = partial
        self.tokens = [normalize(t) for t in _TOKEN_RE.findall(surface)]
        self.tokens = [_canon_token(t) for t in self.tokens if t]
        self.keys = [phonetic_key(t) for t in self.tokens]
        self.n = len(self.tokens)


def _expand(surface: str, entity: str) -> list[_Value]:
    """A child introduced as "Samira Hasan" will also be called just "Samira".

    Each component of a known multi-token name becomes a known value in its
    own right, because half a name is still an identifier.
    """
    if not any(marker in entity for marker in _NAME_ENTITIES):
        return []
    tokens = [t for t in _TOKEN_RE.findall(surface) if normalize(t) not in _HONORIFICS]
    if len(tokens) < 2:
        return []
    parts = [
        t for t in tokens if len(normalize(t)) >= 4 and normalize(t) not in _GENERIC_COMPONENTS
    ]
    return [_Value(p, entity, partial=True) for p in parts]


@register("known_values")
class KnownValuesDetector(Detector):
    """Deterministic matching of values the caller already knows."""

    def load(self) -> None:
        self.fuzzy = bool(self.spec.get("fuzzy", True))
        self.threshold = float(self.spec.get("fuzzy_threshold", 0.82))
        self.phonetic = bool(self.spec.get("phonetic", True))
        self.min_token_len = int(self.spec.get("min_token_len", 3))
        self.indexed_matching = bool(self.spec.get("indexed_matching", False))
        self._config_values = self._collect(self.config.known_values)

    # ------------------------------------------------------------------
    @staticmethod
    def _collect(mapping: Any) -> list[_Value]:
        """Accept {entity: [values]}, {entity: value} or [{value, entity}]."""
        out: list[_Value] = []
        if not mapping:
            return out
        if isinstance(mapping, dict):
            for entity, values in mapping.items():
                if entity.startswith("_"):
                    continue
                if isinstance(values, (str, int, float)):
                    values = [values]
                for v in values or []:
                    if isinstance(v, dict):
                        surface = str(v.get("value", "")).strip()
                        ent = str(v.get("entity", entity))
                    else:
                        surface, ent = str(v).strip(), entity
                    if surface:
                        out.append(_Value(surface, ent.upper()))
                        out.extend(_expand(surface, ent.upper()))
        elif isinstance(mapping, list):
            for item in mapping:
                if isinstance(item, dict) and item.get("value"):
                    surface = str(item["value"]).strip()
                    ent = str(item.get("entity", "PERSON_NAME")).upper()
                    out.append(_Value(surface, ent))
                    out.extend(_expand(surface, ent))
        return out

    def values_for(self, doc: Document) -> list[_Value]:
        extra = self._collect(doc.meta.get("known_values"))
        return self._config_values + extra

    # ------------------------------------------------------------------
    def detect(self, doc: Document) -> list[Detection]:
        values = self.values_for(doc)
        if not values:
            return []
        text = doc.text
        spans = [
            (m.start(), m.end(), _canon_token(normalize(m.group(0))))
            for m in _TOKEN_RE.finditer(text)
        ]
        if not spans:
            return []
        norm_tokens = [s[2] for s in spans]
        keys = [phonetic_key(t) for t in norm_tokens]

        index = self._start_index(norm_tokens) if self.indexed_matching else None

        out: list[Detection] = []
        for val in values:
            if val.n == 0:
                continue
            candidates = self._candidate_starts(val, index) if index is not None else None
            out.extend(self._match_value(doc, val, spans, norm_tokens, keys, candidates))
        out.extend(self._match_despoken(doc, values))
        return out

    # ------------------------------------------------------------------
    def _match_value(
        self,
        doc: Document,
        val: _Value,
        spans: list[tuple[int, int, str]],
        norm_tokens: list[str],
        keys: list[str],
        candidates=None,
    ) -> list[Detection]:
        found: list[Detection] = []
        next_start = 0
        n_tokens = len(spans)
        for i in range(n_tokens) if candidates is None else candidates:
            if i < next_start:
                continue
            res = self._align(val, norm_tokens, i)
            if res is None:
                continue
            end_index, channels = res
            if end_index <= i:
                continue
            channel, score = _channel_score(channels, val)
            start, end = spans[i][0], spans[end_index - 1][1]
            found.append(
                self.make(
                    doc,
                    start,
                    end,
                    val.entity,
                    score,
                    raw_label="known_value",
                    channel=channel,
                    known_value=True,
                    matched=val.surface,
                    partial=val.partial,
                )
            )
            next_start = end_index
        return found

    @staticmethod
    def _start_index(tokens):
        """Index repeated tokens and adjacent pairs once per document."""
        singles, pairs = {}, {}
        for i, word in enumerate(tokens):
            singles.setdefault(word, []).append(i)
            if i + 1 < len(tokens):
                pairs.setdefault(word + tokens[i + 1], []).append(i)
        return singles, pairs, {}

    def _candidate_starts(self, val, index):
        """A lossless prefilter for the three entry branches of _align.

        Run the exact existing comparison predicates on unique tokens/pairs.
        Every possible first transition remains eligible for full alignment;
        only starts for which all three transitions fail are skipped.
        """
        singles, pairs, cache = index
        want = val.tokens[0]
        next_want = val.tokens[1] if val.n > 1 else None
        merged_name = val.entity.endswith("_NAME")
        key = (want, next_want, merged_name)
        if key in cache:
            return cache[key]
        found = set()
        for got, positions in singles.items():
            if self._compare(want, got)[0] or (
                next_want is not None and self._compare(want + next_want, got)[0]
            ):
                found.update(positions)
        for got, positions in pairs.items():
            if (
                abs(len(got) - len(want)) <= 2
                and self._compare(want, got, strict=True, merged_name=merged_name)[0]
            ):
                found.update(positions)
        result = sorted(found)
        cache[key] = result
        return result

    def _window_match(
        self, val: _Value, window: list[str], window_keys: list[str]
    ) -> tuple[str | None, float]:
        """Deprecated 1:1 matcher, kept for direct callers."""
        res = self._align(val, window, 0)
        if res is None:
            return None, 0.0
        _end, channels = res
        return _channel_score(channels, val)

    def _align(self, val: _Value, tokens: list[str], start: int) -> tuple[int, set[str]] | None:
        """Align a known value onto the transcript allowing merges and splits.

        ASR does not preserve token boundaries: "Greenfield" comes back as
        "green field", "Da Silva" as "dasilva".  A strict token-for-token
        comparison misses exactly the values the caller told us to protect,
        so each known token may consume up to two transcript tokens, and two
        known tokens may be covered by one transcript token.
        """
        n_tokens = len(tokens)

        def rec(k: int, t: int, channels: set[str], depth: int) -> tuple[int, set[str]] | None:
            if k >= val.n:
                return t, channels
            if t >= n_tokens or depth > max(8, val.n):
                return None
            want = val.tokens[k]
            for consume in (1, 2):
                if t + consume > n_tokens:
                    break
                got = "".join(tokens[t : t + consume])
                # a merge is only credible if the result is about the same
                # length as the target - otherwise "to greenfield" fuzzily
                # matches "greenfield" and swallows the preceding word
                if consume == 2 and abs(len(got) - len(want)) > 2:
                    continue
                ok, ch = self._compare(
                    want,
                    got,
                    strict=(consume == 2),
                    merged_name=(consume == 2 and val.entity.endswith("_NAME")),
                )
                if not ok:
                    continue
                res = rec(
                    k + 1, t + consume, channels | {ch if consume == 1 else "merged"}, depth + 1
                )
                if res:
                    return res
            if k + 1 < val.n:  # two known tokens spoken as one
                ok, ch = self._compare(want + val.tokens[k + 1], tokens[t])
                if ok:
                    res = rec(k + 2, t + 1, channels | {"merged"}, depth + 1)
                    if res:
                        return res
            return None

        return rec(0, start, set(), 0)

    def _compare(
        self, want: str, got: str, strict: bool = False, merged_name: bool = False
    ) -> tuple[bool, str]:
        if want == got:
            return True, "exact"
        if not self.fuzzy:
            return False, "none"
        if len(want) < self.min_token_len or len(got) < self.min_token_len:
            return False, "none"
        threshold = max(self.threshold, 0.9) if strict else self.threshold
        # Known names are routinely split phonetically by ASR: Osei -> o say,
        # Mei-Ling -> may ling. The existing merge length bound still applies.
        ok, _score, ch = fuzzy_match(
            want, got, threshold, self.phonetic and (not strict or merged_name)
        )
        return ok, ch

    # ------------------------------------------------------------------
    def _match_despoken(self, doc: Document, values: list[_Value]) -> list[Detection]:
        """Catch dictated emails/handles: 'sam dot hasan at gmail dot com'."""
        out: list[Detection] = []
        targets = [
            v for v in values if v.entity in {"EMAIL", "ONLINE_USERNAME", "SOCIAL_HANDLE", "URL"}
        ]
        if not targets:
            return out
        for m in re.finditer(
            r"[\w.'-]+(?:\s+(?:dot|at|underscore|dash|hyphen)\s+[\w.'-]+)+", doc.text, re.I
        ):
            candidate = despoken_email(m.group(0)).lower()
            for val in targets:
                if fuzzy_match(candidate, val.surface, 0.88, False)[0]:
                    out.append(
                        self.make(
                            doc,
                            m.start(),
                            m.end(),
                            val.entity,
                            1.0,
                            raw_label="known_value_spoken",
                            known_value=True,
                            matched=val.surface,
                        )
                    )
                    break
        return out
