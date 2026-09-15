"""Pseudonym allocation and generalisation.

Replacing every identifier with the same ``[PII]`` token destroys the thing
that made the transcript worth keeping: who said what about whom.  Consistent
role-aware pseudonyms keep the relational structure while removing identity.

Scope is a deliberate privacy dial:

    per_turn          strongest unlinkability, unusable for analysis
    per_conversation  the default

If a reversible mapping is not needed, none is kept.  If it is, the key lives
in a KMS/env var and identifiers are HMAC-derived - never a bare hash of the
name, which is trivially dictionary-attacked.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .normalize import fuzzy_match, normalize
from .quasi import age_bucket
from .types import (
    ACTION_GENERALIZE,
    ACTION_KEEP,
    ACTION_REDACT,
    Document,
    Span,
)

SURROGATE_POOL_FILE = "first_names.txt"

_DAYPARTS = [
    (0, 5, "NIGHT"),
    (5, 12, "MORNING"),
    (12, 17, "AFTERNOON"),
    (17, 21, "EVENING"),
    (21, 24, "NIGHT"),
]

_ROLE_PREFIX = {
    "child": "CHILD",
    "parent": "PARENT",
    "sibling": "SIBLING",
    "clinician": "CLINICIAN",
    "teacher": "TEACHER",
}


@dataclass
class Allocation:
    label: str
    prefix: str
    canonical: str
    variants: list[str] = field(default_factory=list)


class SurrogateAllocator:
    """Allocates a stable replacement for every distinct real-world referent."""

    def __init__(self, config: Config, doc: Document | None = None):
        self.config = config
        self.doc = doc
        self.scope = config.get("surrogates.scope", "per_conversation")
        if self.scope not in {"per_turn", "per_conversation"}:
            raise ValueError("Scope must be per_turn or per_conversation")
        self.style = config.get("surrogates.style", "placeholder")
        self.width = int(config.get("surrogates.number_width", 2))
        self.link_threshold = float(config.get("surrogates.link_threshold", 0.86))
        self.role_aware = bool(config.get("surrogates.role_aware", True))
        self._by_prefix: dict[str, list[Allocation]] = {}
        self._counters: dict[str, int] = {}
        self._pool: list[str] = []
        self._pool_used: dict[str, int] = {}
        key_env = config.get("surrogates.hmac_key_env", "KRYPTOS_HMAC_KEY")
        raw_key = os.environ.get(key_env or "", "")
        self.hmac_key = raw_key.encode("utf-8") if raw_key else None

    # ------------------------------------------------------------------
    def prefix_for(self, span: Span) -> str:
        policy = self.config.taxonomy.policy(span.entity)
        prefix = policy.prefix
        if not self.role_aware:
            return prefix
        # a generic PERSON said by a known role gets the role's pool, so the
        # child does not end up as PERSON_03 in half the transcript
        if span.entity == "PERSON_NAME" and span.speaker:
            role = self.doc.meta.get("speaker_roles", {}).get(span.speaker) if self.doc else None
            if role in _ROLE_PREFIX and role in ("clinician", "teacher"):
                return _ROLE_PREFIX[role]
        return prefix

    # ------------------------------------------------------------------
    def allocate(self, span: Span) -> str:
        action = span.action
        if action == ACTION_KEEP:
            return span.text
        if action == ACTION_GENERALIZE:
            return self.generalize(span)
        if action == ACTION_REDACT:
            return self._redaction_token(span)
        return self.pseudonym(span)

    # ------------------------------------------------------------------
    def pseudonym(self, span: Span) -> str:
        prefix = self.prefix_for(span)
        surface = normalize(span.canonical_hint or span.text)
        if self.scope == "per_turn":
            key = f"{surface}|{span.turn_index}"
        else:
            key = surface
        alloc = self._find(prefix, key, span)
        if alloc is None:
            alloc = self._create(prefix, key, span)
        elif key not in alloc.variants:
            alloc.variants.append(key)
        return alloc.label

    def _find(self, prefix: str, key: str, span: Span) -> Allocation | None:
        hit = self._find_in(self._by_prefix.get(prefix, []), key, fuzzy=True)
        if hit is not None:
            return hit
        # A name first seen as CHILD_NAME and later only as a generic PERSON
        # (a bare vocative, "Aisha, come here") is the same child.  Cross-pool
        # linking requires an exact or sub-name match, never a fuzzy one, so
        # two different people who merely sound alike stay separate.
        if prefix in _NAME_PREFIXES:
            for other, allocs in self._by_prefix.items():
                if other == prefix or other not in _NAME_PREFIXES:
                    continue
                hit = self._find_in(allocs, key, fuzzy=False)
                if hit is not None:
                    return hit
        return None

    def _find_in(self, bucket: list[Allocation], key: str, fuzzy: bool) -> Allocation | None:
        for alloc in bucket:
            if key == alloc.canonical or key in alloc.variants:
                return alloc
            for known in [alloc.canonical] + alloc.variants:
                # link ASR variants of the same referent onto one pseudonym
                if fuzzy and fuzzy_match(key, known, self.link_threshold)[0]:
                    return alloc
                # "Samira" and "Samira Hasan" are the same child
                if _is_subname(key, known):
                    return alloc
        return None

    def _create(self, prefix: str, key: str, span: Span) -> Allocation:
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        label = self._render_label(prefix, n, key, span)
        alloc = Allocation(label=label, prefix=prefix, canonical=key, variants=[key])
        self._by_prefix.setdefault(prefix, []).append(alloc)
        return alloc

    def _render_label(self, prefix: str, n: int, key: str, span: Span) -> str:
        if self.style == "surrogate" and prefix in _NAME_PREFIXES:
            return self._surrogate_name(prefix, n, key)
        if self.style == "hmac":
            return f"<{prefix}_{self._hmac_id(prefix, key)}>"
        return f"<{prefix}_{n:0{self.width}d}>"

    def _hmac_id(self, prefix: str, key: str) -> str:
        if not self.hmac_key or len(self.hmac_key) < 32:
            raise RuntimeError(
                "surrogates.style='hmac' requires at least 32 bytes in key env var {!r}; "
                "a bare hash of a name is dictionary-attackable and is not supported".format(
                    self.config.get("surrogates.hmac_key_env", "KRYPTOS_HMAC_KEY")
                )
            )
        if not self.doc or not self.doc.doc_id:
            raise RuntimeError("HMAC pseudonyms require an explicit opaque document ID")
        context = f"{self.doc.doc_id}|{prefix}|{key}"
        mac = hmac.new(self.hmac_key, context.encode(), hashlib.sha256)
        return mac.hexdigest()[:32].upper()

    def _surrogate_name(self, prefix: str, n: int, key: str) -> str:
        if not self._pool:
            self._pool = _load_pool()
        if not self._pool:
            return f"<{prefix}_{n:0{self.width}d}>"
        idx = (n - 1 + _stable_offset(key)) % len(self._pool)
        for _ in range(len(self._pool)):
            cand = self._pool[idx].title()
            if cand not in self._pool_used:
                self._pool_used[cand] = 1
                return cand
            idx = (idx + 1) % len(self._pool)
        return f"<{prefix}_{n:0{self.width}d}>"

    # ------------------------------------------------------------------
    def generalize(self, span: Span) -> str:
        entity = span.entity
        policy = self.config.taxonomy.policy(entity)

        if entity == "EXACT_AGE":
            val = _first_int(span.text)
            if val is None:
                from .normalize import spoken_cardinal

                val = spoken_cardinal(span.text)
            if val is not None:
                return age_bucket(val, int(self.config.get("quasi.age_bucket_size", 3)))
            return "<AGE>"

        if entity == "PRECISE_TIME":
            gran = self.config.get("surrogates.time_granularity", "daypart")
            if gran == "daypart":
                part = _daypart(span.text)
                if part:
                    return f"<{part}>"
            return "<TIME>"

        if entity in ("PRECISE_DATE", "DATE_OF_BIRTH"):
            gran = self.config.get(
                "surrogates.date_granularity", "none" if entity == "DATE_OF_BIRTH" else "year"
            )
            if gran == "year":
                year = _first_year(span.text)
                if year:
                    return f"<YEAR_{year}>"
            if gran == "month":
                year = _first_year(span.text)
                if year:
                    return f"<MONTH_OF_{year}>"
            return "<DOB>" if entity == "DATE_OF_BIRTH" else "<DATE>"

        return policy.generalized or f"<{entity}>"

    def _redaction_token(self, span: Span) -> str:
        policy = self.config.taxonomy.policy(span.entity)
        return policy.generalized or f"<{span.entity}>"

    # ------------------------------------------------------------------
    def mapping(self) -> dict[str, str]:
        """{pseudonym: original} - only ever returned when explicitly kept."""
        out: dict[str, str] = {}
        for allocs in self._by_prefix.values():
            for a in allocs:
                out[a.label] = a.canonical
        return out

    def inventory(self) -> dict[str, Any]:
        """Non-reversible summary: how many distinct referents per pool."""
        return {prefix: len(allocs) for prefix, allocs in self._by_prefix.items()}


_NAME_PREFIXES = {
    "CHILD",
    "PARENT",
    "SIBLING",
    "RELATIVE",
    "TEACHER",
    "THERAPIST",
    "DOCTOR",
    "COACH",
    "FRIEND",
    "PERSON",
    "CLINICIAN",
}


def _load_pool() -> list[str]:
    from .detectors.context_rules import GAZ_DIR

    path = os.path.join(GAZ_DIR, SURROGATE_POOL_FILE)
    out: list[str] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                out.extend(t for t in line.split() if t.isalpha() and len(t) > 2)
    return out


def _stable_offset(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:6], 16)


def _is_subname(a: str, b: str) -> bool:
    """'samira' vs 'samira hasan' - same referent, different mention length."""
    ta, tb = a.split(), b.split()
    if not ta or not tb or ta == tb:
        return False
    short, long_ = (ta, tb) if len(ta) < len(tb) else (tb, ta)
    if len(short) != 1 or len(long_) > 3:
        return False
    return short[0] == long_[0] and len(short[0]) > 2


def _first_int(text: str) -> int | None:
    m = re.search(r"\d{1,3}", text)
    return int(m.group(0)) if m else None


def _first_year(text: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", text)
    return int(m.group(0)) if m else None


def _daypart(text: str) -> str | None:
    m = re.search(r"\b([01]?\d|2[0-3])\s*[:.]\s*([0-5]\d)?\s*(am|pm)?", text, re.I)
    if not m:
        m2 = re.search(r"\b(1[0-2]|[1-9])\s*(am|pm)\b", text, re.I)
        if not m2:
            return None
        hour = int(m2.group(1))
        mer = m2.group(2).lower()
    else:
        hour = int(m.group(1))
        mer = (m.group(3) or "").lower()
    if mer == "pm" and hour < 12:
        hour += 12
    elif mer == "am" and hour == 12:
        hour = 0
    elif not mer and hour <= 7:
        hour += 12  # "finishes at 3:15" in a school transcript means 15:15
    for low, high, name in _DAYPARTS:
        if low <= hour < high:
            return name
    return None
