"""Quasi-identifier risk pass.

Direct identifiers are the easy half.  The half that actually re-identifies
children is the combination that contains no name, no number and no address:

    "I'm nine and I'm the only goalkeeper on the under-10 Tigers.
     Dad owns the bakery opposite Westbrook Primary."

Every span there is individually innocuous.  Together they describe one child.
This pass scores combinations across the whole conversation - a conversation
is one identity, so locality is a weighting factor, not a gate - and escalates
the action on the spans that participate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .types import ACTION_GENERALIZE, ACTION_KEEP, Document, Span

# entities whose *precision* is the problem, not their existence
GENERALISABLE = {
    "EXACT_AGE",
    "PRECISE_DATE",
    "PRECISE_TIME",
    "NEIGHBORHOOD",
    "LOCAL_LANDMARK",
    "CITY",
    "REGION",
    "OCCUPATION",
    "SPORTS_TEAM",
    "CLUB_NAME",
    "CLASS_NAME",
    "BUS_ROUTE",
    "UNIQUE_EVENT",
    "MEDICAL_CONDITION",
    "RELIGION",
    "ETHNICITY",
    "NATIONALITY",
    "ORGANIZATION",
}

# phrases that assert uniqueness - a population of one is not anonymous
UNIQUENESS_RE = re.compile(
    r"\b(the\s+only|only\s+one|one\s+of\s+(?:the\s+)?(?:two|three)|the\s+first|"
    r"the\s+youngest|the\s+oldest|the\s+tallest|the\s+best|the\s+worst|unique|"
    r"nobody\s+else|no\s+one\s+else)\b",
    re.I,
)

SMALL_POPULATION_RE = re.compile(
    r"\b(?:class|team|group|club|year|squad|troop|scout|choir|band)\b", re.I
)


@dataclass
class QuasiReport:
    risk_score: float = 0.0
    entity_counts: dict[str, int] = field(default_factory=dict)
    combinations: list[dict[str, Any]] = field(default_factory=list)
    escalated: int = 0
    uniqueness_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 3),
            "entity_counts": self.entity_counts,
            "combinations": self.combinations,
            "escalated_spans": self.escalated,
            "uniqueness_claims": self.uniqueness_claims,
        }


def apply(doc: Document, spans: list[Span], config: Config) -> QuasiReport:
    report = QuasiReport()
    if not config.get("quasi.enabled", True):
        return report

    counts: dict[str, int] = {}
    for s in spans:
        counts[s.entity] = counts.get(s.entity, 0) + 1
    report.entity_counts = counts
    present = set(counts)

    window = int(config.get("quasi.window_chars", 600))
    combos: list[Sequence[str]] = config.get("quasi.risky_combinations", []) or []
    participating: dict[int, list[str]] = {}

    for combo in combos:
        combo = list(combo)
        if not all(e in present for e in combo):
            continue
        members = [s for s in spans if s.entity in combo]
        proximity = _closest_grouping(members, combo, window)
        weight = 1.0 if proximity is not None else 0.6
        report.combinations.append(
            {
                "entities": combo,
                "co_located": proximity is not None,
                "weight": weight,
                "example": [m.entity for m in (proximity or members)][: len(combo)],
            }
        )
        report.risk_score += weight
        for s in proximity or members:
            participating.setdefault(id(s), []).append("+".join(combo))

    # uniqueness assertions multiply the risk of everything near them
    for m in UNIQUENESS_RE.finditer(doc.text):
        claim = doc.text[m.start() : min(len(doc.text), m.start() + 80)].split("\n")[0]
        report.uniqueness_claims.append(claim.strip())
        report.risk_score += 0.8
        for s in spans:
            if abs(s.start - m.start()) <= window:
                participating.setdefault(id(s), []).append("uniqueness_claim")

    # small-population context ("only goalkeeper on the team")
    for m in SMALL_POPULATION_RE.finditer(doc.text):
        near_unique = UNIQUENESS_RE.search(doc.text[max(0, m.start() - 120) : m.start() + 120])
        if not near_unique:
            continue
        report.risk_score += 0.3
        for s in spans:
            if abs(s.start - m.start()) <= 200:
                participating.setdefault(id(s), []).append("small_population")

    threshold = int(config.get("quasi.combination_threshold", 2))
    quasi_present = {e for e in present if config.taxonomy.policy(e).tier == "quasi"}
    broad_escalation = (
        config.get("quasi.generalize_on_combination", True) and len(quasi_present) >= threshold
    )

    for s in spans:
        reasons = participating.get(id(s), [])
        if reasons:
            s.risk_reasons.extend(sorted(set(reasons)))
            s.risk = min(1.0, s.risk + 0.35 * len(set(reasons)))
        if s.tier != "quasi" and s.entity not in GENERALISABLE:
            continue
        should = bool(reasons) or (broad_escalation and s.entity in GENERALISABLE)
        if should and s.action == ACTION_KEEP:
            s.action = ACTION_GENERALIZE
            if broad_escalation and not reasons:
                s.risk_reasons.append("quasi_combination_in_document")
            report.escalated += 1

    report.risk_score = min(10.0, report.risk_score)
    return report


def _closest_grouping(members: list[Span], combo: Sequence[str], window: int) -> list[Span] | None:
    """Find one span per combo entity inside a single window, if one exists."""
    by_entity: dict[str, list[Span]] = {}
    for m in members:
        by_entity.setdefault(m.entity, []).append(m)
    if not all(e in by_entity for e in combo):
        return None
    anchors = by_entity[combo[0]]
    for anchor in anchors:
        picked = [anchor]
        ok = True
        for other in combo[1:]:
            candidates = [c for c in by_entity[other] if abs(c.start - anchor.start) <= window]
            if not candidates:
                ok = False
                break
            picked.append(min(candidates, key=lambda c: abs(c.start - anchor.start)))
        if ok:
            return picked
    return None


def age_bucket(value: int, size: int = 3) -> str:
    """9 -> '<AGE_9_11>' with size 3; buckets are aligned to the size grid."""
    size = max(1, size)
    low = (value // size) * size
    high = low + size - 1
    return f"<AGE_{low}_{high}>"
