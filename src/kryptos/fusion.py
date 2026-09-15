"""Span fusion.

All detectors saw the same original text, so their claims must now be
reconciled into one non-overlapping set of spans.  Three decisions matter:

*Boundaries* - in ``union`` mode the surviving span covers everything the
cluster claimed.  Over-covering a name costs a slightly uglier transcript;
under-covering it leaks half a surname.

*Entity* - a specific label beats a generic one from the same family
(CHILD_NAME over PERSON_NAME), because the downstream policy and risk model
differ even though the text to remove is identical.

*Score* - agreement between independent detectors is evidence, so each extra
detector adds a bounded bonus.  Known values are deterministic and pinned
at 1.0.
"""

from __future__ import annotations

from collections.abc import Sequence

from .config import Config
from .types import Detection, Document, Span

# generic label -> the specific labels that should outrank it
SPECIALISATIONS: dict[str, set[str]] = {
    "PERSON_NAME": {
        "CHILD_NAME",
        "PARENT_NAME",
        "SIBLING_NAME",
        "RELATIVE_NAME",
        "TEACHER_NAME",
        "THERAPIST_NAME",
        "DOCTOR_NAME",
        "COACH_NAME",
        "FRIEND_NAME",
    },
    "ORGANIZATION": {
        "SCHOOL_NAME",
        "DAYCARE_NAME",
        "CLUB_NAME",
        "SPORTS_TEAM",
        "PARENT_WORKPLACE",
        "CLASS_NAME",
    },
    "CITY": {"NEIGHBORHOOD", "LOCAL_LANDMARK", "HOME_ADDRESS", "STREET_ADDRESS"},
    "PRECISE_DATE": {"DATE_OF_BIRTH"},
    "ACCOUNT_ID": {"STUDENT_ID", "MEDICAL_ID", "CREDIT_CARD", "IBAN", "NATIONAL_ID"},
    "STREET_ADDRESS": {"HOME_ADDRESS"},
}


def fuse(detections: Sequence[Detection], doc: Document, config: Config) -> list[Span]:
    """Cluster overlapping detections into final, non-overlapping spans."""
    mode = config.get("fusion.mode", "union")
    keep = [d for d in detections if d.entity and d.entity != "KEEP"]
    if not keep:
        return []

    clusters = _cluster(keep)
    spans: list[Span] = []
    for cluster in clusters:
        spans.extend(_resolve_cluster(cluster, doc, config, mode))

    spans = _merge_adjacent(spans, doc, config)
    spans.sort(key=lambda s: (s.start, -s.end))
    return _drop_nested(spans, config)


# ---------------------------------------------------------------------------


def _cluster(dets: Sequence[Detection]) -> list[list[Detection]]:
    ordered = sorted(dets, key=lambda d: (d.start, -d.end))
    clusters: list[list[Detection]] = []
    current: list[Detection] = []
    reach = -1
    for d in ordered:
        if current and d.start < reach:
            current.append(d)
            reach = max(reach, d.end)
        else:
            if current:
                clusters.append(current)
            current = [d]
            reach = d.end
    if current:
        clusters.append(current)
    return clusters


def _resolve_cluster(
    cluster: list[Detection], doc: Document, config: Config, mode: str
) -> list[Span]:
    """Resolve a cluster, then re-resolve anything it left behind.

    Overlapping detections are not necessarily *the same* entity: "owns the
    bakery opposite Westbrook Primary" carries a workplace and a school that
    merely touch.  Letting the winner delete the loser would silently drop a
    direct identifier, so the leftovers get their own pass.
    """
    out: list[Span] = []
    remaining = list(cluster)
    for _ in range(len(cluster)):
        if not remaining:
            break
        span, supporting = _resolve(remaining, doc, config, mode)
        consumed = set(id(d) for d in supporting)
        leftovers: list[Detection] = []
        for d in remaining:
            if id(d) in consumed:
                continue
            clipped = _clip(d, span)
            if clipped is not None:
                leftovers.append(clipped)
        if span is not None:
            out.append(span)
        if len(leftovers) == len(remaining):
            break
        remaining = leftovers
    return out


def _resolve(
    cluster: list[Detection], doc: Document, config: Config, mode: str
) -> tuple[Span | None, list[Detection]]:
    tax = config.taxonomy
    entity = _vote_entity(cluster, config)
    if entity is None:
        return None, list(cluster)

    supporting = [d for d in cluster if _compatible(d.entity, entity)] or cluster
    detectors = {d.detector for d in supporting}

    if mode == "intersection":
        start = max(d.start for d in supporting)
        end = min(d.end for d in supporting)
        if end <= start:
            start = min(d.start for d in supporting)
            end = max(d.end for d in supporting)
    else:
        start = min(d.start for d in supporting)
        end = max(d.end for d in supporting)

    start, end = _trim(doc.text, start, end)
    if end <= start:
        return None, supporting

    base = max(d.score for d in supporting)
    bonus = float(config.get("fusion.agreement_bonus", 0.12)) * max(0, len(detectors) - 1)
    bonus = min(bonus, float(config.get("fusion.max_agreement_bonus", 0.3)))
    score = min(1.0, base + bonus)
    if any(d.meta.get("known_value") for d in supporting):
        score = 1.0

    if score < config.entity_threshold(entity):
        return None, supporting

    policy = tax.policy(entity)
    turn = doc.turn_at(start)
    span = Span(
        start=start,
        end=end,
        entity=entity,
        tier=policy.tier,
        score=score,
        text=doc.text[start:end],
        sources=[
            {
                "detector": d.detector,
                "entity": d.entity,
                "score": round(d.score, 4),
                "raw_label": d.raw_label,
                "start": d.start,
                "end": d.end,
                **({"channel": d.meta["channel"]} if "channel" in d.meta else {}),
            }
            for d in sorted(supporting, key=lambda x: -x.score)
        ],
        action=config.action_for(entity),
        canonical_hint=_canonical_hint(supporting),
        speaker=turn.speaker if turn else "",
        turn_index=turn.index if turn else -1,
    )
    return span, supporting


def _canonical_hint(supporting: list[Detection]) -> str:
    """The known value behind this span, if a detector matched one.

    "Sameera", "Samira" and "samira hassan" are three surface forms of one
    child.  Keying the pseudonym on the surface would give her three
    pseudonyms; keying it on the value the caller supplied gives her one.
    """
    best = ""
    for d in supporting:
        matched = d.meta.get("matched")
        if not matched:
            continue
        if not d.meta.get("partial"):
            return str(matched)
        best = best or str(matched)
    return best


def _clip(det: Detection, span: Span | None) -> Detection | None:
    """Cut a leftover detection free of the span that just won.

    Dropping it outright is what loses "the bakery" to "Westbrook Primary";
    clipping keeps the uncovered remainder eligible for its own span.
    """
    if span is None:
        return det
    if det.end <= span.start or det.start >= span.end:
        return det
    left = span.start - det.start
    right = det.end - span.end
    if max(left, right) < 3:
        return None
    if left >= right:
        det.end = span.start
    else:
        det.start = span.end
    det.text = ""
    return det if det.end - det.start >= 3 else None


def _compatible(entity_a: str, entity_b: str) -> bool:
    if entity_a == entity_b:
        return True
    return entity_a in SPECIALISATIONS.get(entity_b, ()) or entity_b in SPECIALISATIONS.get(
        entity_a, ()
    )


# entities that must never be split - half an email address is still an
# email address, and the remainder reads as if it were something else
ATOMIC = {
    "EMAIL",
    "URL",
    "PHONE",
    "IBAN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "NATIONAL_ID",
    "PASSPORT",
    "SOCIAL_HANDLE",
}


def _vote_entity(cluster: list[Detection], config: Config) -> str | None:
    tax = config.taxonomy
    # one vote per detector per entity: a detector that fires four times on
    # the same text (a full name plus each of its components) is still one
    # opinion, and summing raw scores would let it outvote the ensemble
    per_detector: dict[str, dict[str, float]] = {}
    for d in cluster:
        node = per_detector.setdefault(d.entity, {})
        node[d.detector] = max(node.get(d.detector, 0.0), d.score)
    weights = {entity: sum(scores.values()) for entity, scores in per_detector.items()}
    if not weights:
        return None

    # an atomic entity that spans the whole cluster wins outright - but only
    # if it is also the strongest vote, or "943 476 5919" after "NHS number"
    # would be relabelled from a medical id to a phone number
    top_weight = max(weights.values())
    atomic = [e for e in weights if e in ATOMIC and weights[e] >= top_weight - 1e-9]
    if atomic:
        cluster_start = min(d.start for d in cluster)
        cluster_end = max(d.end for d in cluster)
        for entity in sorted(atomic, key=lambda e: -weights[e]):
            covering = [d for d in cluster if d.entity == entity]
            if (
                min(d.start for d in covering) <= cluster_start
                and max(d.end for d in covering) >= cluster_end
            ):
                return entity

    promote = bool(config.get("fusion.promote_specific_over_generic", True))
    best_generic = max(weights.items(), key=lambda kv: (kv[1], tax.priority(kv[0])))[0]
    if not promote:
        return best_generic

    # a specific label anywhere in the cluster outranks its generic parent,
    # provided it carries real support rather than a stray low-score hit
    specifics = [
        e for e in weights if any(e in SPECIALISATIONS.get(g, ()) for g in weights if g != e)
    ]
    if specifics:
        top_specific = max(specifics, key=lambda e: (weights[e], tax.priority(e)))
        if weights[top_specific] >= 0.35 * weights[best_generic]:
            return top_specific
    return best_generic


_TRIM_CHARS = " \t\n\r,.;:!?\"'“”‘’()[]{}<>-–—"


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _TRIM_CHARS:
        start += 1
    while end > start and text[end - 1] in _TRIM_CHARS:
        end -= 1
    return start, end


def _merge_adjacent(spans: list[Span], doc: Document, config: Config) -> list[Span]:
    gap = int(config.get("fusion.merge_adjacent_gap", 2))
    if gap <= 0 or not spans:
        return spans
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    out: list[Span] = [spans[0]]
    for s in spans[1:]:
        prev = out[-1]
        between = doc.text[prev.end : s.start]
        if s.entity == prev.entity and len(between) <= gap and between.strip(" \t-'") == "":
            prev.end = max(prev.end, s.end)
            prev.text = doc.text[prev.start : prev.end]
            prev.score = max(prev.score, s.score)
            prev.sources.extend(s.sources)
        else:
            out.append(s)
    return out


def _drop_nested(spans: list[Span], config: Config) -> list[Span]:
    """Guarantee a non-overlapping set; the longer/stronger span wins."""
    out: list[Span] = []
    for s in spans:
        if not out:
            out.append(s)
            continue
        prev = out[-1]
        if s.start < prev.end:
            keep_prev = (prev.length, prev.score, config.taxonomy.priority(prev.entity)) >= (
                s.length,
                s.score,
                config.taxonomy.priority(s.entity),
            )
            if keep_prev:
                prev.sources.extend(s.sources)
                prev.end = max(prev.end, s.end)
            else:
                s.sources.extend(prev.sources)
                s.start = min(prev.start, s.start)
                out[-1] = s
        else:
            out.append(s)
    return out
