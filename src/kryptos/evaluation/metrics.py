"""Evaluation metrics for transcript de-identification.

Three deliberate departures from the usual PII leaderboard:

*F2, not F1.*  For a minor, a false positive costs a slightly uglier
transcript; a false negative costs identity.  Recall is weighted accordingly.

*Conversation leakage rate.*  The unit of privacy failure is the
conversation, not the token.  100 transcripts of which 99 are perfect and one
still carries a full name and a school can score >99% token F1 and still be
an unacceptable system.  This is the release-blocking number.

*Partial credit is reported separately, never blended in.*  A span that
catches "Rahman" but not "Nafisa" is a leak, and exact-match recall is the
number that says so.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

Span = tuple[int, int, str]  # (start, end, entity)


def _overlap(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    def f(self, beta: float = 1.0) -> float:
        p, r = self.precision(), self.recall()
        if p + r == 0:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision(), 4),
            "recall": round(self.recall(), 4),
            "f1": round(self.f(1.0), 4),
            "f2": round(self.f(2.0), 4),
        }


def match(
    gold: Sequence[Span], pred: Sequence[Span], mode: str = "exact", typed: bool = True
) -> tuple[Counts, list[Span], list[Span]]:
    """Greedy one-to-one matching.  mode: exact | partial | boundary."""
    counts = Counts()
    used: set[int] = set()
    missed: list[Span] = []
    for g in gold:
        hit = None
        for i, p in enumerate(pred):
            if i in used:
                continue
            if typed and p[2] != g[2]:
                continue
            if mode == "exact" and (p[0], p[1]) == (g[0], g[1]):
                hit = i
            elif mode in ("partial", "boundary") and _overlap(g, p) > 0:
                if mode == "boundary" and _overlap(g, p) < 0.5 * (g[1] - g[0]):
                    continue
                hit = i
            if hit is not None:
                break
        if hit is None:
            counts.fn += 1
            missed.append(g)
        else:
            counts.tp += 1
            used.add(hit)
    spurious = [p for i, p in enumerate(pred) if i not in used]
    counts.fp = len(spurious)
    return counts, missed, spurious


@dataclass
class Evaluation:
    overall: dict[str, Counts] = field(default_factory=dict)  # by match mode
    by_entity: dict[str, Counts] = field(default_factory=dict)
    by_tier: dict[str, Counts] = field(default_factory=dict)
    documents: int = 0
    leaking_documents: int = 0
    leaks: list[dict[str, Any]] = field(default_factory=list)
    false_redactions: int = 0
    predicted_spans: int = 0

    def conversation_leakage_rate(self) -> float:
        return self.leaking_documents / self.documents if self.documents else 0.0

    def false_redaction_rate(self) -> float:
        return self.false_redactions / self.predicted_spans if self.predicted_spans else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "conversation_leakage_rate": round(self.conversation_leakage_rate(), 4),
            "leaking_documents": self.leaking_documents,
            "false_redaction_rate": round(self.false_redaction_rate(), 4),
            "overall": {k: v.to_dict() for k, v in self.overall.items()},
            "by_entity": {
                k: v.to_dict()
                for k, v in sorted(self.by_entity.items(), key=lambda kv: kv[1].recall())
            },
            "by_tier": {k: v.to_dict() for k, v in self.by_tier.items()},
            "worst_leaks": self.leaks[:25],
        }


def evaluate(
    pairs: Iterable[tuple[str, Sequence[Span], Sequence[Span]]],
    tier_of: dict[str, str] | None = None,
    leak_tiers: Sequence[str] = ("direct",),
) -> Evaluation:
    """pairs: (doc_id, gold spans, predicted spans)."""
    tier_of = tier_of or {}
    ev = Evaluation()
    for mode in ("exact", "partial"):
        ev.overall[mode] = Counts()
    ev.overall["untyped_partial"] = Counts()

    for doc_id, gold, pred in pairs:
        ev.documents += 1
        ev.predicted_spans += len(pred)

        for mode in ("exact", "partial"):
            c, missed, spurious = match(gold, pred, mode=mode)
            ev.overall[mode].tp += c.tp
            ev.overall[mode].fp += c.fp
            ev.overall[mode].fn += c.fn
            if mode == "partial":
                ev.false_redactions += len(spurious)

        c_untyped, _, _ = match(gold, pred, mode="partial", typed=False)
        ev.overall["untyped_partial"].tp += c_untyped.tp
        ev.overall["untyped_partial"].fp += c_untyped.fp
        ev.overall["untyped_partial"].fn += c_untyped.fn

        entities = {g[2] for g in gold} | {p[2] for p in pred}
        # A gold identifier must be covered in full. Partial overlap can
        # leave a surname or digits behind. Coverage is independent of type.
        doc_leak: list[Span] = []
        for g in gold:
            cursor = g[0]
            for p in sorted(pred):
                if p[0] > cursor:
                    break
                if p[1] > cursor:
                    cursor = p[1]
            if cursor < g[1] and tier_of.get(g[2], "unknown") in leak_tiers:
                doc_leak.append(g)
        for entity in entities:
            g_e = [g for g in gold if g[2] == entity]
            p_e = [p for p in pred if p[2] == entity]
            c, missed, _ = match(g_e, p_e, mode="partial")
            node = ev.by_entity.setdefault(entity, Counts())
            node.tp += c.tp
            node.fp += c.fp
            node.fn += c.fn
            tier = tier_of.get(entity, "unknown")
            tnode = ev.by_tier.setdefault(tier, Counts())
            tnode.tp += c.tp
            tnode.fp += c.fp
            tnode.fn += c.fn

        if doc_leak:
            ev.leaking_documents += 1
            ev.leaks.append(
                {
                    "doc_id": doc_id,
                    "missed": [{"start": s[0], "end": s[1], "entity": s[2]} for s in doc_leak],
                }
            )
    return ev


def pii_token_wer(
    reference_words: Sequence[str], hypothesis_words: Sequence[str], pii_indices: Sequence[int]
) -> dict[str, float]:
    """WER restricted to the tokens that carry PII.

    Global WER hides the failure that matters: an ASR system can be excellent
    overall and still mangle every name, and a mangled name is a name your
    detectors will not find.
    """
    from ..normalize import normalize

    ref = [normalize(w) for w in reference_words]
    hyp = [normalize(w) for w in hypothesis_words]
    d = _align(ref, hyp)
    pii = set(pii_indices)
    errors = sum(1 for op, index in d if index in pii and op != "ok")
    total = len(pii) or 1
    overall_err = sum(1 for op, _ in d if op != "ok")
    return {
        "pii_token_wer": round(errors / total, 4),
        "wer": round(overall_err / (len(ref) or 1), 4),
        "pii_tokens": len(pii),
    }


def _align(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str, int]]:
    """Edit operations; insertions belong to the preceding reference token.

    Leading insertions belong to the first reference token. This attribution
    makes the PII-specific measure reproducible when alignment has ties.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
        ):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1))
            i -= 1
        else:
            ops.append(("ins", max(0, i - 1)))
            j -= 1
    return list(reversed(ops))
