"""Run the pipeline over a labelled set and report release metrics.

Gold file format - JSON Lines, one conversation per line:

    {"doc_id": "c001",
     "turns": [{"speaker": "CHILD", "text": "I'm Aisha"}],
     "spans": [{"start": 4, "end": 9, "entity": "CHILD_NAME"}],
     "known_values": {"CHILD_NAME": ["Aisha"]}}

``start``/``end`` are character offsets into the flat text the loader builds
(turn texts joined by newlines, speaker labels excluded) - print it with
``python -m kryptos offsets <file>`` if you are annotating by hand.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..config import Config
from ..io_formats import parse_json
from ..pipeline import Pipeline
from .metrics import Evaluation, evaluate


def load_gold(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run(
    path: str,
    config: Config | None = None,
    pipeline: Pipeline | None = None,
    leak_tiers: Sequence[str] = ("direct",),
) -> tuple[Evaluation, list[dict[str, Any]]]:
    config = config or Config.load()
    pipeline = pipeline or Pipeline(config)
    records = load_gold(path)

    pairs = []
    per_doc: list[dict[str, Any]] = []
    for rec in records:
        doc = parse_json(rec, doc_id=str(rec.get("doc_id", "")))
        result = pipeline.process(doc, known_values=rec.get("known_values"))
        gold = [(int(s["start"]), int(s["end"]), s["entity"]) for s in rec.get("spans", [])]
        pred = [(s.start, s.end, s.entity) for s in result.spans if s.action != "keep"]
        pairs.append((doc.doc_id or str(len(pairs)), gold, pred))
        per_doc.append(
            {
                "doc_id": doc.doc_id,
                "status": result.status,
                "n_gold": len(gold),
                "n_pred": len(pred),
                "qa_fail": sum(1 for f in result.qa if f.severity == "fail"),
            }
        )

    tier_of = {name: pol.tier for name, pol in config.taxonomy.entities.items()}
    return evaluate(pairs, tier_of=tier_of, leak_tiers=leak_tiers), per_doc
