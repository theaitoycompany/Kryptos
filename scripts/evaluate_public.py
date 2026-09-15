"""Evaluate a fixed sample from Gretel's public synthetic test split, without tuning."""

import argparse
import ast
import hashlib
import json
import random
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq

from kryptos import Config, Pipeline, Turn, __version__
from kryptos.io_formats import build_document

DATASET = "gretelai/gretel-pii-masking-en-v1"
REVISION = "e06eb1499ca8d54470f085021cd8e54f9efac7fd"
DATA_SHA256 = "c89a6bffa838edb0f2fc6686bf35da4930944519ac97308b0b6e32298594b5b1"
# These categories can be retained or generalized by Kryptos's contextual policy.
QUASI_TYPES = {"city", "company_name", "country", "date", "date_time", "postcode", "state", "time"}


def coverage(gold, predicted):
    """Count complete gold-span coverage by the union of transformed character spans."""
    covered = {i for start, end in predicted for i in range(start, end)}
    return [all(i in covered for i in range(start, end)) for start, end, _ in gold]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["core", "gliner"], required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != DATA_SHA256:
        raise SystemExit("Dataset checksum does not match the pinned test split")
    rows = pq.read_table(args.data, columns=["text", "entities", "domain"]).to_pylist()
    if len(rows) != 5000:
        raise SystemExit("Unexpected test split size")
    indices = sorted(random.Random(20260915).sample(range(len(rows)), 200))
    config = Config.load()
    if args.profile == "gliner":
        import torch

        torch.manual_seed(20260915)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        for spec in config.raw["detectors"]:
            if spec["name"] == "gliner_child":
                spec.update(enabled=True, device="cpu")
    pipeline = Pipeline(config)
    statuses, totals, domains = Counter(), Counter(), Counter()
    per_entity = {}
    latencies = []
    for number, index in enumerate(indices, 1):
        row = rows[index]
        doc = build_document([Turn("SPEAKER_UNKNOWN", row["text"])])
        gold = []
        for annotation in ast.literal_eval(row["entities"]):
            surface = unicodedata.normalize("NFC", annotation["entity"])
            if not surface or doc.text.count(surface) != 1 or len(annotation["types"]) != 1:
                raise ValueError("Annotation alignment requires review; no rows were excluded")
            start = doc.text.index(surface)
            gold.append((start, start + len(surface), annotation["types"][0]))
        started = time.perf_counter()
        result = pipeline.process(doc)
        latencies.append(1000 * (time.perf_counter() - started))
        predicted = [(span.start, span.end) for span in result.spans if span.action != "keep"]
        covered = coverage(gold, predicted)
        retained = [
            doc.text[start:end].casefold() in result.sanitized_text.casefold()
            for start, end, _ in gold
        ]
        direct = [i for i, (_, _, label) in enumerate(gold) if label not in QUASI_TYPES]
        statuses[result.status] += 1
        domains[row["domain"]] += 1
        totals["annotations"] += len(gold)
        totals["fully_covered_annotations"] += sum(covered)
        totals["direct_annotations"] += len(direct)
        totals["fully_covered_direct_annotations"] += sum(covered[i] for i in direct)
        totals["documents_with_direct_annotations"] += bool(direct)
        totals["documents_with_uncovered_direct_span"] += any(not covered[i] for i in direct)
        has_retained = any(retained[i] for i in direct)
        totals["documents_with_retained_direct_surface"] += has_retained
        totals["passed_with_retained_direct_surface"] += has_retained and result.status == "passed"
        totals["passed_with_uncovered_direct_span"] += (
            any(not covered[i] for i in direct) and result.status == "passed"
        )
        for i, (_, _, label) in enumerate(gold):
            entity = per_entity.setdefault(label, Counter())
            entity["annotations"] += 1
            entity["fully_covered"] += covered[i]
            entity["surface_retained"] += retained[i]
        if number % 25 == 0:
            print(f"Evaluated {number}/200 public synthetic documents", flush=True)
    root = Path(__file__).resolve().parents[1]
    source = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src" / "kryptos").rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json", ".txt"}
    }
    report = {
        "schema_version": 1,
        "version": __version__,
        "profile": args.profile,
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "dataset_sha256": DATA_SHA256,
        "split": "test",
        "license": "apache-2.0",
        "synthetic_only": True,
        "known_values_provided": False,
        "seed": 20260915,
        "selected_row_indices": indices,
        "documents": len(indices),
        "excluded_documents": 0,
        "quasi_types_excluded_from_direct_metrics": sorted(QUASI_TYPES),
        "statuses": dict(statuses),
        "totals": dict(totals),
        "per_entity": dict(sorted(per_entity.items())),
        "domains": dict(sorted(domains.items())),
        "latency_ms_median": round(statistics.median(latencies), 3),
        "pyarrow": pyarrow.__version__,
        "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "detectors": [detector.describe() for detector in pipeline.detectors],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved aggregate public-dataset evaluation; no transcript text or source IDs included")


if __name__ == "__main__":
    main()
