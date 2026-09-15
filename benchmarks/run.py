"""Measure synthetic coverage, output retention, utility, and runtime."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import statistics
import time
from collections import Counter
from pathlib import Path

from kryptos import Config, Pipeline, __version__
from kryptos.evaluation import evaluate
from kryptos.io_formats import parse_json

from . import gen_multilingual, gen_synth


def corpora(seed=20260915, per_language=12):
    clean, asr, negative = gen_synth.build(n_docs=120, seed=seed)
    result = {"written": clean, "asr": asr, "negative": negative}
    for language in gen_multilingual.L:
        language_seed = seed + int.from_bytes(hashlib.sha256(language.encode()).digest()[:4], "big")
        for condition in ("clean", "asr"):
            rng = random.Random(language_seed)
            records = []
            for i in range(per_language):
                record = gen_multilingual.build_document(language, rng, asr=condition == "asr")
                record["doc_id"] = f"{language}-{condition}-{i}"
                records.append(record)
            result[f"{language}_{condition}"] = records
    for condition in ("clean", "asr"):
        rng = random.Random(seed + 7)
        result[f"codeswitch_{condition}"] = []
        for i in range(per_language):
            record = gen_multilingual.build_codeswitched(rng, asr=condition == "asr")
            record["doc_id"] = f"codeswitch-{condition}-{i}"
            result[f"codeswitch_{condition}"].append(record)
    return result


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def stable_result(result):
    report = result.to_dict()
    report["stats"].pop("timings", None)
    return report


def source_digest():
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for directory in (root / "src", root / "benchmarks"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".txt"}:
                sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(sources)


def score(records, pipeline, use_known):
    pairs, statuses, milliseconds = [], Counter(), []
    surface_retention, passed_retention, passed_uncovered = 0, 0, 0
    negative_redactions, negative_characters = 0, 0
    total_characters, direct_documents = 0, 0
    output_hash = hashlib.sha256()
    tier_of = {name: policy.tier for name, policy in pipeline.config.taxonomy.entities.items()}
    for original in records:
        record = copy.deepcopy(original)
        if not use_known:
            record["known_values"] = {}
        document = parse_json(record)
        gold = [(span["start"], span["end"], span["entity"]) for span in record["spans"]]
        assert all(0 <= start < end <= len(document.text) for start, end, _ in gold)
        start_time = time.perf_counter()
        result = pipeline.process(document)
        milliseconds.append(1000 * (time.perf_counter() - start_time))
        pred = [(s.start, s.end, s.entity) for s in result.spans if s.action != "keep"]
        pairs.append((record["doc_id"], gold, pred))
        statuses[result.status] += 1
        total_characters += len(document.text)
        output_hash.update(json.dumps(stable_result(result), sort_keys=True).encode())
        direct = [g for g in gold if tier_of.get(g[2]) == "direct"]
        direct_documents += bool(direct)
        # This separately tests literal surface survival in transformed text.
        # It does not detect all paraphrases, identity combinations, or variants.
        retained = any(
            document.text[s:e].casefold() in result.sanitized_text.casefold() for s, e, _ in direct
        )
        surface_retention += retained
        passed_retention += retained and result.status == "passed"
        doc_score = evaluate([(record["doc_id"], gold, pred)], tier_of=tier_of)
        passed_uncovered += doc_score.leaking_documents > 0 and result.status == "passed"
        if not gold:
            negative_redactions += result.stats["redacted_characters"]
            negative_characters += len(document.text)
    metrics = evaluate(pairs, tier_of=tier_of).to_dict()
    metrics.pop("worst_leaks")
    metrics.update(
        {
            "direct_identifier_documents": direct_documents,
            "statuses": dict(statuses),
            "retained_direct_surface_documents": surface_retention,
            "passed_with_retained_direct_surface": passed_retention,
            "passed_with_uncovered_direct_span": passed_uncovered,
            "negative_redacted_character_fraction": round(
                negative_redactions / negative_characters, 6
            )
            if negative_characters
            else None,
            "output_sha256_without_timings": output_hash.hexdigest(),
            "latency_ms_p50": round(statistics.median(milliseconds), 3),
            "latency_ms_p95": round(sorted(milliseconds)[int(0.95 * (len(milliseconds) - 1))], 3),
            "characters_per_second": round(total_characters / (sum(milliseconds) / 1000), 1),
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["core", "gliner"], default="core")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    cfg = Config.load()
    if args.profile == "gliner":
        import torch

        torch.manual_seed(args.seed)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        for detector in cfg.raw["detectors"]:
            if detector["name"] == "gliner_child":
                detector.update(enabled=True, device=args.device)
    cold_start = time.perf_counter()
    pipeline = Pipeline(cfg)
    load_seconds = time.perf_counter() - cold_start
    fixtures = corpora(args.seed)
    versions = {}
    for name in ("torch", "transformers", "gliner", "rapidfuzz"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "schema_version": 1,
        "synthetic_only": True,
        "version": __version__,
        "profile": args.profile,
        "device": args.device,
        "seed": args.seed,
        "python": platform.python_version(),
        "os": platform.system(),
        "machine": platform.machine(),
        "dependencies": versions,
        "source_sha256": source_digest(),
        "corpus_sha256": digest(fixtures),
        "detectors": [d.describe() for d in pipeline.detectors],
        "model_load_seconds": round(load_seconds, 3),
        "results": {},
    }
    for name, records in fixtures.items():
        for use_known in (False, True):
            key = f"{name}/{'known' if use_known else 'no_known'}"
            report["results"][key] = score(records, pipeline, use_known)
            print(f"completed {key}: {len(records)} synthetic documents", flush=True)
    fixture = fixtures["written"][0]
    first = pipeline.process(parse_json(copy.deepcopy(fixture)))
    second = pipeline.process(parse_json(copy.deepcopy(fixture)))
    report["repeatability_sample_passed"] = stable_result(first) == stable_result(second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["repeatability_sample_passed"]:
        raise SystemExit("Repeatability check failed")
    print(f"Saved {args.profile} benchmark report", flush=True)


if __name__ == "__main__":
    main()
