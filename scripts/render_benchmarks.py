"""Render the public summary from the checked-in, content-free measurements."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = {
        name: json.loads((ROOT / "docs" / "benchmarks" / f"{name}.json").read_text())
        for name in ("core", "gliner")
    }
    if reports["core"]["corpus_sha256"] != reports["gliner"]["corpus_sha256"]:
        raise SystemExit("Corpus hashes differ; do not compare these runs")
    if reports["core"]["source_sha256"] != reports["gliner"]["source_sha256"]:
        raise SystemExit("Source hashes differ; rerun the stale profile")
    lines = [
        "# Synthetic benchmark results",
        "",
        "## Scope",
        "",
        "These are reproducible regression diagnostics, not independent real-world accuracy estimates. "
        "The fixtures use hand-authored templates and invented identities. Some templates informed "
        "the rules; there is no held-out accuracy claim or confidence interval.",
        "",
        "Each profile runs 544 synthetic conversations twice: with and without known values. "
        "The suite includes 120 written-form, 120 ASR-style, 40 negative controls, "
        "240 language probes (ten languages, written and ASR), and 24 mixed-language probes. "
        "That is **2,176 document evaluations across two detector profiles**. Repeated templates "
        "and the two knowledge configurations are not independent samples.",
        "",
        "Both reports record seed, corpus and source hashes, dependency versions, detector revisions, "
        "status counts, per-entity metrics, output hashes, and runtime. All model evaluation runs "
        "locally; no private conversations are used.",
        "",
        "## Main results",
        "",
        "Rows below use the 120-document written and ASR suites with known values. "
        "Full breakdowns, including runs without known values and each language, are in "
        "[core.json](benchmarks/core.json) and [gliner.json](benchmarks/gliner.json).",
        "",
        "| Profile | Input | Exact typed recall | Full-span coverage miss rate | Whole direct surface retained | Passed with retained surface | Median ms/doc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for profile, report in reports.items():
        for condition in ("written", "asr"):
            row = report["results"][f"{condition}/known"]
            lines.append(
                f"| {profile} | {condition} | {row['overall']['exact']['recall']:.1%} | "
                f"{row['conversation_leakage_rate']:.1%} | "
                f"{row['retained_direct_surface_documents']}/{row['documents']} | "
                f"{row['passed_with_retained_direct_surface']}/{row['documents']} | "
                f"{row['latency_ms_p50']:.1f} |"
            )
    lines += [
        "",
        "## How to read the metrics",
        "",
        "- **Exact typed recall** requires both the gold span boundaries and entity label to match. "
        "Partial and untyped matching are reported separately in JSON.",
        "- **Full-span coverage miss rate** is the fraction of conversations with an annotated direct "
        "identifier not covered in full by the union of transformed spans. The JSON field retains "
        "the name `conversation_leakage_rate`. It is conservative: even a retained article or space "
        "inside a gold span counts as uncovered. It is not measured re-identification probability.",
        "- **Whole direct surface retained** checks whether any complete annotated direct identifier "
        "appears verbatim, case-insensitively, in candidate output. This catches exact surface "
        "survival but can miss partial names, variants, and combinations of contextual details.",
        "- **Passed with retained surface** counts those cases assigned `passed`. Zero in a row "
        "does not prove zero unknown PII. Review and blocked outcomes are explicitly separate.",
        "- **Negative controls** contain no annotated identifiers. Their character redaction "
        "fraction measures loss of ordinary text on this fixture set.",
        "",
        "## Utility and gates",
        "",
    ]
    for profile, report in reports.items():
        row = report["results"]["negative/no_known"]
        passed = sum(
            row["passed_with_retained_direct_surface"] for row in report["results"].values()
        )
        lines.append(
            f"- **{profile}:** {row['negative_redacted_character_fraction']:.2%} of negative-control "
            f"characters removed; {passed} passed documents retained a whole annotated direct "
            "surface across all buckets/configurations."
        )
    lines += [
        "",
        "Long uncased text and unvalidated languages/scripts require review. "
        "That withholding is a workflow safeguard, not successful multilingual anonymization. "
        "The GLiNER profile can increase coverage and over-redaction simultaneously. "
        "Use the full per-language reports when choosing a profile; do not extrapolate from English.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv sync --locked --extra ml --extra fast",
        "uv run --locked --extra ml --extra fast python -m benchmarks.run --profile core \\",
        "  --output docs/benchmarks/core.json",
        "HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \\",
        "uv run --locked --extra ml --extra fast python -m benchmarks.run --profile gliner \\",
        "  --device cpu --output docs/benchmarks/gliner.json",
        "uv run --locked python scripts/render_benchmarks.py",
        "```",
        "",
        "The first model run downloads pinned weights and tokenizer/backbone files. "
        "Afterward, add `HF_HUB_OFFLINE=1` to require local assets. Published runs use Python "
        f"{reports['core']['python']} on {reports['core']['os']} {reports['core']['machine']}, "
        "with four PyTorch CPU threads and the optional native string matcher enabled. "
        "Runtime numbers are hardware- and load-dependent. Cross-platform bit-identical "
        "floating-point inference is not promised.",
        "",
        "The runner also compares repeated output on one fixed sample, excluding timing fields. "
        "This is a repeatability smoke check, not a multi-seed robustness experiment. "
        "The checked-in reports retain all bucket results rather than filtering misses.",
        "",
        "Core and model source hash: `" + reports["core"]["source_sha256"] + "`.",
        "",
        "Corpus hash: `" + reports["core"]["corpus_sha256"] + "`.",
        "",
    ]
    (ROOT / "docs" / "BENCHMARKS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Rendered benchmark summary from matching source/corpus reports")


if __name__ == "__main__":
    main()
