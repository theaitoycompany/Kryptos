"""Command line interface.

python -m kryptos run transcript.txt --known known.json -o out.json
python -m kryptos run - < transcript.txt --text-only
python -m kryptos detectors
python -m kryptos eval gold.jsonl
python -m kryptos offsets transcript.txt
python -m kryptos audio-plan out.json --input audio.wav --output clean.wav
python -m kryptos selftest
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile

from .audio import ffmpeg_command
from .config import Config
from .detectors.base import build_detectors
from .io_formats import load_file, load_text
from .pipeline import Pipeline

EXIT_OK = 0
EXIT_REVIEW = 10
EXIT_BLOCKED = 20
EXIT_ERROR = 2


def _load_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)
    if getattr(args, "known", None):
        with open(args.known, encoding="utf-8") as fh:
            cfg.raw["known_values"] = json.load(fh)
    for setting in getattr(args, "set", None) or []:
        key, _, value = setting.partition("=")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        cfg.set(key.strip(), parsed)
    names = {spec.get("name") for spec in cfg.raw.get("detectors", [])}
    requested = set(getattr(args, "enable", None) or []) | set(getattr(args, "disable", None) or [])
    if requested - names:
        raise ValueError("Unknown detector name")
    if getattr(args, "enable", None):
        wanted = set(args.enable)
        for spec in cfg.raw.get("detectors", []):
            if spec.get("name") in wanted:
                spec["enabled"] = True
    if getattr(args, "disable", None):
        unwanted = set(args.disable)
        for spec in cfg.raw.get("detectors", []):
            if spec.get("name") in unwanted:
                spec["enabled"] = False
    if getattr(args, "scope", None):
        cfg.set("surrogates.scope", args.scope)
    if getattr(args, "style", None):
        cfg.set("surrogates.style", args.style)
    if getattr(args, "no_qa", False):
        cfg.set("qa.enabled", False)
    return cfg


# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    pipe = Pipeline(cfg, only=args.only)

    if args.input == "-":
        doc = load_text(sys.stdin.read(), doc_id=args.doc_id or "stdin", fmt=args.format)
    else:
        doc = load_file(args.input, fmt=args.format)
        if args.doc_id:
            doc.doc_id = args.doc_id

    result = pipe.process(doc)
    show_text = result.status == "passed" or (result.status == "review" and args.allow_review)

    if args.text_only:
        payload = result.sanitized_text if show_text else ""
    elif args.turns:
        payload = (
            "\n".join("{}: {}".format(t["speaker"], t["text"]) for t in result.sanitized_turns)
            if show_text
            else ""
        )
    else:
        report = result.to_dict(include_detections=args.include_detections)
        if not show_text:
            report["sanitized_text"] = ""
            report["sanitized_turns"] = []
            report["text_withheld"] = True
        payload = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        _write_private(args.output, payload)
        print(
            f"output written (status={result.status}, {len(result.spans)} spans)",
            file=sys.stderr,
        )
    else:
        print(payload)

    if not args.quiet:
        _print_summary(result)

    if result.status == "blocked":
        return EXIT_BLOCKED
    if result.status == "review":
        return EXIT_REVIEW
    return EXIT_OK


def _write_private(path: str, payload: str) -> None:
    """Atomically replace the output with a file readable only by its owner."""
    parent = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix=".kryptos-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload + ("\n" if not payload.endswith("\n") else ""))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _print_summary(result) -> None:
    stats = result.stats
    out = sys.stderr
    print("", file=out)
    print(f"status            : {result.status.upper()}", file=out)
    print(f"spans             : {stats['n_spans']} ({stats['direct_spans']} direct)", file=out)
    print(
        "entities          : {}".format(
            ", ".join(f"{k}={v}" for k, v in list(stats["spans_by_entity"].items())[:12])
        ),
        file=out,
    )
    print(
        "detectors used    : {}".format(", ".join(sorted(stats["detector_contribution"]))), file=out
    )
    unavailable = [d["name"] for d in stats["detectors"] if not d["available"]]
    if unavailable:
        print("detectors offline : {}".format(", ".join(unavailable)), file=out)
    print(
        "quasi risk        : {:.2f}  {}".format(
            stats["quasi"]["risk_score"],
            ", ".join("+".join(c["entities"]) for c in stats["quasi"]["combinations"][:4]),
        ),
        file=out,
    )
    for f in result.qa:
        print(f"  [{f.severity.upper()}] {f.check:<26} {f.message}", file=out)


def cmd_detectors(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    for spec in cfg.raw.get("detectors", []):
        if args.all:
            spec["enabled"] = True
    dets = build_detectors(cfg)
    print(f"{'NAME':<14} {'TYPE':<13} {'STATUS':<9} MODEL / REASON")
    for d in dets:
        info = d.describe()
        status = "ready" if info["available"] else "offline"
        detail = info["model"] if info["available"] else info["reason"]
        print(f"{info['name']:<14} {info['type']:<13} {status:<9} {detail}")
    return EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation.runner import run as run_eval

    cfg = _load_config(args)
    ev, per_doc = run_eval(args.gold, config=cfg)
    report = ev.to_dict()
    report["per_document"] = per_doc
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    rate = ev.conversation_leakage_rate()
    print(f"\nconversation leakage rate: {rate:.4f}  (documents={ev.documents})", file=sys.stderr)
    if args.max_leakage is not None and rate > args.max_leakage:
        print(
            f"FAIL: leakage {rate:.4f} exceeds --max-leakage {args.max_leakage:.4f}",
            file=sys.stderr,
        )
        return EXIT_BLOCKED
    return EXIT_OK


def cmd_offsets(args: argparse.Namespace) -> int:
    doc = load_text(sys.stdin.read()) if args.input == "-" else load_file(args.input)
    print(
        json.dumps(
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "turns": [
                    {
                        "speaker": t.speaker,
                        "char_start": t.char_start,
                        "char_end": t.char_end,
                        "text": t.text,
                    }
                    for t in doc.turns
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return EXIT_OK


def cmd_audio_plan(args: argparse.Namespace) -> int:
    with open(args.result, encoding="utf-8") as fh:
        payload = json.load(fh)
    intervals = payload.get("audio_plan", [])
    if not intervals:
        print(
            f"no audio plan in {args.result} (the transcript carried no word timings)",
            file=sys.stderr,
        )
        return EXIT_ERROR
    print(ffmpeg_command(args.input, args.output, intervals, mode=args.mode))
    return EXIT_OK


def cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import run_selftest

    return run_selftest(verbose=not args.quiet)


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kryptos", description="Child/parent transcript de-identification"
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-c", "--config", help="JSON/YAML config layered over defaults")
        sp.add_argument("-k", "--known", help="JSON file of known values (deny list)")
        sp.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="override a config key, e.g. --set fusion.mode=intersection",
        )
        sp.add_argument("--enable", action="append", metavar="DETECTOR")
        sp.add_argument("--disable", action="append", metavar="DETECTOR")

    r = sub.add_parser("run", help="de-identify a transcript")
    common(r)
    r.add_argument("input", help="transcript file, or - for stdin")
    r.add_argument("-o", "--output")
    r.add_argument(
        "-f", "--format", default="auto", choices=["auto", "text", "json", "jsonl", "srt", "vtt"]
    )
    r.add_argument("--doc-id")
    r.add_argument("--only", action="append", metavar="DETECTOR", help="run only these detectors")
    r.add_argument("--text-only", action="store_true", help="print sanitized text only")
    r.add_argument(
        "--allow-review",
        action="store_true",
        help="include review-status candidate text for local inspection",
    )
    r.add_argument("--turns", action="store_true", help="print SPEAKER: text lines")
    r.add_argument(
        "--include-detections",
        action="store_true",
        help="include per-detector offsets and scores, without original matched text",
    )
    r.add_argument("--scope", choices=["per_turn", "per_conversation"])
    r.add_argument("--style", choices=["placeholder", "surrogate", "hmac"])
    r.add_argument("--no-qa", action="store_true")
    r.add_argument("-q", "--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("detectors", help="show detector availability")
    common(d)
    d.add_argument("--all", action="store_true", help="probe every detector, not just enabled")
    d.set_defaults(func=cmd_detectors)

    e = sub.add_parser("eval", help="score against a labelled gold set")
    common(e)
    e.add_argument("gold", help="JSONL gold file")
    e.add_argument("-o", "--output")
    e.add_argument(
        "--max-leakage", type=float, help="exit non-zero if conversation leakage exceeds this"
    )
    e.set_defaults(func=cmd_eval)

    o = sub.add_parser("offsets", help="print the flat text and turn offsets")
    o.add_argument("input")
    o.set_defaults(func=cmd_offsets)

    a = sub.add_parser("audio-plan", help="emit the ffmpeg command for an audio plan")
    a.add_argument("result", help="JSON result written by 'kryptos run'")
    a.add_argument("--input", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--mode", default="mute", choices=["mute", "beep"])
    a.set_defaults(func=cmd_audio_plan)

    s = sub.add_parser("selftest", help="run the built-in end-to-end checks")
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except BrokenPipeError:
        return EXIT_OK
    except Exception as exc:
        print(
            f"error: {type(exc).__name__}; check input, configuration, and installed detectors",
            file=sys.stderr,
        )
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
