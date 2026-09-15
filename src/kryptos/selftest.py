"""Built-in end-to-end checks.

These are leak tests, not unit tests: each case is a transcript plus the
strings that must not survive.  They run with whatever detectors are
installed, so the same suite reports honestly on a stdlib-only box and on a
full ML install.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from .config import Config
from .pipeline import Pipeline

# (name, transcript, known_values, must_not_appear, should_appear)
CASES: list[tuple[str, str, dict[str, Any], Sequence[str], Sequence[str]]] = [
    (
        "direct identifiers",
        "CHILD: I'm Aisha and Mum picks me up from Greenfield School.\n"
        "PARENT: Call me on 07911 123456 or samira.hasan@gmail.com.",
        {"CHILD_NAME": ["Aisha"], "SCHOOL_NAME": ["Greenfield School"]},
        ["Aisha", "Greenfield", "07911", "123456", "samira.hasan@gmail.com"],
        ["<CHILD_01>", "<SCHOOL_01>"],
    ),
    (
        "asr-corrupted known names",
        "PARENT: Sameera goes to Green field School, her teacher is Miss Roopa.\n"
        "CHILD: my name is samira hassan.",
        {
            "CHILD_NAME": ["Samira Hasan"],
            "SCHOOL_NAME": ["Greenfield School"],
            "TEACHER_NAME": ["Ms Rupa"],
        },
        ["Sameera", "Roopa", "samira", "hassan"],
        [],
    ),
    (
        "spoken identifiers",
        "PARENT: my number is oh seven nine one one one two three four five six, "
        "and my email is samira dot hasan at gmail dot com.",
        {},
        ["oh seven nine one one", "at gmail dot com"],
        [],
    ),
    (
        "quasi-identifier combination",
        "CHILD: I'm nine and I'm the only goalkeeper on the under-10 Tigers team.\n"
        "PARENT: Dad owns the bakery opposite Westbrook Primary.",
        {},
        ["nine", "Westbrook"],
        ["<AGE_9_11>"],
    ),
    (
        "role typing and pseudonym consistency",
        "CHILD: my teacher is Ms Rupa and my brother Zaid is in year 4.\n"
        "PARENT: Zaid started at the same school as Aisha.",
        {"CHILD_NAME": ["Aisha"]},
        ["Zaid", "Rupa", "Aisha"],
        ["<SIBLING_01>", "<TEACHER_01>", "<CHILD_01>"],
    ),
    (
        "structured identifiers",
        "PARENT: student number 4459-2, NHS number 943 476 5919, "
        "we live at 21 Oak Street, SW1A 1AA. Her DOB is 12/03/2016.",
        {},
        ["4459-2", "943 476 5919", "21 Oak Street", "SW1A 1AA", "12/03/2016"],
        [],
    ),
]


def run_selftest(verbose: bool = True, config: Config | None = None) -> int:
    cfg = config or Config.load()
    failures = 0
    checks = 0

    for name, transcript, known, forbidden, expected in CASES:
        case_cfg = Config.load(overrides=cfg.raw)
        case_cfg.raw["known_values"] = known
        pipe = Pipeline(case_cfg)
        result = pipe.process_text(transcript, doc_id="selftest")
        out = result.sanitized_text
        problems: list[str] = []
        for bad in forbidden:
            checks += 1
            if bad.lower() in out.lower():
                problems.append(f"LEAK {bad!r}")
        for good in expected:
            checks += 1
            if good not in out:
                problems.append(f"MISSING {good!r}")
        if problems:
            failures += 1
        if verbose:
            mark = "FAIL" if problems else "ok  "
            print(f"[{mark}] {name:<38} status={result.status:<8} spans={len(result.spans)}")
            if problems:
                for p in problems:
                    print(f"        {p}")
                for line in out.splitlines():
                    print(f"        | {line}")
    if verbose:
        print(f"\n{len(CASES) - failures}/{len(CASES)} cases passed, {checks} assertions")
        avail = [d.name for d in Pipeline(cfg).detectors if d.available]
        print("detectors active: {}".format(", ".join(avail)))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(run_selftest())
