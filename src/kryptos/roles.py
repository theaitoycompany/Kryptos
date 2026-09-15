"""Speaker role inference.

Role matters twice over.  It types the names a speaker says ("my teacher is
X" said by the child), and it drives pseudonym allocation (<CHILD_01> vs
<PARENT_01>) so the sanitized transcript keeps the conversational structure
that makes it worth analysing at all.

Roles come from three sources, in decreasing order of trust:
  1. an explicit role in the input (``role`` field, or a label like ``CHILD:``)
  2. a speaker label that matches a known participant name
  3. linguistic cues in what the speaker says
"""

from __future__ import annotations

import re

from .normalize import fuzzy_match, normalize
from .types import Document, Turn

ROLE_CHILD = "child"
ROLE_PARENT = "parent"
ROLE_CLINICIAN = "clinician"
ROLE_TEACHER = "teacher"
ROLE_SIBLING = "sibling"
ROLE_UNKNOWN = "unknown"

LABEL_ROLES: dict[str, str] = {
    "child": ROLE_CHILD,
    "kid": ROLE_CHILD,
    "boy": ROLE_CHILD,
    "girl": ROLE_CHILD,
    "son": ROLE_CHILD,
    "daughter": ROLE_CHILD,
    "student": ROLE_CHILD,
    "pupil": ROLE_CHILD,
    "c": ROLE_CHILD,
    "minor": ROLE_CHILD,
    "parent": ROLE_PARENT,
    "mum": ROLE_PARENT,
    "mom": ROLE_PARENT,
    "mother": ROLE_PARENT,
    "mummy": ROLE_PARENT,
    "dad": ROLE_PARENT,
    "father": ROLE_PARENT,
    "daddy": ROLE_PARENT,
    "caregiver": ROLE_PARENT,
    "carer": ROLE_PARENT,
    "guardian": ROLE_PARENT,
    "adult": ROLE_PARENT,
    "p": ROLE_PARENT,
    "doctor": ROLE_CLINICIAN,
    "dr": ROLE_CLINICIAN,
    "clinician": ROLE_CLINICIAN,
    "therapist": ROLE_CLINICIAN,
    "nurse": ROLE_CLINICIAN,
    "counsellor": ROLE_CLINICIAN,
    "counselor": ROLE_CLINICIAN,
    "interviewer": ROLE_CLINICIAN,
    "researcher": ROLE_CLINICIAN,
    "assessor": ROLE_CLINICIAN,
    "teacher": ROLE_TEACHER,
    "ta": ROLE_TEACHER,
    "tutor": ROLE_TEACHER,
    "sibling": ROLE_SIBLING,
    "brother": ROLE_SIBLING,
    "sister": ROLE_SIBLING,
}

CHILD_CUES = [
    (
        re.compile(r"\bmy (mum|mummy|mom|mommy|dad|daddy|teacher|miss|nan|nanny|granny)\b", re.I),
        2.0,
    ),
    (re.compile(r"\bi'?m (in )?(year|grade|reception|nursery)\b", re.I), 2.0),
    (re.compile(r"\bat (school|nursery|playtime|break time|lunchtime)\b", re.I), 1.0),
    (
        re.compile(
            r"\bi'?m (\d|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.I
        ),
        1.5,
    ),
    (re.compile(r"\b(my toy|my teddy|playtime|homework|my best friend)\b", re.I), 1.0),
]
PARENT_CUES = [
    (re.compile(r"\bmy (son|daughter|child|kid|boy|girl|little one|eldest|youngest)\b", re.I), 2.5),
    (
        re.compile(
            r"\b(she|he|they) (goes|go|started|starts|attends) (to )?(school|nursery)\b", re.I
        ),
        1.5,
    ),
    (re.compile(r"\b(i|we) (work|works|working)\b", re.I), 1.2),
    (
        re.compile(
            r"\b(pick (him|her|them) up|drop (him|her|them) off|his teacher|her teacher)\b", re.I
        ),
        1.5,
    ),
    (re.compile(r"\b(mortgage|nursery fees|appointment|gp|our address)\b", re.I), 1.0),
]
CLINICIAN_CUES = [
    (
        re.compile(
            r"\b(can you tell me|how (does|did) that (make you )?feel|"
            r"i'?m going to ask|on a scale of|next question|thank you for coming)\b",
            re.I,
        ),
        2.0,
    ),
    (re.compile(r"\b(the (assessment|session|appointment) (today|now))\b", re.I), 1.5),
]


def _label_role(label: str) -> str | None:
    key = normalize(label).replace(".", "").strip()
    if not key:
        return None
    if key in LABEL_ROLES:
        return LABEL_ROLES[key]
    # "CHILD_1", "Parent 2", "SPEAKER_CHILD"
    for part in re.split(r"[\s_\-0-9]+", key):
        if part in LABEL_ROLES:
            return LABEL_ROLES[part]
    return None


def _known_name_role(label: str, known: dict[str, list[str]]) -> str | None:
    mapping = {
        "CHILD_NAME": ROLE_CHILD,
        "PARENT_NAME": ROLE_PARENT,
        "SIBLING_NAME": ROLE_SIBLING,
        "TEACHER_NAME": ROLE_TEACHER,
        "DOCTOR_NAME": ROLE_CLINICIAN,
        "THERAPIST_NAME": ROLE_CLINICIAN,
    }
    for entity, role in mapping.items():
        for value in known.get(entity, []) or []:
            if fuzzy_match(label, str(value), 0.85)[0]:
                return role
    return None


def _cue_scores(text: str) -> dict[str, float]:
    scores = {ROLE_CHILD: 0.0, ROLE_PARENT: 0.0, ROLE_CLINICIAN: 0.0}
    for rx, w in CHILD_CUES:
        scores[ROLE_CHILD] += w * len(rx.findall(text))
    for rx, w in PARENT_CUES:
        scores[ROLE_PARENT] += w * len(rx.findall(text))
    for rx, w in CLINICIAN_CUES:
        scores[ROLE_CLINICIAN] += w * len(rx.findall(text))
    return scores


def assign_roles(
    doc: Document,
    known_values: dict[str, list[str]] | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assign a role to every speaker; returns {speaker_label: role}."""
    known = known_values or {}
    overrides = {normalize(k): v for k, v in (overrides or {}).items()}

    by_speaker: dict[str, list[Turn]] = {}
    for t in doc.turns:
        by_speaker.setdefault(t.speaker, []).append(t)

    roles: dict[str, str] = {}
    cue_totals: dict[str, dict[str, float]] = {}
    for speaker, turns in by_speaker.items():
        if normalize(speaker) in overrides:
            roles[speaker] = overrides[normalize(speaker)]
            continue
        role = _label_role(speaker) or _known_name_role(speaker, known)
        text = " ".join(t.text for t in turns)
        cue_totals[speaker] = _cue_scores(text)
        roles[speaker] = role or ROLE_UNKNOWN

    # resolve the unknowns from cues, strongest signal first
    unknown = [s for s, r in roles.items() if r == ROLE_UNKNOWN]
    for speaker in unknown:
        scores = cue_totals.get(speaker, {})
        if not scores:
            continue
        best_role, best = max(scores.items(), key=lambda kv: kv[1])
        if best >= 2.0:
            roles[speaker] = best_role

    # a two-speaker conversation where exactly one side is known is symmetric:
    # the other side is almost certainly the counterpart
    if len(by_speaker) == 2:
        items = list(roles.items())
        (s1, r1), (s2, r2) = items
        if r1 == ROLE_UNKNOWN and r2 in (ROLE_CHILD, ROLE_PARENT):
            roles[s1] = ROLE_PARENT if r2 == ROLE_CHILD else ROLE_CHILD
        elif r2 == ROLE_UNKNOWN and r1 in (ROLE_CHILD, ROLE_PARENT):
            roles[s2] = ROLE_PARENT if r1 == ROLE_CHILD else ROLE_CHILD

    for t in doc.turns:
        t.role = roles.get(t.speaker, ROLE_UNKNOWN)
    doc.meta["speaker_roles"] = roles
    return roles
