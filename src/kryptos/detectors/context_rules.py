"""Context/role detector - the dependency-free NER layer.

Two jobs no generic PII model does well for this domain:

1. **Role typing.**  A model that finds "Rupa" tells you PERSON.  For child
   data you need to know it is the *teacher* rather than the child, because
   the transformation policy and the re-identification risk differ.  Role is
   recovered from the surrounding trigger words ("my teacher is ...", "Ms
   ...", "my mum ...") and from who is speaking.

2. **Lower-cased names.**  ASR output frequently has no reliable
   capitalisation, which silently destroys the main feature capitalisation-
   dependent NER relies on.  Trigger context plus a first-name gazetteer
   recovers those.

It is deliberately recall-biased; precision is recovered at fusion time by
the agreement bonus and per-entity thresholds.
"""

from __future__ import annotations

import os
import re

from ..normalize import normalize
from ..types import Detection, Document
from .base import Detector, register

HERE = os.path.dirname(os.path.abspath(__file__))
GAZ_DIR = os.path.join(os.path.dirname(HERE), "resources")

# ---------------------------------------------------------------------------
# lexicon
# ---------------------------------------------------------------------------

# words that look like names because they are capitalised but never are
STOP_CAPS: set[str] = set(
    """
i i'm im i've id ill the a an and but or so then now well yes no okay ok oh ah um uh erm
hmm yeah yep nope right sure maybe just really very too also because if when where what
who why how which that this these those there here he she it they we you my your his her
their our its mum mummy mom mommy dad daddy papa mama nan nana nanny granny grandma grandad
grandpa grandad brother sister baby son daughter kid kids child children boy girl man woman
people friend friends teacher teachers doctor nurse coach school class today tomorrow
yesterday morning afternoon evening night week weekend month year monday tuesday wednesday
thursday friday saturday sunday january february march april may june july august september
october november december good bad nice great cool fine sorry thanks thank please hello hi
hey bye goodbye love like want need got get go going went come came said say says tell told
know knew think thought see saw look looked make made take took give gave put let can could
would should will shall must might do does did done have has had be been being am are was
were not don't didn't can't won't isn't it's that's what's there's let's dont didnt cant wont
isnt thats whats theres lets one two three four five six seven eight nine ten first second
third next last other another some any all both each every no none more most much many few
""".split()
)

# Common verbs, expanded morphologically.  A name span must never run into the
# verb that follows it ("Dr Ahmed sees her" -> "Dr Ahmed").
_VERB_STEMS = """
go come see look watch play work live like love want need know think say tell ask talk
speak walk run jump sit stand sleep eat drink read write draw help start finish begin end
take give get put pick drop call text ring visit meet leave arrive stay wait move turn open
close win lose learn teach study practise practice train ride drive bring send show wear
buy pay cost feel seem look sound become stop keep hold carry laugh cry smile shout
""".split()
for _stem in _VERB_STEMS:
    STOP_CAPS.add(_stem)
    STOP_CAPS.add(_stem + "s")
    STOP_CAPS.add(_stem + "es")
    STOP_CAPS.add(_stem + "ed")
    STOP_CAPS.add(_stem + "ing")
    if _stem.endswith("e"):
        STOP_CAPS.add(_stem[:-1] + "ing")
        STOP_CAPS.add(_stem + "d")
    if _stem.endswith("y"):
        STOP_CAPS.add(_stem[:-1] + "ies")
        STOP_CAPS.add(_stem[:-1] + "ied")

for _day in (
    "monday tuesday wednesday thursday friday saturday sunday january february "
    "march april may june july august september october november december "
    "morning afternoon evening night week weekend"
).split():
    STOP_CAPS.add(_day + "s")

# Domain nouns that routinely open a sentence in this kind of transcript.
STOP_CAPS.update(
    """
homework reading writing maths math spelling science history geography art music drama
swimming football soccer rugby cricket netball basketball gymnastics dancing dance karate
swimming lunch lunchtime dinner breakfast tea snack playtime break assembly registration
nursery reception school class classroom playground bus car walk bedtime bath story
holiday holidays weekend birthday party christmas easter ramadan eid diwali hanukkah
therapy session appointment medicine tablet inhaler asthma allergy speech language
everyone everybody someone somebody anyone anybody nothing something anything nobody
""".split()
)

# Closed-class words that routinely open a sentence and are never names.
# Sentence-initial capitalisation is the weakest evidence the detector has, so
# anything on this list is excluded before it can reach the fusion stage.
STOP_CAPS.update(
    """
sometimes usually often always never rarely seldom occasionally normally generally maybe perhaps possibly probably certainly definitely apparently obviously clearly actually basically honestly frankly personally recently lately currently previously initially finally eventually suddenly immediately meanwhile afterwards otherwise anyway however although though because since unless until whenever wherever whereas therefore besides moreover furthermore instead nevertheless nonetheless yesterday today tonight tomorrow now then there here everywhere anywhere somewhere everything nothing anything something everyone anyone someone nobody this that these those they them their theirs there's here's what when where which while who whom whose why how whether after before during between around through above below under over please thanks thank sorry yes yeah yep no nope okay ok right well oh ah um erm mine yours ours hers his its
""".split()
)

ROLE_WORDS: dict[str, str] = {
    "mum": "PARENT_NAME",
    "mummy": "PARENT_NAME",
    "mom": "PARENT_NAME",
    "mommy": "PARENT_NAME",
    "mama": "PARENT_NAME",
    "mother": "PARENT_NAME",
    "dad": "PARENT_NAME",
    "daddy": "PARENT_NAME",
    "papa": "PARENT_NAME",
    "father": "PARENT_NAME",
    "stepmum": "PARENT_NAME",
    "stepdad": "PARENT_NAME",
    "brother": "SIBLING_NAME",
    "sister": "SIBLING_NAME",
    "twin": "SIBLING_NAME",
    "stepbrother": "SIBLING_NAME",
    "stepsister": "SIBLING_NAME",
    "grandma": "RELATIVE_NAME",
    "grandad": "RELATIVE_NAME",
    "grandpa": "RELATIVE_NAME",
    "granny": "RELATIVE_NAME",
    "nan": "RELATIVE_NAME",
    "nana": "RELATIVE_NAME",
    "auntie": "RELATIVE_NAME",
    "aunty": "RELATIVE_NAME",
    "aunt": "RELATIVE_NAME",
    "uncle": "RELATIVE_NAME",
    "cousin": "RELATIVE_NAME",
    "godmother": "RELATIVE_NAME",
    "teacher": "TEACHER_NAME",
    "headteacher": "TEACHER_NAME",
    "principal": "TEACHER_NAME",
    "ta": "TEACHER_NAME",
    "tutor": "TEACHER_NAME",
    "keyworker": "TEACHER_NAME",
    "therapist": "THERAPIST_NAME",
    "counsellor": "THERAPIST_NAME",
    "counselor": "THERAPIST_NAME",
    "psychologist": "THERAPIST_NAME",
    "doctor": "DOCTOR_NAME",
    "dr": "DOCTOR_NAME",
    "gp": "DOCTOR_NAME",
    "paediatrician": "DOCTOR_NAME",
    "pediatrician": "DOCTOR_NAME",
    "nurse": "DOCTOR_NAME",
    "dentist": "DOCTOR_NAME",
    "coach": "COACH_NAME",
    "instructor": "COACH_NAME",
    "friend": "FRIEND_NAME",
    "bestfriend": "FRIEND_NAME",
    "classmate": "FRIEND_NAME",
    "babysitter": "RELATIVE_NAME",
    "childminder": "RELATIVE_NAME",
    "nanny": "RELATIVE_NAME",
    "sitter": "RELATIVE_NAME",
    "neighbour": "PERSON_NAME",
    "neighbor": "PERSON_NAME",
}

TITLES: dict[str, str] = {
    "mr": "PERSON_NAME",
    "mister": "PERSON_NAME",
    "mrs": "PERSON_NAME",
    "ms": "PERSON_NAME",
    "miss": "TEACHER_NAME",
    "sir": "TEACHER_NAME",
    "dr": "DOCTOR_NAME",
    "doctor": "DOCTOR_NAME",
    "nurse": "DOCTOR_NAME",
    "professor": "PERSON_NAME",
    "prof": "PERSON_NAME",
    "coach": "COACH_NAME",
    "auntie": "RELATIVE_NAME",
    "aunty": "RELATIVE_NAME",
    "uncle": "RELATIVE_NAME",
}

WORKPLACE_TRIGGER = re.compile(
    r"\b(?:works?|working|worked|job|shift|office)\s+(?:at|for|in)\s+"
    r"(?:the\s+)?((?:[\w'&-]+\s?){1,4})",
    re.I,
)
OWNS_TRIGGER = re.compile(
    r"\b(?:owns?|owned|runs?|manages?)\s+(?:the\s+|a\s+|his\s+|her\s+|their\s+)?"
    r"((?:[\w'&-]+\s?){1,4})",
    re.I,
)
# "works at", "runs the" and friends match a lot of ordinary speech ("runs out
# of battery", "worked for a fortnight").  A workplace object has to look like
# an organisation: a proper noun, or a venue noun.
VENUE_NOUNS = frozenset(
    """
bakery shop store supermarket market pharmacy chemist garage factory warehouse
office bank hospital clinic surgery practice school college university nursery
restaurant cafe café takeaway pub bar hotel salon barbers barber gym studio
depot depot's centre center library museum theatre cinema station post office
airport farm workshop laboratory lab dealership motors haulage joinery plumbing
council nhs hmrc firm agency company ltd limited plc llp inc corporation charity
""".split()
)
_TIME_NOUNS = frozenset(
    """
second seconds minute minutes hour hours day days week weeks fortnight month
months year years term terms while bit ages age time times moment moments night
nights morning afternoon evening weekend weekends holiday holidays
""".split()
)
_PARTICLES = frozenset(
    """
out into up down off away back over under through around about along across
""".split()
)

NEIGHBOURHOOD_TRIGGER = re.compile(
    r"\b(?:in|from|near|around|opposite|next\s+to|behind|by)\s+"
    r"(?-i:([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2}))",
    re.I,
)
UNIQUE_EVENT_TRIGGER = re.compile(
    r"\b(?:the\s+only|one\s+of\s+the\s+only|the\s+first|the\s+youngest|the\s+oldest"
    r"|the\s+best)\s+([^.,;!?\n]{4,60})",
    re.I,
)

# Unicode-aware: "Aïcha" and "Étienne" are names too, and an ASCII-only
# character class silently drops every accented name in the corpus.
_WORD = r"[^\W\d_][\w'’\u00ad\u200b\u200c\u200d\u2060\ufeff-]*"

# A name span must not swallow the words that follow it.  Without this guard a
# greedy sequence eats "Farhana and my" and hides the next match entirely.
_NAME_BOUNDARY = (
    "and|or|but|so|then|the|a|an|my|our|your|his|her|their|its|is|are|was|were|am|be|been|"
    "to|at|in|on|of|for|from|with|by|about|into|over|under|after|before|because|if|when|"
    "who|that|this|these|those|there|here|goes|go|going|went|comes|come|came|said|says|say|"
    "told|tells|tell|does|do|did|has|have|had|will|would|can|could|should|just|really|very|"
    "not|no|yes|okay|ok|well|like|got|get|gets|takes|take|took|picks|pick|picked|finishes|"
    "finish|finished|starts|start|started|works|work|worked|plays|play|played|lives|live|"
    "lived|likes|likes|loves|love|wants|want|needs|need|thinks|think|knows|know|too|also|"
    "always|never|sometimes|usually|today|tomorrow|yesterday|now|still|again|back|up|down|"
    "out|off|on|over|all|both|each|every|some|any|one|two|three|four|five"
)
_NOT_BOUNDARY = rf"(?!(?:{_NAME_BOUNDARY})\b)"
_WORD_NS = _NOT_BOUNDARY + _WORD
_NAME_SEQ = rf"(?:{_WORD_NS})(?:\s+(?:{_WORD_NS})){{0,2}}"

# "my teacher is X" / "our doctor, X" / "his coach X"
ROLE_IS = re.compile(
    r"\b(?:my|our|his|her|their|your|the)\s+(?P<role>{})"
    r"(?:'s\s+name\s+is|\s*,\s*|\s+is\s+called\s+|\s+is\s+|\s+named\s+|\s+)"
    r"(?P<name>{})".format("|".join(sorted(ROLE_WORDS, key=len, reverse=True)), _NAME_SEQ),
    re.I,
)

# "Ms Rupa", "Dr Ahmed", "Coach Sam"
TITLE_NAME = re.compile(
    r"\b(?P<title>{})\.?\s+(?P<name>{})".format(
        "|".join(sorted(TITLES, key=len, reverse=True)), _NAME_SEQ
    ),
    re.I,
)

# self / other introduction
INTRO = re.compile(
    r"\b(?:my\s+name\s+is|i'?m|i\s+am|this\s+is|that'?s|call\s+me|it'?s|"
    rf"here\s+with|joined\s+by|speaking\s+to)\s+(?P<name>{_NAME_SEQ})",
    re.I,
)

# "Aisha, come here" / "come on, Aisha" - direct address
VOCATIVE = re.compile(rf"(?:^|[.!?]\s+)(?P<name>{_WORD})\s*[,!](?=\s)")
VOCATIVE_TAIL = re.compile(rf",\s*(?P<name>{_WORD})\s*[.!?]")

_WORD_RE = re.compile(_WORD, re.UNICODE)


def _load_gazetteer(name: str) -> set[str]:
    path = os.path.join(GAZ_DIR, name)
    out: set[str] = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for tok in line.split():
                    n = normalize(tok)
                    if n:
                        out.add(n)
    return out


@register("context")
class ContextDetector(Detector):
    def load(self) -> None:
        self.first_names = _load_gazetteer("first_names.txt")
        self.extra_stop = {normalize(w) for w in self.spec.get("stopwords", [])}
        self.min_cap_score = float(self.spec.get("cap_score", 0.3))
        self.sentence_initial_score = float(self.spec.get("sentence_initial_score", 0.24))
        self.enable_cap_sweep = bool(self.spec.get("cap_sweep", True))
        self.enable_gazetteer = bool(self.spec.get("gazetteer", True))

    # ------------------------------------------------------------------
    def detect(self, doc: Document) -> list[Detection]:
        text = doc.text
        # A token that also appears lower-cased somewhere in the conversation
        # is a common word, not a name - the cheapest available signal for
        # deciding whether a sentence-initial capital means anything.
        lower_elsewhere = {
            m.group(0).lower() for m in _WORD_RE.finditer(text) if m.group(0).islower()
        }
        out: list[Detection] = []
        out.extend(self._role_names(doc, text))
        out.extend(self._title_names(doc, text))
        out.extend(self._introductions(doc, text))
        out.extend(self._vocatives(doc, text))
        out.extend(self._workplaces(doc, text))
        out.extend(self._places(doc, text))
        out.extend(self._unique_events(doc, text))
        if self.enable_gazetteer and self.first_names:
            out.extend(self._gazetteer_names(doc, text))
        if self.enable_cap_sweep:
            out.extend(self._capitalised_sweep(doc, text, lower_elsewhere))
        return out

    # ------------------------------------------------------------------
    def _clean_name(self, raw: str, start: int) -> tuple[int, int, str] | None:
        """Trim leading/trailing function words off a candidate name span."""
        toks = list(re.finditer(_WORD, raw))
        if not toks:
            return None
        keep: list[re.Match] = []
        for t in toks:
            n = normalize(t.group(0))
            if n in STOP_CAPS or n in self.extra_stop:
                if keep:
                    break
                continue
            keep.append(t)
        if not keep:
            return None
        s = start + keep[0].start()
        e = start + keep[-1].end()
        return s, e, raw[keep[0].start() : keep[-1].end()]

    def _emit_name(
        self, doc: Document, raw: str, offset: int, entity: str, score: float, rule: str
    ) -> Detection | None:
        cleaned = self._clean_name(raw, offset)
        if not cleaned:
            return None
        s, e, surface = cleaned
        if len(surface) < 2:
            return None
        # a fully lower-case single token is only a name if context is strong
        return self.make(doc, s, e, entity, score, raw_label=rule, rule=rule)

    # ------------------------------------------------------------------
    def _role_names(self, doc: Document, text: str) -> list[Detection]:
        out = []
        for m in ROLE_IS.finditer(text):
            role = normalize(m.group("role"))
            entity = ROLE_WORDS.get(role, "PERSON_NAME")
            name = m.group("name")
            if normalize(name) in STOP_CAPS:
                continue
            det = self._emit_name(doc, name, m.start("name"), entity, 0.75, "role_is")
            if det:
                det.meta["role"] = role
                out.append(det)
        return out

    def _title_names(self, doc: Document, text: str) -> list[Detection]:
        out = []
        for m in TITLE_NAME.finditer(text):
            title = normalize(m.group("title"))
            entity = TITLES.get(title, "PERSON_NAME")
            name = m.group("name")
            if normalize(name.split()[0]) in STOP_CAPS:
                continue
            # keep the title inside the span: "Ms Rupa" reads better redacted whole
            det = self._emit_name(doc, m.group(0), m.start(), entity, 0.8, "title_name")
            if det:
                det.meta["title"] = title
                out.append(det)
        return out

    def _introductions(self, doc: Document, text: str) -> list[Detection]:
        out = []
        for m in INTRO.finditer(text):
            name = m.group("name")
            first = normalize(name.split()[0])
            if first in STOP_CAPS and first not in self.first_names:
                continue
            turn = doc.turn_at(m.start("name"))
            entity = "PERSON_NAME"
            trigger = normalize(m.group(0).split()[0])
            if turn is not None and trigger in {"my", "i'm", "im", "i"}:
                entity = {"child": "CHILD_NAME", "parent": "PARENT_NAME"}.get(
                    turn.role, "PERSON_NAME"
                )
            score = 0.8 if name[:1].isupper() or first in self.first_names else 0.55
            det = self._emit_name(doc, name, m.start("name"), entity, score, "introduction")
            if det:
                out.append(det)
        return out

    def _vocatives(self, doc: Document, text: str) -> list[Detection]:
        out = []
        for rx, rule in ((VOCATIVE, "vocative"), (VOCATIVE_TAIL, "vocative_tail")):
            for m in rx.finditer(text):
                name = m.group("name")
                n = normalize(name)
                if n in STOP_CAPS:
                    continue
                if not (name[:1].isupper() or n in self.first_names):
                    continue
                det = self._emit_name(doc, name, m.start("name"), "PERSON_NAME", 0.5, rule)
                if det:
                    out.append(det)
        return out

    def _workplaces(self, doc: Document, text: str) -> list[Detection]:
        out = []
        for rx, entity, score in (
            (WORKPLACE_TRIGGER, "PARENT_WORKPLACE", 0.65),
            (OWNS_TRIGGER, "PARENT_WORKPLACE", 0.55),
        ):
            for m in rx.finditer(text):
                raw = m.group(1)
                cleaned = self._clean_name(raw, m.start(1))
                if not cleaned:
                    continue
                s, e, surface = cleaned
                if len(normalize(surface)) < 3:
                    continue
                if not _looks_like_workplace(raw, surface):
                    continue
                out.append(self.make(doc, s, e, entity, score, raw_label="workplace"))
        return out

    def _places(self, doc: Document, text: str) -> list[Detection]:
        out = []
        for m in NEIGHBOURHOOD_TRIGGER.finditer(text):
            surface = m.group(1)
            if normalize(surface.split()[0]) in STOP_CAPS:
                continue
            prep = normalize(m.group(0).split()[0])
            entity = (
                "LOCAL_LANDMARK"
                if prep in {"opposite", "next", "behind", "by", "near", "around"}
                else "NEIGHBORHOOD"
            )
            out.append(
                self.make(
                    doc,
                    m.start(1),
                    m.end(1),
                    entity,
                    0.45,
                    raw_label="place_trigger",
                    preposition=prep,
                )
            )
        return out

    def _unique_events(self, doc: Document, text: str) -> list[Detection]:
        out = []
        for m in UNIQUE_EVENT_TRIGGER.finditer(text):
            out.append(
                self.make(
                    doc, m.start(), m.end(), "UNIQUE_EVENT", 0.5, raw_label="uniqueness_phrase"
                )
            )
        return out

    def _gazetteer_names(self, doc: Document, text: str) -> list[Detection]:
        """Catch names in lower-cased ASR output via a first-name list."""
        out = []
        for m in re.finditer(_WORD, text):
            n = normalize(m.group(0))
            if len(n) < 3 or n not in self.first_names:
                continue
            if n in STOP_CAPS:
                continue
            out.append(
                self.make(
                    doc, m.start(), m.end(), "PERSON_NAME", 0.45, raw_label="gazetteer_first_name"
                )
            )
        return out

    def _capitalised_sweep(
        self, doc: Document, text: str, lower_elsewhere: set[str]
    ) -> list[Detection]:
        """Low-confidence net for capitalised tokens that survive the stoplist."""
        out: list[Detection] = []
        matches = list(_WORD_RE.finditer(text))
        i = 0
        while i < len(matches):
            if not _is_capitalised(matches[i].group(0)):
                i += 1
                continue
            j = i
            while (
                j + 1 < len(matches)
                and j - i < 2
                and _is_capitalised(matches[j + 1].group(0))
                and text[matches[j].end() : matches[j + 1].start()].strip() in ("", "-")
            ):
                j += 1
            start, end = matches[i].start(), matches[j].end()
            surface = text[start:end]
            toks = [m.group(0) for m in matches[i : j + 1]]
            if all(normalize(t) in STOP_CAPS for t in toks):
                i = j + 1
                continue
            prev = text[:start].rstrip()
            sentence_initial = (not prev) or prev[-1] in ".!?\n"
            cleaned = self._clean_name(surface, start)
            if cleaned:
                s_, e_, kept = cleaned
                score = self.min_cap_score
                if normalize(kept.split()[0]) in self.first_names:
                    score = 0.5
                elif len(kept.split()) > 1:
                    score = 0.38
                if sentence_initial and len(toks) == 1:
                    # Sentence-initial capitalisation is not evidence of a
                    # name.  It is still emitted, at a score low enough that
                    # it only survives fusion unopposed, because the
                    # alternative is missing every name that opens a turn.
                    if normalize(toks[0]) in self.first_names:
                        score = 0.45
                    elif toks[0].lower() in lower_elsewhere:
                        i = j + 1
                        continue
                    else:
                        score = self.sentence_initial_score
                out.append(
                    self.make(doc, s_, e_, "PERSON_NAME", score, raw_label="capitalised_span")
                )
            i = j + 1
        return out


def _looks_like_workplace(raw: str, surface: str) -> bool:
    """Filter the objects of "works at" / "runs the" that are not workplaces.

    Ordinary speech trips these triggers constantly - "the tablet runs out of
    battery", "it worked for a fortnight".  A real workplace is a proper noun
    or a venue noun; a duration or a particle never is.
    """
    words = [normalize(w) for w in re.findall(_WORD, raw) if w.strip()]
    if not words:
        return False
    if words[0] in _PARTICLES:
        return False
    # positive evidence first: "Day Nursery" and "Sunday School" are workplaces
    # even though they contain a time word
    if any(w in VENUE_NOUNS for w in words):
        return True
    if any(_is_capitalised(tok) for tok in re.findall(_WORD, surface)):
        return True
    return False


def _is_capitalised(token: str) -> bool:
    return len(token) > 1 and token[0].isupper() and not token.isupper()
