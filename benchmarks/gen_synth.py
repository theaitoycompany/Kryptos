"""Generate the in-domain child/parent benchmark.

Three corpora, written in the repository's own gold JSONL format so that both
`python -m kryptos eval` and this harness can score them:

  synth_clean.jsonl  written-form transcripts, correct spellings
  synth_asr.jsonl    the same conversations as ASR output: lower-cased, no
                     punctuation, phonetically corrupted names, spoken-form
                     phone numbers / emails / dates
  synth_neg.jsonl    conversations with no identifiers at all - the
                     over-masking control

Nothing here is real. Names, schools, numbers and addresses are invented; the
ASR variants are hand-authored corruptions of the kind Whisper produces on
child speech (name substitution, compound splitting, digit-to-word).
"""

from __future__ import annotations

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", ".build", "gold")

# --------------------------------------------------------------------------
# pools: (written form, ASR form)
# --------------------------------------------------------------------------

CHILDREN = [
    ("Aisha Hasan", "aisha hassan"),
    ("Nafisa Rahman", "nafeesa ramen"),
    ("Tomasz Nowak", "tomash novak"),
    ("Mei-Ling Chen", "may ling chen"),
    ("Olusegun Adeyemi", "olu segun adeyemi"),
    ("Sofia Marquez", "sofia markes"),
    ("Yusuf Iqbal", "yousef ikbal"),
    ("Zainab Ali", "zaynub ali"),
    ("Ffion Llewellyn", "fion lewellyn"),
    ("Darragh O'Neill", "dara oneill"),
]
PARENTS = [
    ("Farhana Hasan", "farhana hassan"),
    ("Samira Begum", "sameera begum"),
    ("Katarzyna Nowak", "katarina novak"),
    ("Adaeze Adeyemi", "ada easy adeyemi"),
    ("Priya Chandran", "prea chandran"),
    ("Gulnara Yusupova", "gulnara you supova"),
]
SIBLINGS = [
    ("Zaid", "zade"),
    ("Kwame", "kwarmay"),
    ("Lucja", "loocha"),
    ("Hamza", "hamzer"),
    ("Niamh", "neev"),
    ("Bilal", "belal"),
]
TEACHERS = [
    ("Ms Rupa", "miss roopa"),
    ("Mrs Okonkwo", "missus okonkwo"),
    ("Mr Whitfield", "mister whitfield"),
    ("Miss Dhaliwal", "miss dally wall"),
]
DOCTORS = [
    ("Dr Ahmed", "doctor ahmad"),
    ("Dr Fitzgerald", "doctor fitzgerald"),
    ("Dr Osei", "doctor o say"),
]
COACHES = [("Coach Bennett", "coach bennet"), ("Coach Idris", "coach idriss")]
FRIENDS = [
    ("Maryam", "maryum"),
    ("Jacob", "jaycob"),
    ("Ayesha", "eye eesha"),
    ("Oliwia", "olivia"),
    ("Tomasz", "tomash"),
]
SCHOOLS = [
    ("Greenfield Primary School", "green field primary school"),
    ("St Aidan's Academy", "saint aidans academy"),
    ("Westbrook Primary", "west brook primary"),
    ("Holy Trinity Junior School", "holy trinity junior school"),
    ("Marsh Lane Primary", "marsh lane primary"),
]
NURSERIES = [("Sunbeams Nursery", "sun beams nursery"), ("Little Acorns", "little acorns")]
TEAMS = [
    ("under-10 Tigers", "under ten tigers"),
    ("U11 Rockets", "you eleven rockets"),
    ("under-9 Falcons", "under nine falcons"),
]
CLUBS = [
    ("Riverside Judo Club", "riverside judo club"),
    ("the Thursday chess club", "the thursday chess club"),
]
WORKPLACES = [
    ("the bakery on Mill Lane", "the bakery on mill lane"),
    ("Patel Motors", "pat tell motors"),
    ("the pharmacy inside Riverside Retail Park", "the pharmacy inside riverside retail park"),
]
STREETS = [
    ("21 Oak Street", "twenty one oak street"),
    ("14 Beechwood Avenue", "fourteen beechwood avenue"),
    ("3 Sandringham Close", "three sandringham close"),
]
POSTCODES = [
    ("LS6 2QT", "l s six two q t"),
    ("M14 5RB", "m fourteen five r b"),
    ("BS7 9DA", "b s seven nine d a"),
]
LANDMARKS = [
    ("Oakwood Park", "oakwood park"),
    ("the blue mosque on Carlton Road", "the blue mosque on carlton road"),
    ("Beckett Library", "beckett library"),
]
CITIES = [("Leeds", "leeds"), ("Bradford", "bradford"), ("Coventry", "coventry")]

PHONES = [
    ("07911 123456", "oh seven nine one one one two three four five six"),
    ("07700 900461", "oh seven seven double oh nine double oh four six one"),
    ("0113 496 0812", "oh one one three four nine six oh eight one two"),
]
EMAILS = [
    ("farhana.hasan@gmail.com", "farhana dot hasan at gmail dot com"),
    ("k.nowak88@outlook.com", "k dot nowak eighty eight at outlook dot com"),
    ("priya.c@yahoo.co.uk", "priya dot c at yahoo dot co dot uk"),
]
NHS = [
    ("943 476 5919", "nine four three four seven six five nine one nine"),
    ("485 777 3456", "four eight five triple seven three four five six"),
]
DOBS = [
    ("12/03/2016", "the twelfth of march twenty sixteen"),
    ("04/09/2015", "the fourth of september twenty fifteen"),
    ("28/11/2017", "the twenty eighth of november twenty seventeen"),
]
AGES = [("nine", "nine"), ("seven", "seven"), ("eleven", "eleven"), ("8", "eight")]
USERNAMES = [
    ("farhana.hasan", "farhana dot hasan"),
    ("nowak_k88", "nowak underscore k eighty eight"),
]
HANDLES = [("@aisha_draws", "at aisha underscore draws"), ("@teamnowak", "at team nowak")]
STUDENT_IDS = [
    ("GP-2019-4471", "g p twenty nineteen forty four seventy one"),
    ("SCH0092381", "s c h double oh nine two three eight one"),
]


class Turn:
    def __init__(self, speaker):
        self.speaker = speaker
        self.parts = []  # (clean, asr, entity|None)

    def t(self, clean, asr=None):
        self.parts.append((clean, asr if asr is not None else asr_plain(clean), None))
        return self

    def e(self, pair, entity):
        self.parts.append((pair[0], pair[1], entity))
        return self


def render(turns, asr):
    """Return (turn dicts, spans) with exact character offsets into the flat text."""
    out_turns, spans = [], []
    cursor = 0
    for turn in turns:
        pieces = []
        local = 0
        for clean, asr_form, entity in turn.parts:
            surface = asr_form if asr else clean
            if entity:
                spans.append(
                    {
                        "start": cursor + local,
                        "end": cursor + local + len(surface),
                        "entity": entity,
                    }
                )
            pieces.append(surface)
            local += len(surface)
        text = "".join(pieces)
        out_turns.append({"speaker": turn.speaker, "text": text})
        cursor += len(text) + 1
    return out_turns, spans


def _p(rnd, pool):
    return pool[rnd.randrange(len(pool))]


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------


def scenario_clinic(rnd, cast):
    child, _parent, doc, school = cast["child"], cast["parent"], cast["doctor"], cast["school"]
    t = []
    a = Turn("PARENT")
    a.t("Right ")
    a.e(child, "CHILD_NAME")
    a.t(", tell ")
    a.e(doc, "DOCTOR_NAME")
    a.t(" how school has been this term.")
    t.append(a)
    b = Turn("CHILD")
    b.t("It's okay. My teacher is ")
    b.e(cast["teacher"], "TEACHER_NAME")
    b.t(" and she says my reading is better.")
    t.append(b)
    c = Turn("CHILD")
    c.t("I'm ")
    c.e(cast["age"], "EXACT_AGE")
    c.t(" now. I go to ")
    c.e(school, "SCHOOL_NAME")
    c.t(" with my brother ")
    c.e(cast["sibling"], "SIBLING_NAME")
    c.t(".")
    t.append(c)
    d = Turn("PARENT")
    d.t("Her date of birth is ")
    d.e(cast["dob"], "DATE_OF_BIRTH")
    d.t(" and the NHS number is ")
    d.e(cast["nhs"], "MEDICAL_ID")
    d.t(".")
    t.append(d)
    e = Turn("PARENT")
    e.t("If you need us, my mobile is ")
    e.e(cast["phone"], "PHONE")
    e.t(", or email ")
    e.e(cast["email"], "EMAIL")
    e.t(".")
    t.append(e)
    return t


def scenario_quasi(rnd, cast):
    t = []
    a = Turn("CHILD")
    a.t("I'm the only goalkeeper in the ")
    a.e(cast["team"], "SPORTS_TEAM")
    a.t(" so I always play.")
    t.append(a)
    b = Turn("PARENT")
    b.t("My husband owns ")
    b.e(cast["workplace"], "PARENT_WORKPLACE")
    b.t(", so he does the pick ups.")
    t.append(b)
    c = Turn("CHILD")
    c.t("I'm ")
    c.e(cast["age"], "EXACT_AGE")
    c.t(" and we live near ")
    c.e(cast["landmark"], "LOCAL_LANDMARK")
    c.t(".")
    t.append(c)
    d = Turn("PARENT")
    d.t("She is the only child in ")
    d.e(cast["school"], "SCHOOL_NAME")
    d.t(" who uses a wheelchair, so everyone knows her.")
    t.append(d)
    return t


def scenario_admin(rnd, cast):
    t = []
    a = Turn("PARENT")
    a.t("We live at ")
    a.e(cast["street"], "STREET_ADDRESS")
    a.t(", ")
    a.e(cast["city"], "CITY")
    a.t(" ")
    a.e(cast["postcode"], "POSTCODE")
    a.t(".")
    t.append(a)
    b = Turn("PARENT")
    b.t("The school portal login is ")
    b.e(cast["username"], "ONLINE_USERNAME")
    b.t(" and her pupil number is ")
    b.e(cast["student_id"], "STUDENT_ID")
    b.t(".")
    t.append(b)
    c = Turn("CHILD")
    c.t("My drawing account is ")
    c.e(cast["handle"], "SOCIAL_HANDLE")
    c.t(" but mum checks it.")
    t.append(c)
    d = Turn("PARENT")
    d.t("Call me on ")
    d.e(cast["phone"], "PHONE")
    d.t(" if ")
    d.e(cast["child"], "CHILD_NAME")
    d.t(" is unwell.")
    t.append(d)
    return t


def scenario_playground(rnd, cast):
    t = []
    a = Turn("CHILD")
    a.t("Me and ")
    a.e(cast["friend"], "FRIEND_NAME")
    a.t(" play at breaktime and ")
    a.e(cast["teacher"], "TEACHER_NAME")
    a.t(" tells us off.")
    t.append(a)
    b = Turn("CHILD")
    b.t("My brother ")
    b.e(cast["sibling"], "SIBLING_NAME")
    b.t(" is in year six at ")
    b.e(cast["school"], "SCHOOL_NAME")
    b.t(".")
    t.append(b)
    c = Turn("PARENT")
    c.t("Say your name properly for the doctor.")
    t.append(c)
    d = Turn("CHILD")
    d.t("I'm ")
    d.e(cast["child"], "CHILD_NAME")
    d.t(" and I'm ")
    d.e(cast["age"], "EXACT_AGE")
    d.t(".")
    t.append(d)
    e = Turn("PARENT")
    e.t("Her coach is ")
    e.e(cast["coach"], "COACH_NAME")
    e.t(" at ")
    e.e(cast["club"], "CLUB_NAME")
    e.t(" on Saturdays.")
    t.append(e)
    return t


def scenario_nursery(rnd, cast):
    t = []
    a = Turn("PARENT")
    a.t("She moved up from ")
    a.e(cast["nursery"], "DAYCARE_NAME")
    a.t(" in September.")
    t.append(a)
    b = Turn("PARENT")
    b.t("I'm ")
    b.e(cast["parent"], "PARENT_NAME")
    b.t(", her mum, and her dad works at ")
    b.e(cast["workplace"], "PARENT_WORKPLACE")
    b.t(".")
    t.append(b)
    c = Turn("CHILD")
    c.t("Nana lives on ")
    c.e(cast["street"], "STREET_ADDRESS")
    c.t(" near ")
    c.e(cast["landmark"], "LOCAL_LANDMARK")
    c.t(".")
    t.append(c)
    d = Turn("PARENT")
    d.t("You can reach me at ")
    d.e(cast["email"], "EMAIL")
    d.t(".")
    t.append(d)
    return t


def scenario_repeat(rnd, cast):
    """The same child named four times, twice corrupted - tests linking."""
    child = cast["child"]
    first = (child[0].split()[0], child[1].split()[0])
    t = []
    a = Turn("PARENT")
    a.t("This is ")
    a.e(child, "CHILD_NAME")
    a.t(".")
    t.append(a)
    b = Turn("DOCTOR")
    b.t("Hello ")
    b.e(first, "CHILD_NAME")
    b.t(", how old are you?")
    t.append(b)
    c = Turn("CHILD")
    c.t("I'm ")
    c.e(cast["age"], "EXACT_AGE")
    c.t(" and I go to ")
    c.e(cast["school"], "SCHOOL_NAME")
    c.t(".")
    t.append(c)
    d = Turn("PARENT")
    d.e(first, "CHILD_NAME")
    d.t(" has swimming after school with ")
    d.e(cast["friend"], "FRIEND_NAME")
    d.t(".")
    t.append(d)
    return t


SCENARIOS = [
    scenario_clinic,
    scenario_quasi,
    scenario_admin,
    scenario_playground,
    scenario_nursery,
    scenario_repeat,
]


def make_cast(rnd):
    return {
        "child": _p(rnd, CHILDREN),
        "parent": _p(rnd, PARENTS),
        "sibling": _p(rnd, SIBLINGS),
        "teacher": _p(rnd, TEACHERS),
        "doctor": _p(rnd, DOCTORS),
        "coach": _p(rnd, COACHES),
        "friend": _p(rnd, FRIENDS),
        "school": _p(rnd, SCHOOLS),
        "nursery": _p(rnd, NURSERIES),
        "team": _p(rnd, TEAMS),
        "club": _p(rnd, CLUBS),
        "workplace": _p(rnd, WORKPLACES),
        "street": _p(rnd, STREETS),
        "postcode": _p(rnd, POSTCODES),
        "landmark": _p(rnd, LANDMARKS),
        "city": _p(rnd, CITIES),
        "phone": _p(rnd, PHONES),
        "email": _p(rnd, EMAILS),
        "nhs": _p(rnd, NHS),
        "dob": _p(rnd, DOBS),
        "age": _p(rnd, AGES),
        "username": _p(rnd, USERNAMES),
        "handle": _p(rnd, HANDLES),
        "student_id": _p(rnd, STUDENT_IDS),
    }


NEG_LINES = [
    ("PARENT", "She has been eating better this week, mostly pasta and apples."),
    ("CHILD", "Sometimes my tummy hurts after running but then it stops."),
    ("PARENT", "Bedtime is easier now that the evenings are lighter."),
    ("CHILD", "We did painting and my picture had a big yellow sun on it."),
    ("PARENT", "He gets frustrated when the tablet runs out of battery."),
    ("CHILD", "My favourite dinner is chicken and rice with peas."),
    ("PARENT", "Reading together helps, although she rushes the long words."),
    ("CHILD", "I like playing football at breaktime with everyone."),
    ("PARENT", "Mornings are hard because getting dressed takes ages."),
    ("CHILD", "The swings are my favourite because you go really high."),
    ("PARENT", "We tried the sticker chart and it worked for a fortnight."),
    ("CHILD", "Sometimes I feel shy and then I don't want to talk."),
    ("PARENT", "Her appetite dips when she is tired, otherwise she is fine."),
    ("CHILD", "I can count backwards from twenty without stopping now."),
    ("PARENT", "Nothing has changed at home, no new stress that we know of."),
]


def asr_plain(text):
    out = text.lower()
    for ch in ".,?!":
        out = out.replace(ch, "")
    return out


def build(n_docs=120, seed=7):
    rnd = random.Random(seed)
    clean, asr = [], []
    for i in range(n_docs):
        cast = make_cast(rnd)
        scenario = SCENARIOS[i % len(SCENARIOS)]
        turns = scenario(rnd, cast)
        kv = {
            "CHILD_NAME": [cast["child"][0]],
            "SCHOOL_NAME": [cast["school"][0]],
            "PARENT_NAME": [cast["parent"][0]],
        }
        for corpus, is_asr in ((clean, False), (asr, True)):
            t_dicts, spans = render(turns, asr=is_asr)
            corpus.append(
                {"doc_id": f"synth-{i:03d}", "turns": t_dicts, "spans": spans, "known_values": kv}
            )
    neg = []
    for i in range(40):
        lines = [NEG_LINES[(i + j) % len(NEG_LINES)] for j in range(rnd.randint(4, 7))]
        neg.append(
            {
                "doc_id": f"neg-{i:03d}",
                "turns": [{"speaker": s, "text": t} for s, t in lines],
                "spans": [],
                "known_values": {},
            }
        )
    return clean, asr, neg


def main():
    os.makedirs(OUT, exist_ok=True)
    clean, asr, neg = build()
    for name, corpus in (("synth_clean", clean), ("synth_asr", asr), ("synth_neg", neg)):
        path = os.path.join(OUT, name + ".jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for rec in corpus:
                fh.write(json.dumps(rec) + "\n")
        print(name, len(corpus), path)


if __name__ == "__main__":
    main()
