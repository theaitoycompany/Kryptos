"""Deterministic rule detectors.

These carry the highest confidence in the ensemble because they cannot
hallucinate, and they cover the two things ML PII models are weakest at in
speech transcripts: structured identifiers, and *spoken* renderings of them
("oh seven eight one ...", "samira dot hasan at gmail dot com").
"""

from __future__ import annotations

import re
from typing import Any

from ..normalize import MULTIPLIER_WORDS, NUMBER_WORDS, spoken_cardinal, spoken_digits
from ..normalize import normalize as normalize_first
from ..types import Detection, Document
from .base import Detector, register

# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

_DIGIT_WORD = r"(?:zero|oh|nought|naught|one|two|three|four|five|six|seven|eight|nine|double|triple|treble|\d)"
_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_STREET_TYPES = (
    r"street|st|road|rd|avenue|ave|lane|ln|drive|dr|close|court|ct|crescent|"
    r"way|place|pl|terrace|grove|gardens?|park|hill|row|square|sq|boulevard|blvd|"
    r"parade|walk|mews|rise|view|circle|cir|highway|hwy|route|apartment|apt|flat|unit"
)
_UNIT_WORDS = r"(?:flat|apt|apartment|unit|suite|no\.?|number|house)"

# ---- spoken numerals -------------------------------------------------------
# Transcripts render identifiers as words: "twenty one oak street", "the twelfth
# of march twenty sixteen", "l s six two q t".  These components let the written
# rules above have spoken twins.
_ONES = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
)
_TENS = r"twenty|thirty|forty|fourty|fifty|sixty|seventy|eighty|ninety"
_SPOKEN_INT = rf"(?:(?:{_TENS})(?:[\s-](?:{_ONES}))?|{_ONES})"
_ORDINAL = (
    r"(?:(?:twenty|thirty)[\s-])?(?:first|second|third|fourth|fifth|sixth|"
    r"seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|"
    r"fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|thirtieth)"
)
_YEAR_SPOKEN = (
    rf"(?:nineteen|twenty)(?:[\s-](?:hundred|and|oh|zero|{_TENS}|{_ONES}))"
    r"{1,3}"
)
_SPOKEN_DATE = rf"(?:the\s+)?(?:{_ORDINAL})\s+of\s+(?:{_MONTHS})(?:\s+(?:{_YEAR_SPOKEN}|\d{{4}}))?"

# Every rule is compiled case-INSENSITIVELY; where capitalisation is real
# evidence (proper nouns) the pattern says so explicitly with (?-i:...).
_CAP = r"(?-i:[A-Z][\w'&-]*)"  # a capitalised token
_UPPER_ID = r"(?-i:[A-Z0-9][A-Z0-9-]{2,19})"  # an identifier-looking token

# (name, pattern, entity, score)
RULES: list[tuple[str, str, str, float]] = [
    # ---------------- structured identifiers ----------------
    ("email", r"\b[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+\b", "EMAIL", 0.99),
    (
        "email_spoken",
        r"\b[\w.'-]+(?:\s+(?:dot|underscore|dash|hyphen)\s+[\w.'-]+)*\s+at\s+"
        r"[\w-]+(?:\s+dot\s+[\w-]+)+\b",
        "EMAIL",
        0.9,
    ),
    ("url", r"\b(?:https?://|www\.)[^\s<>\"')]+", "URL", 0.97),
    (
        "url_spoken",
        r"\b(?:www|double\s+u\s+double\s+u\s+double\s+u)\s+dot\s+"
        r"[\w-]+(?:\s+dot\s+[\w-]+)+\b",
        "URL",
        0.85,
    ),
    (
        "ipv4",
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
        "IP_ADDRESS",
        0.97,
    ),
    ("ipv6", r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b", "IP_ADDRESS", 0.95),
    ("mac", r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b", "MAC_ADDRESS", 0.97),
    ("iban", r"(?-i:\b[A-Z]{2}\d{2}[ ]?(?:[A-Z0-9]{4}[ ]?){2,7}[A-Z0-9]{1,4}\b)", "IBAN", 0.9),
    ("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", "NATIONAL_ID", 0.97),
    ("uk_nhs", r"\b\d{3}[ -]?\d{3}[ -]?\d{4}\b", "MEDICAL_ID", 0.55),
    (
        "uk_ni",
        r"(?-i:\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b)",
        "NATIONAL_ID",
        0.9,
    ),
    ("passport", rf"\bpassport\s*(?:number|no\.?|#)?\s*[:=]?\s*({_UPPER_ID})\b", "PASSPORT", 0.9),
    ("plate_uk", r"(?-i:\b[A-Z]{2}\d{2}\s?[A-Z]{3}\b)", "LICENSE_PLATE", 0.6),
    ("card", r"\b(?:\d[ -]?){13,19}\b", "CREDIT_CARD", 0.55),
    (
        "secret",
        r"\b(?:password|passcode|pin(?:\s*code)?|api[ _-]?key|secret|token)\s*"
        r"(?:is|=|:)\s*[\"']?([^\s\"',.;]{3,64})",
        "SECRET",
        0.9,
    ),
    # ---------------- contact ----------------
    (
        "phone_intl",
        r"(?<![\w.])\+\d{1,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?){2,5}\d{2,4}(?![\w])",
        "PHONE",
        0.95,
    ),
    ("phone_pairs", r"(?<![\w.])0\d(?:[\s.-]\d{2}){4}(?![\w])", "PHONE", 0.9),
    (
        "phone_local",
        r"(?<![\w.])(?:\(?0\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}|\d{3}[\s.-]\d{3}[\s.-]\d{4}"
        r"|\(\d{3}\)\s?\d{3}[\s.-]?\d{4})(?![\w])",
        "PHONE",
        0.9,
    ),
    # ---------------- online identity ----------------
    ("handle", r"(?<![\w@])@[A-Za-z][\w.]{2,29}\b", "SOCIAL_HANDLE", 0.9),
    (
        "username_ctx",
        r"\b(?:username|user\s*name|handle|gamertag|screen\s*name|login|user\s*id)\s*"
        r"(?:is|:|=)?\s+([A-Za-z0-9_.-]{3,32})\b",
        "ONLINE_USERNAME",
        0.85,
    ),
    (
        "username_spoken",
        r"\b(?:username|user\s*name|gamertag|screen\s*name|login|user\s*id|account)\s*"
        r"(?:is|:|=)?\s+([A-Za-z0-9'-]{2,32}"
        r"(?:\s+(?:dot|underscore|dash|hyphen)\s+[A-Za-z0-9'-]{1,32})+)",
        "ONLINE_USERNAME",
        0.85,
    ),
    (
        "handle_spoken",
        r"\bat\s+[A-Za-z0-9'-]{2,20}(?:\s+(?:dot|dash)\s+[A-Za-z0-9'-]{2,20})*"
        r"(?:\s+underscore\s+[A-Za-z0-9'-]{2,20})+"
        r"(?:\s+(?:dot|dash|underscore)\s+[A-Za-z0-9'-]{2,20})*",
        "SOCIAL_HANDLE",
        0.7,
    ),
    # ---------------- ids in context ----------------
    (
        "student_id",
        r"\b(?:student|pupil|admission|enrol(?:l)?ment|roll|register|upn)\s*"
        r"(?:id|number|no\.?|#)\s*(?:is|:|=)?\s*([\w-]{3,20})\b",
        "STUDENT_ID",
        0.9,
    ),
    (
        "account_id",
        r"\b(?:account|customer|reference|policy|case|member(?:ship)?|order|booking|ticket)\s*"
        r"(?:id|number|no\.?|#|code)\s*(?:is|:|=)?\s*([\w-]{3,24})\b",
        "ACCOUNT_ID",
        0.88,
    ),
    (
        "medical_id",
        r"\b(?:nhs|medicare|medicaid|patient|medical\s*record|mrn|chart|insurance)\s*"
        r"(?:id|number|no\.?|#)\s*(?:is|:|=)?\s*([\w-]{3,20})\b",
        "MEDICAL_ID",
        0.88,
    ),
    # ---------------- address ----------------
    (
        "street_full",
        rf"\b(?:{_UNIT_WORDS}\s*\d+[a-z]?\s*,?\s*)?\d{{1,4}}[a-z]?\s+(?:{_CAP}\s+){{0,3}}"
        rf"(?:{_STREET_TYPES})\b\.?",
        "STREET_ADDRESS",
        0.85,
    ),
    (
        "street_spoken",
        rf"\b(?:{_SPOKEN_INT})\s+(?:[\w'-]+\s+){{0,2}}(?:{_STREET_TYPES})\b",
        "STREET_ADDRESS",
        0.6,
    ),
    (
        "street_named",
        r"\b(?:{}\s+){{1,3}}(?-i:(?:{}))\b".format(_CAP, _STREET_TYPES.title().replace("|", "|")),
        "STREET_ADDRESS",
        0.5,
    ),
    ("postcode_uk", r"(?-i:\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b)", "POSTCODE", 0.92),
    ("postcode_us", r"\b\d{5}(?:-\d{4})?\b", "POSTCODE", 0.4),
    ("postcode_ca", r"(?-i:\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b)", "POSTCODE", 0.85),
    (
        "live_at",
        rf"\b(?:we|i)\s+live\s+(?:at|on|in)\s+((?:{_CAP}\s?){{1,4}}|\d{{1,4}}[^,.;\n]{{2,40}})",
        "HOME_ADDRESS",
        0.6,
    ),
    # ---------------- dates / times / age ----------------
    (
        "date_numeric",
        r"\b(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)?\d{2}\b",
        "PRECISE_DATE",
        0.9,
    ),
    (
        "date_iso",
        r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b",
        "PRECISE_DATE",
        0.95,
    ),
    (
        "date_written",
        rf"\b(?:(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})|"
        rf"(?:{_MONTHS})\s+(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?)"
        r"(?:,?\s+(?:19|20)\d{2})?\b",
        "PRECISE_DATE",
        0.85,
    ),
    (
        "dob_ctx",
        r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|born\s+on|birthday\s+is)\s*(?:is|:|=)?\s*"
        r"((?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})"
        rf"|(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})(?:,?\s+(?:19|20)?\d{{2}})?)"
        rf"|(?:(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+(?:19|20)?\d{{2}})?)"
        r"|(?:(?:19|20)\d{2}-\d{2}-\d{2})"
        rf"|(?:{_SPOKEN_DATE}))",
        "DATE_OF_BIRTH",
        0.92,
    ),
    ("date_spoken", rf"\b{_SPOKEN_DATE}\b", "PRECISE_DATE", 0.8),
    (
        "time_clock",
        r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\s?(?:am|pm|a\.m\.|p\.m\.)?\b",
        "PRECISE_TIME",
        0.85,
    ),
    ("time_ampm", r"\b(?:1[0-2]|0?[1-9])\s?(?:am|pm|a\.m\.|p\.m\.)\b", "PRECISE_TIME", 0.8),
    (
        "time_spoken",
        r"\b(?:quarter|half|ten|twenty|five|twenty-five)\s+(?:past|to|after)\s+"
        r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})\b",
        "PRECISE_TIME",
        0.8,
    ),
    (
        "age_digit",
        r"\b(?:(?:i'?m|im|i am|he'?s|she'?s|they'?re|is|was|aged?|turning|turns?)\s+)"
        r"(\d{1,2})(?![\d/.:-])(?:\s*(?:years?\s*old|yrs?|y/?o))?\b",
        "EXACT_AGE",
        0.7,
    ),
    ("age_years", r"\b(\d{1,2})\s*(?:years?\s*old|yrs?\s*old|y/?o)\b", "EXACT_AGE", 0.9),
    (
        "age_word",
        r"\b(?:(?:i'?m|im|i am|he'?s|she'?s|(?:\w+)\s+(?:is|was)|"
        r"aged?|turning|turns?)\s+)"
        r"((?:twenty[\s-])?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty))"
        r"(?:\s*years?\s*old)?\b",
        "EXACT_AGE",
        0.6,
    ),
    (
        "year_group",
        r"\b(?:year|grade|class|form)\s+(?:\d{1,2}|(?-i:[A-Z])\b|one|two|three|four|five|six)"
        r"|\breception\s+class\b",
        "CLASS_NAME",
        0.6,
    ),
    # ---------------- child-context organisations ----------------
    (
        "school_named",
        r"(?-i:\b(?:[A-Z][\w'&-]+\s+){1,4}"
        r"(?:School|Academy|College|Primary|Secondary|Nursery|Preschool|Pre-school|"
        r"Kindergarten|Daycare|Day\s?Care|Montessori|Grammar|Elementary|Institute)\b)",
        "SCHOOL_NAME",
        0.8,
    ),
    (
        "school_lower",
        r"\b(?:at|to|from|in|attends?|enrolled\s+at)\s+"
        r"((?:[\w'&-]+\s+){1,3}(?:school|academy|primary|secondary|nursery|college|"
        r"preschool|pre-school|kindergarten|montessori|daycare))\b",
        "SCHOOL_NAME",
        0.5,
    ),
    (
        "school_ctx",
        r"\b(?:goes?\s+to|attends?|starts?\s+at|enrolled\s+(?:at|in)|transferred\s+to)\s+"
        rf"((?:{_CAP}\s?){{1,4}})\b",
        "SCHOOL_NAME",
        0.4,
    ),
    (
        "team_named",
        r"(?-i:\b(?:[A-Z][\w'-]+\s+){1,2}"
        r"(?:FC|F\.C\.|United|Rovers|Tigers|Lions|Eagles|Hawks|Sharks|Dragons|Warriors|"
        r"Panthers|Wolves|Bears|Colts|Cubs|Rangers|Athletic)\b)",
        "SPORTS_TEAM",
        0.7,
    ),
    (
        "team_ctx",
        rf"\b(?:under[\s-]?\d{{1,2}}|u\d{{1,2}})s?\s+((?:{_CAP}\s?){{1,2}})",
        "SPORTS_TEAM",
        0.65,
    ),
    # ---------------- lower-cased speech variants of place / club names ------
    # ASR output is routinely lower-cased, which removes the capitalisation the
    # written-form rules rely on.  These fire on the head noun instead.
    (
        "club_lower",
        r"\b(?:at|joins?|joined|goes?\s+to|attends?|plays?\s+for|member\s+of|training\s+at)\s+"
        r"((?:the\s+)?(?:[\w'-]+\s+){1,3}"
        r"(?:club|society|scouts|guides|brownies|cubs|troop|squad|academy))\b",
        "CLUB_NAME",
        0.5,
    ),
    (
        "landmark_lower",
        r"\b(?:near|opposite|behind|next\s+to|beside|across\s+from|by)\s+"
        r"((?:the\s+)?(?:[\w'-]+\s+){1,3}"
        r"(?:park|library|mosque|church|chapel|temple|synagogue|gurdwara|station|"
        r"centre|center|hall|bridge|market|square|gardens?|playground|pool|baths))\b",
        "LOCAL_LANDMARK",
        0.5,
    ),
    (
        "city_lower",
        r"\b(?:we|i|they|she|he)\s+(?:live|lives|lived|moved|grew\s+up)\s+"
        r"(?:in|to|from|near)\s+((?:[\w'-]+)(?:\s+[\w'-]+)?)\b",
        "CITY",
        0.5,
    ),
    (
        "bus_route",
        r"\b(?:(?:the\s+)?(?:number\s+)?\d{1,3}\s+bus|bus\s+(?:number\s+)?\d{1,3}|"
        r"route\s+\d{1,3})\b",
        "BUS_ROUTE",
        0.7,
    ),
]

# Spoken contact details need their own multilingual rules. A long sequence
# of number words is redacted even without a nearby phone cue; counting aloud
# may be over-redacted, but dictated contact details must not remain readable.
_MULTILINGUAL_NUMBER = (
    "zero|oh|one|two|three|four|five|six|seven|eight|nine|"
    "cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|"
    "zéro|un|deux|trois|quatre|cinq|sept|huit|neuf|dix|onze|douze|"
    "treize|quatorze|quinze|seize|vingt|trente|quarante|cinquante|soixante|"
    "null|eins|zwei|drei|vier|fünf|funf|sechs|sieben|acht|neun|"
    "due|tre|quattro|cinque|sei|sette|otto|nove|"
    "nul|een|twee|drie|vijf|zes|zeven|negen|"
    "um|dois|três|quatro|sete|oito|"
    "ноль|нуль|один|два|три|четыре|пять|шесть|семь|восемь|девять"
)
RULES.extend(
    [
        (
            "multilingual_spoken_phone",
            rf"(?<!\w)(?:{_MULTILINGUAL_NUMBER})(?:[ \t,-]+(?:{_MULTILINGUAL_NUMBER})){{6,}}(?!\w)",
            "PHONE",
            0.96,
        ),
        (
            "multilingual_spoken_email",
            r"\b[\w.\'-]+(?:[ \t]+(?:dot|punto|point|punkt|punt|ponto|точка)[ \t]+[\w.\'-]+)*"
            r"[ \t]+(?:at|arroba|arobase|chiocciola|apenstaartje|собака)[ \t]+"
            r"[\w-]+(?:[ \t]+(?:dot|punto|point|punkt|punt|ponto|точка)[ \t]+[\w-]+)+\b",
            "EMAIL",
            0.96,
        ),
        (
            "spoken_initial_name",
            r"\b(?:doctor|dr|mister|mr|miss|ms|mrs|professor)[ \t]+[a-z][ \t]+[a-z]{2,20}\b",
            "PERSON_NAME",
            0.92,
        ),
    ]
)

# Titles anchor unfamiliar names even when speech recognition removes case.
# Stop before common clause boundaries rather than swallowing following prose.
_FOREIGN_TITLE = (
    "señorita|senyorita|señor|senyor|señora|senyora|doctora|"
    "madame|monsieur|docteur|docteure|frau|herr|doktor|"
    "maestra|maestro|dottor|dottoressa|juf|meester|dokter|doutor|doutora|professora"
)
_FOREIGN_STOP = r"(?:and|how|is|she|he|it|the|my|your|y|e|et|und|en|es|est|elle|il|ist|sie|hij|zij|ela|ele|ha|has)"
_FOREIGN_NAME_WORD = rf"(?!(?:{_FOREIGN_STOP})\b)[\w\'-]+"
RULES.append(
    (
        "multilingual_titled_name",
        rf"\b(?:(?:la|el|le|il|lo|a|o)[ \t]+)?(?:{_FOREIGN_TITLE})[ \t]+{_FOREIGN_NAME_WORD}(?:[ \t]+{_FOREIGN_NAME_WORD}){{0,2}}",
        "PERSON_NAME",
        0.9,
    )
)

_COMPILED = [
    (name, re.compile(pat, re.IGNORECASE), entity, score) for name, pat, entity, score in RULES
]

# rules whose *capture group 1* is the actual identifier (the lead-in words are context)
_GROUP_RULES = {
    "passport",
    "secret",
    "username_ctx",
    "student_id",
    "account_id",
    "medical_id",
    "live_at",
    "dob_ctx",
    "age_digit",
    "age_years",
    "age_word",
    "school_ctx",
    "school_lower",
    "team_ctx",
    "username_spoken",
    "club_lower",
    "landmark_lower",
    "city_lower",
}


_SCHOOL_MODIFIERS = {
    "the",
    "a",
    "an",
    "my",
    "our",
    "his",
    "her",
    "their",
    "its",
    "big",
    "new",
    "old",
    "same",
    "another",
    "local",
    "little",
    "main",
    "that",
    "this",
    "other",
    "nearby",
    "usual",
}
_SCHOOL_SUFFIXES = {
    "school",
    "academy",
    "primary",
    "secondary",
    "nursery",
    "college",
    "preschool",
    "pre-school",
    "kindergarten",
    "montessori",
    "daycare",
}


# Ordinary nouns that follow "we live in ..." and are not place names.
# Words that follow a place name rather than belonging to it.
_TRAILING_WORDS = frozenset(
    """
last this next ago again now then when because and but so with for from before
after during since while year years month months week weeks summer winter term
""".split()
)

_GENERIC_PLACE_WORDS = frozenset(
    """
a an the town city village hamlet suburb area estate street road lane close avenue
house home flat apartment bungalow maisonette caravan block council countryside
country side centre center middle north south east west england scotland wales
ireland uk britain here there nearby locally together alone school hospital
""".split()
)


_SPELLED_CODE_TOKEN = (
    rf"(?:[a-z]|{_SPOKEN_INT}|hundred|thousand|double|triple|oh|zero|nought|\d{{1,4}})"
)
_SPELLED_CODE_RE = re.compile(
    rf"\b(?:{_SPELLED_CODE_TOKEN})(?:[\s,-]+(?:{_SPELLED_CODE_TOKEN})){{4,}}\b", re.I
)


def luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 13:
        return False
    total, parity = 0, len(nums) % 2
    for i, n in enumerate(nums):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


@register("regex")
class RegexDetector(Detector):
    """Pattern + spoken-form deterministic detector."""

    def load(self) -> None:
        self.extra: list[tuple[str, Any, str, float]] = []
        for spec in self.spec.get("patterns", []) or []:
            self.extra.append(
                (
                    spec.get("name", "custom"),
                    re.compile(spec["pattern"], re.IGNORECASE if spec.get("ignore_case") else 0),
                    spec["entity"],
                    float(spec.get("score", 0.9)),
                )
            )

    def detect(self, doc: Document) -> list[Detection]:
        out: list[Detection] = []
        text = doc.text
        for name, rx, entity, score in list(_COMPILED) + list(self.extra):
            for m in rx.finditer(text):
                start, end = m.span()
                if name in _GROUP_RULES and m.groups():
                    try:
                        gs, ge = m.span(1)
                        if ge > gs:
                            start, end = gs, ge
                    except (IndexError, ValueError):
                        pass
                frag = text[start:end]
                adj = self._adjust(name, entity, frag, text, start, end)
                if adj is None:
                    continue
                # a rule may also shorten its own match ("bradford last year")
                span_end = end
                if len(adj) == 3:
                    entity_final, score_final, keep = adj
                    keep = int(keep)
                    if keep <= 0:
                        continue
                    span_end = min(end, start + keep)
                else:
                    entity_final, score_final = adj
                out.append(
                    self.make(
                        doc,
                        start,
                        span_end,
                        entity_final,
                        score_final if score_final is not None else score,
                        raw_label=name,
                        rule=name,
                    )
                )
        out.extend(self._spoken_digit_runs(doc))
        out.extend(self._spelled_code_runs(doc))
        return out

    # ------------------------------------------------------------------
    def _adjust(
        self, rule: str, entity: str, frag: str, text: str, start: int, end: int
    ) -> tuple[Any, ...] | None:
        """Rule-specific validation.  Returning None drops the match."""
        stripped = frag.strip()
        if not stripped:
            return None

        if rule == "card":
            digits = re.sub(r"\D", "", frag)
            if len(digits) < 13 or len(digits) > 19:
                return None
            if luhn_ok(digits):
                return entity, 0.97
            # long digit run that is not a card is still an identifier of some kind
            return ("ACCOUNT_ID", 0.5) if len(digits) >= 12 else None

        if rule in ("club_lower", "landmark_lower", "city_lower"):
            head = [w for w in re.findall(r"[\w'-]+", stripped)]
            if not head:
                return None
            if rule == "city_lower":
                # "we live in a flat" is not a city; require a name-like object
                if any(normalize_first(w) in _GENERIC_PLACE_WORDS for w in head):
                    return None
                if len(head) > 2:
                    return None
                if len(head) == 2 and normalize_first(head[1]) in _TRAILING_WORDS:
                    # "we moved to bradford last year" - the place is one token
                    return entity, None, len(head[0])
            return entity, None

        if rule == "postcode_us":
            # bare 5-digit numbers are usually not postcodes in speech
            before = text[max(0, start - 30) : start].lower()
            if re.search(r"\b(zip|postcode|postal|post\s?code)\b", before):
                return entity, 0.9
            return None

        if rule == "uk_nhs":
            before = text[max(0, start - 40) : start].lower()
            if re.search(r"\b(nhs|patient|medical|health)\b", before):
                return entity, 0.95
            return "PHONE", 0.6

        if rule == "plate_uk":
            before = text[max(0, start - 40) : start].lower()
            if re.search(r"\b(car|van|plate|registration|reg|vehicle|number\s?plate)\b", before):
                return entity, 0.9
            return None

        if rule in ("age_digit", "age_word"):
            val = spoken_cardinal(stripped) if not stripped.isdigit() else int(stripped)
            if val is None or not (0 <= val <= 120):
                return None
            after = text[end : end + 24].lower()
            before = text[max(0, start - 40) : start].lower()
            if re.search(
                r"^\s*(o'?clock|am|pm|p\.m|a\.m|minutes?|seconds?|hours?|"
                r"pounds?|dollars?|percent|times?)",
                after,
            ):
                return None
            if re.search(r"\b(at|around|about|by|until|till)\s*$", before) and val <= 12:
                return None  # "at three" is a time, not an age
            conf = 0.85 if re.search(r"^\s*(years?\s*old|yrs?)", after) else None
            return entity, conf

        if rule == "street_named":
            # "Oak Street" without a number is weaker evidence
            if re.match(r"^(the|a|an|my|our|his|her)\b", stripped, re.I):
                return None
            return entity, 0.5

        if rule == "school_lower":
            words = [normalize_first(w) for w in stripped.split()]
            # "at the same nursery" names nothing; "at greenfield primary" does
            distinctive = [
                w for w in words if w not in _SCHOOL_MODIFIERS and w not in _SCHOOL_SUFFIXES
            ]
            if not distinctive:
                return None
            return entity, None

        if rule == "school_ctx":
            if len(stripped.split()) > 5:
                return None
            if re.match(r"^(the|a|an|my|our|his|her|school|bed|sleep|work)\b", stripped, re.I):
                return None
            return entity, 0.45

        if rule == "email_spoken":
            if " at " not in frag.lower():
                return None
            return entity, 0.9

        if rule == "phone_local":
            digits = re.sub(r"\D", "", frag)
            if len(digits) < 9:
                return None
            return entity, None

        if rule == "secret":
            if len(stripped) < 3:
                return None
            return entity, None

        if rule == "handle":
            if re.match(r"^@(?:gmail|yahoo|hotmail|outlook|icloud)", stripped, re.I):
                return None
            return entity, None

        return entity, None

    # ------------------------------------------------------------------
    _CODE_CUES = (
        (r"post\s*code|postal\s*code|zip", "POSTCODE", 0.85),
        (r"pupil|student|admission|enrol|upn|roll", "STUDENT_ID", 0.85),
        (r"nhs|patient|medical|hospital", "MEDICAL_ID", 0.85),
        (r"account|reference|policy|membership|booking|case", "ACCOUNT_ID", 0.85),
    )

    def _spelled_code_runs(self, doc: Document) -> list[Detection]:
        """Identifiers spelled out letter-by-letter: "l s six two q t".

        Spoken postcodes, pupil numbers and reference codes alternate letters
        and numbers, which no written-form pattern matches and which the digit
        run scanner skips precisely because of the letters.
        """
        text = doc.text
        out: list[Detection] = []
        for m in _SPELLED_CODE_RE.finditer(text):
            frag = m.group(0)
            words = re.findall(r"[\w]+", frag)
            singles = sum(1 for w in words if len(w) == 1 and w.isalpha())
            if singles < 2 or len(words) < 5:
                continue  # a pure number run; handled elsewhere
            entity, score = "ACCOUNT_ID", 0.6
            before = text[max(0, m.start() - 60) : m.start()].lower()
            for cue, ent, sc in self._CODE_CUES:
                if re.search(cue, before):
                    entity, score = ent, sc
                    break
            out.append(
                self.make(
                    doc,
                    m.start(),
                    m.end(),
                    entity,
                    score,
                    raw_label="spelled_code",
                    tokens=len(words),
                )
            )
        return out

    def _spoken_digit_runs(self, doc: Document) -> list[Detection]:
        """Catch phone/account numbers dictated as words."""
        out: list[Detection] = []
        text = doc.text
        pattern = re.compile(rf"\b{_DIGIT_WORD}(?:[\s,-]+{_DIGIT_WORD}){{5,}}\b", re.I)
        for m in pattern.finditer(text):
            frag = m.group(0)
            words = re.split(r"[\s,-]+", frag)
            if (
                sum(
                    1
                    for w in words
                    if w.lower() in NUMBER_WORDS or w.lower() in MULTIPLIER_WORDS or w.isdigit()
                )
                < 6
            ):
                continue
            digits = spoken_digits(words)
            if len(digits) < 6:
                continue
            # look only at the immediately preceding words - a keyword further
            # back usually belongs to a different sentence
            before = text[max(0, m.start() - 32) : m.start()].lower()
            entity = "PHONE"
            score = 0.75
            if re.search(r"\b(account|reference|policy|card|member|customer|order)\b", before):
                entity, score = "ACCOUNT_ID", 0.8
            elif re.search(r"\b(number|phone|mobile|cell|call|reach|text)\b", before):
                score = 0.9
            elif len(digits) >= 9:
                score = 0.8
            out.append(
                self.make(
                    doc,
                    m.start(),
                    m.end(),
                    entity,
                    score,
                    raw_label="spoken_digits",
                    digits=len(digits),
                )
            )
        return out
