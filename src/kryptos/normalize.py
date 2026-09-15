"""Text normalisation + ASR-noise-tolerant matching.

The single most under-appreciated failure mode in transcript de-identification
is that ASR mangles exactly the tokens that matter most: names.  "Nafisa
Rahman" comes back as "nafeesa ramen".  Dictionary and known-value matching
therefore runs on three channels - exact, edit-distance and phonetic - and a
hit on any channel counts.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)
try:
    from rapidfuzz.distance import Levenshtein as _native_levenshtein
except ImportError:
    _native_levenshtein = None

# ---------------------------------------------------------------------------
# basic normalisation
# ---------------------------------------------------------------------------


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# zero-width and other format characters are invisible, occur in copied text,
# and are a trivial way to hide a name from a matcher ("Ais<ZWSP>ha")
_FORMAT_CHARS = re.compile(r"[\u00ad\u200b\u200c\u200d\u2060\ufeff]")


def strip_format_chars(text: str) -> str:
    return _FORMAT_CHARS.sub("", text)


def normalize(text: str, keep_punct: bool = False) -> str:
    out = strip_accents(strip_format_chars(text)).lower()
    if not keep_punct:
        out = _PUNCT.sub(" ", out)
    return _WS.sub(" ", out).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


# ---------------------------------------------------------------------------
# edit distance
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str, cap: int | None = None) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1
    if _native_levenshtein is not None:
        return _native_levenshtein.distance(a, b, score_cutoff=cap)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(val)
            best = min(best, val)
        prev = cur
        if cap is not None and best > cap:
            return cap + 1
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """Normalised similarity in [0, 1]."""
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    longest = max(len(a), len(b))
    return 1.0 - (levenshtein(a, b) / float(longest))


# ---------------------------------------------------------------------------
# phonetic key (double-metaphone-lite): survives most ASR name corruption
# ---------------------------------------------------------------------------

_PHONETIC_RULES: Sequence[tuple[str, str]] = (
    ("ough", "F"),
    ("ph", "F"),
    ("gh", "K"),
    ("ck", "K"),
    ("sch", "SK"),
    ("sh", "X"),
    ("ch", "X"),
    ("tch", "X"),
    ("th", "0"),
    ("wr", "R"),
    ("kn", "N"),
    ("gn", "N"),
    ("wh", "W"),
    ("qu", "KW"),
    ("dg", "J"),
    ("ee", "I"),
    ("ea", "I"),
    ("ie", "I"),
    ("ei", "I"),
    ("oo", "U"),
    ("ou", "U"),
    ("aa", "A"),
    ("ah", "A"),
)

_CONSONANT_MAP = {
    "b": "B",
    "c": "K",
    "d": "T",
    "f": "F",
    "g": "K",
    "j": "J",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "p": "P",
    "q": "K",
    "r": "R",
    "s": "S",
    "t": "T",
    "v": "F",
    "w": "W",
    "x": "KS",
    "z": "S",
}
_VOWELS = set("aeiouy")  # y is treated as a vowel: ASR flips Aisha/Ayesha freely


def phonetic_key(word: str) -> str:
    """Cheap, dependency-free phonetic key tuned for name confusions."""
    w = normalize(word).replace(" ", "")
    if not w:
        return ""
    for src, dst in _PHONETIC_RULES:
        w = w.replace(src, dst)
    out: list[str] = []
    prev = ""
    for i, ch in enumerate(w):
        if ch in _VOWELS or ch in "AEIOU":
            # vowels carry almost no signal through ASR; keep only the onset
            code = "A" if i == 0 else ""
        elif ch.isupper() or ch in "0":
            code = ch.upper()
        elif ch == "h":
            code = ""
        else:
            code = _CONSONANT_MAP.get(ch, ch.upper() if ch.isalnum() else "")
        if code and code != prev:
            out.append(code)
        if code:
            prev = code
    key = "".join(out)
    # collapse voiced/unvoiced pairs that ASR routinely swaps
    key = key.replace("Z", "S").replace("D", "T").replace("G", "K").replace("V", "F")
    return key


def phonetic_match(a: str, b: str) -> bool:
    ka, kb = phonetic_key(a), phonetic_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    # allow one phonetic edit for longer keys
    if min(len(ka), len(kb)) >= 4 and levenshtein(ka, kb, cap=1) <= 1:
        return True
    return False


def fuzzy_match(
    a: str, b: str, threshold: float = 0.82, phonetic: bool = True
) -> tuple[bool, float, str]:
    """Return (matched, score, channel)."""
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True, 1.0, "exact"
    sim = similarity(na, nb)
    if sim >= threshold:
        return True, sim, "fuzzy"
    if phonetic and phonetic_match(na, nb):
        return True, max(sim, threshold), "phonetic"
    return False, sim, "none"


# ---------------------------------------------------------------------------
# spoken numbers -> digits ("oh seven eight one" / "double four")
# ---------------------------------------------------------------------------

NUMBER_WORDS: dict[str, str] = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "nought": "0",
    "naught": "0",
    "one": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "three": "3",
    "four": "4",
    "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
}
MULTIPLIER_WORDS = {"double": 2, "triple": 3, "treble": 3}

TEEN_TENS: dict[str, int] = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
UNITS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


def spoken_digits(words: Iterable[str]) -> str:
    """Convert a run of spoken digit words into a digit string."""
    out: list[str] = []
    pending = 1
    for raw in words:
        w = normalize(raw)
        if w in MULTIPLIER_WORDS:
            pending = MULTIPLIER_WORDS[w]
            continue
        if w.isdigit():
            out.append(w * pending)
            pending = 1
            continue
        if w in NUMBER_WORDS:
            out.append(NUMBER_WORDS[w] * pending)
            pending = 1
            continue
        if w in TEEN_TENS:
            out.append(str(TEEN_TENS[w]) * pending)
            pending = 1
            continue
        pending = 1
    return "".join(out)


def spoken_cardinal(text: str) -> int | None:
    """Parse a small spoken cardinal ('nine', 'twenty three') -> int."""
    parts = [normalize(p) for p in re.split(r"[\s-]+", text) if p.strip()]
    if not parts:
        return None
    total = 0
    seen = False
    for p in parts:
        if p in TEEN_TENS:
            total += TEEN_TENS[p]
            seen = True
        elif p in UNITS:
            total += UNITS[p]
            seen = True
        elif p.isdigit():
            total += int(p)
            seen = True
        else:
            return None
    return total if seen else None


def despoken_email(text: str) -> str:
    """'name at gmail dot com' -> 'name@gmail.com' for matching purposes."""
    out = re.sub(r"\s+at\s+", "@", text, flags=re.I)
    out = re.sub(r"\s+dot\s+", ".", out, flags=re.I)
    out = re.sub(r"\s+underscore\s+", "_", out, flags=re.I)
    out = re.sub(r"\s+(dash|hyphen)\s+", "-", out, flags=re.I)
    return out.strip()
