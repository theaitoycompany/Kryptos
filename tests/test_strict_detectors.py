import unittest

from kryptos.config import Config
from kryptos.detectors.base import Detector, run_detectors
from kryptos.detectors.gliner_det import _chunks
from kryptos.detectors.known_values import KnownValuesDetector
from kryptos.detectors.regex_rules import RegexDetector
from kryptos.qa import _check_known_values
from kryptos.types import Document


class BrokenDetector(Detector):
    def detect(self, doc):
        raise ValueError("sensitive error details")


class StrictModelGate(unittest.TestCase):
    def test_spoken_identifiers_cover_complete_multilingual_values(self):
        detector = RegexDetector({"name": "rules"}, Config.load())
        detector.load()
        for text, entity in [
            ("zéro six douze trente quatre cinquante six soixante dix huit", "PHONE"),
            ("marieke punt devries apenstaartje gmail punt com", "EMAIL"),
            ("doctor o say", "PERSON_NAME"),
        ]:
            found = detector.detect(Document(text))
            self.assertTrue(
                any(d.start == 0 and d.end == len(text) and d.entity == entity for d in found), text
            )

    def test_known_names_cover_phonetic_split_including_last_syllable(self):
        config = Config.load()
        detector = KnownValuesDetector({"name": "known_values"}, config)
        detector.load()
        doc = Document(
            "tell doctor o say and may ling chen hello",
            meta={"known_values": {"DOCTOR_NAME": ["Dr Osei"], "CHILD_NAME": ["Mei-Ling Chen"]}},
        )
        found = detector.detect(doc)
        for target in ("doctor o say", "may ling chen"):
            start = doc.text.index(target)
            self.assertTrue(
                any(d.start <= start and d.end >= start + len(target) for d in found), target
            )

    def test_known_audit_reports_all_exact_spans_without_short_phonetic_false_alarm(self):
        config = Config.load()
        doc = Document(
            "Ava is here. Ava likes pictures of cats.",
            meta={"known_values": {"CHILD_NAME": ["Ava"]}},
        )
        findings = _check_known_values(doc.text, doc, config)
        self.assertEqual([(f.detail["at"], f.detail["end"]) for f in findings], [(0, 3), (13, 16)])

    def test_detector_failure_cannot_produce_empty_safe_result(self):
        config = Config.load(overrides={"runtime": {"strict_detectors": True}})
        with self.assertRaisesRegex(RuntimeError, "Required detector failed"):
            run_detectors([BrokenDetector({"name": "test"}, config)], Document("private"))

    def test_long_unstructured_text_has_no_unscanned_gaps(self):
        text = ("a " * 1800) + " APrivateNameNearTheEnd"
        coverage = [False] * len(text)
        chunks = _chunks(Document(text), 1200)
        for chunk, start in chunks:
            self.assertEqual(text[start : start + len(chunk)], chunk)
            self.assertLessEqual(len(chunk), 1200)
            self.assertLessEqual(len(chunk.split()), 180)
            coverage[start : start + len(chunk)] = [True] * len(chunk)
        self.assertTrue(all(coverage))
        self.assertIn("APrivateNameNearTheEnd", chunks[-1][0])
