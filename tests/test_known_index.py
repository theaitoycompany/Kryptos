import random
import unittest

from kryptos.config import Config
from kryptos.detectors.known_values import KnownValuesDetector
from kryptos.types import Document


class IndexedKnownValues(unittest.TestCase):
    def test_independent_qa_retains_all_occurrences_and_order(self):
        from kryptos.qa import _check_known_values

        rng = random.Random(914)
        a = Config.load(overrides={"qa": {"indexed_known_values": False}})
        b = Config.load(overrides={"qa": {"indexed_known_values": True}})
        values = {
            "CHILD_NAME": ["Mei-Ling Chen", "Amina Shah", "Ava", "Green Field"],
            "DOCTOR_NAME": ["Osei", "Ahmed"],
            "ACCOUNT_ID": ["alpha beta gamma"],
        }
        phrases = [
            "mei-ling chen",
            "mina shah",
            "ava",
            "of",
            "green field",
            "ahmad",
            "hello",
            "nothing here",
            "alpha beta gamma",
            "shah",
        ]
        for _ in range(80):
            doc = Document(
                " ".join(rng.choices(phrases, k=rng.randint(1, 80))), meta={"known_values": values}
            )
            self.assertEqual(
                _check_known_values(doc.text, doc, a), _check_known_values(doc.text, doc, b)
            )

    def test_index_retains_all_reference_detections_and_order(self):
        rng = random.Random(205)
        a = KnownValuesDetector({"name": "known", "indexed_matching": False}, Config.load())
        b = KnownValuesDetector({"name": "known", "indexed_matching": True}, Config.load())
        a.load()
        b.load()
        values = {
            "CHILD_NAME": ["Mei-Ling Chen", "Amina Shah", "Ava", "Green Field"],
            "DOCTOR_NAME": ["Dr Osei", "Dr Ahmed"],
            "SCHOOL_NAME": ["Greenfield Primary School"],
            "ACCOUNT_ID": ["alpha beta gamma delta epsilon zeta eta theta iota kappa"],
        }
        phrases = [
            "may ling chen",
            "doctor o say",
            "mina shah",
            "ava",
            "of",
            "greenfield primary school",
            "green field",
            "dr ahmad",
            "hello",
            "one two three",
            "nothing here",
            "greenfield",
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
        ]
        for _ in range(80):
            doc = Document(
                " ".join(rng.choices(phrases, k=rng.randint(5, 60))), meta={"known_values": values}
            )

            def result(d):
                return [(x.start, x.end, x.entity, x.score, x.meta) for x in d.detect(doc)]

            self.assertEqual(result(a), result(b))
