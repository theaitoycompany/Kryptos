"""End-to-end and unit tests.  Run: python -m unittest discover -s tests -v"""

from __future__ import annotations

import json
import unittest

from kryptos import Config, Pipeline  # noqa: E402
from kryptos.detectors.regex_rules import RegexDetector, luhn_ok  # noqa: E402
from kryptos.evaluation import evaluate, match, pii_token_wer  # noqa: E402
from kryptos.fusion import fuse  # noqa: E402
from kryptos.io_formats import load_text  # noqa: E402
from kryptos.normalize import (  # noqa: E402
    fuzzy_match,
    spoken_cardinal,
    spoken_digits,
)
from kryptos.quasi import age_bucket  # noqa: E402
from kryptos.roles import assign_roles  # noqa: E402
from kryptos.types import Detection  # noqa: E402


def run(text, known=None, **overrides):
    cfg = Config.load()
    if known:
        cfg.raw["known_values"] = known
    for k, v in overrides.items():
        cfg.set(k.replace("__", "."), v)
    return Pipeline(cfg).process_text(text, doc_id="t")


class TestNormalize(unittest.TestCase):
    def test_phonetic_survives_asr_name_corruption(self):
        for a, b in [
            ("Nafisa", "nafeesa"),
            ("Rahman", "ramen"),
            ("Aisha", "Ayesha"),
            ("Rupa", "Roopa"),
            ("Ahmed", "Ahmad"),
            ("Samira", "Sameera"),
        ]:
            self.assertTrue(fuzzy_match(a, b)[0], f"{a} !~ {b}")

    def test_phonetic_does_not_collapse_unrelated_words(self):
        for a, b in [("bakery", "library"), ("school", "pool"), ("Westbrook", "Ashford")]:
            self.assertFalse(fuzzy_match(a, b)[0], f"{a} ~ {b}")

    def test_spoken_numbers(self):
        self.assertEqual(spoken_digits("oh seven double one nine".split()), "07119")
        self.assertEqual(spoken_cardinal("twenty three"), 23)
        self.assertIsNone(spoken_cardinal("banana"))


class TestLoaders(unittest.TestCase):
    def test_speaker_lines_and_offsets(self):
        doc = load_text("CHILD: hello there\nPARENT: hi Aisha")
        self.assertEqual(len(doc.turns), 2)
        self.assertEqual(doc.text[doc.turns[1].char_start : doc.turns[1].char_end], "hi Aisha")
        self.assertNotIn("CHILD:", doc.text)  # labels never enter the text

    def test_sentence_colon_is_not_a_speaker(self):
        doc = load_text("and then I said: we should go")
        self.assertEqual(len(doc.turns), 1)

    def test_whisper_json_word_offsets(self):
        payload = json.dumps(
            {
                "segments": [
                    {
                        "speaker": "S0",
                        "text": "call miss rahman",
                        "words": [
                            {"word": "call", "start": 1.0, "end": 1.2},
                            {"word": "miss", "start": 1.3, "end": 1.5},
                            {"word": "rahman", "start": 1.6, "end": 2.0},
                        ],
                    }
                ]
            }
        )
        doc = load_text(payload)
        word = doc.turns[0].words[-1]
        self.assertEqual(doc.text[word.char_start : word.char_end], "rahman")
        self.assertTrue(doc.has_timings())

    def test_srt(self):
        doc = load_text("1\n00:00:01,000 --> 00:00:03,000\nCHILD: hello\n")
        self.assertEqual(doc.turns[0].speaker, "CHILD")
        self.assertEqual(doc.turns[0].start, 1.0)


class TestRoles(unittest.TestCase):
    def test_roles_from_cues_only(self):
        doc = load_text(
            "SPEAKER_00: I'm nine and my mum picks me up.\n"
            "SPEAKER_01: my daughter finishes at three."
        )
        roles = assign_roles(doc)
        self.assertEqual(roles["SPEAKER_00"], "child")
        self.assertEqual(roles["SPEAKER_01"], "parent")

    def test_roles_from_known_names(self):
        doc = load_text("AISHA: hello\nFARHANA: hello love")
        roles = assign_roles(
            doc, known_values={"CHILD_NAME": ["Aisha"], "PARENT_NAME": ["Farhana Hasan"]}
        )
        self.assertEqual(roles["AISHA"], "child")
        self.assertEqual(roles["FARHANA"], "parent")


class TestRules(unittest.TestCase):
    def setUp(self):
        self.det = RegexDetector({"name": "rules", "family": "regex"}, Config.load())
        self.det.load()

    def entities(self, text):
        doc = load_text("S: " + text)
        return {(d.entity, d.text) for d in self.det.detect(doc)}

    def test_luhn(self):
        self.assertTrue(luhn_ok("4111111111111111"))
        self.assertFalse(luhn_ok("4111111111111112"))

    def test_structured(self):
        got = self.entities("email me at a.b@c.com or call 07911 123456")
        self.assertIn(("EMAIL", "a.b@c.com"), got)
        self.assertIn(("PHONE", "07911 123456"), got)

    def test_spoken_phone(self):
        got = self.entities("my number is oh seven nine one one two three four five")
        self.assertTrue(any(e == "PHONE" for e, _ in got))

    def test_age_not_confused_with_time(self):
        got = self.entities("we leave at 3 and she is 9 years old")
        self.assertIn(("EXACT_AGE", "9"), got)
        self.assertNotIn(("EXACT_AGE", "3"), got)

    def test_date_context_does_not_swallow_the_clause(self):
        got = self.entities("her date of birth is 12/03/2016 and the number is 5")
        self.assertIn(("DATE_OF_BIRTH", "12/03/2016"), got)


class TestFusion(unittest.TestCase):
    def test_specific_beats_generic(self):
        doc = load_text("S: Rupa teaches her")
        cfg = Config.load()
        dets = [Detection(3, 7, "PERSON_NAME", 0.9, "a"), Detection(3, 7, "TEACHER_NAME", 0.6, "b")]
        spans = fuse(dets, doc, cfg)
        self.assertEqual(spans[0].entity, "TEACHER_NAME")

    def test_agreement_raises_score(self):
        doc = load_text("S: Rupa teaches her")
        cfg = Config.load()
        one = fuse([Detection(3, 7, "PERSON_NAME", 0.4, "a")], doc, cfg)[0]
        two = fuse(
            [Detection(3, 7, "PERSON_NAME", 0.4, "a"), Detection(3, 7, "PERSON_NAME", 0.4, "b")],
            doc,
            cfg,
        )[0]
        self.assertGreater(two.score, one.score)

    def test_leftovers_are_not_swallowed(self):
        doc = load_text("S: the bakery opposite Westbrook Primary is ours")
        cfg = Config.load()
        dets = [
            Detection(4, 32, "PARENT_WORKPLACE", 0.6, "ctx"),
            Detection(22, 39, "SCHOOL_NAME", 0.9, "rules"),
        ]
        spans = fuse(dets, doc, cfg)
        self.assertEqual({s.entity for s in spans}, {"PARENT_WORKPLACE", "SCHOOL_NAME"})

    def test_spans_never_overlap(self):
        result = run("PARENT: Farhana Hasan on 07911 123456, samira.hasan@gmail.com")
        ends = [(s.start, s.end) for s in sorted(result.spans, key=lambda x: x.start)]
        for (s1, e1), (s2, _e2) in zip(ends, ends[1:]):
            self.assertLessEqual(e1, s2)


class TestPipeline(unittest.TestCase):
    def test_direct_identifiers_removed(self):
        r = run(
            "CHILD: I'm Aisha and I go to Greenfield School.\n"
            "PARENT: ring 07911 123456 or samira.hasan@gmail.com",
            known={"CHILD_NAME": ["Aisha"], "SCHOOL_NAME": ["Greenfield School"]},
        )
        for leak in ("Aisha", "Greenfield", "123456", "gmail.com"):
            self.assertNotIn(leak.lower(), r.sanitized_text.lower())

    def test_pseudonyms_are_consistent_and_role_aware(self):
        r = run(
            "CHILD: I'm Aisha.\nPARENT: Aisha, come here. Zaid too.",
            known={"CHILD_NAME": ["Aisha"], "SIBLING_NAME": ["Zaid"]},
        )
        self.assertEqual(r.sanitized_text.count("<CHILD_01>"), 2)  # both mentions
        self.assertEqual(r.sanitized_turns[0]["speaker"], "<CHILD_01>")  # and the label
        self.assertIn("<SIBLING_01>", r.sanitized_text)

    def test_speaker_labels_do_not_collide_with_span_pseudonyms(self):
        r = run(
            "AISHA: hi\nFARHANA: hello", known={"CHILD_NAME": ["Aisha"], "PARENT_NAME": ["Farhana"]}
        )
        labels = [t["speaker"] for t in r.sanitized_turns]
        self.assertEqual(len(set(labels)), 2)
        self.assertNotIn("Aisha", r.sanitized_text)

    def test_asr_variants_link_to_one_pseudonym(self):
        r = run(
            "PARENT: Sameera is here. Samira went out. samira hassan is nine.",
            known={"CHILD_NAME": ["Samira Hasan"]},
        )
        self.assertEqual(r.sanitized_text.count("<CHILD_01>"), 3)

    def test_quasi_combination_is_generalised(self):
        r = run(
            "CHILD: I'm nine and I play for the under-10 Tigers.\n"
            "PARENT: dad works at the bakery in Westbrook."
        )
        self.assertNotIn("nine", r.sanitized_text)
        self.assertTrue(any("AGE_" in s.replacement for s in r.spans))
        self.assertGreater(r.stats["quasi"]["risk_score"], 0)

    def test_qa_blocks_on_leak(self):
        cfg = Config.load()
        cfg.raw["known_values"] = {"CHILD_NAME": ["Aisha"]}
        # disable every detector that could find the name, so it survives
        for spec in cfg.raw["detectors"]:
            spec["enabled"] = spec["name"] == "rules"
        r = Pipeline(cfg).process_text("PARENT: Aisha is here", doc_id="t")
        self.assertEqual(r.status, "blocked")
        self.assertTrue(any(f.check == "known_value_leak" for f in r.qa))

    def test_report_does_not_contain_the_pii(self):
        r = run("CHILD: I'm Aisha.", known={"CHILD_NAME": ["Aisha"]})
        blob = r.to_json()
        self.assertNotIn("Aisha", blob)

    def test_mapping_withheld_unless_requested(self):
        r = run("CHILD: I'm Aisha.", known={"CHILD_NAME": ["Aisha"]})
        self.assertEqual(r.pseudonyms, {})
        r2 = run(
            "CHILD: I'm Aisha.", known={"CHILD_NAME": ["Aisha"]}, surrogates__keep_mapping=True
        )
        self.assertTrue(r2.pseudonyms)

    def test_metadata_is_stripped(self):
        payload = json.dumps(
            {
                "audio_file": "/recordings/hasan_family.wav",
                "device_id": "MBP-8821",
                "segments": [{"speaker": "CHILD", "text": "hello"}],
            }
        )
        r = run(payload)
        self.assertNotIn("hasan_family", json.dumps(r.stats))

    def test_audio_plan_from_word_timings(self):
        payload = json.dumps(
            {
                "segments": [
                    {
                        "speaker": "CHILD",
                        "text": "call miss rahman now",
                        "words": [
                            {"word": "call", "start": 1.0, "end": 1.2},
                            {"word": "miss", "start": 1.3, "end": 1.5},
                            {"word": "rahman", "start": 1.6, "end": 2.0},
                            {"word": "now", "start": 2.1, "end": 2.3},
                        ],
                    }
                ]
            }
        )
        r = run(payload)
        self.assertTrue(r.audio_plan)
        iv = r.audio_plan[0]
        self.assertLessEqual(iv["start"], 1.3)
        self.assertGreaterEqual(iv["end"], 2.0)

    def test_turn_scoped_pseudonyms_do_not_link(self):
        r = run(
            "CHILD: I'm Aisha.\nPARENT: Aisha, come here.",
            known={"CHILD_NAME": ["Aisha"]},
            surrogates__scope="per_turn",
        )
        self.assertIn("<CHILD_02>", r.sanitized_text)

    def test_self_introduction_links_to_the_speaker(self):
        r = run("CHILD: my name is Aisha.\nPARENT: Aisha, come here.")
        self.assertEqual(r.sanitized_turns[0]["speaker"], "<CHILD_01>")
        self.assertEqual(r.sanitized_text.count("<CHILD_01>"), 2)

    def test_generic_mention_links_to_the_typed_one(self):
        r = run("CHILD: I'm Samira.\nPARENT: Samira, come here.", known={"CHILD_NAME": ["Samira"]})
        self.assertNotIn("<PERSON_01>", r.sanitized_text)

    def test_accented_names_are_detected(self):
        r = run("CHILD: je m'appelle Aicha et Aïcha est ma soeur.")
        self.assertNotIn("Aïcha", r.sanitized_text)

    def test_common_nouns_survive_sentence_initial_capitals(self):
        r = run("CHILD: Swimming was fun. Homework is hard. Sameera helped me.")
        self.assertIn("Swimming", r.sanitized_text)
        self.assertIn("Homework", r.sanitized_text)
        self.assertNotIn("Sameera", r.sanitized_text)

    def test_empty_input_is_safe(self):
        r = run("")
        self.assertEqual(r.sanitized_text.strip(), "")
        self.assertEqual(r.spans, [])


class TestQuasi(unittest.TestCase):
    def test_age_bucket(self):
        self.assertEqual(age_bucket(9, 3), "<AGE_9_11>")
        self.assertEqual(age_bucket(11, 3), "<AGE_9_11>")
        self.assertEqual(age_bucket(12, 3), "<AGE_12_14>")


class TestMetrics(unittest.TestCase):
    def test_conversation_leakage_is_per_document(self):
        pairs = [
            ("a", [(0, 5, "CHILD_NAME")], [(0, 5, "CHILD_NAME")]),
            ("b", [(0, 5, "CHILD_NAME")], []),
        ]
        ev = evaluate(pairs, tier_of={"CHILD_NAME": "direct"})
        self.assertEqual(ev.conversation_leakage_rate(), 0.5)

    def test_f2_weights_recall(self):
        c, _, _ = match([(0, 3, "X"), (5, 8, "X")], [(0, 3, "X")])
        self.assertGreater(c.f(1.0), c.f(2.0))  # recall is the weak side here

    def test_pii_token_wer(self):
        out = pii_token_wer(["my", "name", "is", "nafisa"], ["my", "name", "is", "nafeesa"], [3])
        self.assertEqual(out["pii_token_wer"], 1.0)
        self.assertLess(out["wer"], 1.0)


class TestSpokenIdentifiers(unittest.TestCase):
    """Speech renders identifiers as words; the written patterns never see them."""

    def setUp(self):
        self.pipeline = Pipeline(Config.load())

    def _entities(self, text):
        return {s.entity for s in self.pipeline.process_text(text).spans}

    def test_spoken_date_of_birth(self):
        ents = self._entities("PARENT: her date of birth is the twelfth of march twenty sixteen")
        self.assertIn("DATE_OF_BIRTH", ents)

    def test_spoken_street_address(self):
        ents = self._entities("PARENT: we live at twenty one oak street")
        self.assertIn("STREET_ADDRESS", ents)

    def test_spelled_postcode(self):
        ents = self._entities("PARENT: our post code is l s six two q t")
        self.assertIn("POSTCODE", ents)

    def test_spelled_pupil_number(self):
        ents = self._entities(
            "PARENT: her pupil number is g p twenty nineteen forty four seventy one"
        )
        self.assertIn("STUDENT_ID", ents)

    def test_spoken_username_and_handle(self):
        self.assertIn("ONLINE_USERNAME", self._entities("PARENT: the login is farhana dot hasan"))
        self.assertIn(
            "SOCIAL_HANDLE", self._entities("CHILD: my account is at aisha underscore draws")
        )

    def test_lower_cased_place_and_club(self):
        self.assertIn("CITY", self._entities("CHILD: we live in leeds now"))
        self.assertIn("LOCAL_LANDMARK", self._entities("CHILD: nana lives near oakwood park"))
        self.assertIn("CLUB_NAME", self._entities("PARENT: she trains at riverside judo club"))

    def test_ordinary_speech_is_not_an_identifier(self):
        for text in (
            "PARENT: we tried the sticker chart and it worked for a fortnight.",
            "PARENT: he gets frustrated when the tablet runs out of battery.",
            "CHILD: Sometimes my tummy hurts after running.",
            "PARENT: we live in a flat and it is fine.",
        ):
            self.assertEqual(self.pipeline.process_text(text).spans, [], text)

    def test_a_real_workplace_still_survives_the_filter(self):
        self.assertIn(
            "PARENT_WORKPLACE", self._entities("PARENT: my husband owns the bakery on Mill Lane.")
        )
        self.assertIn("PARENT_WORKPLACE", self._entities("PARENT: she works at Patel Motors."))


class TestTurnIndex(unittest.TestCase):
    """Document.turn_at is indexed; the index must not go stale."""

    def _doc(self):
        from kryptos.io_formats import load_text

        return load_text(
            "PARENT: one two three\nCHILD: four five six\nPARENT: seven eight nine", fmt="text"
        )

    def test_lookup_matches_a_linear_scan(self):
        doc = self._doc()
        for i in range(len(doc.text)):
            found = doc.turn_at(i)
            expected = next((t for t in doc.turns if t.char_start <= i <= t.char_end), None)
            self.assertIs(found, expected, f"char {i}")

    def test_index_survives_in_place_mutation(self):
        doc = self._doc()
        self.assertIsNotNone(doc.turn_at(0))  # populates the index
        for turn in doc.turns:  # shift every turn
            turn.char_start += 100
            turn.char_end += 100
        found = doc.turn_at(doc.turns[1].char_start)
        self.assertIs(found, doc.turns[1])


class TestUnicodeIntegrity(unittest.TestCase):
    """Text arrives in whatever normalisation form its source produced."""

    def setUp(self):
        self.pipeline = Pipeline(Config.load())

    def test_decomposed_input_masks_the_whole_name(self):
        import unicodedata

        name = "PARENT: this is Zo\u00eb Ekstr\u00f6m from Krak\u00f3w"
        nfc = self.pipeline.process_text(unicodedata.normalize("NFC", name))
        nfd = self.pipeline.process_text(unicodedata.normalize("NFD", name))
        self.assertEqual(nfc.sanitized_text, nfd.sanitized_text)
        for fragment in ("Ekstr", "\u0308m", "Krak"):
            self.assertNotIn(fragment, nfd.sanitized_text)

    def test_zero_width_characters_do_not_hide_a_name(self):
        hidden = "PARENT: my name is Ais\u200bha Has\u200ban and my number is 07911 123456"
        out = self.pipeline.process_text(hidden).sanitized_text
        self.assertNotIn("Has", out)
        self.assertNotIn("ha ", out.replace("<", " <"))

    def test_output_stays_valid_across_scripts(self):
        import unicodedata

        for text in (
            "PARENT: \u0645\u0631\u062d\u0628\u0627 07911 123456",
            "PARENT: \u5a18\u306f\u7530\u4e2d\u3055\u304f\u3089\u3067\u3059",
            "CHILD: family \U0001f468\u200d\U0001f469\u200d\U0001f467 and Aisha Hasan",
        ):
            out = self.pipeline.process_text(text).sanitized_text
            self.assertTrue(unicodedata.is_normalized("NFC", out), text)
            self.assertFalse(any(unicodedata.category(c) in ("Mn", "Mc") for c in out[:1]), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
