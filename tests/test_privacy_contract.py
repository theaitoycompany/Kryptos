"""Publication gates using invented identifiers and injected detector failures."""

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kryptos import Config, Pipeline
from kryptos.detectors.base import Detector, DetectorUnavailable, build_detectors, run_detectors
from kryptos.evaluation import evaluate, pii_token_wer
from kryptos.io_formats import load_file
from kryptos.qa import _check_known_values, _inside
from kryptos.types import Detection, Document


class Broken(Detector):
    def detect(self, doc):
        raise ValueError("secret fixture must never appear in error messages")


class NonFinite(Detector):
    def detect(self, doc):
        return [Detection(0, 2, "PERSON_NAME", math.nan, "synthetic")]


class PrivacyContract(unittest.TestCase):
    def test_default_requires_every_enabled_detector(self):
        cfg = Config.load()
        with self.assertRaisesRegex(RuntimeError, "Required detector failed") as error:
            run_detectors([Broken({"name": "fixture"}, cfg)], Document("test"))
        self.assertNotIn("secret fixture", str(error.exception))
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            run_detectors([NonFinite({}, cfg)], Document("test"))
        cfg.raw["detectors"] = [{"name": "fixture", "type": "does-not-exist"}]
        with self.assertRaises(DetectorUnavailable):
            build_detectors(cfg)

    def test_reports_drop_original_speakers_ids_and_uniqueness_snippets(self):
        payload = {
            "id": "private-record-123",
            "turns": [
                {
                    "speaker": "ZEFRINA",
                    "text": "I am the only Zefrina at this school.",
                    "words": [{"word": "Zefrina", "start": 1, "end": 2}],
                },
            ],
        }
        result = Pipeline().process_text(
            json.dumps(payload), known_values={"CHILD_NAME": ["Zefrina"]}
        )
        report = result.to_json()
        self.assertNotIn("Zefrina", report)
        self.assertNotIn("ZEFRINA", report)
        self.assertNotIn("private-record", report)
        self.assertNotIn("uniqueness_claims", report)

    def test_file_basename_does_not_become_document_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private-person.txt"
            path.write_text("hello", encoding="utf-8")
            self.assertEqual(load_file(str(path)).doc_id, "")

    def test_known_value_rescan_does_not_mutate_config(self):
        cfg = Config.load(overrides={"known_values": {"CHILD_NAME": "Ava"}})
        doc = Document("Ava and Mina", meta={"known_values": {"CHILD_NAME": ["Mina"]}})
        for _ in range(2):
            self.assertEqual(len(_check_known_values(doc.text, doc, cfg)), 2)
        self.assertEqual(cfg.known_values, {"CHILD_NAME": "Ava"})

    def test_disabled_quality_checks_never_pass(self):
        for overrides in (
            {"qa": {"enabled": False}},
            {"qa": {"rescan": False}},
            {"runtime": {"strict_detectors": False}},
        ):
            result = Pipeline(Config.load(overrides=overrides)).process_text("hello")
            self.assertEqual(result.status, "review")

    def test_non_latin_script_requires_review(self):
        result = Pipeline().process_text("hello नमस्ते")
        self.assertNotEqual(result.status, "passed")
        self.assertIn("language_review", {f.check for f in result.qa})

    def test_lowercase_asr_requires_review(self):
        result = Pipeline().process_text(
            "we went to the park and then we came back to play with our toys"
        )
        self.assertEqual(result.status, "review")
        self.assertIn("uncased_transcript_review", {f.check for f in result.qa})

    def test_short_known_names_are_checked(self):
        config = Config.load(overrides={"known_values": {"CHILD_NAME": ["Li"]}})
        self.assertEqual(len(_check_known_values("Li is here", Document("Li is here"), config)), 1)

    def test_hmac_respects_conversation_scope(self):
        cfg = Config.load(overrides={"surrogates": {"style": "hmac"}})
        with patch.dict(os.environ, {"KRYPTOS_HMAC_KEY": "synthetic-test-key-for-tests-only-32"}):
            pipeline = Pipeline(cfg)
            a = pipeline.process_text("My name is Aisha.", doc_id="synthetic-a")
            same = pipeline.process_text("My name is Aisha.", doc_id="synthetic-a")
            b = pipeline.process_text("My name is Aisha.", doc_id="synthetic-b")
            self.assertEqual(a.sanitized_text, same.sanitized_text)
            self.assertNotEqual(a.sanitized_text, b.sanitized_text)
            with self.assertRaisesRegex(RuntimeError, "opaque document ID"):
                pipeline.process_text("My name is Aisha.")

    def test_partial_placeholder_overlap_is_not_ignored(self):
        self.assertFalse(_inside(4, 15, [(0, 10)]))
        self.assertTrue(_inside(2, 8, [(0, 10)]))

    def test_partial_identifier_removal_counts_as_a_miss(self):
        tier = {"CHILD_NAME": "direct"}
        for predicted, expected in [
            ([(0, 4, "PERSON_NAME")], 1),
            ([(0, 4, "PERSON_NAME"), (4, 10, "PERSON_NAME")], 0),
            ([(0, 4, "PERSON_NAME"), (5, 10, "PERSON_NAME")], 1),
        ]:
            result = evaluate([("synthetic", [(0, 10, "CHILD_NAME")], predicted)], tier)
            self.assertEqual(result.leaking_documents, expected)

    def test_cli_withholds_blocked_text_and_creates_private_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "detectors": [{"name": "rules", "type": "regex"}],
                        "known_values": {"CHILD_NAME": ["Zefrina"]},
                    }
                ),
                encoding="utf-8",
            )
            output = Path(directory) / "result.json"
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kryptos",
                    "run",
                    "-",
                    "-c",
                    str(config),
                    "-o",
                    str(output),
                    "--allow-review",
                ],
                input="Zefrina is here",
                text=True,
                capture_output=True,
            )
            self.assertEqual(run.returncode, 20, run.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["sanitized_text"], "")
            self.assertTrue(report["text_withheld"])
            self.assertNotIn("Zefrina", output.read_text())
            if os.name == "posix":
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_config_resources_exist_after_install(self):
        cfg = Config.load()
        self.assertGreater(len(cfg.taxonomy.entities), 40)
        context = next(d for d in Pipeline(cfg).detectors if d.name == "context")
        self.assertIn("aisha", context.first_names)

    def test_word_error_rate_counts_insertions(self):
        score = pii_token_wer(["Aisha"], ["hello", "Aisha", "Hasan"], [0])
        self.assertEqual(score["wer"], 2)
        self.assertEqual(score["pii_token_wer"], 2)

    def test_cross_turn_identifier_is_fully_removed(self):
        class AcrossTurns(Detector):
            def detect(self, doc):
                if doc.text == "Aisha\nHasan":
                    return [self.make(doc, 0, len(doc.text), "CHILD_NAME", 1.0)]
                return []

        cfg = Config.load()
        pipeline = Pipeline(cfg, detectors=[AcrossTurns({"name": "synthetic"}, cfg)])
        result = pipeline.process_text("CHILD: Aisha\nCHILD: Hasan")
        self.assertNotIn("Aisha", result.sanitized_text)
        self.assertNotIn("Hasan", result.sanitized_text)
        self.assertEqual(len(result.spans), 2)

    def test_unaligned_document_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "aligned turns"):
            Pipeline().process(Document("hello"))

    def test_simple_json_preserves_language_and_known_identifiers(self):
        payload = json.dumps(
            {
                "text": "Bonjour Zefrina",
                "language": "fr",
                "known_values": {"CHILD_NAME": ["Zefrina"]},
            }
        )
        result = Pipeline().process_text(payload)
        self.assertNotIn("Zefrina", result.sanitized_text)
        self.assertIn("language_review", {finding.check for finding in result.qa})

    def test_unknown_detector_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not enabled"):
            build_detectors(Config.load(), only=["rules", "does-not-exist"])
        run = subprocess.run(
            [sys.executable, "-m", "kryptos", "run", "-", "--enable", "does-not-exist"],
            input="hello",
            text=True,
            capture_output=True,
        )
        self.assertEqual(run.returncode, 2)
        self.assertEqual(run.stdout, "")
