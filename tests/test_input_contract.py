"""Synthetic regressions for malformed input and privacy-sensitive output paths."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kryptos import Config, Detection, Document, Pipeline, Turn
from kryptos.detectors.base import Detector, run_detectors
from kryptos.io_formats import load_text


class InputContractTests(unittest.TestCase):
    def test_input_placeholders_cannot_bypass_review(self):
        result = Pipeline().process_text("Hello <ZAFRINA>.")
        self.assertNotEqual(result.status, "passed")
        self.assertIn("input_placeholder_review", {finding.check for finding in result.qa})

    def test_reversed_detector_offsets_and_invalid_weights_are_rejected(self):
        class Reversed(Detector):
            def detect(self, doc):
                return [Detection(4, 0, "CHILD_NAME", 1, self.name)]

        config = Config.load()
        with self.assertRaises(RuntimeError):
            run_detectors([Reversed({}, config)], Document("hello"))
        with self.assertRaises(ValueError):
            Detector({"weight": float("nan")}, config)

    def test_explicit_roles_are_honored_and_invalid_roles_are_rejected(self):
        payload = json.dumps(
            {"text": "SPEAKER_01: Hello", "speaker_roles": {"SPEAKER_01": "teacher"}}
        )
        result = Pipeline().process_text(payload)
        self.assertEqual(result.sanitized_turns[0]["role"], "teacher")
        with self.assertRaises(ValueError):
            Pipeline().process_text(
                "hello", speaker_roles={"SPEAKER_UNKNOWN": "synthetic-private-name"}
            )
        with self.assertRaises(ValueError):
            Pipeline().process_text("hello", known_values={"synthetic-private-name": ["hello"]})

    def test_invalid_json_segments_cannot_pass_as_empty_text(self):
        for payload in (
            {},
            {"segments": "hello"},
            {"segments": ["hello"]},
            {"segments": [{}]},
            {"text": 42},
            {"segments": [{"text": "hello", "words": [42]}]},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                Pipeline().process_text(json.dumps(payload))

    def test_bracketed_timestamp_is_plain_text(self):
        result = Pipeline().process_text("[00:01:12] CHILD: My name is Aisha.")
        self.assertNotIn("Aisha", result.sanitized_text)
        self.assertEqual(result.sanitized_turns[0]["start"], 72)

    def test_short_vtt_timestamps_entities_and_numeric_speech(self):
        doc = load_text(
            "WEBVTT\n\ncue-one\n00:01.000 --> 00:03.000 align:start\n<v CHILD>Aisha &amp; Hasan</v>\n\n00:04.000 --> 00:05.000\n12345\n"
        )
        self.assertEqual(doc.text, "Aisha & Hasan\n12345")
        self.assertEqual([(turn.start, turn.end) for turn in doc.turns], [(1, 3), (4, 5)])

    def test_invalid_subtitles_are_rejected(self):
        for text in (
            "WEBVTT\n\ninvalid cue",
            "1\n00:61:00,000 --> 00:62:00,000\nhello",
            "1\n00:00:03,000 --> 00:00:01,000\nhello",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_text(text, fmt="srt")

    def test_misaligned_document_is_rejected(self):
        for doc in (
            Document("Aisha", [Turn("CHILD", "hello", char_end=5)]),
            Document("hello", [Turn("CHILD", "hello", char_start=1, char_end=6)]),
        ):
            with self.assertRaisesRegex(ValueError, "aligned turns"):
                Pipeline().process(doc)

    def test_invalid_timings_cannot_enter_reports(self):
        for value in (float("nan"), float("inf"), -1, "invalid", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Pipeline().process_text(
                    json.dumps({"segments": [{"text": "hello", "start": value}]})
                )
        with self.assertRaises(ValueError):
            Pipeline().process_text('{"segments": [{"text":"hello", "start":5, "end":1}]}')

    def test_nfd_words_align_to_normalized_text(self):
        doc = load_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "text": "Ekstro\u0308m hello",
                            "words": [
                                {"word": "Ekstro\u0308m", "start": 0, "end": 1},
                                {"word": "hello", "start": 1, "end": 2},
                            ],
                        }
                    ]
                }
            )
        )
        for word, expected in zip(doc.turns[0].words, ["Ekström", "hello"]):
            self.assertEqual(doc.text[word.char_start : word.char_end], expected)

    def test_partial_audio_alignment_requires_review(self):
        result = Pipeline().process_text(
            json.dumps(
                {
                    "segments": [
                        {"text": "Aisha Hasan", "words": [{"word": "Aisha", "start": 0, "end": 1}]}
                    ]
                }
            ),
            known_values={"CHILD_NAME": ["Aisha Hasan"]},
        )
        self.assertNotEqual(result.status, "passed")
        self.assertIn("audio_alignment_review", {finding.check for finding in result.qa})

    def test_explicit_unavailable_detector_is_not_silently_dropped(self):
        config = Config.load()
        unavailable = Detector({"name": "unavailable"}, config)
        unavailable.available = False
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            Pipeline(config, detectors=[*Pipeline(config).detectors, unavailable])

    def test_per_turn_names_and_speakers_have_separate_scopes(self):
        config = Config.load(overrides={"surrogates": {"scope": "per_turn"}})
        result = Pipeline(config).process_text(
            "CHILD: My name is Alexandrina.\nCHILD: My name is Alexandrina.",
            known_values={"CHILD_NAME": ["Alexandrina"]},
        )
        first, second = result.sanitized_turns
        self.assertNotEqual(first["speaker"], second["speaker"])
        self.assertNotEqual(first["text"], second["text"])
        self.assertIn(first["speaker"], first["text"])
        self.assertIn(second["speaker"], second["text"])

    def test_evaluation_report_has_no_source_ids_and_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "gold.jsonl", Path(directory) / "report.json"
            source.write_text(
                json.dumps({"doc_id": "synthetic-private-identifier", "text": "hello", "spans": []})
                + "\n"
            )
            run = subprocess.run(
                [sys.executable, "-m", "kryptos", "eval", str(source), "-o", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertNotIn("synthetic-private-identifier", output.read_text())
            self.assertEqual(json.loads(output.read_text())["per_document"][0]["document_index"], 0)
            if os.name == "posix":
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_invalid_evaluation_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "gold.jsonl"
            for content in (
                "",
                '{"text":"hello"}\n',
                '{"text":"hello", "spans":[{"start":0,"end":9,"entity":"CHILD_NAME"}]}\n',
            ):
                source.write_text(content)
                run = subprocess.run(
                    [sys.executable, "-m", "kryptos", "eval", str(source)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(run.returncode, 2)
                self.assertEqual(run.stdout, "")
