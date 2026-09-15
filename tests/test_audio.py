"""Execute emitted audio commands against a generated, non-speech waveform."""

import array
import math
import shlex
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from kryptos.audio import ffmpeg_command


class AudioTests(unittest.TestCase):
    def test_invalid_intervals_and_modes_are_rejected(self):
        for start, end in [(1, 0), (-1, 2), (0, float("nan")), (0, float("inf")), (0, 0)]:
            with self.assertRaises(ValueError):
                ffmpeg_command("in.wav", "out.wav", [{"start": start, "end": end}])
        with self.assertRaises(ValueError):
            ffmpeg_command("same.wav", "same.wav", [])
        with self.assertRaises(ValueError):
            ffmpeg_command("in.wav", "out.wav", [], mode="invalid")

    @unittest.skipUnless(
        shutil.which("ffmpeg"), "ffmpeg is required for waveform integration checks"
    )
    def test_mute_and_beep_change_only_the_selected_interval(self):
        rate = 48000
        samples = array.array(
            "h", [int(4000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(rate * 3)]
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source with ' quote.wav"
            with wave.open(str(source), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(rate)
                stream.writeframes(samples.tobytes())
            for mode in ("mute", "beep"):
                target = Path(directory) / (mode + ".wav")
                command = ffmpeg_command(
                    str(source), str(target), [{"start": 1, "end": 2}], mode=mode
                )
                subprocess.run(shlex.split(command), check=True, capture_output=True, timeout=20)
                with wave.open(str(target), "rb") as stream:
                    result = array.array("h", stream.readframes(stream.getnframes()))
                self.assertEqual(len(result), len(samples))
                for offset in (rate // 4, rate * 9 // 4):
                    self.assertLessEqual(
                        max(
                            abs(a - b)
                            for a, b in zip(
                                result[offset : offset + 1000], samples[offset : offset + 1000]
                            )
                        ),
                        2,
                    )
                middle = result[rate * 3 // 2 : rate * 3 // 2 + 1000]
                self.assertEqual(max(map(abs, middle)) == 0, mode == "mute")
