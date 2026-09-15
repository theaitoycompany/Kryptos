import random
import unittest
from unittest.mock import patch

import kryptos.normalize as normal


class MatchingAcceleration(unittest.TestCase):
    def test_native_distance_preserves_fallback_results(self):
        if normal._native_levenshtein is None:
            self.skipTest("Optional native matcher is not installed")
        randomizer = random.Random(7823)
        pairs = [("nafeesa", "nafisa"), ("জয়া", "জিয়া"), ("", "example"), ("same", "same")]
        pairs += [
            (
                "".join(randomizer.choices("abcéñ", k=randomizer.randrange(1, 15))),
                "".join(randomizer.choices("abcéñ", k=randomizer.randrange(1, 15))),
            )
            for _ in range(100)
        ]
        for a, b in pairs:
            for cap in (None, 0, 1, 3):
                actual = normal.levenshtein(a, b, cap)
                with patch.object(normal, "_native_levenshtein", None):
                    expected = normal.levenshtein(a, b, cap)
                # A cutoff promises exact distances only up to the cutoff;
                # the fallback may return either cap+1 or a larger distance.
                self.assertEqual(
                    actual if cap is None else min(actual, cap + 1),
                    expected if cap is None else min(expected, cap + 1),
                )

    def test_cached_matching_keeps_asr_and_exact_channels(self):
        for _ in range(2):
            self.assertEqual(normal.fuzzy_match("Aisha", "aisha")[2], "exact")
            self.assertTrue(normal.fuzzy_match("Nafisa", "nafeesa")[0])
            self.assertFalse(normal.fuzzy_match("Qureshi", "dragon")[0])
