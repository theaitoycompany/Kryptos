"""Build artifacts must not change the identity of evaluated source code."""

import tempfile
import unittest
from pathlib import Path

from benchmarks.run import source_digest


class BenchmarkProvenanceTests(unittest.TestCase):
    def test_hash_ignores_generated_metadata_and_detects_source_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src" / "kryptos"
            source.mkdir(parents=True)
            module = source / "example.py"
            module.write_text("value = 1\n")
            original = source_digest(root)
            metadata = root / "src" / "kryptos_pii.egg-info"
            metadata.mkdir()
            (metadata / "SOURCES.txt").write_text("generated build inventory\n")
            self.assertEqual(source_digest(root), original)
            module.write_text("value = 2\n")
            self.assertNotEqual(source_digest(root), original)
