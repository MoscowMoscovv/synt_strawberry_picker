import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from batch_generate_strawberries import external_main, parse_args


class BatchGenerationTests(unittest.TestCase):
    def test_face_limit_is_forwarded_to_blender(self):
        with tempfile.TemporaryDirectory() as temporary:
            blend = Path(temporary) / "source.blend"
            blend.touch()
            args = parse_args(["--blend", str(blend), "--count", "10",
                               "--max-faces", "10000"])
            with patch("batch_generate_strawberries.subprocess.run") as run:
                external_main(args)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--max-faces") + 1], "10000")

    def test_full_detail_is_preserved_without_face_limit(self):
        self.assertIsNone(parse_args(["--count", "1"]).max_faces)

    def test_face_limit_rejects_degenerate_triangle_budget(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--count", "1", "--max-faces", "3"])


if __name__ == "__main__":
    unittest.main()
