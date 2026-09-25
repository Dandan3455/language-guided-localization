import tempfile
import unittest
from pathlib import Path

from src.run_paths import create_run_directory


class RunPathsTests(unittest.TestCase):
    def test_reusing_name_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as base:
            run = create_run_directory(base, "desk_left")
            result = run / "results.json"
            result.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                create_run_directory(base, "desk_left")
            self.assertEqual(result.read_text(encoding="utf-8"), "original")
            self.assertNotEqual(run, create_run_directory(base, "desk_right"))

    def test_automatic_runs_are_distinct(self):
        with tempfile.TemporaryDirectory() as base:
            self.assertNotEqual(create_run_directory(base), create_run_directory(base))

    def test_rejects_paths_and_windows_device_names(self):
        with tempfile.TemporaryDirectory() as base:
            for name in ("../escape", "a/b", "a\\b", "", "CON", "NUL", "a."):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    create_run_directory(base, name)
            self.assertEqual(list(Path(base).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
