from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from src.resume import prepare_resume_catalogs


class ResumeDiscoveryTests(unittest.TestCase):
    def test_discovers_prefixed_directory_result_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "dataset" / "Hastika_Icon_2026_Viper"
            output = project / "outputs" / "task_a" / "seed_42"
            checkpoint = project / "checkpoints" / "task_a" / "seed_42"
            output.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            (output / "seed_summary.json").write_text("{}", encoding="utf-8")
            catalogs = prepare_resume_catalogs(
                [root / "dataset"], root / "scratch", "task_a", "task_a"
            )
            self.assertEqual(len(catalogs), 1)
            self.assertEqual(catalogs[0].output_dir, output.parent.resolve())
            self.assertEqual(catalogs[0].checkpoint_dir, checkpoint.parent.resolve())

    def test_extracts_only_requested_task_from_result_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "result-7.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(
                    "repo/outputs/task_a/seed_42/seed_summary.json", '{"seed": 42}'
                )
                output.writestr(
                    "repo/checkpoints/task_a/seed_42/best_holdout_adapter/config.json",
                    "{}",
                )
                output.writestr(
                    "repo/outputs/task_b/seed_42/seed_summary.json", "ignored"
                )
                output.writestr("../../unsafe.txt", "ignored")
            scratch = root / "scratch"
            catalogs = prepare_resume_catalogs(
                [archive], scratch, "task_a", "task_a"
            )
            self.assertEqual(len(catalogs), 1)
            self.assertTrue(
                (catalogs[0].output_dir / "seed_42" / "seed_summary.json").is_file()
            )
            self.assertFalse((scratch / "unsafe.txt").exists())
            self.assertFalse(any(scratch.rglob("task_b")))
            self.assertFalse(any(scratch.rglob("best_holdout_adapter")))


if __name__ == "__main__":
    unittest.main()
