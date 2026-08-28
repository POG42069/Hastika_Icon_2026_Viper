"""Fast tests for preprocessing and organizer submission formatting."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from config import PreprocessConfig
from src.preprocessing import normalize_text
from src.submission import write_submission


class PreprocessingTests(unittest.TestCase):
    """Verify conservative text cleaning behavior."""

    def test_normalization_preserves_case_and_hashtag_content(self) -> None:
        """MuRIL-cased input should keep case and hashtag words."""

        text = "HeLLo #Kannada!!! @person https://example.com <br> 😊"
        cleaned = normalize_text(text, PreprocessConfig())
        self.assertIn("HeLLo", cleaned)
        self.assertIn("Kannada", cleaned)
        self.assertNotIn("#", cleaned)
        self.assertIn("USER", cleaned)
        self.assertIn("URL", cleaned)
        self.assertNotIn("😊", cleaned)

    def test_all_emoji_sequences_are_removed(self) -> None:
        """Flags, skin tones, keycaps, and joined emoji must not survive."""

        cleaned = normalize_text(
            "before 😂 👨‍👩‍👧‍👦 👍🏽 🇻🇳 1️⃣ after",
            PreprocessConfig(),
        )
        self.assertEqual(cleaned, "before after")

    def test_hashtag_inside_mention_is_normalized(self) -> None:
        """Removing a hashtag marker must not expose an untreated mention."""

        cleaned = normalize_text("hello @#ps", PreprocessConfig())
        self.assertEqual(cleaned, "hello USER")

    def test_repeated_characters_are_limited(self) -> None:
        """Three or more repeated characters should be shortened to two."""

        cleaned = normalize_text("soooo goood!!!!!", PreprocessConfig())
        self.assertEqual(cleaned, "soo good!!")


class SubmissionTests(unittest.TestCase):
    """Verify exact CSV and ZIP structure required by the organizer."""

    def test_submission_contains_only_predictions_csv(self) -> None:
        """The ZIP root must contain one id,label CSV in original id order."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            csv_path, zip_path = write_submission(
                ids=["001", "958"],
                labels=["Non-Hate", "Hate"],
                allowed_labels=("Non-Hate", "Hate"),
                output_dir=output_dir,
                zip_name="task_a_submission.zip",
            )
            frame = pd.read_csv(csv_path, dtype={"id": str})
            self.assertEqual(frame.columns.tolist(), ["id", "label"])
            self.assertEqual(frame["id"].tolist(), ["001", "958"])
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["predictions.csv"])


if __name__ == "__main__":
    unittest.main()
