"""Fast tests for paper-style preprocessing and submission formatting."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from config import PreprocessConfig
from src.preprocessing import ensure_nltk_resources, normalize_text
from src.submission import write_submission


class PreprocessingTests(unittest.TestCase):
    """Verify the reproducible approximation of the paper preprocessing."""

    @classmethod
    def setUpClass(cls) -> None:
        """Download the small NLTK corpora once before preprocessing tests."""

        ensure_nltk_resources(download_missing=True)

    def test_normalization_removes_published_noise(self) -> None:
        """Case, HTML, URL, hashtag, mention, emoji and punctuation are removed."""

        text = "HeLLo #Kannada!!! @person https://example.com <br> 😊 CARS"
        cleaned = normalize_text(text, PreprocessConfig())
        self.assertEqual(cleaned, "hello car")

    def test_stopwords_protected_context_and_kannada_are_handled(self) -> None:
        """Remove configured stopwords but retain paper examples and Kannada text."""

        text = "The mattu ಮತ್ತು Devru not ಕನ್ನಡ"
        cleaned = normalize_text(text, PreprocessConfig())
        self.assertEqual(cleaned, "devru not ಕನ್ನಡ")

    def test_repeated_characters_and_words_are_normalized(self) -> None:
        """Limit long character runs and collapse adjacent repeated tokens."""

        cleaned = normalize_text("soooo goood goood!!!!!", PreprocessConfig())
        self.assertEqual(cleaned, "soo good")


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
