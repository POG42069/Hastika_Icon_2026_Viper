"""Run a dependency-light leakage-safe sanity check on Task A data variants.

This is not a substitute for the Qwen LoRA experiment.  It trains a simple
character n-gram multinomial Naive Bayes model on folds 1-4 and evaluates only
original fold 5, allowing obvious augmentation regressions to be detected
without CUDA or model downloads.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd


LABELS = ("Non-Hate", "Hate")


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip().lower()


def features(value: object) -> Counter[str]:
    text = f" {normalize(value)} "
    output: Counter[str] = Counter()
    for size in (3, 4, 5):
        output.update(
            text[index : index + size]
            for index in range(len(text) - size + 1)
        )
    return output


class CharacterNaiveBayes:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_documents: Counter[str] = Counter()
        self.class_features = {label: Counter() for label in LABELS}
        self.class_feature_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, frame: pd.DataFrame) -> None:
        for comment, label in zip(frame["Comment"], frame["Label"], strict=True):
            row_features = features(comment)
            self.class_documents[label] += 1
            self.class_features[label].update(row_features)
            self.class_feature_totals[label] += sum(row_features.values())
            self.vocabulary.update(row_features)

    def predict_one(self, comment: object) -> str:
        row_features = features(comment)
        total_documents = sum(self.class_documents.values())
        vocabulary_size = max(1, len(self.vocabulary))
        scores = {}
        for label in LABELS:
            score = math.log(self.class_documents[label] / total_documents)
            denominator = (
                self.class_feature_totals[label] + self.alpha * vocabulary_size
            )
            counts = self.class_features[label]
            for token, count in row_features.items():
                score += count * math.log(
                    (counts[token] + self.alpha) / denominator
                )
            scores[label] = score
        return max(LABELS, key=lambda label: scores[label])

    def predict(self, comments: pd.Series) -> list[str]:
        return [self.predict_one(comment) for comment in comments]


def metrics(actual: list[str], predicted: list[str]) -> dict[str, object]:
    per_class = {}
    matrix = {actual_label: Counter() for actual_label in LABELS}
    for truth, guess in zip(actual, predicted, strict=True):
        matrix[truth][guess] += 1
    for label in LABELS:
        true_positive = matrix[label][label]
        false_positive = sum(
            matrix[other][label] for other in LABELS if other != label
        )
        false_negative = sum(
            matrix[label][other] for other in LABELS if other != label
        )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }
    accuracy = sum(matrix[label][label] for label in LABELS) / len(actual)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(per_class[label]["f1"] for label in LABELS) / len(LABELS),
        "per_class": per_class,
        "confusion_matrix": {
            truth: {guess: matrix[truth][guess] for guess in LABELS}
            for truth in LABELS
        },
    }


def run(data_dir: Path) -> dict[str, object]:
    originals = pd.concat(
        [
            pd.read_csv(data_dir / f"binary_original_{fold}.csv").assign(fold=fold)
            for fold in range(1, 6)
        ],
        ignore_index=True,
        sort=False,
    )
    augments = pd.concat(
        [pd.read_csv(data_dir / f"binary_augment{fold}.csv") for fold in range(1, 6)],
        ignore_index=True,
        sort=False,
    )
    train_originals = originals.loc[originals["fold"].ne(5)].copy()
    validation = originals.loc[originals["fold"].eq(5)].copy()
    train_hate_augments = augments.loc[
        augments["fold"].ne(5) & augments["Label"].eq("Hate")
    ].copy()
    train_all_augments = augments.loc[augments["fold"].ne(5)].copy()
    variants = {
        "original_only": train_originals,
        "hate_augmented": pd.concat(
            [train_originals, train_hate_augments], ignore_index=True, sort=False
        ),
        "balanced_augmented": pd.concat(
            [train_originals, train_all_augments], ignore_index=True, sort=False
        ),
    }
    result = {
        "model": "character_3_to_5_gram_multinomial_naive_bayes",
        "purpose": "data sanity check only; not a replacement for Qwen LoRA",
        "validation": "original fold 5 only",
        "validation_rows": len(validation),
        "variants": {},
    }
    for name, train in variants.items():
        model = CharacterNaiveBayes(alpha=1.0)
        model.fit(train)
        predictions = model.predict(validation["Comment"])
        result["variants"][name] = {
            "training_rows": len(train),
            "training_label_counts": train["Label"].value_counts().to_dict(),
            **metrics(validation["Label"].tolist(), predictions),
        }
    output_path = data_dir / "sanity_experiment_results.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root.parent / "task_a_balanced",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments.data_dir.resolve()), ensure_ascii=False, indent=2))
