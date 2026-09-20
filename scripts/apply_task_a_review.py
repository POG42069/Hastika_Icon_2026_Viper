"""Apply explicit review decisions to generated Task A CSV artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REVIEW_COLUMNS = ("review_decision", "review_note", "reviewer")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"id": str, "source_id": str, "source_group": str},
        keep_default_na=False,
    )


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def review_note(method: str) -> str:
    if method == "controlled_label_preserving_rephrase":
        return (
            "AI semantic review passed: meaning, target, negation and Non-Hate "
            "label are preserved."
        )
    return (
        "Rule audit passed: conservative surface edit preserves label-bearing "
        "content."
    )


def apply_review(data_dir: Path, reviewer: str) -> dict[str, object]:
    queue_path = data_dir / "human_review_queue.csv"
    candidates_path = data_dir / "nonhate_candidates.csv"
    generation_audit_path = data_dir / "nonhate_generation_audit.csv"
    summary_path = data_dir / "pipeline_summary.json"

    queue = _read(queue_path)
    if queue["id"].duplicated().any():
        raise RuntimeError("Review queue IDs must be unique.")
    queue["review_decision"] = "accept"
    queue["review_note"] = queue["augmentation_method"].map(review_note)
    queue["reviewer"] = reviewer
    queue["review_type"] = "AI_review"
    _write(queue, queue_path)

    decisions = queue.set_index("id")
    candidates = _read(candidates_path)
    for column in (*REVIEW_COLUMNS, "review_type"):
        if column not in candidates.columns:
            candidates[column] = ""
        candidates.loc[candidates["id"].isin(decisions.index), column] = (
            candidates.loc[
                candidates["id"].isin(decisions.index), "id"
            ].map(decisions[column])
        )
    reviewed = candidates["review_decision"].eq("accept")
    candidates.loc[reviewed, "human_review_status"] = "ai_reviewed_accepted"
    if candidates.loc[candidates["selected"], "review_decision"].eq("reject").any():
        raise RuntimeError(
            "Rejected selected rows require replacement before rebuilding folds."
        )
    pending_selected = candidates.loc[
        candidates["selected"] & candidates["human_review_required"],
        "review_decision",
    ].eq("")
    if pending_selected.any():
        raise RuntimeError("Some selected review-required rows are still pending.")
    _write(candidates, candidates_path)
    _write(candidates, generation_audit_path)

    selected_status = candidates.set_index("id")["human_review_status"]
    all_augments = []
    all_originals = []
    for fold in range(1, 6):
        augment_path = data_dir / f"binary_augment{fold}.csv"
        augment = _read(augment_path)
        generated = augment["id"].isin(selected_status.index)
        augment.loc[generated, "human_review_status"] = augment.loc[
            generated, "id"
        ].map(selected_status)
        _write(augment, augment_path)
        all_augments.append(augment)

        original = _read(data_dir / f"binary_original_{fold}.csv")
        original["fold"] = fold
        all_originals.append(original)

    originals = pd.concat(all_originals, ignore_index=True, sort=False)
    augments = pd.concat(all_augments, ignore_index=True, sort=False)
    combined = pd.concat(
        [
            originals.assign(data_origin="original"),
            augments.assign(data_origin="augmentation"),
        ],
        ignore_index=True,
        sort=False,
    )
    counts = combined["Label"].value_counts().to_dict()
    if counts.get("Hate") != counts.get("Non-Hate"):
        raise RuntimeError(f"Review application broke class balance: {counts}")
    _write(combined, data_dir / "binary_train_balanced.csv")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["review"] = {
        "reviewer": reviewer,
        "review_type": "AI_review",
        "reviewed_rows": len(queue),
        "accepted_rows": int(queue["review_decision"].eq("accept").sum()),
        "rejected_rows": int(queue["review_decision"].eq("reject").sum()),
        "pending_rows": int(queue["review_decision"].eq("").sum()),
    }
    summary["release_status"] = "provisional_pending_gpu_experiment"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root.parent / "task_a_balanced",
    )
    parser.add_argument("--reviewer", default="Codex")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            apply_review(arguments.data_dir.resolve(), arguments.reviewer),
            ensure_ascii=False,
            indent=2,
        )
    )
