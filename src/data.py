"""HASTIKA CSV loading, validation, and light text normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_config import DATA_DIR, OUTPUT_ROOT, TaskConfig
from src.preprocessing import normalize_text


def resolve_data_file(filename: str, configured_dir: Path = DATA_DIR) -> Path:
    """Find an input locally or in an attached Kaggle Dataset."""

    candidates = [
        configured_dir / filename,
        Path(__file__).resolve().parents[1] / "data" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        matches = sorted(kaggle_root.glob(f"**/{filename}"))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            rendered = "\n  - ".join(str(path) for path in matches)
            raise FileNotFoundError(
                f"Found multiple copies of {filename}. Set HASTIKA_DATA_DIR to "
                f"the intended directory:\n  - {rendered}"
            )

    raise FileNotFoundError(
        f"Could not find {filename}. Attach the private HASTIKA dataset on "
        "Kaggle or set the HASTIKA_DATA_DIR environment variable."
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"id": str, "source_id": str, "source_group": str},
    )


def _validate_columns(
    frame: pd.DataFrame,
    path: Path,
    required: set[str],
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def _validate_and_encode_frames(
    train_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    task: TaskConfig,
    train_description: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_train = {"id", "Comment", task.label_column}
    required_prediction = {"id", "Comment"}
    _validate_columns(train_df, Path(train_description), required_train)
    _validate_columns(
        prediction_df,
        Path(task.prediction_filename),
        required_prediction,
    )
    if train_df["id"].isna().any() or prediction_df["id"].isna().any():
        raise ValueError("Missing IDs were found; every row needs an ID.")
    if train_df["id"].duplicated().any():
        duplicates = train_df.loc[train_df["id"].duplicated(False), "id"].unique()
        raise ValueError(
            "Duplicate labeled IDs were found across the Task B fold files: "
            f"{duplicates[:10].tolist()}"
        )
    if prediction_df["id"].duplicated().any():
        raise ValueError("Duplicate prediction IDs were found; each ID must be unique.")

    unknown_labels = sorted(
        set(train_df[task.label_column].dropna()) - set(task.labels)
    )
    if unknown_labels:
        raise ValueError(f"Unexpected labels in {train_description}: {unknown_labels}")
    if train_df[task.label_column].isna().any():
        raise ValueError(f"Missing labels were found in {train_description}.")

    train_df = train_df.copy()
    prediction_df = prediction_df.copy()
    train_df["text"] = train_df["Comment"].map(normalize_text)
    prediction_df["text"] = prediction_df["Comment"].map(normalize_text)
    label_to_id = {label: index for index, label in enumerate(task.labels)}
    train_df["label_id"] = train_df[task.label_column].map(label_to_id).astype(int)
    return train_df, prediction_df


def _normalise_for_audit(values: pd.Series) -> pd.Series:
    """Create a conservative text key for exact-duplicate checks."""

    return (
        values.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _audit_augmentation_quality(
    originals: pd.DataFrame,
    augments: pd.DataFrame,
    task: TaskConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit augmentation quality and optionally filter flagged rows.

    This audit is deliberately conservative: ``report`` is the default action,
    so questionable rows are written to a report but remain in training data.
    """

    if task.augmentation_audit_action not in {"report", "drop", "error"}:
        raise ValueError(
            "augmentation_audit_action must be one of: report, drop, error"
        )
    if not 0 < task.augmentation_near_duplicate_threshold <= 1:
        raise ValueError("augmentation_near_duplicate_threshold must be in (0, 1]")
    if not 0 <= task.augmentation_min_source_similarity <= 1:
        raise ValueError("augmentation_min_source_similarity must be in [0, 1]")

    audit = augments[
        ["id", "source_id", "source_group", "fold", task.label_column, "Comment"]
    ].copy()
    audit["audit_reason"] = ""
    audit["audit_status"] = "keep"
    audit["text_length"] = audit["Comment"].fillna("").astype(str).str.len()
    audit["word_count"] = audit["Comment"].fillna("").astype(str).str.split().str.len()
    audit.loc[audit["word_count"] < 2, "audit_reason"] += "too_short;"

    all_texts = pd.concat(
        [originals["Comment"].fillna(""), augments["Comment"].fillna("")],
        ignore_index=True,
    ).astype(str)
    all_keys = _normalise_for_audit(all_texts)
    duplicate_counts = all_keys.value_counts()
    augment_keys = _normalise_for_audit(audit["Comment"])
    audit["exact_duplicate"] = augment_keys.map(duplicate_counts).gt(1)
    audit.loc[audit["exact_duplicate"], "audit_reason"] += "exact_duplicate;"

    # Character n-grams work for Kannada-English code-mixed text without
    # requiring a separate embedding model or network download.
    try:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
        matrix = vectorizer.fit_transform(all_texts.tolist())
        original_matrix = matrix[: len(originals)]
        augment_matrix = matrix[len(originals) :]
        similarity = cosine_similarity(augment_matrix, original_matrix)
    except ValueError:
        # An all-empty or too-short corpus has no character n-gram vocabulary.
        similarity = np.zeros((len(augments), len(originals)), dtype=float)
    original_ids = originals["id"].astype(str).tolist()
    original_labels = originals[task.label_column].tolist()
    source_similarity = []
    nearest_similarity = []
    nearest_label = []
    for row_index, source_id in enumerate(audit["source_id"].astype(str)):
        source_positions = [
            index for index, value in enumerate(original_ids) if value == source_id
        ]
        source_position = source_positions[0] if source_positions else None
        source_score = (
            float(similarity[row_index, source_position])
            if source_position is not None
            else 0.0
        )
        source_similarity.append(source_score)
        comparison = similarity[row_index].copy()
        if source_position is not None:
            comparison[source_position] = -1.0
        nearest_position = int(comparison.argmax())
        nearest_score = float(comparison[nearest_position])
        nearest_similarity.append(nearest_score)
        nearest_label.append(original_labels[nearest_position])
    audit["source_similarity"] = source_similarity
    audit["nearest_original_similarity"] = nearest_similarity
    audit["nearest_original_label"] = nearest_label
    audit.loc[
        audit["source_similarity"] < task.augmentation_min_source_similarity,
        "audit_reason",
    ] += "semantic_drift;"
    near_duplicate = (
        audit["nearest_original_similarity"]
        >= task.augmentation_near_duplicate_threshold
    )
    audit.loc[near_duplicate, "audit_reason"] += "near_duplicate;"
    cross_class = near_duplicate & audit[task.label_column].ne(
        audit["nearest_original_label"]
    )
    audit.loc[cross_class, "audit_reason"] += "cross_class_overlap;"
    audit["audit_status"] = audit["audit_reason"].ne("").map(
        {True: "review", False: "keep"}
    )

    report_dir = OUTPUT_ROOT / task.output_subdir
    report_dir.mkdir(parents=True, exist_ok=True)
    audit_path = report_dir / "augmentation_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    flagged = audit["audit_status"].eq("review")
    if flagged.any() and task.augmentation_audit_action == "error":
        raise ValueError(
            f"Augmentation audit flagged {int(flagged.sum())} rows; "
            f"see {audit_path}"
        )
    if task.augmentation_audit_action == "drop":
        augments = augments.loc[~flagged.to_numpy()].copy()
    return augments, audit


def _load_predefined_folds(task: TaskConfig) -> pd.DataFrame:
    if task.use_augmentation and len(task.original_fold_filenames) != len(
        task.augmentation_fold_filenames
    ):
        raise ValueError(
            "original_fold_filenames and augmentation_fold_filenames must have "
            "the same number of files."
        )
    if not task.original_fold_filenames:
        raise ValueError("At least one predefined fold is required.")
    if task.selection_validation_fold not in range(
        1, len(task.original_fold_filenames) + 1
    ):
        raise ValueError("selection_validation_fold does not identify a configured fold.")

    fold_frames: list[pd.DataFrame] = []
    original_by_fold: dict[int, pd.DataFrame] = {}
    augment_by_fold: dict[int, pd.DataFrame] = {}
    for fold, original_name in enumerate(task.original_fold_filenames, start=1):
        original_path = resolve_data_file(original_name)
        original = _read_csv(original_path)
        _validate_columns(
            original,
            original_path,
            {"id", "Comment", task.label_column},
        )
        original = original.copy()
        original["fold"] = fold
        original["data_origin"] = "original"
        original_by_fold[fold] = original
        fold_frames.append(original)

        if not task.use_augmentation:
            continue

        augment_path = resolve_data_file(task.augmentation_fold_filenames[fold - 1])
        augment = _read_csv(augment_path)
        _validate_columns(
            augment,
            augment_path,
            {
                "id",
                "Comment",
                task.label_column,
                "source_id",
                "source_group",
                "fold",
            },
        )
        if augment["source_id"].isna().any() or augment["source_group"].isna().any():
            raise ValueError(f"{augment_path} contains missing augmentation lineage.")
        parsed_folds = pd.to_numeric(augment["fold"], errors="coerce")
        if parsed_folds.isna().any() or not parsed_folds.eq(fold).all():
            raise ValueError(
                f"{augment_path} must contain only rows whose fold value is {fold}."
            )

        augment = augment.copy()
        augment["fold"] = fold
        augment["data_origin"] = "augmentation"
        augment_by_fold[fold] = augment
        fold_frames.append(augment)

    originals = pd.concat(original_by_fold.values(), ignore_index=True, sort=False)
    if not task.use_augmentation:
        return originals

    original_ids = set(originals["id"])
    for fold, augment in augment_by_fold.items():
        missing_sources = sorted(set(augment["source_id"]) - original_ids)
        if missing_sources:
            raise ValueError(
                f"augment fold {fold} refers to unknown source IDs: "
                f"{missing_sources[:10]}"
            )
        source_folds = originals.set_index("id")["fold"]
        wrong_fold = augment["source_id"].map(source_folds).ne(fold)
        if wrong_fold.any():
            bad_ids = augment.loc[wrong_fold, "id"].head(10).tolist()
            raise ValueError(
                f"augment fold {fold} contains sources from another fold: {bad_ids}"
            )
        source_labels = originals.set_index("id")[task.label_column]
        wrong_label = augment["source_id"].map(source_labels).ne(
            augment[task.label_column]
        )
        if wrong_label.any():
            bad_ids = augment.loc[wrong_label, "id"].head(10).tolist()
            raise ValueError(
                f"augment fold {fold} changed its source label: {bad_ids}"
            )

    all_augments = pd.concat(augment_by_fold.values(), ignore_index=True, sort=False)
    if task.enable_augmentation_audit:
        all_augments, _ = _audit_augmentation_quality(
            originals,
            all_augments,
            task,
        )
    groups_per_fold = all_augments.groupby("source_group")["fold"].nunique()
    crossing_groups = groups_per_fold[groups_per_fold > 1]
    if not crossing_groups.empty:
        raise ValueError(
            "Augmentation source groups cross predefined folds: "
            f"{crossing_groups.index[:10].tolist()}"
        )
    # Use the audited (and possibly filtered) augmentation frame, rather than
    # the pre-audit fold list accumulated while reading each file.
    return pd.concat([originals, all_augments], ignore_index=True, sort=False)


def _compact_original_paths(task: TaskConfig) -> tuple[Path, Path] | None:
    """Find the two-file original-only layout when both files are available."""

    if not task.compact_train_filename or not task.compact_validation_filename:
        return None
    try:
        train_path = resolve_data_file(task.compact_train_filename)
        validation_path = resolve_data_file(task.compact_validation_filename)
    except FileNotFoundError:
        return None
    if not train_path.is_file() or not validation_path.is_file():
        return None
    if train_path.parent != validation_path.parent:
        raise ValueError("Compact train and validation files must share one directory.")
    return train_path, validation_path


def labeled_input_filenames(task: TaskConfig) -> tuple[str, ...]:
    """List labeled inputs actually used, including the compact layout."""

    if _compact_original_paths(task) is not None:
        assert task.compact_train_filename and task.compact_validation_filename
        return task.compact_train_filename, task.compact_validation_filename
    return task.labeled_filenames()


def _load_compact_originals(task: TaskConfig, paths: tuple[Path, Path]) -> pd.DataFrame:
    if task.selection_validation_fold is None:
        raise ValueError("Compact originals require a predefined validation fold.")
    frames = []
    for path, fold in zip(paths, (0, task.selection_validation_fold), strict=True):
        frame = _read_csv(path)
        _validate_columns(frame, path, {"id", "Comment", task.label_column})
        frame = frame.copy()
        # The merged file has no per-source fold column. Zero means the combined
        # training group; the untouched holdout keeps its configured fold number.
        frame["fold"] = fold
        frame["data_origin"] = "original"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_task_frames(task: TaskConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and normalize train/development CSVs consistently."""

    prediction_path = resolve_data_file(task.prediction_filename)
    prediction_df = _read_csv(prediction_path)
    compact_paths = _compact_original_paths(task)
    if compact_paths is not None:
        train_df = _load_compact_originals(task, compact_paths)
        train_description = "merged original training file and original holdout"
    elif task.original_fold_filenames:
        train_df = _load_predefined_folds(task)
        train_description = (
            "predefined original and augmentation folds"
            if task.use_augmentation
            else "predefined original folds"
        )
    else:
        train_path = resolve_data_file(task.train_filename)
        train_df = _read_csv(train_path)
        train_description = str(train_path)
    return _validate_and_encode_frames(
        train_df,
        prediction_df,
        task,
        train_description,
    )
