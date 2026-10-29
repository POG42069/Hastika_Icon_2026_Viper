"""HASTIKA CSV loading, validation, and light text normalization."""

from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_config import (
    DATA_DIR, OUTPUT_ROOT, PROJECT_ROOT, TASK_A_OPEN_VALIDATION_FILENAME,
    TASK_B_OPEN_VALIDATION_FILENAME, TaskConfig,
)
from src.preprocessing import normalize_text


class MissingDataFileError(FileNotFoundError):
    """An input is absent, rather than ambiguous among attached datasets."""


def resolve_data_file(filename: str, configured_dir: Path = DATA_DIR) -> Path:
    """Find an input locally or in an attached Kaggle Dataset."""

    if filename in {TASK_A_OPEN_VALIDATION_FILENAME, TASK_B_OPEN_VALIDATION_FILENAME}:
        # This named evaluation set ships with the experiment branches. Keep
        # training in HASTIKA_DATA_DIR, while inference and gold use this exact
        # paired open set rather than a legacy raw-only inference file.
        bundled = PROJECT_ROOT / "data" / "open_validation" / filename
        if not bundled.is_file():
            letter = "A" if filename == TASK_A_OPEN_VALIDATION_FILENAME else "B"
            raise MissingDataFileError(
                f"Bundled Task {letter} open validation is missing: {bundled}. "
                "Update or re-clone this branch including data/open_validation/."
            )
        return bundled.resolve()

    selected = configured_dir / filename
    if selected.is_file():
        return selected.resolve()
    if "HASTIKA_DATA_DIR" in os.environ:
        # An explicitly selected release must not borrow a missing compact
        # file (or any other input) from an older attached dataset.
        raise MissingDataFileError(
            f"Could not find {filename} in the selected HASTIKA_DATA_DIR: {configured_dir}"
        )
    candidates = [Path(__file__).resolve().parents[1] / "data" / filename]
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

    raise MissingDataFileError(
        f"Could not find {filename}. Attach the private HASTIKA dataset on "
        "Kaggle or set the HASTIKA_DATA_DIR environment variable."
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        keep_default_na=False,
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
    if task.input_mode not in {"english", "bilingual"}:
        raise ValueError("input_mode must be english or bilingual")
    if task.input_mode == "bilingual":
        required_train.add(task.original_comment_column)
        required_prediction.add(task.original_comment_column)
    _validate_columns(train_df, Path(train_description), required_train)
    _validate_columns(
        prediction_df,
        Path(task.prediction_filename),
        required_prediction,
    )
    if any(frame["id"].isna().any() or frame["id"].eq("").any() for frame in (train_df, prediction_df)):
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
    if task.evaluate_prediction_labels and task.label_column in prediction_df:
        invalid = ~prediction_df[task.label_column].isin(task.labels)
        if invalid.any():
            raise ValueError("Open-validation prediction inputs have missing or unknown gold labels.")

    train_df = train_df.copy()
    prediction_df = prediction_df.copy()
    train_df["text"] = train_df["Comment"].map(normalize_text)
    prediction_df["text"] = prediction_df["Comment"].map(normalize_text)
    if task.input_mode == "bilingual":
        for frame in (train_df, prediction_df):
            frame["original_text"] = frame[task.original_comment_column].map(normalize_text)
            missing_original = frame["original_text"].eq("") & frame["text"].ne("")
            if missing_original.any():
                raise ValueError(
                    "Bilingual input has missing original comments for IDs: "
                    f"{frame.loc[missing_original, 'id'].head(10).tolist()}"
                )
    label_to_id = {label: index for index, label in enumerate(task.labels)}
    train_df["label_id"] = train_df[task.label_column].map(label_to_id).astype(int)
    return train_df, prediction_df


def prompt_inputs(frame: pd.DataFrame, task: TaskConfig) -> list:
    """Return explicit English/original pairs for every pipeline phase."""

    if task.input_mode == "bilingual":
        return list(zip(frame["text"].tolist(), frame["original_text"].tolist(), strict=True))
    return frame["text"].tolist()


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
    if task.fixed_full_retrain_epochs is None and task.selection_validation_fold not in range(
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


def _optional_data_file(name: str) -> Path | None:
    try:
        path = resolve_data_file(name)
    except MissingDataFileError:
        # A missing optional compact file permits the original-fold fallback.
        # Ambiguous dataset paths and other resolution errors still propagate.
        return None
    return path if path.is_file() else None


def _compact_original_paths(task: TaskConfig) -> tuple[Path, Path] | None:
    """Select the compact originals only when both files are available."""

    if task.use_augmentation or task.fixed_full_retrain_epochs is not None:
        return None
    names = (task.compact_train_filename, task.compact_validation_filename)
    if not any(names):
        return None
    if not all(names):
        raise ValueError("Both compact original filenames must be configured.")
    paths = [_optional_data_file(name) for name in names]
    if all(path is None for path in paths):
        return None
    if any(path is None for path in paths):
        # The compact holdout name is also the final legacy fold name. A full
        # legacy fold set is safe to use when the merged 1–4 file is absent.
        if paths[0] is None and task.original_fold_filenames:
            if all(
                _optional_data_file(name) is not None
                for name in task.original_fold_filenames
            ):
                return None
        missing = [name for name, path in zip(names, paths) if path is None]
        raise FileNotFoundError(
            "The compact original dataset is incomplete; missing: "
            + ", ".join(missing)
        )
    train_path, validation_path = paths
    assert train_path is not None and validation_path is not None
    return train_path, validation_path


def labeled_input_filenames(task: TaskConfig) -> tuple[str, ...]:
    """Return the labeled CSVs actually selected for loading and fingerprints."""

    if _compact_original_paths(task) is not None:
        assert task.compact_train_filename and task.compact_validation_filename
        return (task.compact_train_filename, task.compact_validation_filename)
    if task.original_fold_filenames:
        augmentation_files = (
            task.augmentation_fold_filenames if task.use_augmentation else ()
        )
        return task.original_fold_filenames + augmentation_files
    return (task.train_filename,)


def _load_compact_originals(
    task: TaskConfig, paths: tuple[Path, Path]
) -> pd.DataFrame:
    if task.selection_validation_fold != 5:
        raise ValueError("Compact original files require validation fold 5.")
    frames = []
    for fold, path in zip((1, 5), paths):
        frame = _read_csv(path)
        _validate_columns(frame, path, {"id", "Comment", task.label_column})
        frame = frame.copy()
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
        train_description = "compact original train and validation files"
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
    train_df, prediction_df = _validate_and_encode_frames(
        train_df,
        prediction_df,
        task,
        train_description,
    )
    if task.training_exclusion_filename is not None:
        reference_path = resolve_data_file(task.training_exclusion_filename)
        reference = _read_csv(reference_path)
        _validate_columns(reference, reference_path, {"id"})
        if reference['id'].eq('').any() or reference['id'].duplicated().any():
            raise ValueError("Training exclusion reference requires unique, nonempty IDs.")
        excluded = train_df['id'].isin(reference['id'])
        audit_columns = [column for column in ('id', 'fold', task.label_column) if column in train_df]
        audit = {
            'policy': 'exclude-reference-ids-v1',
            'reference': task.training_exclusion_filename,
            'loaded_rows': len(train_df),
            'excluded_rows': int(excluded.sum()),
            'remaining_rows': int((~excluded).sum()),
            'excluded_records': train_df.loc[excluded, audit_columns].to_dict(orient='records'),
        }
        train_df = train_df.loc[~excluded].reset_index(drop=True)
        if train_df.empty:
            raise ValueError("No training rows remain after excluding open-validation IDs.")
        train_df.attrs['training_exclusions'] = audit
    return train_df, prediction_df


def load_epoch_validation_frame(task: TaskConfig) -> pd.DataFrame | None:
    """Align external gold by ID to the bilingual development inputs.

    Gold comments may be raw code-mixed, while this release stores English in
    Comment. Keep gold row order, use the paired input text, and check the raw
    source when present. Never use prediction-file overrides as this reference.
    """
    if task.epoch_validation_filename is None:
        return None
    if task.epoch_validation_required:
        path = resolve_data_file(task.epoch_validation_filename)
    else:
        path = _optional_data_file(task.epoch_validation_filename)
        if path is None:
            return None
    gold = _read_csv(path)
    _validate_columns(gold, path, {"id", task.label_column})
    if gold.empty or gold['id'].eq('').any() or gold['id'].duplicated().any():
        raise ValueError("Official validation requires nonempty data and unique, nonmissing IDs.")
    if not set(gold[task.label_column]).issubset(task.labels):
        raise ValueError("Official validation has missing or unknown labels.")
    name = task.epoch_validation_input_filename or task.prediction_filename
    input_path = resolve_data_file(name)
    inputs = _read_csv(input_path)
    required = {'id', 'Comment'}
    if task.input_mode == 'bilingual':
        required.add(task.original_comment_column)
    _validate_columns(inputs, input_path, required)
    if inputs['id'].eq('').any() or inputs['id'].duplicated().any():
        raise ValueError("Validation input requires unique, nonmissing IDs.")
    reference = inputs.set_index('id')
    missing = ~gold['id'].isin(reference.index)
    if missing.any():
        raise ValueError(f"Official validation IDs are missing from development: {gold.loc[missing, 'id'].head(10).tolist()}")
    aligned = reference.loc[gold['id']].reset_index()
    if task.input_mode == 'bilingual' and 'Comment' in gold:
        raw = aligned[task.original_comment_column].map(normalize_text)
        translated = aligned['Comment'].map(normalize_text)
        authored = gold['Comment'].map(normalize_text)
        if (~(authored.eq(raw) | authored.eq(translated))).any():
            raise ValueError("Official validation comments differ from paired development inputs.")
    aligned[task.label_column] = gold[task.label_column].to_numpy()
    validated, _ = _validate_and_encode_frames(aligned, inputs, task, str(path))
    if validated['text'].eq('').any() or (
        task.input_mode == 'bilingual' and validated['original_text'].eq('').any()
    ):
        raise ValueError("Official validation inputs contain empty comments.")
    return validated
