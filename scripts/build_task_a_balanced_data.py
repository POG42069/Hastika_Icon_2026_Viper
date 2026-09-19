"""Build a leakage-safe, balanced Task A dataset with auditable augmentation.

The script keeps the supplied CSVs immutable.  It recovers Task B source rows
needed by the existing Hate augmentations, creates deterministic five-fold
original files, generates conservative Non-Hate surface variants, runs quality
checks, and writes balanced Task A fold files plus audit artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import re
import shutil
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


SEED = 42
FOLDS = (1, 2, 3, 4, 5)
BUFFER_RATIO = 0.20

TRANSLITERATION_RULES = (
    ("adhu", "adu"),
    ("adu", "adhu"),
    ("idhu", "idu"),
    ("idu", "idhu"),
    ("ee", "e"),
    ("nivu", "neevu"),
    ("neevu", "nivu"),
    ("madi", "maadi"),
    ("maadi", "madi"),
    ("nodu", "noodu"),
    ("noodu", "nodu"),
    ("tara", "thara"),
    ("thara", "tara"),
    ("inda", "indha"),
    ("indha", "inda"),
    ("antha", "anta"),
    ("anta", "antha"),
    ("buddhi", "buddi"),
    ("buddi", "buddhi"),
    ("summane", "sumne"),
    ("sumne", "summane"),
    ("illa", "ila"),
    ("adre", "aadre"),
    ("aadre", "adre"),
    ("yake", "yaake"),
    ("yaake", "yake"),
    ("nija", "nijaa"),
    ("sari", "sarii"),
    ("ella", "yella"),
    ("yella", "ella"),
    ("nam", "namm"),
    ("namm", "nam"),
    ("ninu", "neenu"),
    ("neenu", "ninu"),
    ("hogi", "hoogi"),
    ("hoogi", "hogi"),
    ("maklu", "makkalu"),
    ("makkalu", "maklu"),
    ("thu", "thuu"),
    ("thuu", "thu"),
    ("nin", "ninna"),
    ("mele", "meele"),
    ("meele", "mele"),
    ("modlu", "modalu"),
    ("modalu", "modlu"),
    ("munde", "mundhe"),
    ("mundhe", "munde"),
    ("enu", "yenu"),
    ("yenu", "enu"),
    ("nodi", "noodi"),
    ("noodi", "nodi"),
    ("ondu", "ondhu"),
    ("ondhu", "ondu"),
    ("kelsa", "kelasa"),
    ("kelasa", "kelsa"),
    ("andre", "andhre"),
    ("andhre", "andre"),
    ("nayi", "naayi"),
    ("naayi", "nayi"),
    ("avaru", "avru"),
    ("avru", "avaru"),
    ("nimma", "nima"),
    ("nima", "nimma"),
    ("ide", "idhe"),
    ("idhe", "ide"),
    ("nimge", "nimage"),
    ("nimage", "nimge"),
    ("ade", "adhe"),
    ("adhe", "ade"),
    ("jasti", "jaasti"),
    ("jaasti", "jasti"),
    ("navu", "naavu"),
    ("naavu", "navu"),
    ("yaru", "yaaru"),
    ("yaaru", "yaru"),
    ("agi", "aagi"),
    ("aagi", "agi"),
    ("namge", "namage"),
    ("namage", "namge"),
    ("nanu", "naanu"),
    ("naanu", "nanu"),
    ("mado", "maado"),
    ("maado", "mado"),
    ("ninge", "ninage"),
    ("ninage", "ninge"),
)

INFORMAL_RULES = (
    ("people", "ppl"),
    ("please", "pls"),
    ("your", "ur"),
    ("you", "u"),
    ("government", "govt"),
    ("because", "cuz"),
    ("before", "b4"),
)

REPHRASE_RULES = (
    ("thank you", "thanks"),
    ("thanks", "thank you"),
    ("do not", "don't"),
    ("does not", "doesn't"),
    ("did not", "didn't"),
    ("cannot", "can't"),
    ("can not", "can't"),
    ("will not", "won't"),
    ("i am", "I'm"),
    ("it is", "it's"),
    ("that is", "that's"),
    ("what is", "what's"),
    ("there is", "there's"),
    ("i think", "in my opinion"),
    ("in my opinion", "I think"),
    ("very good", "really good"),
    ("really good", "very good"),
    ("very nice", "really nice"),
    ("really nice", "very nice"),
    ("good job", "well done"),
    ("well done", "good job"),
)

RISK_RE = re.compile(
    r"\b(?:muslim|hindu|islam|christian|religion|caste|jati|jaati|gender|women|"
    r"woman|men|man|pakistan|india|kannada|tamil|kill|koll|sayis|rape|terror|"
    r"encounter|violence|hate|beda|illa|not|no|never|dont|don't)\b",
    flags=re.IGNORECASE,
)

METHOD_ALIASES = {
    "controlled_transliteration_variant": "controlled_transliteration_variant",
    "controlled_punctuation_variant": "controlled_punctuation_variant",
    "gpt56sol_label_preserving_paraphrase": "controlled_label_preserving_rephrase",
    "controlled_informal_english_variant": "controlled_informal_english_variant",
    "controlled_informal_english+transliteration_variant": (
        "controlled_informal_english+transliteration_variant"
    ),
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"id": str, "source_id": str, "source_group": str},
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def normalized_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = html.unescape(unicodedata.normalize("NFKC", text))
    return re.sub(r"\s+", " ", text).strip().lower()


def canonical_text(value: object) -> str:
    return re.sub(r"[^\w]+", "", normalized_text(value), flags=re.UNICODE)


def stable_number(*values: object) -> int:
    payload = "\x1f".join(map(str, values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def restore_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_one_rule(
    text: str,
    rules: tuple[tuple[str, str], ...],
    key: object,
) -> str | None:
    matches: list[tuple[str, str]] = []
    for source, replacement in rules:
        if re.search(rf"(?<!\w){re.escape(source)}(?!\w)", text, flags=re.I):
            matches.append((source, replacement))
    if not matches:
        return None
    source, replacement = matches[stable_number(key, text) % len(matches)]
    pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", flags=re.I)
    return pattern.sub(
        lambda match: restore_case(match.group(0), replacement), text, count=1
    )


def punctuation_variant(text: str, key: object) -> str | None:
    stripped = text.rstrip()
    if not stripped:
        return None
    if stripped.endswith("..."):
        result = stripped + "."
    elif stripped.endswith("?"):
        result = stripped + "?"
    elif stripped.endswith("!"):
        result = stripped + "!"
    elif stripped.endswith("."):
        result = stripped + "."
    else:
        result = stripped + ("..." if stable_number(key) % 2 else ".")
    return result if result != text else None


def transliteration_variant(text: str, key: object) -> str | None:
    return apply_one_rule(text, TRANSLITERATION_RULES, key)


def informal_variant(text: str, key: object) -> str | None:
    return apply_one_rule(text, INFORMAL_RULES, key)


def rephrase_variant(text: str, key: object) -> str | None:
    result = apply_one_rule(text, REPHRASE_RULES, key)
    if result is not None:
        return result
    # Moving a discourse address changes style, not the proposition.
    match = re.match(r"^(bro|guru|sir|madam)[, ]+(.+)$", text, flags=re.I)
    if match:
        if re.search(
            rf"(?<!\w){re.escape(match.group(1))}(?!\w)",
            match.group(2),
            flags=re.I,
        ):
            return None
        return f"{match.group(2).rstrip(' ,')}, {match.group(1)}"
    return None


def combined_variant(text: str, key: object) -> str | None:
    informal = informal_variant(text, (key, "informal"))
    if informal is None:
        return None
    return transliteration_variant(informal, (key, "transliteration"))


TRANSFORMS: dict[str, Callable[[str, object], str | None]] = {
    "controlled_transliteration_variant": transliteration_variant,
    "controlled_punctuation_variant": punctuation_variant,
    "controlled_label_preserving_rephrase": rephrase_variant,
    "controlled_informal_english_variant": informal_variant,
    "controlled_informal_english+transliteration_variant": combined_variant,
}

MIN_SEQUENCE_SIMILARITY = {
    "controlled_transliteration_variant": 0.60,
    "controlled_punctuation_variant": 0.75,
    "controlled_label_preserving_rephrase": 0.45,
    "controlled_informal_english_variant": 0.70,
    "controlled_informal_english+transliteration_variant": 0.65,
}


def largest_remainder_counts(weights: dict[object, int], total: int) -> dict:
    if total < 0 or not weights or sum(weights.values()) <= 0:
        raise ValueError("weights must be non-empty and total must be non-negative")
    denominator = sum(weights.values())
    raw = {key: total * value / denominator for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(
        weights,
        key=lambda key: (raw[key] - result[key], str(key)),
        reverse=True,
    )
    for key in order[:remaining]:
        result[key] += 1
    return result


def assign_original_folds(
    originals: pd.DataFrame,
    known_folds: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    result = originals.copy()
    result["fold"] = result["id"].map(known_folds)
    rng = random.Random(seed)
    for label in sorted(result["Label"].unique()):
        label_mask = result["Label"].eq(label)
        counts = Counter(
            int(value) for value in result.loc[label_mask, "fold"].dropna().tolist()
        )
        for fold in FOLDS:
            counts.setdefault(fold, 0)
        unassigned = result.index[label_mask & result["fold"].isna()].tolist()
        rng.shuffle(unassigned)
        tie_order = list(FOLDS)
        rng.shuffle(tie_order)
        tie_rank = {fold: index for index, fold in enumerate(tie_order)}
        for row_index in unassigned:
            fold = min(FOLDS, key=lambda item: (counts[item], tie_rank[item]))
            result.at[row_index, "fold"] = fold
            counts[fold] += 1
    result["fold"] = result["fold"].astype(int)
    result["source_group"] = result["id"].astype(str)
    return result


def audit_hate_augmentations(
    task_b_originals: pd.DataFrame,
    hate_augments: pd.DataFrame,
    prior_audit: pd.DataFrame | None,
) -> pd.DataFrame:
    sources = task_b_originals.set_index("id")
    audit = hate_augments.copy()
    audit["source_comment"] = audit["source_id"].map(sources["Comment"])
    audit["source_label"] = audit["source_id"].map(sources["Hate Category"])
    audit["source_fold"] = audit["source_id"].map(sources["fold"])
    audit["source_exists"] = audit["source_comment"].notna()
    audit["label_matches_source"] = audit["Hate Category"].eq(audit["source_label"])
    audit["fold_matches_source"] = audit["fold"].eq(audit["source_fold"])
    audit["exact_source_equal"] = [
        normalized_text(comment) == normalized_text(source)
        for comment, source in zip(
            audit["Comment"], audit["source_comment"], strict=True
        )
    ]
    audit["canonical_source_equal"] = [
        canonical_text(comment) == canonical_text(source)
        for comment, source in zip(
            audit["Comment"], audit["source_comment"], strict=True
        )
    ]
    audit["sequence_ratio"] = [
        SequenceMatcher(
            None, normalized_text(source), normalized_text(comment)
        ).ratio()
        for comment, source in zip(
            audit["Comment"], audit["source_comment"], strict=True
        )
    ]
    normalized = audit["Comment"].map(normalized_text)
    audit["duplicate_augment_text"] = normalized.duplicated(False)
    audit["manual_review_recommended"] = (
        audit["augmentation_method"].eq("gpt56sol_label_preserving_paraphrase")
        & audit["sequence_ratio"].lt(0.50)
    )
    failures = (
        ~audit["source_exists"]
        | ~audit["label_matches_source"]
        | ~audit["fold_matches_source"]
        | audit["exact_source_equal"]
        | audit["duplicate_augment_text"]
    )
    audit["qa_status"] = np.where(failures, "reject", "accepted")
    if prior_audit is not None and "id" in prior_audit.columns:
        prior_columns = [
            column
            for column in ("id", "action", "reasons", "previous_action")
            if column in prior_audit.columns
        ]
        audit = audit.merge(
            prior_audit[prior_columns].drop_duplicates("id"),
            on="id",
            how="left",
            suffixes=("", "_prior"),
        )
    return audit


def source_pool_frame(originals: pd.DataFrame) -> pd.DataFrame:
    pool = originals.loc[originals["Label"].eq("Non-Hate")].copy()
    pool["word_count"] = pool["Comment"].fillna("").astype(str).str.split().str.len()
    pool["character_count"] = pool["Comment"].fillna("").astype(str).str.len()
    pool["normalized_key"] = pool["Comment"].map(normalized_text)
    pool["canonical_key"] = pool["Comment"].map(canonical_text)
    all_keys = originals.assign(
        canonical_key=originals["Comment"].map(canonical_text)
    )
    conflicting = set(
        all_keys.groupby("canonical_key")["Label"].nunique().loc[lambda value: value > 1].index
    )
    pool["eligibility_reason"] = ""
    pool.loc[pool["word_count"].lt(3), "eligibility_reason"] += "too_short;"
    pool.loc[pool["word_count"].gt(50), "eligibility_reason"] += "too_long;"
    pool.loc[pool["canonical_key"].eq(""), "eligibility_reason"] += "empty;"
    pool.loc[
        pool["canonical_key"].isin(conflicting), "eligibility_reason"
    ] += "cross_label_duplicate;"
    duplicate = pool["canonical_key"].duplicated(keep="first")
    pool.loc[duplicate, "eligibility_reason"] += "duplicate_nonhate_source;"
    pool["eligible"] = pool["eligibility_reason"].eq("")
    return pool


def candidate_quality(
    source_text: str,
    candidate_text: str,
    method: str,
    occupied_normalized: set[str],
    hate_canonical: set[str],
) -> tuple[bool, str, float, float]:
    reasons: list[str] = []
    source_norm = normalized_text(source_text)
    candidate_norm = normalized_text(candidate_text)
    source_canonical = canonical_text(source_text)
    candidate_canonical = canonical_text(candidate_text)
    if not candidate_norm or candidate_norm == source_norm:
        reasons.append("unchanged_or_empty")
    if candidate_norm in occupied_normalized:
        reasons.append("exact_duplicate")
    if candidate_canonical in hate_canonical:
        reasons.append("cross_label_canonical_overlap")
    if method == "controlled_punctuation_variant":
        if candidate_canonical != source_canonical:
            reasons.append("punctuation_changed_lexical_content")
    elif candidate_canonical == source_canonical:
        reasons.append("no_lexical_change")
    source_words = max(1, len(source_norm.split()))
    candidate_words = len(candidate_norm.split())
    length_ratio = candidate_words / source_words
    if candidate_words < 2 or not 0.55 <= length_ratio <= 1.45:
        reasons.append("length_out_of_range")
    sequence_ratio = SequenceMatcher(None, source_norm, candidate_norm).ratio()
    if sequence_ratio < MIN_SEQUENCE_SIMILARITY[method]:
        reasons.append("source_similarity_too_low")
    return not reasons, ";".join(reasons), sequence_ratio, length_ratio


def slots_from_counts(counts: dict[object, int], seed: int) -> list:
    slots = [key for key, count in counts.items() for _ in range(count)]
    random.Random(seed).shuffle(slots)
    return slots


def pair_counts_from_marginals(
    fold_counts: dict[int, int],
    method_counts: dict[str, int],
    seed: int,
) -> Counter[tuple[int, str]]:
    fold_slots = slots_from_counts(fold_counts, seed + 1)
    method_slots = slots_from_counts(method_counts, seed + 2)
    if len(fold_slots) != len(method_slots):
        raise RuntimeError("fold and method totals do not match")
    return Counter(zip(fold_slots, method_slots, strict=True))


def build_nonhate_candidates(
    pool: pd.DataFrame,
    originals: pd.DataFrame,
    hate_augments: pd.DataFrame,
    candidate_pair_counts: dict[tuple[int, str], int],
    seed: int,
) -> pd.DataFrame:
    paired_slots = [
        pair for pair, count in candidate_pair_counts.items() for _ in range(count)
    ]
    random.Random(seed + 3).shuffle(paired_slots)
    rarity = {
        method: sum(
            count
            for (fold, candidate_method), count in candidate_pair_counts.items()
            if candidate_method == method
        )
        for method in {method for _, method in candidate_pair_counts}
    }
    paired_slots.sort(key=lambda item: (rarity[item[1]], stable_number(seed, *item)))

    eligible = pool.loc[pool["eligible"]].copy()
    by_fold: dict[int, list[int]] = {}
    for fold in FOLDS:
        indices = eligible.index[eligible["fold"].eq(fold)].tolist()
        random.Random(seed + fold).shuffle(indices)
        by_fold[fold] = indices

    occupied_normalized = set(originals["Comment"].map(normalized_text))
    occupied_normalized.update(hate_augments["Comment"].map(normalized_text))
    hate_canonical = set(
        originals.loc[originals["Label"].eq("Hate"), "Comment"].map(canonical_text)
    )
    hate_canonical.update(hate_augments["Comment"].map(canonical_text))
    used_sources: set[str] = set()
    rows: list[dict[str, object]] = []

    for candidate_number, (fold, method) in enumerate(paired_slots, start=1):
        selected = None
        for source_index in by_fold[fold]:
            source = eligible.loc[source_index]
            source_id = str(source["id"])
            if source_id in used_sources:
                continue
            candidate_text = TRANSFORMS[method](
                str(source["Comment"]), (seed, candidate_number, source_id, method)
            )
            if candidate_text is None:
                continue
            passed, reasons, sequence_ratio, length_ratio = candidate_quality(
                str(source["Comment"]),
                candidate_text,
                method,
                occupied_normalized,
                hate_canonical,
            )
            if not passed:
                continue
            selected = (
                source,
                source_id,
                candidate_text,
                reasons,
                sequence_ratio,
                length_ratio,
            )
            break
        if selected is None:
            raise RuntimeError(
                f"Could not generate enough {method} candidates in fold {fold}."
            )
        source, source_id, candidate_text, reasons, sequence_ratio, length_ratio = selected
        used_sources.add(source_id)
        occupied_normalized.add(normalized_text(candidate_text))
        candidate_id = f"taska-nh-f{fold}-{source_id}-v1"
        risk_matches = sorted(
            {match.group(0).lower() for match in RISK_RE.finditer(str(source["Comment"]))}
        )
        sampled_review = stable_number(seed, candidate_id, "review") % 5 == 0
        review_reasons = []
        if risk_matches:
            review_reasons.append("sensitive_or_negation_terms")
        if method == "controlled_label_preserving_rephrase":
            review_reasons.append("rephrase_method")
        if sampled_review:
            review_reasons.append("twenty_percent_quality_sample")
        rows.append(
            {
                "id": candidate_id,
                "Comment": candidate_text,
                "Label": "Non-Hate",
                "source_id": source_id,
                "source_group": source_id,
                "fold": int(fold),
                "variant": 1,
                "augmentation_method": method,
                "generation_model": "deterministic_rule_pipeline_v1",
                "prompt_version": "task_a_nonhate_rules_v1",
                "qa_status": "auto_accepted",
                "qa_reasons": reasons,
                "source_sequence_ratio": sequence_ratio,
                "word_length_ratio": length_ratio,
                "risk_terms": ",".join(risk_matches),
                "human_review_required": bool(review_reasons),
                "human_review_reason": ";".join(review_reasons),
                "human_review_status": (
                    "pending" if review_reasons else "not_required_for_conservative_transform"
                ),
                "source_comment": source["Comment"],
            }
        )
    return pd.DataFrame(rows)


def select_balanced_candidates(
    candidates: pd.DataFrame,
    target_pair_counts: dict[tuple[int, str], int],
    seed: int,
) -> pd.DataFrame:
    pairs = [pair for pair, count in target_pair_counts.items() for _ in range(count)]
    random.Random(seed + 13).shuffle(pairs)
    target_method_counts = Counter()
    for (_, method), count in target_pair_counts.items():
        target_method_counts[method] += count
    pairs.sort(
        key=lambda item: (target_method_counts[item[1]], stable_number(seed, "select", *item))
    )
    available = candidates.copy()
    available["selected"] = False
    for fold, method in pairs:
        matches = available.index[
            available["fold"].eq(fold)
            & available["augmentation_method"].eq(method)
            & ~available["selected"]
        ].tolist()
        if not matches:
            raise RuntimeError(
                f"Candidate buffer cannot fill fold={fold}, method={method}."
            )
        chosen = min(matches, key=lambda index: stable_number(seed, available.at[index, "id"]))
        available.at[chosen, "selected"] = True
    available["selection_status"] = np.where(
        available["selected"], "selected_for_balance", "surplus_buffer_not_selected"
    )
    return available


def verify_outputs(
    originals: pd.DataFrame,
    augments: pd.DataFrame,
    expected_per_label: int,
) -> dict[str, object]:
    combined = pd.concat(
        [
            originals.assign(data_origin="original"),
            augments.assign(data_origin="augmentation"),
        ],
        ignore_index=True,
        sort=False,
    )
    if combined["id"].duplicated().any():
        raise RuntimeError("Final Task A IDs are not unique.")
    counts = combined["Label"].value_counts().to_dict()
    if counts != {"Non-Hate": expected_per_label, "Hate": expected_per_label}:
        raise RuntimeError(f"Final labels are not balanced: {counts}")
    source_fold = originals.set_index("id")["fold"]
    missing_sources = sorted(set(augments["source_id"]) - set(originals["id"]))
    if missing_sources:
        raise RuntimeError(f"Missing final source IDs: {missing_sources[:10]}")
    wrong_fold = augments["source_id"].map(source_fold).ne(augments["fold"])
    if wrong_fold.any():
        raise RuntimeError("At least one final augmentation is in the wrong fold.")
    crossing = augments.groupby("source_group")["fold"].nunique()
    if crossing.gt(1).any():
        raise RuntimeError("An augmentation source_group crosses folds.")
    generated = augments.loc[augments["Label"].eq("Non-Hate")]
    if generated["Comment"].map(normalized_text).duplicated().any():
        raise RuntimeError("Generated Non-Hate rows contain exact duplicates.")
    validation = combined.loc[
        combined["fold"].eq(5) & combined["data_origin"].eq("original")
    ]
    return {
        "label_counts": counts,
        "total_rows": len(combined),
        "original_rows": len(originals),
        "augmentation_rows": len(augments),
        "validation_original_rows": len(validation),
        "validation_label_counts": validation["Label"].value_counts().to_dict(),
        "source_groups_crossing_folds": 0,
        "missing_sources": 0,
    }


def build(input_dir: Path, output_dir: Path, seed: int, buffer_ratio: float) -> None:
    binary = _read_csv(input_dir / "binary_train.csv")
    prediction_inputs = _read_csv(input_dir / "binary_validation_inputs.csv")
    required_binary = {"id", "Comment", "Label"}
    if not required_binary.issubset(binary.columns):
        raise ValueError(f"binary_train.csv needs columns {sorted(required_binary)}")

    task_b_parts = []
    hate_parts = []
    for fold in FOLDS:
        task_b_parts.append(
            _read_csv(input_dir / f"original_{fold}.csv").assign(fold=fold)
        )
        hate_parts.append(_read_csv(input_dir / f"augment{fold}.csv"))
    task_b_originals = pd.concat(task_b_parts, ignore_index=True, sort=False)
    hate_augments = pd.concat(hate_parts, ignore_index=True, sort=False)
    prior_audit_path = input_dir / "cleaning_audit.csv"
    prior_audit = _read_csv(prior_audit_path) if prior_audit_path.is_file() else None

    hate_audit = audit_hate_augmentations(
        task_b_originals, hate_augments, prior_audit
    )
    rejected_hate = hate_audit.loc[hate_audit["qa_status"].eq("reject")]
    if not rejected_hate.empty:
        raise RuntimeError(
            f"Existing Hate audit rejected {len(rejected_hate)} rows; inspect the audit."
        )

    binary_ids = set(binary["id"])
    missing_source_ids = sorted(set(hate_augments["source_id"]) - binary_ids)
    recovered = task_b_originals.loc[
        task_b_originals["id"].isin(missing_source_ids),
        ["id", "Comment"],
    ].copy()
    if len(recovered) != len(missing_source_ids):
        raise RuntimeError("Could not recover every missing Hate augmentation source.")
    recovered["Label"] = "Hate"
    recovered["record_origin"] = "task_b_source_recovered"
    binary = binary.copy()
    binary["record_origin"] = "binary_train_original"
    originals = pd.concat([binary, recovered], ignore_index=True, sort=False)
    if originals["id"].duplicated().any():
        raise RuntimeError("Original IDs collide after source recovery.")

    known_folds = dict(
        zip(task_b_originals["id"].astype(str), task_b_originals["fold"], strict=True)
    )
    originals = assign_original_folds(originals, known_folds, seed)

    binary_hate = hate_augments.rename(columns={"Hate Category": "task_b_category"}).copy()
    binary_hate["Label"] = "Hate"
    binary_hate["generation_model"] = "legacy_existing_augmentation"
    binary_hate["prompt_version"] = "legacy_existing"
    binary_hate["qa_status"] = "accepted_existing_audit"
    binary_hate["human_review_status"] = np.where(
        binary_hate["augmentation_method"].eq("gpt56sol_label_preserving_paraphrase"),
        "previous_pipeline_review_recorded",
        "not_required_for_conservative_transform",
    )

    original_counts = originals.groupby(["fold", "Label"]).size().unstack(fill_value=0)
    hate_aug_counts = binary_hate.groupby("fold").size().reindex(FOLDS, fill_value=0)
    target_fold_counts: dict[int, int] = {}
    for fold in FOLDS:
        hate_total = int(original_counts.loc[fold, "Hate"] + hate_aug_counts.loc[fold])
        nonhate_original = int(original_counts.loc[fold, "Non-Hate"])
        target_fold_counts[fold] = hate_total - nonhate_original
        if target_fold_counts[fold] < 0:
            raise RuntimeError(f"Fold {fold} already has more Non-Hate than Hate.")
    target_total = sum(target_fold_counts.values())

    legacy_method_counts = hate_augments["augmentation_method"].value_counts().to_dict()
    aliased_weights: Counter[str] = Counter()
    for method, count in legacy_method_counts.items():
        aliased_weights[METHOD_ALIASES[method]] += int(count)
    target_method_counts = largest_remainder_counts(dict(aliased_weights), target_total)

    candidate_total = math.ceil(target_total * (1 + buffer_ratio))
    target_pair_counts = pair_counts_from_marginals(
        target_fold_counts, target_method_counts, seed
    )
    candidate_pair_counts = largest_remainder_counts(
        dict(target_pair_counts), candidate_total
    )
    candidate_fold_counts = {
        fold: sum(
            count
            for (candidate_fold, _), count in candidate_pair_counts.items()
            if candidate_fold == fold
        )
        for fold in FOLDS
    }
    candidate_method_counts = {
        method: sum(
            count
            for (_, candidate_method), count in candidate_pair_counts.items()
            if candidate_method == method
        )
        for method in target_method_counts
    }

    source_pool = source_pool_frame(originals)
    candidates = build_nonhate_candidates(
        source_pool,
        originals,
        binary_hate,
        candidate_pair_counts,
        seed,
    )
    candidates = select_balanced_candidates(
        candidates, target_pair_counts, seed
    )
    selected_nonhate = candidates.loc[candidates["selected"]].copy()
    selected_nonhate.drop(columns=["selected", "selection_status"], inplace=True)

    final_augment_columns = [
        "id",
        "Comment",
        "Label",
        "source_id",
        "source_group",
        "fold",
        "variant",
        "augmentation_method",
        "generation_model",
        "prompt_version",
        "qa_status",
        "human_review_status",
    ]
    all_augments = pd.concat(
        [binary_hate[final_augment_columns], selected_nonhate[final_augment_columns]],
        ignore_index=True,
        sort=False,
    )
    expected_per_label = int(
        len(originals.loc[originals["Label"].eq("Hate")]) + len(binary_hate)
    )
    verification = verify_outputs(originals, all_augments, expected_per_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(hate_audit, output_dir / "hate_augmentation_audit.csv")
    shutil.copyfile(
        input_dir / "binary_validation_inputs.csv",
        output_dir / "binary_validation_inputs.csv",
    )
    _write_csv(source_pool, output_dir / "nonhate_source_pool.csv")
    _write_csv(candidates, output_dir / "nonhate_candidates.csv")
    _write_csv(candidates, output_dir / "nonhate_generation_audit.csv")
    _write_csv(
        candidates.loc[~candidates["selected"]].copy(),
        output_dir / "nonhate_rejected.csv",
    )
    human_review_queue = candidates.loc[
        candidates["selected"] & candidates["human_review_required"]
    ].copy()
    human_review_queue["review_decision"] = ""
    human_review_queue["review_note"] = ""
    human_review_queue["reviewer"] = ""
    _write_csv(human_review_queue, output_dir / "human_review_queue.csv")
    _write_csv(originals, output_dir / "original_with_folds.csv")

    for fold in FOLDS:
        original_fold = originals.loc[
            originals["fold"].eq(fold), ["id", "Comment", "Label", "record_origin"]
        ].copy()
        augment_fold = all_augments.loc[all_augments["fold"].eq(fold)].copy()
        _write_csv(original_fold, output_dir / f"binary_original_{fold}.csv")
        _write_csv(augment_fold, output_dir / f"binary_augment{fold}.csv")

    combined = pd.concat(
        [
            originals.assign(data_origin="original"),
            all_augments.assign(data_origin="augmentation"),
        ],
        ignore_index=True,
        sort=False,
    )
    _write_csv(combined, output_dir / "binary_train_balanced.csv")

    balance_summary = (
        combined.groupby(["fold", "data_origin", "Label"])
        .size()
        .rename("rows")
        .reset_index()
    )
    _write_csv(balance_summary, output_dir / "balance_summary.csv")

    summary = {
        "seed": seed,
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "binary_original_rows": len(binary),
        "prediction_input_rows": len(prediction_inputs),
        "recovered_hate_source_rows": len(recovered),
        "accepted_hate_augment_rows": len(binary_hate),
        "nonhate_target_rows": target_total,
        "nonhate_candidate_rows": len(candidates),
        "nonhate_selected_rows": len(selected_nonhate),
        "nonhate_surplus_rows": int((~candidates["selected"]).sum()),
        "target_by_fold": target_fold_counts,
        "target_by_method": target_method_counts,
        "candidate_by_fold": candidate_fold_counts,
        "candidate_by_method": candidate_method_counts,
        "target_by_fold_and_method": {
            f"fold_{fold}:{method}": count
            for (fold, method), count in sorted(target_pair_counts.items())
        },
        "human_review_queue_rows": int(
            (
                candidates["selected"] & candidates["human_review_required"]
            ).sum()
        ),
        "release_status": "provisional_pending_human_review_and_gpu_experiment",
        "verification": verification,
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Task A balanced data\n\n"
        "This directory is generated by `scripts/build_task_a_balanced_data.py`.\n\n"
        "- `binary_original_1.csv` ... `binary_original_5.csv`: original rows.\n"
        "- `binary_augment1.csv` ... `binary_augment5.csv`: accepted Hate and "
        "selected Non-Hate augmentations.\n"
        "- `binary_validation_inputs.csv`: unchanged development input copy.\n"
        "- `human_review_queue.csv`: fill `review_decision` with `accept` or "
        "`reject`, plus an optional note and reviewer.\n"
        "- `binary_train_balanced.csv`: provisional combined view for analysis.\n"
        "- `pipeline_summary.json`: quotas, counts, verification, and release status.\n\n"
        "The current release remains provisional until the human-review queue is "
        "completed and the GPU comparison experiment passes. During model selection, "
        "use folds 1-4 for training and original fold 5 only for validation.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root.parent / "hastika_regenerated_augments",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root.parent / "task_a_balanced",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--buffer-ratio", type=float, default=BUFFER_RATIO)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.seed < 0:
        raise ValueError("seed must be non-negative")
    if arguments.buffer_ratio < 0:
        raise ValueError("buffer-ratio must be non-negative")
    build(
        arguments.input_dir.resolve(),
        arguments.output_dir.resolve(),
        arguments.seed,
        arguments.buffer_ratio,
    )
