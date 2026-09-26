"""Resume Task A from results-7.zip in short, independent Kaggle versions.

The archive has four completed selection summaries and no optimizer state.
Full-data retraining starts from a fresh model, as in the original pipeline.
Seed 82 must repeat selection because its archived selection stopped at epoch 2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator


SEEDS = (42, 52, 62, 72, 82)
COMPLETE_SELECTION_SEEDS = SEEDS[:-1]
ARCHIVE_PREFIX = "Hastika_Icon_2026_Viper/outputs/task_a_qwen3_4b_2507_lora"
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/hastika_resume")
DEFAULT_CHECKPOINT_ROOT = Path("/kaggle/working/hastika_resume_checkpoints")


class DirectoryArchive:
    """Read the same artifact paths when Kaggle extracted the ZIP as a Dataset."""

    def __init__(self, directory: Path):
        candidates = [directory / "Hastika_Icon_2026_Viper", directory]
        self.project_root = next(
            (path for path in candidates if (path / "outputs" / "task_a_qwen3_4b_2507_lora").is_dir()),
            None,
        )
        if self.project_root is None:
            for history in directory.rglob("selection_training_history.json"):
                if (history.parent.name == "seed_42"
                        and history.parent.parent.name == "task_a_qwen3_4b_2507_lora"):
                    self.project_root = history.parents[3]
                    break
        if self.project_root is None:
            raise ValueError(f"Không tìm thấy kết quả Task A trong {directory}")

    def _file(self, member: str) -> Path:
        prefix = "Hastika_Icon_2026_Viper/"
        if not member.startswith(prefix):
            raise KeyError(member)
        path = self.project_root / member[len(prefix):]
        if not path.is_file():
            raise KeyError(member)
        return path

    def read(self, member: str) -> bytes:
        return self._file(member).read_bytes()

    def namelist(self) -> list[str]:
        prefix = "Hastika_Icon_2026_Viper/"
        return [prefix + str(path.relative_to(self.project_root))
                for path in self.project_root.rglob("*") if path.is_file()]


@contextmanager
def open_artifacts(path: Path) -> Iterator[zipfile.ZipFile | DirectoryArchive]:
    """Accept the original ZIP, a small metadata ZIP, or an extracted Dataset."""
    if path.is_dir():
        archives = [
            candidate for name in ("task_a_resume_metadata.zip", "results-7.zip")
            for candidate in path.rglob(name)
        ]
        if len(archives) > 1:
            raise ValueError(f"Có nhiều archive trong {path}: {archives}")
        if archives:
            with zipfile.ZipFile(archives[0]) as archive:
                yield archive
        else:
            yield DirectoryArchive(path)
    else:
        with zipfile.ZipFile(path) as archive:
            yield archive


def archive_member(seed: int, name: str) -> str:
    return f"{ARCHIVE_PREFIX}/seed_{seed}/{name}"


def selection_summary(archive: zipfile.ZipFile | DirectoryArchive, seed: int) -> dict:
    member = archive_member(seed, "seed_summary.json")
    try:
        summary = json.loads(archive.read(member))
        history = json.loads(
            archive.read(archive_member(seed, "selection_training_history.json"))
        )
    except KeyError as exc:
        raise ValueError(f"Thiếu selection artifact: {exc}") from exc
    epochs = history.get("epochs", [])
    if summary.get("seed") != seed or not epochs:
        raise ValueError(f"Seed {seed}: summary/history không hợp lệ")
    best = max(epochs, key=lambda row: (row["macro_f1"], -row["loss"], -row["epoch"]))
    if int(summary["best_epoch"]) != int(best["epoch"]):
        raise ValueError(f"Seed {seed}: best_epoch không khớp history")
    if int(summary["full_retrain_rows"]) < 1:
        raise ValueError(f"Seed {seed}: full_retrain_rows không hợp lệ")
    return summary


def write_metadata(source: Path, destination: Path) -> None:
    """Keep only the artifacts needed to resume; do not copy 1.4 GB of adapters."""
    if source.resolve() == destination.resolve():
        raise ValueError("Archive đầu ra phải khác archive nguồn")
    with open_artifacts(source) as archive:
        members = []
        for seed in COMPLETE_SELECTION_SEEDS:
            selection_summary(archive, seed)
            members.extend(
                [archive_member(seed, "seed_summary.json"),
                 archive_member(seed, "selection_training_history.json"),
                 archive_member(seed, "split_manifest.csv")]
            )
        members.append(archive_member(82, "selection_training_history.json"))
        members.append(archive_member(42, "validation_predictions.csv"))
        members.append(
            "Hastika_Icon_2026_Viper/checkpoints/"
            "task_a_qwen3_4b_2507_lora/seed_42/best_holdout_adapter/adapter_config.json"
        )
        source_commit = archive.read(
            "Hastika_Icon_2026_Viper/.git/refs/heads/Nhotest"
        ).decode().strip()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as output:
            for member in members:
                output.writestr(member, archive.read(member))
            output.writestr(
                "RECOVERY_INFO.json",
                json.dumps(
                    {"source_archive": source.name, "source_commit": source_commit,
                     "completed_selection_seeds": list(COMPLETE_SELECTION_SEEDS),
                     "incomplete_selection_seed": 82}, indent=2
                ),
            )
    print(f"Đã tạo {destination} ({destination.stat().st_size:,} bytes)")


def inspect_archive(path: Path) -> None:
    with open_artifacts(path) as archive:
        names = set(archive.namelist())
        for seed in SEEDS:
            history_name = archive_member(seed, "selection_training_history.json")
            history = json.loads(archive.read(history_name))
            summary_name = archive_member(seed, "seed_summary.json")
            if summary_name in names:
                summary = selection_summary(archive, seed)
                print(f"seed {seed}: selection xong, {len(history['epochs'])} epochs; "
                      f"best_epoch={summary['best_epoch']}, "
                      f"macro_f1={summary['macro_f1']:.6f}")
            else:
                print(f"seed {seed}: selection DỞ DANG sau "
                      f"{len(history['epochs'])} epochs; cần chạy lại selection")


def archive_preflight(archive_path: Path, seed: int, frame, task, config) -> dict:
    with open_artifacts(archive_path) as archive:
        summary = selection_summary(archive, seed)
        member = archive_member(seed, "split_manifest.csv")
        manifest = list(csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig"))))
        archived_ids = [row["id"] for row in manifest]
        actual_ids = frame["id"].astype(str).tolist()
        if archived_ids != actual_ids:
            raise ValueError(
                "Dữ liệu train hiện tại không khớp thứ tự ID trong split_manifest "
                f"của seed {seed}; hãy gắn đúng dataset của lần chạy cũ"
            )
        if len(frame) != int(summary["full_retrain_rows"]):
            raise ValueError("Số dòng full retrain khác seed_summary trong archive")
        if summary["downsampled_rows"] != 0 or task.selection_train_label_caps:
            raise ValueError("Resume Task A này chỉ hỗ trợ không downsample")
        adapter_config_member = (
            "Hastika_Icon_2026_Viper/checkpoints/"
            "task_a_qwen3_4b_2507_lora/seed_42/best_holdout_adapter/adapter_config.json"
        )
        if adapter_config_member in archive.namelist():
            saved = json.loads(archive.read(adapter_config_member))
            if (saved["base_model_name_or_path"] != config.model_name
                    or saved["r"] != config.lora_rank
                    or saved["lora_alpha"] != config.lora_alpha):
                raise ValueError("Model hoặc cấu hình LoRA khác archive")
        return summary


def train_completed_seed(args: argparse.Namespace) -> None:
    """Replay one completed seed's full-data phase, without repeating selection."""
    from llm_config import LLM_TRAINING, TASK_A
    from src.data import load_task_frames
    from src.distributed import (
        cleanup_runtime, initialize_runtime, relaunch_with_torchrun_if_needed
    )
    from src.llm_data import ChatEncoder, CompletionCollator, CompletionDataset
    from src.llm_training import (
        _training_state, _wrap_for_ddp, load_lora_model, on_main,
        require_cuda, restore_adapter, save_adapter, score_candidates,
        train_llm_epoch, validate_config
    )
    from src.submission import write_submission
    from src.training import create_train_loader, log, set_global_seed
    from transformers import AutoTokenizer

    config = replace(LLM_TRAINING, use_all_available_gpus=not args.single_gpu)
    validate_config(config)
    relaunch_with_torchrun_if_needed(config)
    runtime = initialize_runtime()
    try:
        require_cuda(runtime)
        task = replace(TASK_A, prediction_filename=args.prediction_file)
        frame, prediction_frame = load_task_frames(task)
        summary = archive_preflight(args.archive, args.seed, frame, task, config)
        epochs = int(summary["best_epoch"])
        output_dir = args.output_root / f"seed_{args.seed}"
        adapter_dir = args.checkpoint_root / f"seed_{args.seed}" / "final_full_data_adapter"
        if (output_dir / "seed_run.json").exists():
            raise ValueError(f"Seed {args.seed} đã hoàn tất ở {output_dir}")
        on_main(lambda: output_dir.mkdir(parents=True, exist_ok=True), runtime)
        set_global_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, revision=config.model_revision, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        encoder = ChatEncoder(tokenizer, task, config.max_length)
        collator = CompletionCollator(tokenizer.pad_token_id)
        full_data = CompletionDataset(
            frame["text"].tolist(), encoder, frame["label_id"].tolist()
        )
        prediction_data = CompletionDataset(prediction_frame["text"].tolist(), encoder)
        loader, sampler = create_train_loader(full_data, collator, config, runtime, args.seed)
        model = _wrap_for_ddp(load_lora_model(config, runtime), runtime)
        optimizer, scheduler, scaler = _training_state(model, loader, epochs, config, runtime)
        history = []
        log(f"Seed {args.seed}: full-data retrain {len(full_data)} rows x {epochs} epochs", runtime)
        for epoch in range(1, epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            loss = train_llm_epoch(
                model, loader, optimizer, scheduler, scaler, config, runtime, epoch,
                f"Full-data retrain seed {args.seed}"
            )
            history.append({"epoch": epoch, "train_loss": loss})
            on_main(
                lambda rows=list(history): (output_dir / "full_data_training_history.json")
                .write_text(json.dumps({"seed": args.seed, "epochs": rows}, indent=2)),
                runtime,
            )
        del optimizer, scheduler, scaler
        on_main(lambda: save_adapter(model, tokenizer, adapter_dir), runtime)
        restore_adapter(model, adapter_dir)
        scores = score_candidates(
            model, prediction_data, collator, config, runtime,
            "Development prediction from full-data adapter"
        )

        def finish() -> None:
            label_ids = scores["probabilities"].argmax(axis=1)
            csv_path, _ = write_submission(
                prediction_frame["id"].tolist(),
                [task.labels[index] for index in label_ids],
                task.labels, output_dir, task.submission_zip_name
            )
            (output_dir / "seed_run.json").write_text(
                json.dumps({
                    "seed": args.seed, "best_epoch": epochs,
                    "selection_macro_f1": summary["macro_f1"],
                    "training_rows": len(full_data),
                    "model": config.model_name, "revision": config.model_revision,
                    "prediction_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                    "selection_archive": args.archive.name,
                }, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Đã xong seed {args.seed}: {csv_path}", flush=True)

        on_main(finish, runtime)
    finally:
        cleanup_runtime(runtime)


def redo_seed_82(args: argparse.Namespace) -> None:
    """Run the original single-seed selection and full retrain for incomplete 82."""
    from llm_config import LLM_TRAINING, TASK_A
    from src.distributed import relaunch_with_torchrun_if_needed
    import src.llm_training as training

    config = replace(LLM_TRAINING, use_all_available_gpus=not args.single_gpu)
    task = replace(
        TASK_A, selection_seeds=(82,), ensemble_all_seeds=False,
        prediction_filename=args.prediction_file,
        output_subdir="seed_82", checkpoint_subdir="seed_82"
    )
    if (args.output_root / "seed_82" / "run_summary.json").exists():
        raise ValueError("Seed 82 đã hoàn tất; không chạy đè")
    training.OUTPUT_ROOT = args.output_root
    training.CHECKPOINT_ROOT = args.checkpoint_root
    training.validate_config(config)
    relaunch_with_torchrun_if_needed(config)
    training.run_llm_task(task, config)


def read_prediction(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError(f"{path}: cần đúng header id,label")
        ordered_ids, by_id = [], {}
        for row in reader:
            sample_id, label = row["id"], row["label"]
            if not sample_id or sample_id in by_id or label not in {"Hate", "Non-Hate"}:
                raise ValueError(f"{path}: ID hoặc nhãn không hợp lệ: {row}")
            ordered_ids.append(sample_id)
            by_id[sample_id] = label
        if not ordered_ids:
            raise ValueError(f"{path}: không có prediction")
    return ordered_ids, by_id


def ensemble(args: argparse.Namespace) -> None:
    if len(args.seed_dir) != 5:
        raise ValueError("Cần đúng năm thư mục seed 42, 52, 62, 72, 82")
    by_seed = {}
    model_keys = set()
    for directory in args.seed_dir:
        if (directory / "seed_run.json").is_file():
            meta = json.loads((directory / "seed_run.json").read_text())
            seed = int(meta["seed"])
            model_keys.add((meta["model"], meta["revision"]))
            csv_path = directory / "predictions.csv"
            if hashlib.sha256(csv_path.read_bytes()).hexdigest() != meta["prediction_sha256"]:
                raise ValueError(f"{directory}: prediction đã thay đổi")
        elif (directory / "run_summary.json").is_file():
            meta = json.loads((directory / "run_summary.json").read_text())
            seed = int(meta["winning_seed"])
            if seed != 82 or meta["full_data_retrain"]["seed"] != 82:
                raise ValueError(f"{directory}: run_summary không phải full retrain seed 82")
            model_keys.add((meta["model"], meta["training_config"]["model_revision"]))
            csv_path = directory / "predictions.csv"
        else:
            raise ValueError(f"{directory}: thiếu seed_run.json hoặc run_summary.json")
        if seed in by_seed or seed not in SEEDS:
            raise ValueError(f"Seed trùng hoặc không hợp lệ: {seed}")
        by_seed[seed] = read_prediction(csv_path)
    if set(by_seed) != set(SEEDS) or len(model_keys) != 1:
        raise ValueError("Thiếu seed hoặc các seed dùng model/revision khác nhau")
    ordered_ids = by_seed[42][0]
    expected = set(ordered_ids)
    for seed, (ids, _) in by_seed.items():
        if set(ids) != expected:
            raise ValueError(f"Seed {seed}: tập ID khác seed 42")
    result = []
    for sample_id in ordered_ids:
        votes = Counter(by_seed[seed][1][sample_id] for seed in SEEDS)
        result.append((sample_id, votes.most_common(1)[0][0]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "label"])
        writer.writerows(result)
    with zipfile.ZipFile(args.output_dir / "task_a_submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_path, arcname="predictions.csv")
    (args.output_dir / "ensemble_summary.json").write_text(
        json.dumps({"seeds": list(SEEDS), "rows": len(result),
                    "method": "five-seed hard majority vote"}, indent=2) + "\n"
    )
    print(f"Đã tạo {prediction_path} và task_a_submission.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--archive", type=Path, required=True)
    metadata_parser = commands.add_parser("metadata")
    metadata_parser.add_argument("--archive", type=Path, required=True)
    metadata_parser.add_argument("--output", type=Path, required=True)
    train_parser = commands.add_parser("train-seed")
    train_parser.add_argument("--archive", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, choices=COMPLETE_SELECTION_SEEDS, required=True)
    seed82_parser = commands.add_parser("redo-seed-82")
    for subparser in (train_parser, seed82_parser):
        subparser.add_argument("--prediction-file", default="binary_validation_inputs.csv")
        subparser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        subparser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
        subparser.add_argument("--single-gpu", action="store_true")
    ensemble_parser = commands.add_parser("ensemble")
    ensemble_parser.add_argument("--seed-dir", type=Path, action="append", required=True)
    ensemble_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            inspect_archive(args.archive)
        elif args.command == "metadata":
            write_metadata(args.archive, args.output)
        elif args.command == "train-seed":
            train_completed_seed(args)
        elif args.command == "redo-seed-82":
            redo_seed_82(args)
        else:
            ensemble(args)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
