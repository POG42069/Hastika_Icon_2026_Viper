"""Discover and validate reusable Kaggle artifacts for interrupted runs."""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class ResumeCatalog:
    """One independently produced result tree."""

    source: str
    output_dir: Path | None
    checkpoint_dir: Path | None

    def seed_output(self, seed: int) -> Path | None:
        path = self.output_dir / f"seed_{seed}" if self.output_dir else None
        return path if path is not None and path.is_dir() else None

    def seed_checkpoint(self, seed: int) -> Path | None:
        path = self.checkpoint_dir / f"seed_{seed}" if self.checkpoint_dir else None
        return path if path is not None and path.is_dir() else None


def _canonical_member(name: str, output_subdir: str, checkpoint_subdir: str):
    """Return a safe path below outputs/ or checkpoints/, ignoring ZIP prefixes."""

    parts = PurePosixPath(name).parts
    for kind, subdir in (
        ("outputs", output_subdir),
        ("checkpoints", checkpoint_subdir),
    ):
        needle = (kind, subdir)
        for index in range(len(parts) - 1):
            if tuple(parts[index : index + 2]) == needle:
                relative = PurePosixPath(*parts[index:])
                if ".." not in relative.parts:
                    return relative
    return None


def _extract_relevant_zip(
    archive_path: Path,
    destination: Path,
    output_subdir: str,
    checkpoint_subdir: str,
) -> bool:
    """Extract only this task's artifacts and reject unsafe archive paths."""

    extracted = False
    with zipfile.ZipFile(archive_path) as archive:
        members = []
        completed_selection_seeds: set[str] = set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = _canonical_member(
                info.filename, output_subdir, checkpoint_subdir
            )
            if relative is None:
                continue
            members.append((info, relative))
            parts = relative.parts
            if (
                parts[0] == "outputs"
                and len(parts) == 4
                and parts[2].startswith("seed_")
                and parts[3] == "seed_summary.json"
            ):
                try:
                    value = json.loads(archive.read(info))
                    if int(value["seed"]) == int(parts[2].removeprefix("seed_")):
                        completed_selection_seeds.add(parts[2])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass

        for info, relative in members:
            parts = relative.parts
            # A completed selection needs only its reports for phase 2. Avoid
            # extracting its large legacy holdout adapter from cumulative ZIPs.
            if (
                parts[0] == "checkpoints"
                and len(parts) > 3
                and parts[2] in completed_selection_seeds
                and parts[3] == "best_holdout_adapter"
            ):
                continue
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted = True
    return extracted


def _task_dirs(root: Path, kind: str, subdir: str) -> list[Path]:
    matches = []
    if root.is_dir() and root.name == subdir and root.parent.name == kind:
        matches.append(root)
    direct = root / kind / subdir
    if direct.is_dir():
        matches.append(direct)
    if root.is_dir():
        for path in root.rglob(subdir):
            if path.is_dir() and path.parent.name == kind and path not in matches:
                matches.append(path)
    return sorted(matches)


def _catalogs_below(
    root: Path, output_subdir: str, checkpoint_subdir: str, source: str
) -> list[ResumeCatalog]:
    outputs = _task_dirs(root, "outputs", output_subdir)
    checkpoints = _task_dirs(root, "checkpoints", checkpoint_subdir)
    catalogs: list[ResumeCatalog] = []
    used_checkpoints: set[Path] = set()
    for output in outputs:
        best = next(
            (
                checkpoint
                for checkpoint in checkpoints
                if checkpoint.parents[1] == output.parents[1]
            ),
            None,
        )
        if best is not None:
            used_checkpoints.add(best)
        catalogs.append(ResumeCatalog(source, output, best))
    for checkpoint in checkpoints:
        if checkpoint not in used_checkpoints:
            catalogs.append(ResumeCatalog(source, None, checkpoint))
    return catalogs


def prepare_resume_catalogs(
    sources: list[Path],
    scratch_root: Path,
    output_subdir: str,
    checkpoint_subdir: str,
) -> list[ResumeCatalog]:
    """Index task artifacts in directories and ZIPs, extracting ZIP members once."""

    scratch_root.mkdir(parents=True, exist_ok=True)
    catalogs: list[ResumeCatalog] = []
    seen_archives: set[Path] = set()
    for source_number, source in enumerate(sources):
        source = source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Resume source does not exist: {source}")
        if source.is_file():
            if not zipfile.is_zipfile(source):
                raise ValueError(f"Resume source file is not a ZIP archive: {source}")
            archives = [source]
            directory_roots: list[Path] = []
        else:
            directory_roots = [source]
            archives = [
                path for path in source.rglob("*.zip") if zipfile.is_zipfile(path)
            ]
        for root in directory_roots:
            catalogs.extend(
                _catalogs_below(root, output_subdir, checkpoint_subdir, str(source))
            )
        for archive_number, archive in enumerate(archives):
            resolved = archive.resolve()
            if resolved in seen_archives:
                continue
            seen_archives.add(resolved)
            destination = (
                scratch_root / f"source_{source_number:03d}_{archive_number:03d}"
            )
            if _extract_relevant_zip(
                archive,
                destination,
                output_subdir,
                checkpoint_subdir,
            ):
                catalogs.extend(
                    _catalogs_below(
                        destination,
                        output_subdir,
                        checkpoint_subdir,
                        str(archive),
                    )
                )
    # Remove duplicate task-directory pairs while preserving discovery order.
    unique: list[ResumeCatalog] = []
    keys = set()
    for catalog in catalogs:
        key = (catalog.output_dir, catalog.checkpoint_dir)
        if key not in keys:
            keys.add(key)
            unique.append(catalog)
    return unique


def write_catalog_manifest(path: Path, catalogs: list[ResumeCatalog]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "source": item.source,
                    "output_dir": str(item.output_dir) if item.output_dir else None,
                    "checkpoint_dir": str(item.checkpoint_dir)
                    if item.checkpoint_dir
                    else None,
                }
                for item in catalogs
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def read_catalog_manifest(path: Path) -> list[ResumeCatalog]:
    values = json.loads(path.read_text(encoding="utf-8"))
    return [
        ResumeCatalog(
            value["source"],
            Path(value["output_dir"]) if value["output_dir"] else None,
            Path(value["checkpoint_dir"]) if value["checkpoint_dir"] else None,
        )
        for value in values
    ]


def copy_seed_outputs(source: Path, destination: Path) -> None:
    """Consolidate reports from the selected best source tree."""

    if source.resolve() == destination.resolve():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
