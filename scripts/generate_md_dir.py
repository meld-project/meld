#!/usr/bin/env python3
"""
Generate md_dir style datasets for LEC training from CAPE markdown reports.

Typical usage:

    python scripts/generate_md_dir.py \
        --malware-dir data/cape_md/raw/cape_report_malware \
        --benign-dir data/cape_md/raw/cape_reports_benign \
        --output-dir data/cape_md/md_dir \
        --splits train:0.8,val:0.1,test:0.1

The script creates <output>/<subset>/{black,white} directories, populates them
with validated markdown files, and writes metadata.json per subset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

LOGGER = logging.getLogger("generate_md_dir")


BLACK_LABEL = "black"  # malicious
WHITE_LABEL = "white"  # benign


@dataclass(frozen=True)
class ReportFile:
    path: Path
    label: str
    size: int

    @property
    def stem(self) -> str:
        return self.path.stem


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CAPE markdown reports to md_dir format."
    )
    parser.add_argument(
        "--malware-dir",
        type=Path,
        required=True,
        help="Directory containing malicious markdown reports (*.md).",
    )
    parser.add_argument(
        "--benign-dir",
        type=Path,
        required=True,
        help="Directory containing benign markdown reports (*.md).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the generated md_dir dataset.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train:1.0",
        help="Comma separated split spec, e.g. train:0.8,val:0.1,test:0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for stratified splitting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing subset directories if present.",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Create symbolic links instead of copying files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report planned actions without touching outputs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity. Repeat for debug output.",
    )
    return parser.parse_args(argv)


def collect_reports(root: Path, label: str) -> List[ReportFile]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected directory: {root}")
    reports: List[ReportFile] = []
    for path in sorted(root.rglob("*.md")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size == 0:
            LOGGER.warning("Skipping empty markdown file: %s", path)
            continue
        reports.append(ReportFile(path=path.resolve(), label=label, size=size))
    if not reports:
        raise ValueError(f"No markdown files found under {root}")
    LOGGER.info("Collected %d %s reports from %s", len(reports), label, root)
    return reports


def detect_duplicates(reports: Iterable[ReportFile]) -> Dict[str, List[Path]]:
    seen: Dict[str, List[Path]] = {}
    for report in reports:
        seen.setdefault(report.stem.lower(), []).append(report.path)
    return {stem: paths for stem, paths in seen.items() if len(paths) > 1}


def parse_split_spec(spec: str) -> List[Tuple[str, float]]:
    splits: List[Tuple[str, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid split spec '{part}'. Expected name:ratio.")
        name, ratio_str = part.split(":", 1)
        name = name.strip()
        try:
            ratio = float(ratio_str)
        except ValueError as exc:
            raise ValueError(f"Invalid ratio '{ratio_str}' in split spec.") from exc
        if ratio <= 0:
            raise ValueError(f"Split ratio must be positive: {part}")
        splits.append((name, ratio))
    if not splits:
        raise ValueError("No valid splits parsed from spec.")
    return splits


def normalize_weights(splits: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
    total = sum(weight for _, weight in splits)
    if total <= 0:
        raise ValueError("Total split weight must be positive.")
    return [(name, weight / total) for name, weight in splits]


def distribute_counts(total: int, weights: Sequence[float]) -> List[int]:
    if total == 0:
        return [0] * len(weights)
    allocations = []
    remainders = []
    assigned = 0
    for idx, weight in enumerate(weights):
        raw = weight * total
        count = int(raw)
        allocations.append(count)
        remainders.append((raw - count, idx))
        assigned += count
    remaining = total - assigned
    for _, idx in sorted(remainders, reverse=True):
        if remaining == 0:
            break
        allocations[idx] += 1
        remaining -= 1
    return allocations


def stratified_split(
    reports: Sequence[ReportFile],
    splits: Sequence[Tuple[str, float]],
    seed: int,
) -> Dict[str, List[ReportFile]]:
    from random import Random

    rng = Random(seed)
    by_label: Dict[str, List[ReportFile]] = {BLACK_LABEL: [], WHITE_LABEL: []}
    for report in reports:
        by_label.setdefault(report.label, []).append(report)
    for label_reports in by_label.values():
        rng.shuffle(label_reports)

    weights = [weight for _, weight in splits]
    subset_buckets: Dict[str, List[ReportFile]] = {name: [] for name, _ in splits}

    for label, label_reports in by_label.items():
        counts = distribute_counts(len(label_reports), weights)
        cursor = 0
        for (subset_name, _), count in zip(splits, counts):
            if count == 0:
                continue
            subset_buckets[subset_name].extend(label_reports[cursor : cursor + count])
            cursor += count

    for subset_name, items in subset_buckets.items():
        rng.shuffle(items)
        LOGGER.info(
            "Subset %s contains %d samples (%d black / %d white).",
            subset_name,
            len(items),
            sum(1 for item in items if item.label == BLACK_LABEL),
            sum(1 for item in items if item.label == WHITE_LABEL),
        )
    return subset_buckets


def ensure_output_subset(path: Path, force: bool, dry_run: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Subset directory {path} already exists. Use --force to overwrite."
            )
        LOGGER.warning("Removing existing subset directory: %s", path)
        if not dry_run:
            shutil.rmtree(path)
    if not dry_run:
        (path / BLACK_LABEL).mkdir(parents=True, exist_ok=True)
        (path / WHITE_LABEL).mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, link: bool, dry_run: bool) -> None:
    if dry_run:
        LOGGER.debug("Dry run: %s -> %s", src, dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def build_metadata(
    subset_name: str,
    files: Sequence[ReportFile],
    output_dir: Path,
    mapping: Dict[Path, Path],
    source_dirs: Dict[str, str],
) -> Dict[str, object]:
    num_black = sum(1 for f in files if f.label == BLACK_LABEL)
    num_white = sum(1 for f in files if f.label == WHITE_LABEL)
    total_size = sum(f.size for f in files)
    metadata_files = []
    for report in files:
        dst = mapping[report.path]
        metadata_files.append(
            {
                "label": report.label,
                "source": str(report.path),
                "destination": str(dst.relative_to(output_dir)),
                "size_bytes": report.size,
                "stem": report.stem,
            }
        )
    created_at = datetime.now(timezone.utc).isoformat()
    return {
        "subset": subset_name,
        "created_at": created_at,
        "num_samples": len(files),
        "num_black": num_black,
        "num_white": num_white,
        "total_size_bytes": total_size,
        "source_directories": source_dirs,
        "files": metadata_files,
    }


def write_metadata(path: Path, metadata: Dict[str, object], dry_run: bool) -> None:
    if dry_run:
        LOGGER.info("Dry run: would write metadata to %s", path)
        return
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=False)
        fh.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    malware_reports = collect_reports(args.malware_dir, BLACK_LABEL)
    benign_reports = collect_reports(args.benign_dir, WHITE_LABEL)

    all_reports = malware_reports + benign_reports
    duplicates = detect_duplicates(all_reports)
    if duplicates:
        LOGGER.warning(
            "Detected %d duplicate stems between reports: %s",
            len(duplicates),
            ", ".join(list(duplicates)[:10]),
        )

    split_pairs = parse_split_spec(args.splits)
    normalized_splits = normalize_weights(split_pairs)
    LOGGER.info(
        "Using splits: %s",
        ", ".join(f"{name}:{weight:.3f}" for name, weight in normalized_splits),
    )
    subsets = stratified_split(all_reports, normalized_splits, args.seed)

    output_dir = args.output_dir
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = {
        BLACK_LABEL: str(args.malware_dir.resolve()),
        WHITE_LABEL: str(args.benign_dir.resolve()),
    }

    overall_stats = {
        BLACK_LABEL: len(malware_reports),
        WHITE_LABEL: len(benign_reports),
        "total": len(all_reports),
    }
    LOGGER.info("Dataset totals: %s", overall_stats)

    for subset_name, reports in subsets.items():
        subset_dir = output_dir / subset_name
        ensure_output_subset(subset_dir, args.force, args.dry_run)
        file_map: Dict[Path, Path] = {}
        for report in reports:
            dst_dir = subset_dir / report.label
            dst_path = dst_dir / (report.path.name)
            copy_or_link(report.path, dst_path, args.link, args.dry_run)
            file_map[report.path] = dst_path

        metadata = build_metadata(
            subset_name=subset_name,
            files=reports,
            output_dir=output_dir,
            mapping=file_map,
            source_dirs=source_dirs,
        )
        write_metadata(subset_dir / "metadata.json", metadata, args.dry_run)
        LOGGER.info(
            "Finished subset %s: %d files (%d black / %d white).",
            subset_name,
            metadata["num_samples"],
            metadata["num_black"],
            metadata["num_white"],
        )

    LOGGER.info("Completed md_dir generation at %s", output_dir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Dataset generation failed: %s", exc)
        sys.exit(1)
