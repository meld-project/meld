#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


LOGGER = logging.getLogger("prepare_family_ood_split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--malicious-manifest", type=Path, required=True)
    parser.add_argument("--mal-dir", type=Path, required=True)
    parser.add_argument("--benign-dir", type=Path, required=True)
    parser.add_argument("--benign-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-start", type=str, default="2024-11-01 00:00:00")
    parser.add_argument("--window-end", type=str, default="2025-04-30 23:59:59")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-family", nargs="*", default=["unknown"])
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def index_md_files(root: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for path in sorted(root.rglob("*.md")):
        index[path.stem.lower()] = str(path.resolve())
    return index


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_timestamp(raw: str) -> datetime | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def stable_fold_seed(base_seed: int, family: str) -> int:
    digest = hashlib.md5(family.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def split_family_ids(
    family_to_ids: Dict[str, List[int]],
    held_out_family: str,
    val_ratio: float,
    seed: int,
) -> tuple[List[int], List[int], List[int]]:
    train_ids: List[int] = []
    val_ids: List[int] = []
    test_ids = sorted(family_to_ids[held_out_family])
    rng = random.Random(stable_fold_seed(seed, held_out_family))
    for family, ids in sorted(family_to_ids.items()):
        if family == held_out_family:
            continue
        shuffled = list(ids)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            train_ids.extend(shuffled)
            continue
        val_count = int(round(len(shuffled) * val_ratio))
        if val_count <= 0:
            val_count = 1
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1
        val_ids.extend(sorted(shuffled[:val_count]))
        train_ids.extend(sorted(shuffled[val_count:]))
    return sorted(train_ids), sorted(val_ids), test_ids


def pick_benign_ids(
    benign_ids: Sequence[int],
    held_out_family: str,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> tuple[List[int], List[int], List[int]]:
    needed = train_count + val_count + test_count
    if needed > len(benign_ids):
        raise ValueError(f"良性样本不足：需要 {needed}，但只有 {len(benign_ids)}")
    pool = list(benign_ids)
    rng = random.Random(stable_fold_seed(seed + 100_000, held_out_family))
    rng.shuffle(pool)
    train_ids = sorted(pool[:train_count])
    val_ids = sorted(pool[train_count:train_count + val_count])
    test_ids = sorted(pool[train_count + val_count:needed])
    return train_ids, val_ids, test_ids


def load_malicious_docs(
    manifest_rows: Iterable[Dict[str, str]],
    md_index: Dict[str, str],
    *,
    window_start: datetime,
    window_end: datetime,
    excluded_families: set[str],
) -> tuple[List[Dict], Dict[str, List[int]], Counter, int]:
    docs: List[Dict] = []
    family_to_ids: Dict[str, List[int]] = defaultdict(list)
    family_counts: Counter = Counter()
    missing_reports = 0
    for row in manifest_rows:
        sha256 = (row.get("sha256") or "").strip().lower()
        family = (row.get("family") or "").strip().lower()
        timestamp = parse_timestamp(row.get("first_seen", ""))
        if not sha256 or not family or family in excluded_families or timestamp is None:
            continue
        if not (window_start <= timestamp <= window_end):
            continue
        path = md_index.get(sha256)
        if not path:
            missing_reports += 1
            continue
        doc_id = len(docs)
        docs.append(
            {
                "path": path,
                "label": 1,
                "sha256": sha256,
                "family": family,
                "source": "malicious",
                "first_seen": row.get("first_seen"),
            }
        )
        family_to_ids[family].append(doc_id)
        family_counts[family] += 1
    return docs, family_to_ids, family_counts, missing_reports


def load_benign_docs(
    benign_dir: Path,
    *,
    start_doc_id: int,
    benign_manifest_rows: Iterable[Dict[str, str]] | None = None,
) -> List[Dict]:
    allowed_sha: set[str] | None = None
    if benign_manifest_rows is not None:
        allowed_sha = {
            (row.get("sha256") or "").strip().lower()
            for row in benign_manifest_rows
            if (row.get("sha256") or "").strip()
        }
    docs: List[Dict] = []
    for path in sorted(benign_dir.rglob("*.md")):
        sha256 = path.stem.lower()
        if allowed_sha is not None and sha256 not in allowed_sha:
            continue
        docs.append(
            {
                "path": str(path.resolve()),
                "label": 0,
                "sha256": sha256,
                "family": "benign",
                "source": "benign",
                "doc_id": start_doc_id + len(docs),
            }
        )
    return docs


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("val_ratio 必须在 (0, 1) 之间")

    window_start = datetime.fromisoformat(args.window_start)
    window_end = datetime.fromisoformat(args.window_end)
    excluded_families = {item.strip().lower() for item in args.exclude_family if item.strip()}

    mal_index = index_md_files(args.mal_dir)
    mal_rows = read_csv(args.malicious_manifest)
    malicious_docs, family_to_ids, family_counts, missing_reports = load_malicious_docs(
        mal_rows,
        mal_index,
        window_start=window_start,
        window_end=window_end,
        excluded_families=excluded_families,
    )
    LOGGER.info(
        "恶意时间窗样本: %d (families=%d, missing_reports=%d)",
        len(malicious_docs),
        len(family_counts),
        missing_reports,
    )
    top_families = [family for family, _count in family_counts.most_common(args.top_k)]
    LOGGER.info("Top-%d families: %s", args.top_k, top_families)

    benign_rows = read_csv(args.benign_manifest) if args.benign_manifest else None
    benign_docs = load_benign_docs(args.benign_dir, start_doc_id=len(malicious_docs), benign_manifest_rows=benign_rows)
    LOGGER.info("良性 Markdown 样本: %d", len(benign_docs))

    docs: List[Dict] = []
    docs.extend(malicious_docs)
    for idx, item in enumerate(benign_docs, start=len(malicious_docs)):
        item = dict(item)
        item.pop("doc_id", None)
        docs.append(item)
    benign_ids = list(range(len(malicious_docs), len(docs)))

    folds: List[Dict] = []
    for fold_idx, held_out_family in enumerate(top_families, start=1):
        train_mal_ids, val_mal_ids, test_mal_ids = split_family_ids(
            family_to_ids,
            held_out_family,
            args.val_ratio,
            args.seed,
        )
        train_ben_ids, val_ben_ids, test_ben_ids = pick_benign_ids(
            benign_ids,
            held_out_family,
            len(train_mal_ids),
            len(val_mal_ids),
            len(test_mal_ids),
            args.seed,
        )
        fold = {
            "name": f"lofo_{held_out_family}",
            "fold_index": fold_idx,
            "held_out_family": held_out_family,
            "train_ids": train_mal_ids + train_ben_ids,
            "val_ids": val_mal_ids + val_ben_ids,
            "test_ids": test_mal_ids + test_ben_ids,
            "counts": {
                "train_malicious": len(train_mal_ids),
                "train_benign": len(train_ben_ids),
                "val_malicious": len(val_mal_ids),
                "val_benign": len(val_ben_ids),
                "test_malicious": len(test_mal_ids),
                "test_benign": len(test_ben_ids),
            },
        }
        folds.append(fold)

    spec = {
        "version": 1,
        "protocol": {
            "name": "family_ood_lofo",
            "window_start": args.window_start,
            "window_end": args.window_end,
            "top_k": args.top_k,
            "excluded_families": sorted(excluded_families),
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "malicious_manifest": str(args.malicious_manifest.resolve()),
            "mal_dir": str(args.mal_dir.resolve()),
            "benign_dir": str(args.benign_dir.resolve()),
            "benign_manifest": str(args.benign_manifest.resolve()) if args.benign_manifest else None,
            "benign_sampling": "random-balanced-without-timestamps",
            "notes": [
                "恶意样本按 family + first_seen 构造 LOFO folds。",
                "良性 manifest 缺少可靠时间戳，因此良性样本使用随机 1:1 配平采样。",
                "Top-k family 从时间窗内、且 Markdown 文件存在的恶意样本中统计。",
            ],
            "top_families": [{family: family_counts[family]} for family in top_families],
            "missing_malicious_reports": missing_reports,
        },
        "docs": docs,
        "folds": folds,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2)
    LOGGER.info("已写出 Family-OOD split spec: %s", args.output)


if __name__ == "__main__":
    main()
