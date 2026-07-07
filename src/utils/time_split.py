"""Helpers for manifest-driven time splits used by MELD experiments."""

from __future__ import annotations

import csv
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("time_split")

TIMESTAMP_KEYS = (
    "first_seen",
    "first_seen_utc",
    "first_seen_at",
    "first_seen_time",
    "timestamp",
    "time",
    "datetime",
    "date",
    "submitted_at",
    "created_at",
    "report_time",
)

ID_KEYS = (
    "sha256",
    "sha_256",
    "sha-256",
    "sample_id",
    "id",
    "hash",
    "file_hash",
    "Malware SHA-256",
)

PATH_KEYS = (
    "md_path",
    "markdown_path",
    "report_path",
    "path",
    "filepath",
)

_MARKDOWN_INDEX_CACHE: Dict[str, Dict[str, str]] = {}


def _normalise_datetime(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def parse_timestamp(value: object) -> datetime:
    if value is None:
        raise ValueError("timestamp is missing")

    text = str(value).strip()
    if not text:
        raise ValueError("timestamp is empty")

    if text.isdigit():
        raw = int(text)
        if len(text) >= 13:
            raw = raw / 1000
        return _normalise_datetime(datetime.fromtimestamp(raw, tz=timezone.utc))

    candidates = [text, text.replace("Z", "+00:00")]
    if "/" in text:
        candidates.append(text.replace("/", "-"))

    for candidate in candidates:
        try:
            return _normalise_datetime(datetime.fromisoformat(candidate))
        except ValueError:
            continue

    fmts = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    )
    for fmt in fmts:
        try:
            return _normalise_datetime(datetime.strptime(text, fmt))
        except ValueError:
            continue
    raise ValueError(f"unsupported timestamp format: {text}")


def _get_value(row: Dict[str, object], keys: Sequence[str]) -> Optional[object]:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key.lower() in lowered:
            value = lowered[key.lower()]
            if value is not None and str(value).strip():
                return value
    return None


def _read_manifest_rows(path: Path) -> List[Dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            if isinstance(payload.get("items"), list):
                return [item for item in payload["items"] if isinstance(item, dict)]
            return [payload]
        raise ValueError(f"Unsupported JSON manifest format: {path}")

    with path.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.excel
        if suffix == ".tsv":
            dialect = csv.excel_tab
        else:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            except csv.Error:
                dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        return [dict(row) for row in reader]


def _build_markdown_index(root_dir: Path) -> Dict[str, str]:
    cache_key = str(root_dir.resolve())
    cached = _MARKDOWN_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    index: Dict[str, str] = {}
    for path in root_dir.rglob("*.md"):
        index[path.stem.lower()] = str(path.resolve())
    _MARKDOWN_INDEX_CACHE[cache_key] = index
    return index


def _resolve_markdown_path(
    row: Dict[str, object],
    *,
    manifest_path: Path,
    md_root: Optional[Path],
) -> Optional[str]:
    raw_path = _get_value(row, PATH_KEYS)
    if raw_path is not None:
        candidate = Path(str(raw_path)).expanduser()
        if not candidate.is_absolute():
            candidate = (manifest_path.parent / candidate).resolve()
        if candidate.exists():
            return str(candidate)

    sample_id = _get_value(row, ID_KEYS)
    if sample_id is None or md_root is None:
        return None

    stem = str(sample_id).strip().lower()
    direct = (md_root / f"{stem}.md").resolve()
    if direct.exists():
        return str(direct)

    index = _build_markdown_index(md_root)
    return index.get(stem)


def _load_single_manifest(
    manifest_path: str | Path,
    *,
    label: int,
    md_root: Optional[str | Path],
) -> List[Dict[str, object]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    root_dir = Path(md_root).expanduser().resolve() if md_root else None
    rows = _read_manifest_rows(path)

    records: List[Dict[str, object]] = []
    missing_paths = 0
    missing_timestamps = 0

    for row in rows:
        resolved_path = _resolve_markdown_path(row, manifest_path=path, md_root=root_dir)
        if resolved_path is None:
            missing_paths += 1
            continue

        sample_id = _get_value(row, ID_KEYS)
        timestamp_raw = _get_value(row, TIMESTAMP_KEYS)
        timestamp = None
        if timestamp_raw is not None:
            try:
                timestamp = parse_timestamp(timestamp_raw)
            except ValueError:
                missing_timestamps += 1
        else:
            missing_timestamps += 1

        records.append(
            {
                "sample_id": str(sample_id).strip() if sample_id is not None else Path(resolved_path).stem,
                "timestamp": timestamp,
                "md_path": resolved_path,
                "label": int(label),
            }
        )

    if missing_paths:
        LOGGER.warning("%s: skipped %d rows without Markdown paths.", path.name, missing_paths)
    if missing_timestamps:
        LOGGER.warning("%s: %d rows are missing usable timestamps.", path.name, missing_timestamps)
    LOGGER.info("Loaded %d usable rows from %s.", len(records), path)
    return records


def load_manifest_with_time(
    malicious_manifest_path: str | Path,
    benign_manifest_path: str | Path,
    *,
    mal_dir: Optional[str | Path] = None,
    benign_dir: Optional[str | Path] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    mal_records = _load_single_manifest(malicious_manifest_path, label=1, md_root=mal_dir)
    ben_records = _load_single_manifest(benign_manifest_path, label=0, md_root=benign_dir)
    return mal_records, ben_records


def _time_partition(
    records: Iterable[Dict[str, object]],
    *,
    train_end: datetime,
    val_end: datetime,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], int]:
    train: List[Dict[str, object]] = []
    val: List[Dict[str, object]] = []
    test: List[Dict[str, object]] = []
    missing = 0

    for record in records:
        timestamp = record.get("timestamp")
        if not isinstance(timestamp, datetime):
            missing += 1
            continue
        if timestamp <= train_end:
            train.append(record)
        elif timestamp <= val_end:
            val.append(record)
        else:
            test.append(record)
    return train, val, test, missing


def _ratio_counts(total: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    weights = [max(0.0, float(r)) for r in ratios]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return total, 0, 0

    counts = [int(total * weight / weight_sum) for weight in weights]
    remainder = total - sum(counts)
    for idx in range(remainder):
        counts[idx % len(counts)] += 1
    return counts[0], counts[1], counts[2]


def _random_partition(
    records: Sequence[Dict[str, object]],
    *,
    train_target: int,
    val_target: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    train_end = min(train_target, len(shuffled))
    val_end = min(train_end + val_target, len(shuffled))
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    return train, val, test


def _records_to_arrays(
    records: Sequence[Dict[str, object]],
) -> Tuple[List[str], List[int], List[str]]:
    ordered = sorted(records, key=lambda item: str(item.get("sample_id")))
    paths = [str(item["md_path"]) for item in ordered]
    labels = [int(item["label"]) for item in ordered]
    ids = [str(item["sample_id"]) for item in ordered]
    return paths, labels, ids


def split_by_time(
    mal_df: Sequence[Dict[str, object]],
    ben_df: Sequence[Dict[str, object]],
    train_end_date: str,
    val_end_date: str,
    *,
    mal_dir: Optional[str | Path] = None,
    benign_dir: Optional[str | Path] = None,
    benign_split_strategy: str = "random",
    random_seed: int = 42,
) -> Tuple[
    List[str],
    List[int],
    List[str],
    List[str],
    List[int],
    List[str],
    List[str],
    List[int],
    List[str],
]:
    del mal_dir, benign_dir

    train_end = parse_timestamp(train_end_date)
    val_end = parse_timestamp(val_end_date)
    if train_end >= val_end:
        raise ValueError("train_end_date must be earlier than val_end_date")

    mal_train, mal_val, mal_test, mal_missing = _time_partition(
        mal_df,
        train_end=train_end,
        val_end=val_end,
    )
    if mal_missing:
        LOGGER.warning("Skipped %d malicious samples without timestamps.", mal_missing)

    if benign_split_strategy == "time":
        ben_train, ben_val, ben_test, ben_missing = _time_partition(
            ben_df,
            train_end=train_end,
            val_end=val_end,
        )
        if ben_missing:
            LOGGER.warning("Skipped %d benign samples without timestamps.", ben_missing)
    elif benign_split_strategy == "random":
        total_mal = max(len(mal_train) + len(mal_val) + len(mal_test), 1)
        ratios = (
            len(mal_train) / total_mal,
            len(mal_val) / total_mal,
            len(mal_test) / total_mal,
        )
        ben_train_target, ben_val_target, _ = _ratio_counts(len(ben_df), ratios)
        ben_train, ben_val, ben_test = _random_partition(
            ben_df,
            train_target=ben_train_target,
            val_target=ben_val_target,
            seed=random_seed,
        )
    else:
        raise ValueError(f"Unsupported benign_split_strategy: {benign_split_strategy}")

    train_records = list(mal_train) + list(ben_train)
    val_records = list(mal_val) + list(ben_val)
    test_records = list(mal_test) + list(ben_test)

    LOGGER.info(
        "Time split sizes - train: %d, val: %d, test: %d",
        len(train_records),
        len(val_records),
        len(test_records),
    )

    if not train_records or not val_records or not test_records:
        raise RuntimeError(
            "Time-based split produced an empty split. "
            "Please check manifest timestamps and split boundaries."
        )

    train_paths, train_labels, train_ids = _records_to_arrays(train_records)
    val_paths, val_labels, val_ids = _records_to_arrays(val_records)
    test_paths, test_labels, test_ids = _records_to_arrays(test_records)

    return (
        train_paths,
        train_labels,
        train_ids,
        val_paths,
        val_labels,
        val_ids,
        test_paths,
        test_labels,
        test_ids,
    )
