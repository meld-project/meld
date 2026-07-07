from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.utils.split_spec import load_named_split
from src.utils.time_split import parse_timestamp, split_by_time


LOGGER = logging.getLogger("json_splits")

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
    "json_path",
    "report_path",
    "path",
    "filepath",
)

_JSON_INDEX_CACHE: Dict[str, Dict[str, str]] = {}


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


def _build_json_index(root_dir: Path) -> Dict[str, str]:
    cache_key = str(root_dir.resolve())
    cached = _JSON_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    index: Dict[str, str] = {}
    for path in root_dir.rglob("*.json"):
        index[path.stem.lower()] = str(path.resolve())
    _JSON_INDEX_CACHE[cache_key] = index
    return index


def _resolve_json_path(
    row: Dict[str, object],
    *,
    manifest_path: Path,
    json_root: Optional[Path],
) -> Optional[str]:
    raw_path = _get_value(row, PATH_KEYS)
    if raw_path is not None:
        candidate = Path(str(raw_path)).expanduser()
        if not candidate.is_absolute():
            candidate = (manifest_path.parent / candidate).resolve()
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    sample_id = _get_value(row, ID_KEYS)
    if sample_id is None or json_root is None:
        return None

    stem = str(sample_id).strip().lower()
    direct = (json_root / f"{stem}.json").resolve()
    if direct.exists():
        return str(direct)

    index = _build_json_index(json_root)
    return index.get(stem)


def _load_single_manifest(
    manifest_path: str | Path,
    *,
    label: int,
    json_root: Optional[str | Path],
) -> List[Dict[str, object]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    root_dir = Path(json_root).expanduser().resolve() if json_root else None
    rows = _read_manifest_rows(path)

    records: List[Dict[str, object]] = []
    missing_paths = 0
    missing_timestamps = 0

    for row in rows:
        resolved_path = _resolve_json_path(row, manifest_path=path, json_root=root_dir)
        if resolved_path is None:
            missing_paths += 1
            continue

        sample_id = _get_value(row, ID_KEYS)
        timestamp_raw = _get_value(
            row,
            (
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
            ),
        )
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
        LOGGER.warning("%s: skipped %d rows without JSON report paths.", path.name, missing_paths)
    if missing_timestamps:
        LOGGER.warning("%s: %d rows are missing usable timestamps.", path.name, missing_timestamps)
    LOGGER.info("Loaded %d usable JSON rows from %s.", len(records), path)
    return records


def load_manifest_with_json(
    malicious_manifest_path: str | Path,
    benign_manifest_path: str | Path,
    *,
    mal_dir: Optional[str | Path] = None,
    benign_dir: Optional[str | Path] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    mal_records = _load_single_manifest(malicious_manifest_path, label=1, json_root=mal_dir)
    ben_records = _load_single_manifest(benign_manifest_path, label=0, json_root=benign_dir)
    return mal_records, ben_records


def load_json_time_ood_splits(
    *,
    malicious_manifest_path: str,
    benign_manifest_path: str,
    mal_dir: str,
    benign_dir: str,
    train_end_date: Optional[str],
    val_end_date: Optional[str],
    benign_split_strategy: str,
    seed: int,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    mal_records, ben_records = load_manifest_with_json(
        malicious_manifest_path,
        benign_manifest_path,
        mal_dir=mal_dir,
        benign_dir=benign_dir,
    )
    train_end = train_end_date or "2025-04-23 08:08:46"
    val_end = val_end_date or "2025-06-14 11:39:39"
    (
        train_paths,
        train_labels,
        _train_ids,
        val_paths,
        val_labels,
        _val_ids,
        test_paths,
        test_labels,
        _test_ids,
    ) = split_by_time(
        mal_records,
        ben_records,
        train_end,
        val_end,
        mal_dir=mal_dir,
        benign_dir=benign_dir,
        benign_split_strategy=benign_split_strategy,
        random_seed=seed,
    )
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def _resolve_json_from_doc(
    doc: Dict[str, object],
    *,
    mal_dir: str,
    benign_dir: str,
) -> str:
    label = int(doc["label"])
    root = Path(mal_dir if label == 1 else benign_dir).expanduser().resolve()
    source_path = str(doc.get("path") or "").strip()
    if not source_path:
        raise ValueError("split spec doc 缺少 path")
    stem = Path(source_path).stem.lower()

    direct = (root / f"{stem}.json").resolve()
    if direct.exists():
        return str(direct)

    index = _build_json_index(root)
    resolved = index.get(stem)
    if resolved:
        return resolved
    raise FileNotFoundError(f"未在 {root} 中找到 stem={stem!r} 对应的 JSON 报告")


def _slice_split_json_docs(
    docs: Sequence[Dict[str, object]],
    ids: Sequence[int],
    *,
    mal_dir: str,
    benign_dir: str,
) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labels: List[int] = []
    for raw_idx in ids:
        idx = int(raw_idx)
        item = docs[idx]
        label = item.get("label")
        if label not in (0, 1):
            raise ValueError(f"doc[{idx}] 的 label 必须是 0/1")
        paths.append(_resolve_json_from_doc(item, mal_dir=mal_dir, benign_dir=benign_dir))
        labels.append(int(label))
    return paths, labels


def load_json_split_spec_splits(
    *,
    split_spec_path: str,
    split_name: str,
    mal_dir: str,
    benign_dir: str,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    docs, fold, _protocol = load_named_split(split_spec_path, split_name)
    train_paths, train_labels = _slice_split_json_docs(docs, fold["train_ids"], mal_dir=mal_dir, benign_dir=benign_dir)
    val_paths, val_labels = _slice_split_json_docs(docs, fold["val_ids"], mal_dir=mal_dir, benign_dir=benign_dir)
    test_paths, test_labels = _slice_split_json_docs(docs, fold["test_ids"], mal_dir=mal_dir, benign_dir=benign_dir)
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
