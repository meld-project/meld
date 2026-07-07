#!/usr/bin/env python3
"""
Hash/identifier ablation for shortcut-feature analysis.

The evaluation protocol intentionally mirrors evaluate_all_layers.py:
train on the training split, select logistic-regression C on the validation
split, and report test metrics. This keeps the "with hash" condition
comparable to the main Time-OOD layer-profile tables.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("hash_ablation")

# SHA-256 正则（64位十六进制）
SHA256_PATTERN = re.compile(r'\b[0-9a-fA-F]{64}\b')
# MD5（32位）
MD5_PATTERN = re.compile(r'\b[0-9a-fA-F]{32}\b')
# SHA-1（40位）
SHA1_PATTERN = re.compile(r'\b[0-9a-fA-F]{40}\b')
# UUID/GUID
UUID_PATTERN = re.compile(
    r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
)
# Other long hexadecimal identifiers. Keep this conservative so regular
# short constants and API parameters are not removed.
LONG_HEX_PATTERN = re.compile(r'\b[0-9a-fA-F]{48,}\b')
# URLs, IPs, domains, registry keys, Windows paths, and long random-looking
# strings are common shortcut identifiers in sandbox logs. These are removed in
# the stricter IOC condition requested by the CJC reviewers.
URL_PATTERN = re.compile(r'\b(?:https?|ftp)://[^\s`|<>)]+', re.IGNORECASE)
IPV4_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b')
DOMAIN_PATTERN = re.compile(
    r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'
    r'(?:com|net|org|biz|info|ru|cn|top|xyz|site|online|io|co|uk|de|jp|kr|'
    r'br|in|us|cc|me|pro|pw|club|live|shop|dev|app)\b',
    re.IGNORECASE,
)
REGISTRY_PATTERN = re.compile(
    r'\b(?:HKEY_LOCAL_MACHINE|HKEY_CURRENT_USER|HKEY_CLASSES_ROOT|'
    r'HKEY_USERS|HKLM|HKCU|HKCR|HKU)\\[^\s`|]+',
    re.IGNORECASE,
)
WINDOWS_PATH_PATTERN = re.compile(
    r'(?:(?:[A-Za-z]:|%[A-Za-z0-9_]+%)\\[^\s`|]+|'
    r'\\\\[A-Za-z0-9_.-]+\\[^\s`|]+)',
    re.IGNORECASE,
)
SANDBOX_ARTIFACT_PATTERN = re.compile(
    r'\b(?:CAPE|Cuckoo|sandbox|analysis|sample|malware|benign|'
    r'cape_report_malware|cape_reports_benign)[\\/][^\s`|]+',
    re.IGNORECASE,
)
RANDOM_TOKEN_PATTERN = re.compile(
    r'\b(?=[A-Za-z0-9_-]{16,}\b)(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_-]{16,}\b'
)


def remove_hashes(text: str) -> str:
    """移除文本中所有哈希字符串。"""
    text = SHA256_PATTERN.sub('[HASH]', text)
    text = SHA1_PATTERN.sub('[HASH]', text)
    text = MD5_PATTERN.sub('[HASH]', text)
    return text


def remove_strong_identifiers(text: str) -> str:
    """移除文件哈希以及保守定义的样本级强标识符。"""
    text = remove_hashes(text)
    text = UUID_PATTERN.sub('[ID]', text)
    text = LONG_HEX_PATTERN.sub('[ID]', text)
    return text


def remove_ioc_identifiers(text: str) -> str:
    """移除哈希以外的 IOC/环境标识符，降低路径、注册表、网络标识符捷径。"""
    text = remove_strong_identifiers(text)
    text = URL_PATTERN.sub('[URL]', text)
    text = IPV4_PATTERN.sub('[IP]', text)
    text = DOMAIN_PATTERN.sub('[DOMAIN]', text)
    text = REGISTRY_PATTERN.sub('[REGISTRY]', text)
    text = WINDOWS_PATH_PATTERN.sub('[PATH]', text)
    text = SANDBOX_ARTIFACT_PATTERN.sub('[SANDBOX_PATH]', text)
    text = RANDOM_TOKEN_PATTERN.sub('[RANDOM_ID]', text)
    return text


def tpr_at_fpr(y_true, y_prob, target_fpr):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = fpr <= target_fpr
    return float(tpr[valid][-1]) if valid.any() else 0.0


def sanitize(X):
    """float16→float32, 处理inf/nan"""
    X = X.astype(np.float32) if X.dtype != np.float32 else X
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def load_cached_embeddings(
    embedding_dir: Path,
    layer: int,
    expected_labels: np.ndarray,
    expected_sample_ids: np.ndarray,
) -> np.ndarray:
    """Load main-experiment embeddings and verify sample alignment."""
    layer_path = embedding_dir / f"layer_{layer:02d}.npy"
    labels_path = embedding_dir / "labels.npy"
    sample_ids_path = embedding_dir / "sample_ids.npy"
    if not layer_path.is_file():
        raise FileNotFoundError(f"cached layer not found: {layer_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"cached labels not found: {labels_path}")
    if not sample_ids_path.is_file():
        raise FileNotFoundError(f"cached sample_ids not found: {sample_ids_path}")

    cached_labels = np.load(labels_path)
    cached_sample_ids = np.load(sample_ids_path, allow_pickle=True)
    if cached_labels.shape != expected_labels.shape or not np.array_equal(cached_labels, expected_labels):
        raise ValueError(f"cached labels do not match current sample order: {embedding_dir}")
    if cached_sample_ids.shape != expected_sample_ids.shape or not np.array_equal(cached_sample_ids, expected_sample_ids):
        raise ValueError(f"cached sample_ids do not match current sample order: {embedding_dir}")

    return np.load(layer_path)


def load_or_init_condition_cache(
    cache_dir: Path,
    condition: str,
    target_layers,
    n_samples: int,
    hidden_size: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Load partial no-hash embeddings if present, otherwise initialize buffers."""
    condition_dir = cache_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    processed_path = condition_dir / "processed_mask.npy"

    processed = np.zeros(n_samples, dtype=bool)
    if processed_path.is_file():
        loaded = np.load(processed_path)
        if loaded.shape == (n_samples,):
            processed = loaded.astype(bool)

    embeddings: Dict[int, np.ndarray] = {}
    for layer in target_layers:
        layer_path = condition_dir / f"layer_{layer:02d}.npy"
        if layer_path.is_file():
            arr = np.load(layer_path)
            if arr.shape == (n_samples, hidden_size):
                embeddings[layer] = arr.astype(np.float16, copy=False)
                continue
        embeddings[layer] = np.zeros((n_samples, hidden_size), dtype=np.float16)

    return embeddings, processed


def save_condition_cache(
    cache_dir: Path,
    condition: str,
    embeddings: Dict[int, np.ndarray],
    processed: np.ndarray,
) -> None:
    condition_dir = cache_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    for layer, arr in embeddings.items():
        np.save(condition_dir / f"layer_{layer:02d}.npy", arr)
    np.save(condition_dir / "processed_mask.npy", processed)


def parse_float_tuple(raw: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise ValueError("C value list is empty")
    return values


def metric_for_selection(metric: str, y_val, val_pred, val_prob) -> float:
    from sklearn.metrics import f1_score

    if metric == "val_macro_f1":
        return float(f1_score(y_val, val_pred, average="macro"))
    if metric == "val_tpr_at_fpr_0.001":
        return tpr_at_fpr(y_val, val_prob, 0.001)
    raise ValueError(f"Unsupported selection metric: {metric}")


def evaluate_embeddings(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    c_values: Tuple[float, ...],
    selection_metric: str,
) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_val, X_test = sanitize(X_train), sanitize(X_val), sanitize(X_test)

    best_c = c_values[0]
    best_score = float("-inf")
    best_val_macro_f1 = 0.0
    best_val_tpr001 = 0.0
    for c in c_values:
        candidate = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=2000, solver="lbfgs")),
        ])
        candidate.fit(X_train, y_train)
        val_prob = candidate.predict_proba(X_val)[:, 1]
        val_pred = candidate.predict(X_val)
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))
        val_tpr001 = tpr_at_fpr(y_val, val_prob, 0.001)
        score = metric_for_selection(selection_metric, y_val, val_pred, val_prob)
        if score > best_score:
            best_score = score
            best_c = c
            best_val_macro_f1 = val_macro_f1
            best_val_tpr001 = val_tpr001

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs")),
    ])
    clf.fit(X_train, y_train)

    test_prob = clf.predict_proba(X_test)[:, 1]
    test_pred = clf.predict(X_test)

    return {
        "best_C": best_c,
        "selection_metric": selection_metric,
        "val_macro_f1": best_val_macro_f1,
        "val_tpr_at_fpr_0.001": best_val_tpr001,
        "macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "auc": float(roc_auc_score(y_test, test_prob)),
        "tpr_at_fpr_0.001": tpr_at_fpr(y_test, test_prob, 0.001),
        "tpr_at_fpr_0.01": tpr_at_fpr(y_test, test_prob, 0.01),
    }


CONDITION_TRANSFORMS: Dict[str, Callable[[str], str]] = {
    "with_hash": lambda text: text,
    "without_hash": remove_hashes,
    "without_strong_identifiers": remove_strong_identifiers,
    "without_ioc_identifiers": remove_ioc_identifiers,
}


def main():
    parser = argparse.ArgumentParser(description="Hash 消融实验")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--mal_md_dir", required=True)
    parser.add_argument("--ben_md_dir", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", type=str, default=None,
                        help="要评估的层号（逗号分隔），默认为最终层")
    parser.add_argument("--conditions", type=str, default="with_hash,without_hash",
                        help="逗号分隔: with_hash,without_hash,without_strong_identifiers")
    parser.add_argument("--c_values", type=str, default="0.01,0.1,1.0",
                        help="验证集选择 C 的候选集合，默认与主逐层实验一致")
    parser.add_argument("--selection_metric", default="val_macro_f1",
                        choices=["val_macro_f1", "val_tpr_at_fpr_0.001"],
                        help="C 的验证集选择指标；默认复现当前主逐层实验脚本")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--baseline_embedding_dir", default=None,
                        help="Optional main-experiment embedding cache for with_hash")
    parser.add_argument("--condition_cache_dir", default=None,
                        help="Optional cache directory for extracted ablation embeddings")
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save partial ablation embeddings every N processed samples")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    c_values = parse_float_tuple(args.c_values)
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    unknown = sorted(set(conditions) - set(CONDITION_TRANSFORMS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}; valid={sorted(CONDITION_TRANSFORMS)}")

    split_path = Path(args.split_file)
    mal_dir = Path(args.mal_md_dir)
    ben_dir = Path(args.ben_md_dir)
    if not split_path.is_file():
        raise FileNotFoundError(f"split_file not found: {split_path}")
    if not mal_dir.is_dir():
        raise FileNotFoundError(f"mal_md_dir not found: {mal_dir}")
    if not ben_dir.is_dir():
        raise FileNotFoundError(f"ben_md_dir not found: {ben_dir}")

    # 加载划分
    with split_path.open() as f:
        split = json.load(f)
    train_idx = np.array(split["train_indices"])
    val_idx = np.array(split["val_indices"])
    test_idx = np.array(split["test_indices"])

    # 收集样本
    mal_paths = sorted(mal_dir.glob("*.md")) or sorted(mal_dir.glob("*.json"))
    ben_paths = sorted(ben_dir.glob("*.md")) or sorted(ben_dir.glob("*.json"))
    if not mal_paths:
        raise FileNotFoundError(f"no markdown files under mal_md_dir: {mal_dir}")
    if not ben_paths:
        raise FileNotFoundError(f"no markdown files under ben_md_dir: {ben_dir}")
    all_paths = list(mal_paths) + list(ben_paths)
    labels = np.array([1] * len(mal_paths) + [0] * len(ben_paths))
    sample_ids = np.array([p.stem for p in all_paths], dtype=object)

    LOGGER.info(f"总样本: {len(all_paths)}, 恶意: {len(mal_paths)}, 良性: {len(ben_paths)}")

    baseline_embedding_dir: Optional[Path] = (
        Path(args.baseline_embedding_dir) if args.baseline_embedding_dir else None
    )
    baseline_meta = None
    if baseline_embedding_dir is not None:
        meta_path = baseline_embedding_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"baseline meta not found: {meta_path}")
        with meta_path.open() as f:
            baseline_meta = json.load(f)
        LOGGER.info(f"with_hash 将复用主实验缓存: {baseline_embedding_dir}")

    needs_model = any(condition != "with_hash" or baseline_embedding_dir is None
                      for condition in conditions)

    extractor = None
    if needs_model:
        import torch
        from lec.feature_extractor import LayerwiseFeatureExtractor

        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        extractor = LayerwiseFeatureExtractor(
            model_dir=args.model_dir, device=device, dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        num_layers = extractor.num_model_layers
        hidden_size = extractor.hidden_size
    elif baseline_meta is not None:
        num_layers = int(baseline_meta["num_layers"])
        hidden_size = int(baseline_meta["hidden_size"])
    else:
        raise ValueError("Cannot determine model layer count without model or baseline cache")

    if args.layers:
        target_layers = [int(x) for x in args.layers.split(",")]
    else:
        target_layers = [num_layers]  # 默认最终层
    for layer in target_layers:
        if layer < 1 or layer > num_layers:
            raise ValueError(f"Layer {layer} outside valid range 1..{num_layers}")

    LOGGER.info(f"评估层: {target_layers}")
    LOGGER.info(f"评估条件: {conditions}")
    LOGGER.info(f"C 候选: {c_values}; 选择指标: {args.selection_metric}")

    # 提取两组嵌入：有hash 和 无hash
    from tqdm import tqdm

    results = {
        "protocol": {
            "model_name": args.model_name,
            "model_dir": args.model_dir,
            "split_file": args.split_file,
            "layers": target_layers,
            "conditions": conditions,
            "c_values": list(c_values),
            "selection_metric": args.selection_metric,
            "baseline_embedding_dir": str(baseline_embedding_dir) if baseline_embedding_dir else None,
            "condition_cache_dir": args.condition_cache_dir,
            "max_tokens": args.max_tokens,
            "stride": args.stride,
            "dtype": args.dtype,
        },
        "sample_counts": {
            "total": int(len(all_paths)),
            "malicious": int(len(mal_paths)),
            "benign": int(len(ben_paths)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }
    for condition in conditions:
        results[condition] = {}

    for condition in conditions:
        transform_fn = CONDITION_TRANSFORMS[condition]
        LOGGER.info(f"=== 条件: {condition} ===")

        if condition == "with_hash" and baseline_embedding_dir is not None:
            embeddings = {
                layer: load_cached_embeddings(
                    baseline_embedding_dir, layer, labels, sample_ids
                )
                for layer in target_layers
            }
        else:
            if extractor is None:
                raise ValueError(f"condition {condition} requires a model extractor")
            cache_root = Path(args.condition_cache_dir) if args.condition_cache_dir else Path(args.output_dir) / "embedding_cache"
            embeddings, processed = load_or_init_condition_cache(
                cache_root, condition, target_layers, len(all_paths), hidden_size
            )
            LOGGER.info(
                f"{condition}: 已有缓存 {int(processed.sum())}/{len(processed)}，"
                f"缓存目录 {cache_root / condition}"
            )

            failed_indices = []
            for idx in tqdm(range(len(all_paths)), desc=condition):
                if processed[idx]:
                    continue
                try:
                    text = all_paths[idx].read_text(encoding="utf-8", errors="replace")
                    text = transform_fn(text)
                    features = extractor.encode_document_layers(
                        text, max_tokens=args.max_tokens, stride=args.stride
                    ).numpy()
                    for layer in target_layers:
                        embeddings[layer][idx] = features[layer - 1].astype(np.float16)
                except Exception as e:
                    failed_indices.append(int(idx))
                    LOGGER.warning(f"失败 [{idx}]: {e}")
                finally:
                    processed[idx] = True

                if args.save_every > 0 and int(processed.sum()) % args.save_every == 0:
                    LOGGER.info(f"{condition}: 保存断点缓存 {int(processed.sum())}/{len(processed)}")
                    save_condition_cache(cache_root, condition, embeddings, processed)

            save_condition_cache(cache_root, condition, embeddings, processed)
            if failed_indices:
                results.setdefault("failed_indices", {})[condition] = failed_indices

        # 评估
        for layer in target_layers:
            X = embeddings[layer].astype(np.float32)
            r = evaluate_embeddings(
                X[train_idx], labels[train_idx],
                X[val_idx], labels[val_idx],
                X[test_idx], labels[test_idx],
                c_values=c_values,
                selection_metric=args.selection_metric,
            )
            r["layer"] = layer
            results[condition][f"layer_{layer}"] = r
            LOGGER.info(
                f"  层{layer}: C={r['best_C']}, F1={r['macro_f1']:.4f}, "
                f"TPR@0.1%={r['tpr_at_fpr_0.001']:.4f}"
            )

    # 计算差异
    comparison = {}
    if "with_hash" not in conditions:
        raise ValueError("comparison requires with_hash condition")
    for layer in target_layers:
        key = f"layer_{layer}"
        base = results["with_hash"][key]
        comparison[key] = {}
        for condition in conditions:
            if condition == "with_hash":
                continue
            current = results[condition][key]
            comparison[key][condition] = {
                "f1_drop": round(base["macro_f1"] - current["macro_f1"], 6),
                "tpr001_drop": round(base["tpr_at_fpr_0.001"] - current["tpr_at_fpr_0.001"], 6),
                "auc_drop": round(base["auc"] - current["auc"], 6),
            }
            LOGGER.info(
                f"层{layer} {condition} 降幅: "
                f"F1={comparison[key][condition]['f1_drop']:.4f}, "
                f"TPR@0.1%={comparison[key][condition]['tpr001_drop']:.4f}"
            )

    results["comparison"] = comparison

    # 保存
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "hash_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"结果已保存至 {out / 'hash_ablation_results.json'}")


if __name__ == "__main__":
    main()
