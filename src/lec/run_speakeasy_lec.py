#!/usr/bin/env python3
"""
Run per-family LEC experiments on the Nebula Speakeasy dataset.

The script follows Nebula's evaluation protocol:
  * Each malware family is treated as a binary task (family vs benign).
  * Validation and test splits both report TPR@FPR=1e-3, AUC, F1, and Accuracy.
  * Layer-wise probes can be compared to Nebula/Neurlux by selecting indices.

Example:
    python src/lec/run_speakeasy_lec.py \
        --manifest experiments/datasets/speakeasy_index.jsonl \
        --families ransomware,coinminer,dropper \
        --model-dir /path/to/qwen \
        --layers 10 12 15 \
        --target-fpr 0.001 \
        --cache-dir cache/speakeasy_features \
        --output experiments/results/lec_speakeasy.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from .feature_extractor import LayerwiseFeatureExtractor
    from .speakeasy_textualizer import SpeakeasyTextualizer
except ImportError:  # pragma: no cover - CLI execution fallback
    from feature_extractor import LayerwiseFeatureExtractor
    from speakeasy_textualizer import SpeakeasyTextualizer

LOGGER = logging.getLogger("run_speakeasy_lec")
BENIGN_FAMILIES = {"clean", "windows_syswow64"}


@dataclass
class ManifestEntry:
    sha256: str
    family: str
    split: str  # train / test
    path: str
    is_benign: bool
    size_bytes: int


@dataclass
class LayerResult:
    family: str
    layer: int
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    threshold: float
    n_train: int
    n_val: int
    n_test: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest produced by scripts/index_speakeasy_dataset.py")
    parser.add_argument("--families", type=str, default="backdoor,coinminer,dropper,keylogger,ransomware,rat,trojan", help="Comma-separated malware families to evaluate.")
    parser.add_argument("--model-dir", type=str, required=True, help="Frozen encoder (e.g., Qwen) directory for LEC feature extraction.")
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 12, 15], help="Layer indices to probe (0-based, negatives allowed).")
    parser.add_argument("--target-fpr", type=float, default=0.001, help="Target false positive rate for threshold calibration.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Hold-out ratio inside the training split for threshold tuning.")
    parser.add_argument("--limit-pos", type=int, default=None, help="Optional cap on malware samples per family (train split).")
    parser.add_argument("--limit-neg", type=int, default=8000, help="Optional cap on benign samples for train split (default 8k to keep encoding manageable).")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional cap on test split per class.")
    parser.add_argument("--max-events", type=int, default=800, help="Maximum Speakeasy API events per document.")
    parser.add_argument("--max-args", type=int, default=4, help="Maximum arguments retained per API call.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per chunk for the encoder.")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride for the encoder.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and splitting.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional directory to cache per-sample feature tensors.")
    parser.add_argument("--until-layer", type=int, default=None, help="Truncate encoder outputs to first N layers (1-based).")
    parser.add_argument("--progress", action="store_true", help="Display encoding progress bars.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file to store aggregated results.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Forward trust_remote_code=True to HuggingFace loader.")
    return parser.parse_args()


def load_manifest(path: Path) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            entries.append(
                ManifestEntry(
                    sha256=obj["sha256"],
                    family=obj["family"],
                    split=obj["split"],
                    path=obj["path"],
                    is_benign=bool(obj["is_benign"]),
                    size_bytes=int(obj.get("size_bytes", 0)),
                )
            )
    if not entries:
        raise ValueError(f"No entries found in manifest {path}")
    return entries


def filter_entries(entries: Iterable[ManifestEntry], *, split: str, family: str | None = None, benign: bool | None = None) -> List[ManifestEntry]:
    result = []
    for entry in entries:
        if entry.split != split:
            continue
        if family is not None and entry.family != family:
            continue
        if benign is not None and entry.is_benign != benign:
            continue
        result.append(entry)
    return result


class FeatureStore:
    def __init__(
        self,
        extractor: LayerwiseFeatureExtractor,
        textualizer: SpeakeasyTextualizer,
        *,
        cache_dir: Path | None,
        max_tokens: int,
        stride: int,
        until_layer: int | None,
        progress: bool = False,
    ) -> None:
        self.extractor = extractor
        self.textualizer = textualizer
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.stride = stride
        self.until_layer = until_layer
        self.progress = progress

    def load(self, entry: ManifestEntry) -> np.ndarray | None:
        key = entry.sha256
        if self.cache_dir:
            cache_path = self.cache_dir / f"{key}.npy"
            if cache_path.exists():
                arr = np.load(cache_path)
                if self.until_layer is not None and arr.shape[0] > self.until_layer:
                    return arr[: self.until_layer]
                return arr
        text = self.textualizer.textualize_file(Path(entry.path), entry.sha256)
        if not text:
            LOGGER.warning("Empty textualization for %s", entry.path)
            return None
        tensor = self.extractor.encode_document_layers(
            text=text,
            max_tokens=self.max_tokens,
            stride=self.stride,
            until_layer=self.until_layer,
        )
        if tensor.is_cuda:
            tensor = tensor.detach().cpu()
        arr = tensor.numpy().astype(np.float32)
        try:  # Clean up GPU memory if available
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if self.cache_dir:
            np.save(self.cache_dir / f"{key}.npy", arr)
        if self.until_layer is not None and arr.shape[0] > self.until_layer:
            return arr[: self.until_layer]
        return arr

    def encode_batch(self, entries: Sequence[ManifestEntry]) -> Tuple[np.ndarray, List[ManifestEntry]]:
        features: List[np.ndarray] = []
        kept: List[ManifestEntry] = []
        iterator: Iterable[ManifestEntry] = entries
        if self.progress:
            iterator = tqdm(entries, desc="Encoding Speakeasy docs", total=len(entries))
        for entry in iterator:
            arr = self.load(entry)
            if arr is None:
                continue
            features.append(arr)
            kept.append(entry)
        if not features:
            raise RuntimeError("No features encoded; aborting.")
        stacked = np.stack(features, axis=0)
        return stacked, kept


def sample_entries(entries: Sequence[ManifestEntry], limit: int | None, rng: random.Random) -> List[ManifestEntry]:
    if limit is None or len(entries) <= limit:
        return list(entries)
    return rng.sample(list(entries), k=limit)


def select_layer(features: np.ndarray, layer_index: int) -> np.ndarray:
    # features shape: [N, L, H]
    total_layers = features.shape[1]
    idx = layer_index if layer_index >= 0 else total_layers + layer_index
    if idx < 0 or idx >= total_layers:
        raise IndexError(f"layer_index {layer_index} out of bounds for {total_layers} layers")
    return features[:, idx, :]


def build_probe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def calibrate_threshold(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    mask_neg = y_true == 0
    neg_scores = scores[mask_neg]
    if neg_scores.size == 0:
        raise ValueError("No negative samples for threshold calibration.")
    quantile = max(0.0, min(1.0, 1 - target_fpr))
    tau = float(np.quantile(neg_scores, quantile))
    return tau


def evaluate_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(int)
    pos = max(1, int((y_true == 1).sum()))
    neg = max(1, int((y_true == 0).sum()))
    tpr = float(((preds == 1) & (y_true == 1)).sum() / pos)
    fpr = float(((preds == 1) & (y_true == 0)).sum() / neg)
    metrics = {
        "tpr": tpr,
        "fpr": fpr,
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        metrics["auroc"] = math.nan
    try:
        metrics["aupr"] = float(average_precision_score(y_true, scores))
    except ValueError:
        metrics["aupr"] = math.nan
    return metrics


def run_family_experiment(
    family: str,
    *,
    manifest: List[ManifestEntry],
    store: FeatureStore,
    args: argparse.Namespace,
    rng: random.Random,
) -> List[LayerResult]:
    train_pos = filter_entries(manifest, split="train", family=family, benign=False)
    if not train_pos:
        raise ValueError(f"No training samples for family {family}")
    train_neg = filter_entries(manifest, split="train", benign=True)
    test_pos = filter_entries(manifest, split="test", family=family, benign=False)
    test_neg = filter_entries(manifest, split="test", benign=True)

    train_pos = sample_entries(train_pos, args.limit_pos, rng)
    train_neg = sample_entries(train_neg, args.limit_neg, rng)
    test_pos = sample_entries(test_pos, args.limit_test, rng)
    test_neg = sample_entries(test_neg, args.limit_test, rng)

    train_entries = train_pos + train_neg
    train_labels = np.concatenate([np.ones(len(train_pos), dtype=int), np.zeros(len(train_neg), dtype=int)])
    train_features, kept_train = store.encode_batch(train_entries)
    if len(kept_train) != len(train_entries):
        LOGGER.warning("Some training samples were dropped during encoding for %s", family)
        valid_labels = []
        for entry in kept_train:
            valid_labels.append(1 if not entry.is_benign else 0)
        train_labels = np.array(valid_labels, dtype=int)

    idx = np.arange(len(train_labels))
    stratify = train_labels if train_labels.min() != train_labels.max() else None
    train_idx, val_idx = train_test_split(
        idx,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=stratify,
    )
    test_entries = test_pos + test_neg
    test_labels = np.concatenate([np.ones(len(test_pos), dtype=int), np.zeros(len(test_neg), dtype=int)])
    test_features, kept_test = store.encode_batch(test_entries)
    if len(kept_test) != len(test_entries):
        LOGGER.warning("Some test samples were dropped during encoding for %s", family)
        valid_labels = []
        for entry in kept_test:
            valid_labels.append(1 if not entry.is_benign else 0)
        test_labels = np.array(valid_labels, dtype=int)

    results: List[LayerResult] = []
    for layer in args.layers:
        X_train = select_layer(train_features[train_idx], layer)
        y_train = train_labels[train_idx]
        X_val = select_layer(train_features[val_idx], layer)
        y_val = train_labels[val_idx]
        X_test = select_layer(test_features, layer)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            LOGGER.warning("Layer %s skipped for %s due to insufficient class diversity.", layer, family)
            continue

        probe = build_probe()
        probe.fit(X_train, y_train)

        val_scores = probe.predict_proba(X_val)[:, 1]
        tau = calibrate_threshold(y_val, val_scores, args.target_fpr)
        val_metrics = evaluate_metrics(y_val, val_scores, tau)

        test_scores = probe.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_metrics(test_labels, test_scores, tau)

        results.append(
            LayerResult(
                family=family,
                layer=layer,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                threshold=tau,
                n_train=len(train_idx),
                n_val=len(val_idx),
                n_test=len(test_labels),
            )
        )

    return results


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    manifest = load_manifest(args.manifest)
    families = [fam.strip().lower() for fam in args.families.split(",") if fam.strip()]
    textualizer = SpeakeasyTextualizer(max_events=args.max_events, max_args=args.max_args)
    device = "cuda" if torch_cuda_available() else "cpu"
    extractor = LayerwiseFeatureExtractor(
        model_dir=args.model_dir,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )
    store = FeatureStore(
        extractor,
        textualizer,
        cache_dir=args.cache_dir,
        max_tokens=args.max_tokens,
        stride=args.stride,
        until_layer=args.until_layer,
        progress=args.progress,
    )

    rng = random.Random(args.seed)
    all_results: List[LayerResult] = []
    for family in families:
        LOGGER.info("=== Running family %s ===", family)
        family_results = run_family_experiment(
            family,
            manifest=manifest,
            store=store,
            args=args,
            rng=rng,
        )
        for res in family_results:
            LOGGER.info(
                "Family=%s Layer=%s ValTPR=%.4f@FPR=%.4f TestTPR=%.4f@FPR=%.4f",
                res.family,
                res.layer,
                res.val_metrics["tpr"],
                res.val_metrics["fpr"],
                res.test_metrics["tpr"],
                res.test_metrics["fpr"],
            )
        all_results.extend(family_results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump([asdict(res) for res in all_results], fh, ensure_ascii=False, indent=2)
        LOGGER.info("Results written to %s", args.output)


def torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
