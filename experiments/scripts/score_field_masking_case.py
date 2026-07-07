#!/usr/bin/env python3
"""Score one field-masking case with cached linear heads.

This script trains the same validation-selected logistic-regression heads used
in the layer analysis from cached full-dataset embeddings, then re-embeds a
small set of masked report variants and reports how their malicious
probabilities change at the selected and final layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scripts.hash_ablation import parse_float_tuple, sanitize  # noqa: E402


def tpr_at_fpr(y_true, y_prob, target_fpr):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = fpr <= target_fpr
    return float(tpr[valid][-1]) if valid.any() else 0.0


def train_head(
    X_train,
    y_train,
    X_val,
    y_val,
    c_values: Tuple[float, ...],
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train = sanitize(X_train)
    X_val = sanitize(X_val)
    best = None
    best_score = float("-inf")
    for c in c_values:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=2000, solver="lbfgs")),
        ])
        clf.fit(X_train, y_train)
        val_pred = clf.predict(X_val)
        val_prob = clf.predict_proba(X_val)[:, 1]
        score = float(f1_score(y_val, val_pred, average="macro"))
        if score > best_score:
            best_score = score
            best = {
                "C": c,
                "val_macro_f1": score,
                "val_tpr_at_fpr_0.001": tpr_at_fpr(y_val, val_prob, 0.001),
                "clf": clf,
            }
    return best


def iter_variants(variant_dir: Path, sample_id: str) -> Iterable[Tuple[str, Path]]:
    for path in sorted(variant_dir.glob(f"{sample_id}.*.md")):
        name = path.name
        prefix = f"{sample_id}."
        suffix = ".md"
        if name.startswith(prefix) and name.endswith(suffix):
            yield name[len(prefix):-len(suffix)], path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score field-masking variants for one sample")
    parser.add_argument("--variant_dir", required=True)
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--baseline_embedding_dir", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="7,28")
    parser.add_argument("--c_values", default="0.01,0.1,1.0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    variant_dir = Path(args.variant_dir)
    baseline_dir = Path(args.baseline_embedding_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    c_values = parse_float_tuple(args.c_values)

    labels = np.load(baseline_dir / "labels.npy")
    sample_ids = np.load(baseline_dir / "sample_ids.npy", allow_pickle=True)
    sample_to_index = {str(s): i for i, s in enumerate(sample_ids)}
    if args.sample_id not in sample_to_index:
        raise ValueError(f"sample_id not found in baseline cache: {args.sample_id}")
    sample_index = sample_to_index[args.sample_id]

    with Path(args.split_file).open() as f:
        split = json.load(f)
    train_idx = np.array(split["train_indices"])
    val_idx = np.array(split["val_indices"])

    heads = {}
    baseline_scores: Dict[str, Dict] = {}
    for layer in layers:
        emb = np.load(baseline_dir / f"layer_{layer:02d}.npy")
        head = train_head(emb[train_idx], labels[train_idx], emb[val_idx], labels[val_idx], c_values)
        heads[layer] = head
        prob = float(head["clf"].predict_proba(sanitize(emb[[sample_index]]))[:, 1][0])
        baseline_scores[f"layer_{layer}"] = {
            "cached_original_probability": prob,
            "selected_C": head["C"],
            "val_macro_f1": head["val_macro_f1"],
            "val_tpr_at_fpr_0.001": head["val_tpr_at_fpr_0.001"],
        }

    import torch
    from lec.feature_extractor import LayerwiseFeatureExtractor

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    extractor = LayerwiseFeatureExtractor(
        model_dir=args.model_dir,
        device=device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    variants = {}
    for variant, path in iter_variants(variant_dir, args.sample_id):
        text = path.read_text(encoding="utf-8", errors="replace")
        feats = extractor.encode_document_layers(
            text, max_tokens=args.max_tokens, stride=args.stride
        ).numpy()
        variants[variant] = {}
        for layer in layers:
            x = sanitize(feats[[layer - 1]])
            prob = float(heads[layer]["clf"].predict_proba(x)[:, 1][0])
            variants[variant][f"layer_{layer}"] = {
                "probability": prob,
                "delta_vs_cached_original": prob - baseline_scores[f"layer_{layer}"]["cached_original_probability"],
            }

    result = {
        "sample_id": args.sample_id,
        "variant_dir": str(variant_dir),
        "baseline_embedding_dir": str(baseline_dir),
        "split_file": args.split_file,
        "model_dir": args.model_dir,
        "layers": layers,
        "baseline_scores": baseline_scores,
        "variants": variants,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
