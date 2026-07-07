#!/usr/bin/env python3
"""Summarize strengthened hash ablation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_LAYERS = [
    ("qwen3", {7: "L7 selected", 28: "L28 final"}),
    ("bert", {11: "L11 selected", 12: "L12 final"}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize hash ablation results")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    rows = []
    missing = []
    for model, label_map in MODEL_LAYERS:
        path = out / model / "hash_ablation_results.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        data = json.loads(path.read_text())
        for layer in data["protocol"]["layers"]:
            key = f"layer_{layer}"
            base = data["with_hash"][key]["tpr_at_fpr_0.001"]
            no_hash = data["without_hash"][key]["tpr_at_fpr_0.001"]
            rows.append((model, label_map.get(layer, f"L{layer}"), base, no_hash, base - no_hash))

    if missing and not args.allow_missing:
        raise FileNotFoundError("missing result files: " + ", ".join(missing))

    md = ["| Model | Layer | With hash | Without hash | Drop |",
          "| --- | --- | ---: | ---: | ---: |"]
    for model, layer, base, no_hash, drop in rows:
        md.append(f"| {model} | {layer} | {base:.4f} | {no_hash:.4f} | {drop*100:.1f} pp |")
    if missing:
        md.append("")
        md.append("Missing result files:")
        md.extend(f"- {item}" for item in missing)

    summary = "\n".join(md) + "\n"
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Summary written to {out / 'summary.md'}")


if __name__ == "__main__":
    main()
