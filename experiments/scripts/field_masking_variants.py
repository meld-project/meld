#!/usr/bin/env python3
"""
Generate behavior-field masking variants for interpretability/case analysis.

The script does not score a model by itself. It creates deterministic variants
of one Markdown behavior report so the same extractor/classifier pipeline can
be run on: original, identifier-redacted, and section-masked inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scripts.hash_ablation import remove_ioc_identifiers


SECTION_GROUPS = {
    "commands": ["Executed Commands"],
    "suspicious": ["Suspicious Activities"],
    "file_ops": ["File Operations"],
    "registry_ops": ["Registry Operations"],
    "network": [
        "Network Activity",
        "Network Connections",
        "DNS Requests",
        "HTTP Requests",
        "TLS Connections",
    ],
    "process": ["Process Timeline", "Process Tree"],
    "other": ["Other Activities"],
    "metadata": ["Summary Statistics", "File Extensions Observed", "Registry Hive Activity"],
}


def split_sections(text: str) -> List[Tuple[str, str]]:
    """Return (heading, block) pairs. Preamble uses heading '__preamble__'."""
    matches = list(re.finditer(r"(?m)^##+\s+(.+?)\s*$", text))
    sections: List[Tuple[str, str]] = []
    if not matches:
        return [("__all__", text)]
    if matches[0].start() > 0:
        sections.append(("__preamble__", text[: matches[0].start()]))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group(1).strip(), text[start:end]))
    return sections


def mask_sections(text: str, headings: Iterable[str]) -> str:
    targets = set(headings)
    output = []
    for heading, block in split_sections(text):
        if heading in targets:
            first_line = block.splitlines()[0] if block.splitlines() else f"## {heading}"
            output.append(f"{first_line}\n[MASKED_{heading.upper().replace(' ', '_')}]\n")
        else:
            output.append(block)
    return "".join(output)


def approx_token_count(text: str) -> int:
    # A tokenizer-free, deterministic proxy used only for reporting input size.
    return len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", text))


def count_indicators(text: str) -> Dict[str, int]:
    patterns = {
        "sha256": r"\b[0-9a-fA-F]{64}\b",
        "md5": r"\b[0-9a-fA-F]{32}\b",
        "url": r"\b(?:https?|ftp)://[^\s`|<>)]+",
        "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b",
        "registry": r"\b(?:HKEY_LOCAL_MACHINE|HKEY_CURRENT_USER|HKEY_CLASSES_ROOT|HKEY_USERS|HKLM|HKCU|HKCR|HKU)\\[^\s`|]+",
        "windows_path": r"(?:(?:[A-Za-z]:|%[A-Za-z0-9_]+%)\\[^\s`|]+|\\\\[A-Za-z0-9_.-]+\\[^\s`|]+)",
    }
    return {name: len(re.findall(pattern, text, re.IGNORECASE)) for name, pattern in patterns.items()}


def build_variants(text: str) -> Dict[str, str]:
    variants = {
        "original": text,
        "redact_ioc_identifiers": remove_ioc_identifiers(text),
    }
    for group, headings in SECTION_GROUPS.items():
        variants[f"mask_{group}"] = mask_sections(text, headings)
        variants[f"redact_ioc_mask_{group}"] = mask_sections(variants["redact_ioc_identifiers"], headings)
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate field masking variants for one CAPE Markdown report.")
    parser.add_argument("--input", required=True, help="Input Markdown behavior report")
    parser.add_argument("--output_dir", required=True, help="Directory for variant Markdown files and summary.json")
    args = parser.parse_args()

    source = Path(args.input)
    out_dir = Path(args.output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = source.read_text(encoding="utf-8", errors="replace")
    variants = build_variants(text)
    summary = {
        "input": str(source),
        "sample_id": source.stem,
        "variants": {},
    }
    base_tokens = approx_token_count(text)
    for name, variant in variants.items():
        path = out_dir / f"{source.stem}.{name}.md"
        path.write_text(variant, encoding="utf-8")
        tokens = approx_token_count(variant)
        summary["variants"][name] = {
            "path": str(path),
            "approx_tokens": tokens,
            "approx_token_delta": tokens - base_tokens,
            "indicator_counts": count_indicators(variant),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
