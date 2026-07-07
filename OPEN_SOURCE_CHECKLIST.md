# Open Source Cleanup Checklist

This checklist records what is currently safe to publish and what still needs a
manual decision before the MELD repository is treated as a release artifact.

## Publish In This Branch

- `README.md`
- `requirements.txt`
- `.gitignore`
- `src/lec/`
- `src/baselines/`
- `src/experiments/`
- `src/utils/`
- `src/capejson2md-rs/`
- `experiments/README.md`
- `experiments/suites/`
- `scripts/generate_md_dir.py`
- `scripts/prepare_family_ood_split.py`
- `scripts/prepare_malware_drift.py`
- Top-level result comparison helpers when they do not contain local paths or
  private data.

## Keep Local Or Move To Private Archive

- `cjc/`, `cjc_v2/`, `latex/`, `meld2.0_latex/`, and `meld_ew2026/`
- Word/PDF manuscript drafts, response letters, journal templates, and revision
  notes.
- Raw sandbox reports, malware samples, local model weights, and result caches.
- Embedded third-party checkouts such as `src/nebula/`; use
  `third_party/nebula` as an explicit external clone instead.

## Before Updating `main`

- Confirm the intended code license.
- Confirm the data license statement and dataset URL.
- Run a syntax/import check on the published Python files.
- Run `python -m src.experiments.run_suite --dry-run` on at least one suite with
  placeholder environment variables set to valid local paths.
- Verify that no local absolute paths, tokens, raw sample hashes requiring
  redaction, or reviewer response drafts are tracked.
