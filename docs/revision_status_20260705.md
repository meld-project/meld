# MELD Public Repository Synchronization Status

Date: 2026-07-05

This note records what has been aligned in the public repository and what still
needs a follow-up pass before the repository can be described as a complete
artifact package for the current manuscript.

## Aligned in this pass

- Added a public `README.md` that explains the current MELD method at a high
  level.
- Added the MELD-DS-448 dataset link.
- Documented the current manuscript-level Time-OOD and Family-OOD headline
  results.
- Documented the main code areas: `lec/`, `baselines/`, `utils/`, and
  `experiments/`.
- Made the early-snapshot limitations explicit so readers do not mistake this
  repository for a fully packaged artifact release.
- Added `requirements.txt` with the common Python dependencies used by the
  current scripts.

## Current limitations

- The current manuscript's exact reproduction workflow is not yet packaged as a
  single command.
- The Rust textualization directory `src/capejson2md-rs/` is still a placeholder
  in this public snapshot.
- Root-level modules and `src/` modules are duplicated.
- Example commands in the README document the intended script interfaces, but
  they require local data, split manifests, and model checkpoints.
- No license file has been added yet.

## Follow-up tasks

1. Replace `src/capejson2md-rs/` with the real CAPE JSON to Markdown converter,
   or remove the placeholder from the public repository.
2. Decide whether the canonical import layout is root-level packages or `src/`,
   then remove the duplicate copy.
3. Add a small smoke-test fixture with sanitized toy CAPE reports.
4. Add exact config files for the current Time-OOD, Family-OOD, ablation, and
   baseline experiments.
5. Add an artifact tag after the manuscript and dataset release are frozen.
6. Add a license after the release policy is confirmed.

