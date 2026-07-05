# MELD

MELD is a research codebase for malware detection with frozen large language
model representations. The project studies whether intermediate hidden states
from a frozen backbone can provide stronger malware evidence than using only the
final layer representation.

The current manuscript version uses CAPE sandbox behavior reports, converts the
reports into structured text, extracts layer-wise embeddings from a frozen LLM,
selects the deployment layer on validation data by TPR at a strict FPR operating
point, and trains a lightweight calibrated classifier on the selected layer.

## Repository Status

This repository is being synchronized with the current manuscript revision.
Several files in this public snapshot are early research scripts. They are useful
for understanding the experiment pipeline, but they should not yet be treated as
a one-command artifact package for every number in the paper.

Current usable components include:

- Layer-wise feature extraction and lightweight classifier training in `lec/`.
- Baseline runners for text, embedding, LSTM, QuoVadis, and Nebula-style inputs
  in `baselines/`.
- Time-based split helpers and reporting utilities in `utils/`.
- Experiment orchestration wrappers in `experiments/`.

Known synchronization gaps:

- The full release workflow for the current CJC revision is still being
  packaged.
- Raw malware binaries are not distributed in this repository.
- `src/capejson2md-rs/` is a placeholder in this early snapshot and is not the
  canonical textualization tool for the current manuscript.
- The root-level packages and `src/` packages currently duplicate many modules;
  this will be cleaned up in a later repository pass.

## Data

The paper uses MELD-DS-448, a CAPE-report dataset covering more than 44,000
sandbox reports across 448 malware families.

Dataset page:

- <https://huggingface.co/datasets/MeldProject/MELD-DS-448>

The dataset release is intended to contain behavior-report artifacts and metadata
needed for research reproduction. It does not distribute executable malware
binaries.

## Method Overview

The MELD pipeline is:

1. Collect CAPE sandbox behavior reports for benign and malicious samples.
2. Convert behavior reports into structured text.
3. Encode each report with a frozen LLM and keep hidden states from every layer.
4. Pool non-padding token representations into one vector per layer.
5. Train a lightweight classifier for each layer.
6. Select the deployment layer on validation data at the target low-FPR
   operating point.
7. Report time-based and family-based out-of-distribution performance.

The manuscript revision reports that, on the Qwen3-0.6B backbone, the selected
intermediate layer improves Time-OOD TPR at 0.1% FPR from 0.8439 at the final
layer to 0.9861. On held-out high-frequency families, the reported mean TPR is
0.9843, with the minimum family TPR of 0.9413.

These numbers are manuscript results. For exact reproduction, use the released
data, the matching model checkpoint, the same split metadata, and the scripts
that will be tagged with the artifact release.

## Installation

Create an isolated Python environment first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install PyTorch according to your CUDA or CPU environment if the default wheel is
not suitable for your machine:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Some baselines require external projects or pretrained weights. For example,
Nebula-based experiments expect an upstream checkout under `third_party/nebula`.

## Example Usage

Layer-wise LEC training on pre-materialized Markdown reports:

```bash
python -m lec.train_lec \
  --model_dir /path/to/Qwen3-0.6B \
  --mal_dir /path/to/malicious_markdown \
  --benign_dir /path/to/benign_markdown \
  --split_mode holdout \
  --target_fpr 0.001 \
  --out results/lec_holdout.json
```

The exact Time-OOD command set for the current manuscript is still being
packaged. Some early scripts already contain time-based split helpers, but the
public repository should be treated as incomplete for exact Time-OOD
reproduction until the artifact tag is added.

Run a baseline directly:

```bash
python -m baselines.text_baselines \
  --mal_dir /path/to/malicious_markdown \
  --benign_dir /path/to/benign_markdown \
  --split_mode holdout \
  --target_fpr 0.001 \
  --out results/tfidf_baseline.json
```

## Citation

The manuscript is currently under revision. A BibTeX entry will be added after
the paper metadata is finalized.

## Security and Ethics

This repository is for malware-behavior research. Do not upload executable
malware binaries, private sandbox reports, API keys, or personally identifiable
information. Use the dataset artifacts and scripts only in an isolated research
environment.
