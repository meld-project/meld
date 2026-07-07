# MalwareDrift 实验数据集指南

本目录用于管理概念漂移实验的数据与结果。核心流程分为三步：

1. **准备 Markdown 行为报告**（`yangben/MalwareDrift/behavior_markdown`）  
   由 `capejson2md` 工具从 CAPE JSON 转换而来。
2. **根据官方标签生成实验子集**  
   使用 `scripts/prepare_malware_drift.py` 将 `pre-drift / post-drift` 报告整理到统一结构。
3. **调用 `src/lec/train_lec.py` 进行训练评估**  
   新训练脚本支持配置文件、特征缓存以及 MPS/GPU。

---

## 1. 元信息与目录

```
experiments/
├── data/                         # 原始数据与标签（保持只读）
│   └── MalwareDrift_Labels.csv   # 官方标签文件
├── datasets/                     # 生成的实验子集会放在这里
├── results/                      # 训练输出（需自行创建）
├── lec_config.json               # 训练示例配置
└── README.md                     # 本文件
```

CSV 字段说明：

| 列名              | 含义                     |
|-------------------|--------------------------|
| `Malware SHA-256` | 样本 SHA-256 哈希        |
| `Family`          | 恶意软件家族（字符串）   |
| `Label`           | 二分类标签（1=恶意）     |
| `State`           | 漂移阶段（`pre-drift` / `post-drift`） |

---

## 2. 生成概念漂移数据集

脚本 `scripts/prepare_malware_drift.py` 会读取元信息，将 Markdown 报告按照 `State` 分类到指定目录，并为每个子集生成 `metadata.json`。

```bash
python scripts/prepare_malware_drift.py \
  --meta experiments/data/MalwareDrift_Labels.csv \
  --md-root yangben/MalwareDrift/behavior_markdown \
  --out-dir experiments/datasets/malware_drift_split
```

运行后目录结构示例：

```
experiments/datasets/malware_drift_split/
├── pre_drift/
│   ├── black/*.md
│   ├── white/*.md
│   └── metadata.json
└── post_drift/
    ├── black/*.md
    ├── white/*.md
    └── metadata.json
```

默认使用符号链接节省空间，可添加 `--copy` 改为复制。部分哈希在行为报告目录中缺失会被自动跳过，日志中会显示数量。

常用可选参数：

| 参数               | 说明                                 |
|--------------------|--------------------------------------|
| `--states`         | 指定导出的状态列表，默认 `pre/post` |
| `--metadata-format`| `json` 或 `csv`                      |
| `--copy`           | 使用复制而非符号链接                |
| `--verbose`        | 输出调试级别日志                    |

---

## 3. 训练脚本使用

生成数据后可使用 `src/lec/train_lec.py` 进行层级特征训练。新版脚本特性：

- 支持 `--config` JSON 配置，所有 CLI 参数均可写入文件；
- 自动检测 CUDA/MPS/CPU，支持 Mac M 系列 GPU；
- 统一的特征缓存与阈值搜索逻辑；
- `--cache_dir` 可开启特征缓存；
- 输出 JSON 汇总，附带完整配置快照。

示例：

```bash
python src/lec/train_lec.py \
  --model_dir ./models/qwen3-0.6b \
  --md_dir experiments/datasets/malware_drift_split/pre_drift \
  --split_mode cv \
  --n_splits 5 \
  --max_tokens 1024 \
  --stride 256 \
  --clf logreg \
  --progress \
  --out experiments/results/pre_drift_cv.json
```

在概念漂移实验中，可将 `pre_drift` 作为训练集，`post_drift` 作为外部评估集。`split_mode dir` 支持显式指定 `--train_dir --val_dir --test_dir`。

---

## 4. 常见问题

- **缺失报告**：日志提示 “缺失 X 个样本”，说明 CSV 中部分哈希在 Markdown 目录下不存在，可检查行为转换是否完整；脚本会自动跳过。
- **MPS/CUDA 设备**：若使用 Mac GPU，确保安装支持 MPS 的 PyTorch，脚本会自动选择 `mps`。
- **特征重复编码**：建议指定 `--cache_dir ./cache/features`，在多次实验时显著加速。
- **结果整合**：所有训练输出 JSON 会保存在 `experiments/results/`，可使用 Pandas 或自定义脚本批量分析。

---

## 5. 参考链接

- [MalwareDrift 官方仓库](https://github.com/MHunt-er/Benchmarking-Malware-Family-Classification)
- [CAPE 沙箱](https://github.com/kevoreilly/CAPEv2)
- [Qwen 模型](https://github.com/QwenLM/Qwen)

---

## 6. 批量调度实验：`run_suite`

为避免逐个脚本调用的重复工作，可使用 `src/experiments/run_suite.py` 按配置批量运行基线与 LEC 任务。

### 6.1 配置结构

`experiments/suites/sample_meld_suite.json` 提供了示例。核心字段如下：

```jsonc
{
  "output_dir": "../results/auto_runs",
  "summary_path": "../results/auto_runs/summary.json",
  "runs": [
    {
      "name": "tfidf_word_cv_pre_drift",
      "runner": "text_baseline",
      "enabled": false,
      "params": {
        "md_dir": "experiments/datasets/malware_drift_split/pre_drift",
        "split_mode": "cv",
        "encoder": "tfidf_word",
        "n_splits": 5,
        "target_fpr": 0.01,
        "seeds": [40, 41, 42]
      }
    }
  ]
}
```

- `runner` 取值：`lec` / `text_baseline` / `embedding_baseline` / `nebula_baseline`。
- `params` 对应各自脚本的 CLI 参数；相对路径会自动基于配置文件所在目录解析。
- `enabled=false` 的条目会被跳过，可用于暂存尚未准备好的实验。

### 6.2 运行方式

```bash
python -m src.experiments.run_suite \
  --config experiments/suites/sample_meld_suite.json \
  --dry-run
```

- 去掉 `--dry-run` 即可真正执行；默认日志写入标准输出。
- `--out` 可覆盖配置中的 `summary_path`，生成包含所有任务状态与关键指标的汇总 JSON。
- 若希望在首个失败后终止，可加 `--fail-fast`。

### 6.3 注意事项

- 在缺省环境下尚未安装 PyYAML 时，请使用 JSON 配置；若需 YAML，记得 `pip install pyyaml`。
- wrapper 仅负责调度，结果 JSON 仍由各自脚本生成；后续分析可复用 `analyze_results.py` 等工具。
- 新增 runner 类型时需在 `src/experiments/run_suite.py` 中注册。
- 若启用 `nebula_baseline`，需提前 `git clone https://github.com/dtrizna/nebula third_party/nebula` 并安装其依赖（torch、sentencepiece、speakeasy-emulator 可选等）。
- `nebula_baseline` 当前已支持 `holdout / cv / dir / time_ood / split_spec` 五种协议；其中 `time_ood` 与 `split_spec` 会把现有 Markdown split 通过同名 stem 映射回原始 JSON 报告。

### 6.4 一次性重跑 Time-OOD 主矩阵

若希望把 Time-OOD 主实验与关键对照一次性跑完，可直接使用
`experiments/suites/time_ood_one_shot.json`。该配置默认包含：

- `lec_time_ood_adaptive`：默认漂移惩罚选层
- `lec_time_ood_no_penalty`：`\lambda=0` 无惩罚选层
- `lec_time_ood_id_only`：仅按 ID 指标选层
- `lec_time_ood_last_layer`：同骨干最后层
- `lec_time_ood_mean_l11_l15`：中间层组均值
- `text_tfidf_word_time_ood` / `text_tfidf_char_time_ood`：同输入文本基线

运行前请先导出 4 个环境变量，避免把本地绝对路径硬编码进仓库：

```bash
export MELD_TIMEOOD_MAL_MANIFEST=/abs/path/to/malicious_manifest.csv
export MELD_TIMEOOD_BEN_MANIFEST=/abs/path/to/benign_manifest.csv
export MELD_TIMEOOD_MAL_MD_DIR=/abs/path/to/malicious_markdown_dir
export MELD_TIMEOOD_BEN_MD_DIR=/abs/path/to/benign_markdown_dir
export MELD_QWEN_MODEL_DIR=/abs/path/to/Qwen3-0.6B
```

建议先做 dry-run：

```bash
python3 -m src.experiments.run_suite \
  --config experiments/suites/time_ood_one_shot.json \
  --dry-run
```

确认路径无误后再正式执行：

```bash
python3 -m src.experiments.run_suite \
  --config experiments/suites/time_ood_one_shot.json
```

说明：

- 所有 LEC 任务共享 `experiments/results/time_ood_one_shot/cache/qwen3_full/` 缓存目录，避免重复抽取同一批层特征。
- `text_baseline` 现已支持 `time_ood` 与 `dir` 两种显式分割模式，因此 TF-IDF 对照和 LEC 使用相同的时间协议。
- suite 中显式写入了 `benign_split_strategy`；当前默认 `random` 以兼容缺少良性时间戳的清单。若你的 benign manifest 也有可靠时间字段，可改为 `time` 以获得更严格的全时序切分。
- 该 one-shot 配置当前聚焦 Time-OOD 与审稿关键对照，Family-OOD/LOFO 尚未接入同一配置文件。

---

## 7. Nebula 训练脚本：`train_nebula.py`

若需在本地 CAPE 数据上细调 Nebula Transformer，可使用 `src/experiments/train_nebula.py`。脚本会：

1. 将本地 JSON 报告编码成 Nebula 预训练 BPE 序列；
2. 按 `holdout / dir / time_ood / split_spec` 准备 train/val/test；
3. 调用 Nebula 官方 `ModelTrainer` 训练 TransformerEncoderChunks；
4. 在验证与测试集上评估并导出模型权重与指标。

示例：

```bash
PYTHONPATH=. python3 -m src.experiments.train_nebula \
  --mal_dir data/cape_json/cape_report_malware \
  --benign_dir data/cape_json/cape_reports_benign \
  --output_dir experiments/results/nebula_finetune \
  --model_out experiments/results/nebula_finetune/meld_nebula.torch \
  --summary_out experiments/results/nebula_finetune/summary.json \
  --epochs 5 \
  --batch_size 96 \
  --progress
```

可选参数：

| 参数 | 说明 |
|------|------|
| `--limit_per_class` | 限制每类样本数量，便于快速调试 |
| `--cache_npz` | 将编码后的特征缓存为 `.npz`，重复运行可直接加载 |
| `--time_budget_min` | 请求训练时间预算（分钟）；当前 wrapper 会记录该预算，并在日志中提示 Nebula upstream 的限制 |
| `--device` | 指定 `cpu`/`cuda`/`mps`，默认自动检测 |
| `--target_fpr` | 评估时报告最接近该 FPR 的 TPR/F1 |

如果希望直接跑官方 Nebula 预训练推理基线，而不是本地微调版本，可使用
`experiments/suites/nebula_external_core.json`。该 suite 会一次性展开：

- `Time-OOD @ 1% FPR`
- `Time-OOD @ 0.1% FPR`
- `8` 个 Family-OOD LOFO folds

运行前请导出以下环境变量：

```bash
export MELD_TIMEOOD_MAL_MANIFEST=/abs/path/to/malicious_manifest.csv
export MELD_TIMEOOD_BEN_MANIFEST=/abs/path/to/benign_manifest.csv
export MELD_FAMILYOOD_SPLIT_SPEC=/abs/path/to/family_ood_lofo_spec.json
export MELD_NEBULA_MAL_JSON_DIR=/abs/path/to/malicious_json_dir
export MELD_NEBULA_BEN_JSON_DIR=/abs/path/to/benign_json_dir
```

建议先 dry-run：

```bash
python3 -m src.experiments.run_suite \
  --config experiments/suites/nebula_external_core.json \
  --dry-run
```

如果希望改跑与当前论文协议对齐的 `Nebula` 微调版，可使用
`experiments/suites/nebula_finetune_core.json`。该 suite 当前包含：

- `nebula_ft_time_ood`：`Time-OOD @ 1% FPR`，内部按 `seeds = {40,41,42}` 重复训练；
- `8` 个 `Family-OOD` LOFO folds：同样按 `seeds = {40,41,42}` 重复训练；
- 统一复用 `cache_npz`，先把 CAPE JSON 适配为 Nebula 可消费的最小 Speakeasy 风格结构，再缓存 token 序列。

可使用与上面相同的环境变量，并先做 dry-run：

```bash
python3 -m src.experiments.run_suite \
  --config experiments/suites/nebula_finetune_core.json \
  --dry-run
```

注意事项：

- 若直接运行脚本，请使用 `python -m ...` 或显式设置 `PYTHONPATH=.`，否则 `src.*` 模块导入可能失败。
- 仍采用 Nebula 默认的 50k BPE tokenizer 与预训练权重作为初始化；脚本会在训练完成后保存新的 `state_dict`。
- 输入 JSON 需包含足够的行为字段（理想情况下接近 Speakeasy 格式）；若字段缺失，编码阶段可能被跳过并写入日志。
- 训练过程会在 `output_dir` 的 `training_files/` 下输出中间指标，可直接复用 Nebula 原始分析脚本查看。
- `time_budget_min` 当前不会直接传给 Nebula upstream trainer，因为 upstream 的该分支在现版本不可用；wrapper 会保留 `requested_time_budget_sec` 供结果记录，并按 `epochs` 执行训练。
- 若需在当前 `Time-OOD / Family-OOD` 协议下补齐一个可复现实验链路完整的外部动态基线，优先对齐 `Nebula`；`API2Vec++ / DA-CFG` 暂未接入与现行 split 完全一致的一键流水线。
- `train_nebula.py` 现已支持 `device=cuda:0` 这类显式 CUDA 设备标识；每次运行会把完整 `fp_rates / tprs / f1s` 一并写入 summary，因此 `0.1% FPR` 可在训练完成后从同一份结果中重评分，无需重复训练。

欢迎根据实验需要扩展字段或添加新的状态切分。
