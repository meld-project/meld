# MELD

MELD（Malware Evidence Layered Detection）是一个面向恶意软件检测的研究代码库。项目将动态沙箱行为报告转换为结构化文本，利用冻结大语言模型逐层表示提取行为语义特征，并在低误报率约束下评估不同层表示的检测效果。

本仓库用于公开 MELD 方法相关的代码、实验入口和复现实验配置；不包含原始恶意样本、商业模型权重、本地论文返修材料或生成的 DOCX/PDF 文件。

## 仓库内容

- `src/lec/`：逐层表示提取、层选择和分类器训练代码。
- `src/baselines/`：TF-IDF、嵌入模型和 Nebula 兼容基线。
- `src/experiments/`：实验套件调度器和 Nebula 微调入口。
- `src/capejson2md-rs/`：将 CAPE JSON 行为报告转换为 Markdown 文本的 Rust 工具。
- `experiments/suites/`：Time-OOD、Family-OOD 和基线对比实验配置。
- `experiments/scripts/`：补充实验、绘图、消融和结果汇总脚本。
- `scripts/`：数据划分与数据集准备辅助脚本。

## 不包含的内容

- 原始恶意软件二进制样本或原始沙箱报告。
- 本地模型权重、Hugging Face 缓存、实验结果缓存。
- 论文草稿、返修回复、期刊模板、审稿意见和生成的 DOCX/PDF 文件。
- 第三方仓库 checkout，例如 Nebula；需要运行相关基线时请单独克隆。

## 数据集

公开数据集单独发布在 Hugging Face：

- <https://huggingface.co/datasets/MeldProject/MELD-DS-448>

数据集发布内容包含元数据和研究复现实验所需的衍生材料。原始恶意样本不随本仓库分发。

## 环境准备

建议使用 Python 3.10 或更新版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要使用 CAPE JSON 到 Markdown 的转换工具，还需要安装 Rust 工具链：

```bash
cargo build --release --manifest-path src/capejson2md-rs/Cargo.toml
```

## 基本使用流程

1. 准备 CAPE 行为报告文本或使用 `src/capejson2md-rs/` 将 JSON 报告转换为 Markdown。
2. 准备时间划分、家族划分或其他 split specification。
3. 设置实验配置中引用的环境变量。
4. 先运行 dry-run 检查配置是否能被正确解析：

```bash
python -m src.experiments.run_suite \
  --config experiments/suites/time_ood_one_shot.json \
  --dry-run
```

5. 确认路径和配置无误后运行实验：

```bash
python -m src.experiments.run_suite \
  --config experiments/suites/time_ood_one_shot.json
```

常用环境变量包括：

- `MELD_QWEN_MODEL_DIR`
- `MELD_TIMEOOD_MAL_MANIFEST`
- `MELD_TIMEOOD_BEN_MANIFEST`
- `MELD_TIMEOOD_MAL_MD_DIR`
- `MELD_TIMEOOD_BEN_MD_DIR`
- `MELD_FAMILYOOD_SPLIT_SPEC`
- `MELD_NEBULA_MAL_JSON_DIR`
- `MELD_NEBULA_BEN_JSON_DIR`

## Nebula 基线

Nebula 作为外部依赖处理。本仓库不直接包含 Nebula 源码；需要运行 Nebula 相关基线时，请在本地执行：

```bash
git clone https://github.com/dtrizna/nebula third_party/nebula
```

MELD 的 Nebula runner 默认从 `third_party/nebula` 加载该依赖。

## 开源边界

当前公开分支只保留复现实验需要的代码和配置。论文返修材料、参考文献原文、生成图表、模型缓存、原始数据和临时实验结果均不进入本仓库。当前清理状态见 `OPEN_SOURCE_CHECKLIST.md`。

## 许可证

代码和数据许可证仍需在正式 release 前最终确认。请不要默认认为本地论文模板、第三方资源或外部数据自动适用于 MELD 代码许可证。
