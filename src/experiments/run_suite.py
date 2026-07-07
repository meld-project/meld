#!/usr/bin/env python3
"""统一调度 MELD 实验脚本与基线的包装器。

配置文件使用 JSON（或已安装 PyYAML 时的 YAML）描述多个实验任务，
每一项指定要调用的训练脚本以及对应参数。本脚本负责：

1. 解析配置并展开相对路径；
2. 依次调用 LEC / TF‑IDF / 嵌入等训练脚本（直接调用其 Python 接口）；
3. 记录运行状态与核心指标，可选写入汇总 JSON。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import fields, replace
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # PyYAML 为可选依赖
    import yaml  # type: ignore

    HAVE_YAML = True
except ImportError:  # pragma: no cover - 环境缺省
    HAVE_YAML = False
    yaml = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:  # 允许直接以脚本方式运行
    sys.path.insert(0, str(ROOT_DIR))

LOGGER = logging.getLogger("run_suite")

# 需要视作路径的参数名后缀 / 关键词
PATH_LIKE_SUFFIXES = ("_dir", "_path", "_file")
PATH_LIKE_KEYS = {
    "md_dir",
    "model_dir",
    "mal_dir",
    "benign_dir",
    "train_dir",
    "val_dir",
    "test_dir",
    "cache_dir",
    "word2vec_path",
    "out",
}

# 指标摘要时优先展示的字段
HIGHLIGHT_KEYS = (
    "macro_f1",
    "f1",
    "auroc",
    "aupr",
    "accuracy",
    "tpr_id",
    "tpr_ood",
    "fpr_id",
    "fpr_ood",
)

INT_KEYS = {
    "seed",
    "n_splits",
    "limit",
    "limit_per_class",
    "max_tokens",
    "stride",
    "batch_size",
    "gpu",
    "bootstrap_samples",
}

FLOAT_KEYS = {
    "val_ratio",
    "test_ratio",
    "target_fpr",
    "lambda_penalty",
}


def load_config_file(path: Path) -> Dict[str, Any]:
    """读取 JSON / YAML 配置并保证返回字典。"""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        if not HAVE_YAML:
            raise RuntimeError("检测到 YAML 配置，但未安装 PyYAML。请安装后重试。")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("配置文件需为 JSON/YAML 对象（顶层键值对）。")
    return data


def should_expand_path(key: str) -> bool:
    if key in PATH_LIKE_KEYS:
        return True
    return any(key.endswith(suffix) for suffix in PATH_LIKE_SUFFIXES)


def resolve_path(value: str, base_dir: Path) -> str:
    value = os.path.expandvars(os.path.expanduser(value))
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def resolve_params(data: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            resolved[key] = resolve_params(value, base_dir)
        elif isinstance(value, list):
            if should_expand_path(key):
                resolved[key] = [
                    resolve_path(item, base_dir) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                resolved[key] = [
                    resolve_params(item, base_dir) if isinstance(item, dict) else item
                    for item in value
                ]
        elif isinstance(value, str) and should_expand_path(key):
            resolved[key] = resolve_path(value, base_dir)
        else:
            resolved[key] = value
    return resolved


def normalise_scalar_types(params: Dict[str, Any]) -> None:
    """就地将部分标量转换为期望类型（int / float / list[int]）。"""
    for key in INT_KEYS:
        if key in params and params[key] is not None:
            params[key] = int(params[key])
    for key in FLOAT_KEYS:
        if key in params and params[key] is not None:
            params[key] = float(params[key])
    if "seeds" in params and params["seeds"] is not None:
        seeds = params["seeds"]
        if isinstance(seeds, str):
            seeds = [s.strip() for s in seeds.split(",")]
        params["seeds"] = [int(s) for s in seeds if str(s).strip()]


def build_dataclass_kwargs(config_cls: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """过滤 dataclass 字段，剔除未知参数。"""
    valid_fields = {f.name for f in fields(config_cls)}
    kwargs = {}
    unknown = []
    for key, value in params.items():
        if key in valid_fields:
            kwargs[key] = value
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"{config_cls.__name__} 不支持参数：{', '.join(sorted(unknown))}")
    return kwargs


def extract_highlights(summary: Dict[str, Any]) -> Dict[str, Any]:
    source = summary.get("best") or summary.get("best_mean") or {}
    highlights = {}
    for key in HIGHLIGHT_KEYS:
        if key in source:
            highlights[key] = source[key]
    return highlights


def run_text_baseline(name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    from src.baselines import text_baselines

    if hasattr(text_baselines, "setup_logging"):
        text_baselines.setup_logging()

    normalise_scalar_types(params)
    kwargs = build_dataclass_kwargs(text_baselines.TrainConfig, params)
    cfg = text_baselines.TrainConfig(**kwargs)
    cfg.validate()

    LOGGER.info("运行文本基线：%s (%s)", name, cfg.encoder)
    if cfg.seeds:
        per_seed = []
        for seed in cfg.seeds:
            seed_cfg = replace(cfg, seed=seed, seeds=None, out=None)
            LOGGER.info(" - 种子 %d", seed)
            seed_summary = text_baselines.run_experiment(seed_cfg)
            seed_summary["seed"] = seed
            per_seed.append(seed_summary)
        summary = text_baselines.aggregate_seed_summaries(cfg, per_seed)
        text_baselines.save_summary(summary, cfg.out)
    else:
        summary = text_baselines.run_experiment(cfg)
        summary["seed"] = cfg.seed
        text_baselines.save_summary(summary, cfg.out)

    return summary, Path(cfg.out) if cfg.out else None


def run_embedding_baseline(name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    from src.baselines import embedding_baselines

    if hasattr(embedding_baselines, "setup_logging"):
        embedding_baselines.setup_logging()

    normalise_scalar_types(params)
    kwargs = build_dataclass_kwargs(embedding_baselines.TrainConfig, params)
    cfg = embedding_baselines.TrainConfig(**kwargs)
    cfg.validate()

    LOGGER.info("运行嵌入基线：%s (%s)", name, cfg.encoder)
    if cfg.seeds:
        per_seed = []
        for seed in cfg.seeds:
            seed_cfg = replace(cfg, seed=seed, seeds=None, out=None)
            LOGGER.info(" - 种子 %d", seed)
            seed_summary = embedding_baselines.run_experiment(seed_cfg)
            seed_summary["seed"] = seed
            per_seed.append(seed_summary)
        summary = embedding_baselines.aggregate_seed_summaries(cfg, per_seed)
        embedding_baselines.save_summary(summary, cfg.out)
    else:
        summary = embedding_baselines.run_experiment(cfg)
        summary["seed"] = cfg.seed
        embedding_baselines.save_summary(summary, cfg.out)

    return summary, Path(cfg.out) if cfg.out else None


def run_nebula_baseline(name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    from src.baselines import nebula_baseline

    if hasattr(nebula_baseline, "setup_logging"):
        nebula_baseline.setup_logging()

    normalise_scalar_types(params)
    kwargs = build_dataclass_kwargs(nebula_baseline.TrainConfig, params)
    cfg = nebula_baseline.TrainConfig(**kwargs)
    cfg.validate()

    LOGGER.info("运行 Nebula 基线：%s (%s)", name, cfg.split_mode)
    if cfg.seeds:
        per_seed = []
        for seed in cfg.seeds:
            seed_cfg = replace(cfg, seed=seed, seeds=None, out=None if cfg.out else None)
            LOGGER.info(" - 种子 %d", seed)
            seed_summary = nebula_baseline.run_experiment(seed_cfg)
            seed_summary["seed"] = seed
            per_seed.append(seed_summary)
        summary = nebula_baseline.aggregate_seed_summaries(cfg, per_seed)
        nebula_baseline.save_summary(summary, cfg.out)
    else:
        summary = nebula_baseline.run_experiment(cfg)
        summary["seed"] = cfg.seed
        nebula_baseline.save_summary(summary, cfg.out)

    return summary, Path(cfg.out) if cfg.out else None


def _seeded_path(path: Optional[str], seed: int) -> Optional[str]:
    if not path:
        return None
    base = Path(path)
    return str(base.with_name(f"{base.stem}_seed{seed}{base.suffix}"))


def run_train_nebula(name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    from src.experiments import train_nebula

    if hasattr(train_nebula, "setup_logging"):
        train_nebula.setup_logging()

    params = dict(params)
    if "out" in params and "summary_out" not in params:
        params["summary_out"] = params.pop("out")
    elif "out" in params:
        params.pop("out")

    normalise_scalar_types(params)
    kwargs = build_dataclass_kwargs(train_nebula.TrainConfig, params)
    cfg = train_nebula.TrainConfig(**kwargs)
    cfg.validate()

    LOGGER.info("运行 Nebula 微调：%s (%s)", name, cfg.split_mode)
    if cfg.seeds:
        per_seed = []
        for seed in cfg.seeds:
            seed_output_dir = str(Path(cfg.output_dir) / f"seed_{seed}")
            seed_cfg = replace(
                cfg,
                seed=seed,
                seeds=None,
                output_dir=seed_output_dir,
                summary_out=None,
                model_out=_seeded_path(cfg.model_out, seed),
            )
            LOGGER.info(" - 种子 %d", seed)
            seed_summary = train_nebula.train_and_evaluate(seed_cfg)
            seed_summary["seed"] = seed
            per_seed.append(seed_summary)
        summary = train_nebula.aggregate_seed_summaries(cfg, per_seed)
        train_nebula.save_summary(summary, cfg.summary_out)
    else:
        summary = train_nebula.train_and_evaluate(cfg)
        summary["seed"] = cfg.seed
        train_nebula.save_summary(summary, cfg.summary_out)

    return summary, Path(cfg.summary_out) if cfg.summary_out else None


def run_lec(name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    from src.lec import train_lec

    if hasattr(train_lec, "setup_logging"):
        train_lec.setup_logging()

    normalise_scalar_types(params)
    kwargs = build_dataclass_kwargs(train_lec.TrainConfig, params)
    cfg = train_lec.TrainConfig(**kwargs)
    cfg.validate()

    LOGGER.info("运行 LEC：%s (%s 模式)", name, cfg.split_mode)
    if cfg.seeds:
        per_seed = []
        for seed in cfg.seeds:
            seed_cfg = replace(cfg, seed=seed, seeds=None, out=None if cfg.out else None)
            LOGGER.info(" - 种子 %d", seed)
            seed_summary = train_lec.run_experiment(seed_cfg)
            seed_summary["seed"] = seed
            per_seed.append(seed_summary)
        summary = train_lec.aggregate_seed_summaries(cfg, per_seed)
        train_lec.save_summary(summary, cfg.out)
    else:
        summary = train_lec.run_experiment(cfg)
        train_lec.save_summary(summary, cfg.out)

    return summary, Path(cfg.out) if cfg.out else None


RUNNERS = {
    "text_baseline": run_text_baseline,
    "embedding_baseline": run_embedding_baseline,
    "lec": run_lec,
    "nebula_baseline": run_nebula_baseline,
    "train_nebula": run_train_nebula,
}


def plan_output_path(name: str, spec: Dict[str, Any], default_root: Optional[Path], base_dir: Path) -> Optional[str]:
    if "out" in spec:
        return resolve_path(spec["out"], base_dir)
    if "output" in spec:
        return resolve_path(spec["output"], base_dir)
    if default_root is not None:
        return str((default_root / f"{name}.json").resolve())
    return None


def execute_run(
    spec: Dict[str, Any],
    *,
    base_dir: Path,
    default_output_dir: Optional[Path],
    dry_run: bool,
) -> Dict[str, Any]:
    if "runner" not in spec:
        raise ValueError("每个任务必须指定 runner 字段。")

    runner_name = spec["runner"]
    name = spec.get("name") or runner_name
    notes = spec.get("notes")

    if not spec.get("enabled", True):
        LOGGER.info("跳过任务 %s（已禁用）。", name)
        return {
            "name": name,
            "runner": runner_name,
            "status": "skipped",
            "reason": "disabled",
            **({"notes": notes} if notes else {}),
        }

    if runner_name not in RUNNERS:
        raise ValueError(f"未知 runner：{runner_name}")

    params = spec.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"任务 {name} 的 params 必须是字典。")

    params = resolve_params(params, base_dir)
    out_path = plan_output_path(name, spec, default_output_dir, base_dir)
    if out_path and "out" not in params:
        params["out"] = out_path

    LOGGER.info("[开始] %s (%s)", name, runner_name)
    if dry_run:
        LOGGER.info("Dry-run 模式，未实际执行。")
        return {
            "name": name,
            "runner": runner_name,
            "status": "dry-run",
            "output": out_path,
            **({"notes": notes} if notes else {}),
        }

    summary, produced_path = RUNNERS[runner_name](name, params)
    highlight = extract_highlights(summary)

    record = {
        "name": name,
        "runner": runner_name,
        "status": "success",
        "output": str(produced_path) if produced_path else out_path,
        "metrics": highlight,
        **({"notes": notes} if notes else {}),
    }

    LOGGER.info("[完成] %s", name)
    if highlight:
        LOGGER.info("  ↳ 关键指标：%s", ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in highlight.items()))
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON/YAML 配置文件路径。")
    parser.add_argument("--out", default=None, help="可选：写入运行汇总的 JSON 路径。")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要执行的任务，不真正运行。")
    parser.add_argument("--fail-fast", action="store_true", help="遇到首个失败立即中止。")
    parser.add_argument("--verbose", action="store_true", help="输出 DEBUG 级别日志。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    config_path = Path(args.config).expanduser().resolve()
    config = load_config_file(config_path)
    base_dir = config_path.parent

    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("配置文件需包含非空的 runs 列表。")

    default_output_dir: Optional[Path] = None
    if "output_dir" in config and config["output_dir"]:
        output_dir_str = resolve_path(str(config["output_dir"]), base_dir)
        default_output_dir = Path(output_dir_str)
        if not args.dry_run:
            default_output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    failures = 0
    for spec in runs:
        try:
            record = execute_run(
                spec,
                base_dir=base_dir,
                default_output_dir=default_output_dir,
                dry_run=args.dry_run,
            )
            results.append(record)
            if record.get("status") not in {"success", "dry-run", "skipped"}:
                failures += 1
        except Exception as exc:  # pragma: no cover - 主流程错误处理
            failures += 1
            LOGGER.exception("任务执行失败：%s", exc)
            record = {
                "name": spec.get("name"),
                "runner": spec.get("runner"),
                "status": "error",
                "error": str(exc),
            }
            results.append(record)
            if args.fail_fast:
                break

    summary = {
        "config": str(config_path),
        "results": results,
        "failures": failures,
    }

    summary_path = args.out or config.get("summary_path")
    if summary_path:
        summary_path_str = resolve_path(str(summary_path), base_dir)
        Path(summary_path_str).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path_str, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        LOGGER.info("运行汇总已写入 %s", summary_path_str)

    if failures:
        LOGGER.warning("执行完成，但有 %d 个任务失败。", failures)


if __name__ == "__main__":  # pragma: no cover
    main()
