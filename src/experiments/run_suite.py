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

# 使用统一的可选依赖处理
from src.utils.optional_imports import TQDM, YAML, MATPLOTLIB

# 并发执行（标准库，总是可用）
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
CONCURRENT_AVAILABLE = True

# 可选依赖的可用性检查
TQDM_AVAILABLE = TQDM.is_available()
HAVE_YAML = YAML.is_available()

# 安全导入可选模块
tqdm = TQDM.safe_import()
yaml = YAML.safe_import()

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


def run_generic_baseline(
    name: str,
    params: Dict[str, Any],
    baseline_module,
    config_class_name: str = "TrainConfig",
    baseline_display_name: str = "",
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """通用基线运行函数，消除代码重复。
    
    Args:
        name: 实验名称
        params: 参数字典
        baseline_module: 基线模块（如 text_baselines）
        config_class_name: 配置类名称（默认 "TrainConfig"）
        baseline_display_name: 显示名称（用于日志）
    
    Returns:
        (summary, output_path) 元组
    """
    if hasattr(baseline_module, "setup_logging"):
        baseline_module.setup_logging()

    normalise_scalar_types(params)
    config_class = getattr(baseline_module, config_class_name)
    kwargs = build_dataclass_kwargs(config_class, params)
    cfg = config_class(**kwargs)
    cfg.validate()

    display_name = baseline_display_name or getattr(cfg, "encoder", getattr(cfg, "baseline", name))
    LOGGER.info("运行%s：%s", baseline_display_name or "基线", name)
    if hasattr(cfg, "encoder"):
        LOGGER.info("  编码器：%s", cfg.encoder)
    
    if cfg.seeds:
        per_seed = []
        for seed in cfg.seeds:
            # 对于 nebula 基线，out 处理不同
            if baseline_display_name and "nebula" in baseline_display_name.lower():
                seed_cfg = replace(cfg, seed=seed, seeds=None, out=None if cfg.out else None)
            else:
                seed_cfg = replace(cfg, seed=seed, seeds=None, out=None)
            LOGGER.info(" - 种子 %d", seed)
            seed_summary = baseline_module.run_experiment(seed_cfg)
            seed_summary["seed"] = seed
            per_seed.append(seed_summary)
        
        # 使用聚合函数（如果存在）
        if hasattr(baseline_module, "aggregate_seed_summaries"):
            summary = baseline_module.aggregate_seed_summaries(cfg, per_seed)
        else:
            # 对于 nebula 基线，使用通用聚合函数
            from src.baselines._nebula_common import aggregate_seed_summaries
            baseline_name = getattr(cfg, "baseline", name.split("-")[0] if "-" in name else name)
            summary = aggregate_seed_summaries(cfg, per_seed, baseline_name)
        
        baseline_module.save_summary(summary, cfg.out)
    else:
        summary = baseline_module.run_experiment(cfg)
        summary["seed"] = cfg.seed
        baseline_module.save_summary(summary, cfg.out)

    return summary, Path(cfg.out) if cfg.out else None


def run_text_baseline(name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    from src.baselines import text_baselines
    return run_generic_baseline(name, params, text_baselines, baseline_display_name="文本基线")


def run_embedding_baseline(name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    from src.baselines import embedding_baselines
    return run_generic_baseline(name, params, embedding_baselines, baseline_display_name="嵌入基线")


def run_nebula_baseline(name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    from src.baselines import nebula_baseline
    return run_generic_baseline(name, params, nebula_baseline, baseline_display_name="Nebula 基线")


def run_lec(name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    from src.lec import train_lec
    return run_generic_baseline(name, params, train_lec, baseline_display_name="LEC")


def run_quovadis_baseline(name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Run QuoVadis baseline model."""
    from src.baselines import quovadis_baseline
    return run_generic_baseline(name, params, quovadis_baseline, baseline_display_name="QuoVadis 基线")


def run_lstm_baseline(name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Run LSTM baseline model."""
    from src.baselines import lstm_baseline
    return run_generic_baseline(name, params, lstm_baseline, baseline_display_name="LSTM 基线")


RUNNERS = {
    "text_baseline": run_text_baseline,
    "embedding_baseline": run_embedding_baseline,
    "lec": run_lec,
    "nebula_baseline": run_nebula_baseline,
    "quovadis_baseline": run_quovadis_baseline,
    "lstm_baseline": run_lstm_baseline,
}


def plan_output_path(name: str, spec: Dict[str, Any], default_root: Optional[Path], base_dir: Path) -> Optional[str]:
    if "out" in spec:
        return resolve_path(spec["out"], base_dir)
    if "output" in spec:
        return resolve_path(spec["output"], base_dir)
    if default_root is not None:
        return str((default_root / f"{name}.json").resolve())
    return None


def classify_task_type(spec: Dict[str, Any]) -> str:
    """判断任务是CPU还是GPU类型。
    
    Returns:
        "cpu" 或 "gpu"
    """
    runner_name = spec.get("runner", "")
    params = spec.get("params", {})
    
    # LEC使用gpu参数
    if runner_name == "lec":
        if params.get("gpu") is not None:
            return "gpu"
        return "cpu"
    
    # 其他基线使用device参数
    device = params.get("device", "").lower()
    if device.startswith("cuda") or device == "mps":
        return "gpu"
    
    # 默认：text_baseline是CPU，其他可能是GPU（需要检查默认值）
    if runner_name == "text_baseline":
        return "cpu"
    
    # Nebula基线使用预训练模型，可能不需要GPU（但推理时可能需要）
    # 为了安全，默认认为是GPU任务
    if runner_name in ["nebula_baseline", "neurlux_baseline", "quovadis_baseline", 
                       "dmds_baseline", "lstm_baseline", "embedding_baseline"]:
        # 检查是否有device参数，如果没有，检查默认值
        if "device" in params:
            device = params.get("device", "").lower()
            if device.startswith("cuda") or device == "mps":
                return "gpu"
        # 如果没有明确指定，根据runner判断
        if runner_name in ["neurlux_baseline", "quovadis_baseline", "dmds_baseline", 
                           "lstm_baseline", "embedding_baseline"]:
            return "gpu"  # 这些基线默认使用GPU
    
    return "cpu"  # 默认是CPU


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

    import time
    start_time = time.time()
    
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

    try:
        summary, produced_path = RUNNERS[runner_name](name, params)
        elapsed_time = time.time() - start_time
        highlight = extract_highlights(summary)

        record = {
            "name": name,
            "runner": runner_name,
            "status": "success",
            "output": str(produced_path) if produced_path else out_path,
            "metrics": highlight,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_minutes": elapsed_time / 60,
            **({"notes": notes} if notes else {}),
        }

        LOGGER.info("[完成] %s (耗时: %.2fs / %.2f分钟)", name, elapsed_time, elapsed_time / 60)
        if highlight:
            LOGGER.info("  ↳ 关键指标：%s", ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in highlight.items()))
        return record
    except Exception as e:
        elapsed_time = time.time() - start_time
        LOGGER.error("[失败] %s (耗时: %.2fs): %s", name, elapsed_time, str(e))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON/YAML 配置文件路径。")
    parser.add_argument("--out", default=None, help="可选：写入运行汇总的 JSON 路径。")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要执行的任务，不真正运行。")
    parser.add_argument("--fail-fast", action="store_true", help="遇到首个失败立即中止。")
    parser.add_argument("--verbose", action="store_true", help="输出 DEBUG 级别日志。")
    parser.add_argument("--parallel", type=int, default=1, help="并行执行的任务数（1=串行，-1=使用所有CPU核心）。")
    parser.add_argument("--gpu-parallel", type=int, default=1, help="GPU任务的并行数（默认1，避免GPU内存竞争）。")
    parser.add_argument("--progress", action="store_true", default=True, help="显示进度条（默认启用）。")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="禁用进度条。")
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
    
    # 过滤启用的任务
    enabled_runs = [spec for spec in runs if spec.get("enabled", True)]
    total_runs = len(enabled_runs)
    
    # 分类任务：CPU vs GPU
    cpu_tasks = []
    gpu_tasks = []
    for spec in enabled_runs:
        task_type = classify_task_type(spec)
        if task_type == "gpu":
            gpu_tasks.append(spec)
        else:
            cpu_tasks.append(spec)
    
    LOGGER.info("任务分类: CPU=%d, GPU=%d", len(cpu_tasks), len(gpu_tasks))
    
    # 确定并行度
    cpu_n_jobs = args.parallel
    if cpu_n_jobs == -1:
        import multiprocessing
        cpu_n_jobs = multiprocessing.cpu_count()
    cpu_n_jobs = max(1, min(cpu_n_jobs, len(cpu_tasks)) if cpu_tasks else 1)
    
    gpu_n_jobs = args.gpu_parallel
    gpu_n_jobs = max(1, min(gpu_n_jobs, len(gpu_tasks)) if gpu_tasks else 1)
    
    # 串行执行
    if (cpu_n_jobs == 1 and gpu_n_jobs == 1) or args.dry_run:
        # 创建迭代器（用于进度条）
        run_iterator = enabled_runs
        if args.progress and TQDM_AVAILABLE and not args.dry_run:
            from tqdm import tqdm as tqdm_func
            run_iterator = tqdm_func(enabled_runs, desc="运行实验", total=total_runs, unit="个")
        
        for spec in run_iterator:
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
    else:
        # 智能并行执行：区分CPU和GPU任务
        if not CONCURRENT_AVAILABLE:
            LOGGER.warning("concurrent.futures 不可用，回退到串行执行")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def execute_run_wrapper(spec):
                """包装 execute_run 以便并行执行"""
                try:
                    return execute_run(
                        spec,
                        base_dir=base_dir,
                        default_output_dir=default_output_dir,
                        dry_run=args.dry_run,
                    )
                except Exception as exc:
                    LOGGER.exception("任务执行失败：%s", exc)
                    return {
                        "name": spec.get("name"),
                        "runner": spec.get("runner"),
                        "status": "error",
                        "error": str(exc),
                    }
            
            # 创建进度条（如果可用）
            pbar = None
            if args.progress and TQDM_AVAILABLE:
                pbar = tqdm(total=total_runs, desc="运行实验", unit="个")
            
            try:
                # 使用两个独立的线程池：一个用于CPU任务，一个用于GPU任务
                # CPU任务可以高度并行，GPU任务需要限制并发以避免内存竞争
                
                all_futures = {}
                
                # CPU任务池：可以高度并行
                if cpu_tasks:
                    LOGGER.info("CPU任务池: %d 个任务，并行度 %d", len(cpu_tasks), cpu_n_jobs)
                    cpu_executor = ThreadPoolExecutor(max_workers=cpu_n_jobs)
                    for spec in cpu_tasks:
                        future = cpu_executor.submit(execute_run_wrapper, spec)
                        all_futures[future] = ("cpu", spec)
                
                # GPU任务池：限制并发（默认1，避免GPU内存竞争）
                if gpu_tasks:
                    LOGGER.info("GPU任务池: %d 个任务，并行度 %d", len(gpu_tasks), gpu_n_jobs)
                    gpu_executor = ThreadPoolExecutor(max_workers=gpu_n_jobs)
                    for spec in gpu_tasks:
                        future = gpu_executor.submit(execute_run_wrapper, spec)
                        all_futures[future] = ("gpu", spec)
                
                # 统一处理所有完成的任务
                for future in as_completed(all_futures):
                    task_type, spec = all_futures[future]
                    try:
                        record = future.result()
                        results.append(record)
                        if record.get("status") not in {"success", "dry-run", "skipped"}:
                            failures += 1
                        if pbar:
                            pbar.update(1)
                    except Exception as exc:
                        failures += 1
                        LOGGER.exception("任务执行异常 [%s]: %s", task_type, exc)
                        record = {
                            "name": spec.get("name"),
                            "runner": spec.get("runner"),
                            "status": "error",
                            "error": str(exc),
                        }
                        results.append(record)
                        if pbar:
                            pbar.update(1)
                    
                    if args.fail_fast and failures > 0:
                        LOGGER.warning("启用 fail-fast，取消剩余任务")
                        for f in all_futures:
                            f.cancel()
                        break
                
                # 关闭执行器
                if cpu_tasks:
                    cpu_executor.shutdown(wait=True)
                if gpu_tasks:
                    gpu_executor.shutdown(wait=True)
                    
            finally:
                if pbar:
                    pbar.close()

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

    # 输出最终统计
    total_time = sum(r.get("elapsed_time_seconds", 0) for r in results if "elapsed_time_seconds" in r)
    successful = sum(1 for r in results if r.get("status") == "success")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    
    LOGGER.info("=" * 70)
    LOGGER.info("实验套件执行完成")
    LOGGER.info("=" * 70)
    LOGGER.info("总任务数: %d", total_runs)
    LOGGER.info("成功: %d", successful)
    LOGGER.info("跳过: %d", skipped)
    LOGGER.info("失败: %d", failures)
    if total_time > 0:
        LOGGER.info("总耗时: %.2f秒 (%.2f分钟)", total_time, total_time / 60)
    LOGGER.info("=" * 70)
    
    if failures:
        LOGGER.warning("执行完成，但有 %d 个任务失败。", failures)


if __name__ == "__main__":  # pragma: no cover
    main()
