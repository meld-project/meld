"""
Nebula基线运行器

集成到MELD实验管理系统的Nebula基线模型
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

# 添加项目路径
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 导入实验分析系统
from src.analysis import AutoAnalyzer

# 导入Nebula基线
try:
    from src.baselines.nebula_baseline import (
        TrainConfig,
        compute_nebula_scores,
        evaluate_at_threshold,
        quantile_threshold,
        run_experiment
    )
    NEBULA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Nebula baseline not available: {e}")
    NEBULA_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


class NebulaRunner:
    """Nebula基线运行器"""

    def __init__(self,
                 mal_dir: str,
                 benign_dir: str,
                 target_fpr: float = 0.01,
                 seq_len: int = 512,
                 tokenizer: str = "bpe"):
        """
        初始化Nebula运行器

        Args:
            mal_dir: 恶意样本目录
            benign_dir: 良性样本目录
            target_fpr: 目标假阳性率
            seq_len: 序列长度
            tokenizer: 分词器类型
        """
        if not NEBULA_AVAILABLE:
            raise RuntimeError("Nebula dependencies not available")

        self.mal_dir = mal_dir
        self.benign_dir = benign_dir
        self.target_fpr = target_fpr
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        LOGGER.info(f"Initialized NebulaRunner with mal_dir={mal_dir}, benign_dir={benign_dir}")

    def run_experiment(self,
                      split_mode: str = "holdout",
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.2,
                      seed: int = 42,
                      limit_per_class: Optional[int] = None,
                      use_analyzer: bool = True,
                      experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        运行Nebula实验

        Args:
            split_mode: 分割模式 ("holdout" 或 "cv")
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
            limit_per_class: 每类样本限制
            use_analyzer: 是否使用实验分析器
            experiment_name: 实验名称

        Returns:
            实验结果字典
        """
        # 创建配置
        config = TrainConfig(
            mal_dir=self.mal_dir,
            benign_dir=self.benign_dir,
            split_mode=split_mode,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            limit_per_class=limit_per_class,
            target_fpr=self.target_fpr,
            seq_len=self.seq_len,
            tokenizer=self.tokenizer,
            use_wandb=False,
            cache_scores=True,
            progress=True
        )

        # 初始化实验分析器
        analyzer = None
        if use_analyzer:
            if experiment_name is None:
                experiment_name = f"nebula_{split_mode}_seed{seed}"

            analyzer = AutoAnalyzer("./experiments/results")
            analyzer.start_experiment(
                experiment_id=experiment_name,
                model_name="nebula",
                dataset_name=f"{Path(self.mal_dir).parent.name}",
                config={
                    "split_mode": split_mode,
                    "target_fpr": self.target_fpr,
                    "seq_len": self.seq_len,
                    "tokenizer": self.tokenizer,
                    "seed": seed,
                    "limit_per_class": limit_per_class
                }
            )

        try:
            LOGGER.info(f"Running Nebula experiment with {split_mode} mode")

            # 运行实验
            results = run_experiment(config)

            # 记录指标到分析器
            if analyzer and "best" in results:
                best_metrics = results["best"]
                analyzer.log_metrics({
                    "accuracy": best_metrics.get("accuracy", 0),
                    "macro_f1": best_metrics.get("macro_f1", 0),
                    "auroc": best_metrics.get("auroc", 0),
                    "aupr": best_metrics.get("aupr", 0),
                    "tpr_at_target_fpr": best_metrics.get("tpr_at_target_fpr", 0)
                })

                # 记录元数据
                analyzer.log_metadata("target_fpr", self.target_fpr)
                analyzer.log_metadata("split_mode", split_mode)
                analyzer.log_metadata("num_samples", results.get("num_samples", 0))

            LOGGER.info("Experiment completed successfully")
            return results

        except Exception as e:
            error_msg = f"Experiment failed: {str(e)}"
            LOGGER.error(error_msg)

            if analyzer:
                analyzer.finish_experiment("failed", error_msg)

            raise

        finally:
            if analyzer:
                analyzer.finish_experiment("completed")

    def run_multiple_seeds(self,
                          seeds: List[int] = [42, 43, 44],
                          split_mode: str = "holdout",
                          **kwargs) -> Dict[str, Any]:
        """
        运行多个随机种子的实验

        Args:
            seeds: 随机种子列表
            split_mode: 分割模式
            **kwargs: 其他实验参数

        Returns:
            聚合结果
        """
        all_results = []

        for i, seed in enumerate(seeds):
            LOGGER.info(f"Running seed {seed} ({i+1}/{len(seeds)})")

            experiment_name = f"nebula_{split_mode}_seed{seed}"
            result = self.run_experiment(
                split_mode=split_mode,
                seed=seed,
                experiment_name=experiment_name,
                **kwargs
            )

            result["seed"] = seed
            all_results.append(result)

        # 聚合结果
        return self._aggregate_results(all_results)

    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """聚合多个种子的结果"""
        if not results:
            return {}

        # 提取所有指标
        metrics = {}
        for key in ["accuracy", "macro_f1", "auroc", "aupr", "tpr_at_target_fpr"]:
            values = []
            for result in results:
                if "best" in result and key in result["best"]:
                    values.append(result["best"][key])

            if values:
                metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values
                }

        return {
            "baseline": "nebula",
            "num_seeds": len(results),
            "seeds": [r["seed"] for r in results],
            "aggregated_metrics": metrics,
            "per_seed_results": results
        }

    def evaluate_single_split(self,
                            mal_paths: List[Path],
                            ben_paths: List[Path],
                            seed: int = 42) -> Dict[str, Any]:
        """
        评估单一数据分割

        Args:
            mal_paths: 恶意样本文件路径
            ben_paths: 良性样本文件路径
            seed: 随机种子

        Returns:
            评估结果
        """
        if not NEBULA_AVAILABLE:
            raise RuntimeError("Nebula dependencies not available")

        # 计算Nebula分数
        scores, labels, ids = compute_nebula_scores(
            mal_dir=str(Path(mal_paths[0]).parent),
            benign_dir=str(Path(ben_paths[0]).parent),
            limit_per_class=None,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            progress=True,
            use_cache=True
        )

        # 计算阈值和指标
        threshold_info = quantile_threshold(labels, scores, self.target_fpr)
        tau = threshold_info["threshold"]

        metrics = evaluate_at_threshold(labels, scores, tau)
        metrics["threshold"] = tau
        metrics["tpr_at_target_fpr"] = threshold_info["tpr"]

        return {
            "predictions": scores.tolist(),
            "labels": labels,
            "threshold": tau,
            "metrics": metrics,
            "sample_ids": ids
        }


def run_nebula_experiment(mal_dir: str,
                         benign_dir: str,
                         output_dir: str = "./experiments/results",
                         **kwargs) -> Dict[str, Any]:
    """
    便捷函数：运行Nebula实验

    Args:
        mal_dir: 恶意样本目录
        benign_dir: 良性样本目录
        output_dir: 输出目录
        **kwargs: 其他实验参数

    Returns:
        实验结果
    """
    runner = NebulaRunner(
        mal_dir=mal_dir,
        benign_dir=benign_dir,
        **{k: v for k, v in kwargs.items() if k in ["target_fpr", "seq_len", "tokenizer"]}
    )

    return runner.run_experiment(**{k: v for k, v in kwargs.items()
                                   if k not in ["target_fpr", "seq_len", "tokenizer"]})


# 示例使用
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Nebula baseline experiment")
    parser.add_argument("--mal_dir", required=True, help="Malicious samples directory")
    parser.add_argument("--benign_dir", required=True, help="Benign samples directory")
    parser.add_argument("--output_dir", default="./experiments/results", help="Output directory")
    parser.add_argument("--split_mode", default="holdout", choices=["holdout", "cv"])
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--tokenizer", default="bpe", choices=["bpe", "whitespace"])

    args = parser.parse_args()

    # 解析种子
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # 运行实验
    runner = NebulaRunner(
        mal_dir=args.mal_dir,
        benign_dir=args.benign_dir,
        target_fpr=args.target_fpr,
        seq_len=args.seq_len,
        tokenizer=args.tokenizer
    )

    if len(seeds) == 1:
        results = runner.run_experiment(
            split_mode=args.split_mode,
            seed=seeds[0]
        )
    else:
        results = runner.run_multiple_seeds(
            seeds=seeds,
            split_mode=args.split_mode
        )

    # 保存结果
    output_path = Path(args.output_dir) / f"nebula_results_{args.split_mode}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")