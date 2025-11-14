"""
实验数据分析系统

提供全面的实验结果分析功能：
- 结果收集和聚合
- 统计分析和可视化
- 模型比较和排名
- 自动生成分析报告
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

LOGGER = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """单个实验结果"""
    experiment_id: str
    model_name: str
    dataset_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str
    runtime_seconds: float
    status: str = "completed"
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ComparisonResult:
    """模型比较结果"""
    model_a: str
    model_b: str
    metric_name: str
    mean_diff: float
    p_value: float
    statistical_test: str
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str


class ExperimentAnalyzer:
    """实验数据分析器"""

    def __init__(self, results_dir: str = "./experiments/results"):
        """
        初始化分析器

        Args:
            results_dir: 实验结果目录
        """
        self.results_dir = Path(results_dir)
        self.results: List[ExperimentResult] = []
        self.df: Optional[pd.DataFrame] = None

        # 创建必要的目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "analysis").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)

        LOGGER.info(f"Initialized analyzer with results directory: {self.results_dir}")

    def load_results(self, pattern: str = "*.json") -> int:
        """
        加载实验结果

        Args:
            pattern: 文件匹配模式

        Returns:
            加载的结果数量
        """
        self.results = []
        result_files = list(self.results_dir.glob(pattern))

        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 转换为ExperimentResult对象
                result = ExperimentResult(
                    experiment_id=data.get("experiment_id", file_path.stem),
                    model_name=data.get("model_name", "unknown"),
                    dataset_name=data.get("dataset_name", "unknown"),
                    config=data.get("config", {}),
                    metrics=data.get("metrics", {}),
                    timestamp=data.get("timestamp", ""),
                    runtime_seconds=data.get("runtime_seconds", 0.0),
                    status=data.get("status", "completed"),
                    error_message=data.get("error_message"),
                    metadata=data.get("metadata")
                )

                self.results.append(result)

            except Exception as e:
                LOGGER.warning(f"Failed to load result from {file_path}: {e}")

        self._create_dataframe()
        LOGGER.info(f"Loaded {len(self.results)} experiment results")
        return len(self.results)

    def save_results(self, filename: str = "aggregated_results.json") -> None:
        """
        保存聚合结果

        Args:
            filename: 保存文件名
        """
        output_path = self.results_dir / "analysis" / filename

        # 转换为可序列化的格式
        serializable_results = []
        for result in self.results:
            result_dict = asdict(result)
            # 确保numpy数组可序列化
            if "metrics" in result_dict:
                for key, value in result_dict["metrics"].items():
                    if isinstance(value, np.ndarray):
                        result_dict["metrics"][key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        result_dict["metrics"][key] = float(value)
            serializable_results.append(result_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"Saved {len(self.results)} results to {output_path}")

    def _create_dataframe(self) -> None:
        """创建pandas DataFrame"""
        if not self.results:
            self.df = pd.DataFrame()
            return

        # 转换为DataFrame
        data = []
        for result in self.results:
            row = {
                "experiment_id": result.experiment_id,
                "model_name": result.model_name,
                "dataset_name": result.dataset_name,
                "timestamp": result.timestamp,
                "runtime_seconds": result.runtime_seconds,
                "status": result.status
            }

            # 添加配置信息
            if result.config:
                for key, value in result.config.items():
                    if isinstance(value, (str, int, float, bool)):
                        row[f"config_{key}"] = value

            # 添加指标
            for metric_name, metric_value in result.metrics.items():
                row[f"metric_{metric_name}"] = metric_value

            data.append(row)

        self.df = pd.DataFrame(data)
        LOGGER.info(f"Created DataFrame with {len(self.df)} rows and {len(self.df.columns)} columns")

    def get_summary_statistics(self, metric_name: str) -> Dict[str, Any]:
        """
        获取指标的汇总统计

        Args:
            metric_name: 指标名称

        Returns:
            汇总统计信息
        """
        if self.df is None or len(self.df) == 0:
            return {}

        metric_col = f"metric_{metric_name}"
        if metric_col not in self.df.columns:
            LOGGER.warning(f"Metric {metric_name} not found in results")
            return {}

        metric_data = self.df[metric_col].dropna()

        summary = {
            "count": len(metric_data),
            "mean": float(metric_data.mean()),
            "std": float(metric_data.std()),
            "min": float(metric_data.min()),
            "max": float(metric_data.max()),
            "median": float(metric_data.median()),
            "q25": float(metric_data.quantile(0.25)),
            "q75": float(metric_data.quantile(0.75))
        }

        # 按模型分组统计
        if "model_name" in self.df.columns:
            model_stats = {}
            for model in self.df["model_name"].unique():
                model_data = self.df[self.df["model_name"] == model][metric_col].dropna()
                if len(model_data) > 0:
                    model_stats[model] = {
                        "count": len(model_data),
                        "mean": float(model_data.mean()),
                        "std": float(model_data.std()),
                        "median": float(model_data.median())
                    }
            summary["by_model"] = model_stats

        return summary

    def compare_models(self, metric_name: str, alpha: float = 0.05) -> List[ComparisonResult]:
        """
        比较模型性能

        Args:
            metric_name: 比较的指标
            alpha: 显著性水平

        Returns:
            比较结果列表
        """
        if not SCIPY_AVAILABLE:
            LOGGER.warning("SciPy not available, cannot perform statistical comparisons")
            return []

        if self.df is None or len(self.df) == 0:
            return []

        metric_col = f"metric_{metric_name}"
        if metric_col not in self.df.columns:
            LOGGER.warning(f"Metric {metric_name} not found in results")
            return []

        comparisons = []
        models = self.df["model_name"].unique()

        # 两两比较所有模型
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                data_a = self.df[self.df["model_name"] == model_a][metric_col].dropna()
                data_b = self.df[self.df["model_name"] == model_b][metric_col].dropna()

                if len(data_a) == 0 or len(data_b) == 0:
                    continue

                # 执行统计检验
                if len(data_a) > 1 and len(data_b) > 1:
                    # 如果样本量足够，使用t检验
                    try:
                        t_stat, p_value = stats.ttest_ind(data_a, data_b)
                        test_name = "independent_t_test"
                    except Exception:
                        # 回退到Mann-Whitney U检验
                        try:
                            u_stat, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
                            test_name = "mann_whitney_u"
                        except Exception:
                            continue
                else:
                    # 样本量不足，跳过
                    continue

                # 计算效应大小
                mean_a = float(data_a.mean())
                mean_b = float(data_b.mean())
                pooled_std = np.sqrt(((len(data_a) - 1) * data_a.var() +
                                    (len(data_b) - 1) * data_b.var()) /
                                   (len(data_a) + len(data_b) - 2))
                effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

                # 计算置信区间
                if len(data_a) > 1 and len(data_b) > 1:
                    se_diff = np.sqrt(data_a.var()/len(data_a) + data_b.var()/len(data_b))
                    ci_lower = (mean_a - mean_b) - 1.96 * se_diff
                    ci_upper = (mean_a - mean_b) + 1.96 * se_diff
                    confidence_interval = (float(ci_lower), float(ci_upper))
                else:
                    confidence_interval = (float(mean_a - mean_b), float(mean_a - mean_b))

                # 解释结果
                is_significant = p_value < alpha
                if is_significant:
                    if mean_a > mean_b:
                        interpretation = f"{model_a} 显著优于 {model_b}"
                    else:
                        interpretation = f"{model_b} 显著优于 {model_a}"
                else:
                    interpretation = f"{model_a} 和 {model_b} 无显著差异"

                comparison = ComparisonResult(
                    model_a=model_a,
                    model_b=model_b,
                    metric_name=metric_name,
                    mean_diff=mean_a - mean_b,
                    p_value=float(p_value),
                    statistical_test=test_name,
                    effect_size=float(effect_size),
                    confidence_interval=confidence_interval,
                    interpretation=interpretation
                )

                comparisons.append(comparison)

        LOGGER.info(f"Generated {len(comparisons)} model comparisons for {metric_name}")
        return comparisons

    def rank_models(self, metric_name: str, higher_is_better: bool = True) -> pd.DataFrame:
        """
        模型排名

        Args:
            metric_name: 排名指标
            higher_is_better: 是否指标越高越好

        Returns:
            排名DataFrame
        """
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()

        metric_col = f"metric_{metric_name}"
        if metric_col not in self.df.columns:
            LOGGER.warning(f"Metric {metric_name} not found in results")
            return pd.DataFrame()

        # 按模型计算统计信息
        model_stats = []
        for model in self.df["model_name"].unique():
            model_data = self.df[self.df["model_name"] == model][metric_col].dropna()
            if len(model_data) > 0:
                stats_dict = {
                    "model_name": model,
                    "mean_score": float(model_data.mean()),
                    "std_score": float(model_data.std()),
                    "median_score": float(model_data.median()),
                    "num_experiments": len(model_data),
                    "min_score": float(model_data.min()),
                    "max_score": float(model_data.max())
                }
                model_stats.append(stats_dict)

        if not model_stats:
            return pd.DataFrame()

        ranking_df = pd.DataFrame(model_stats)

        # 排序
        ascending = not higher_is_better
        ranking_df = ranking_df.sort_values("mean_score", ascending=ascending)
        ranking_df["rank"] = range(1, len(ranking_df) + 1)

        return ranking_df

    def create_visualizations(self, metrics: List[str]) -> None:
        """
        创建可视化图表

        Args:
            metrics: 要可视化的指标列表
        """
        if not PLOTTING_AVAILABLE:
            LOGGER.warning("Matplotlib/Seaborn not available, skipping visualizations")
            return

        if self.df is None or len(self.df) == 0:
            LOGGER.warning("No data available for visualization")
            return

        plt.style.use('default')
        plots_dir = self.results_dir / "plots"

        # 1. 模型性能箱线图
        if "model_name" in self.df.columns:
            for metric in metrics:
                metric_col = f"metric_{metric}"
                if metric_col in self.df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=self.df, x="model_name", y=metric_col)
                    plt.title(f"Model Performance - {metric}")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"boxplot_{metric}.png", dpi=300, bbox_inches='tight')
                    plt.close()

        # 2. 指标相关性热图
        metric_cols = [f"metric_{m}" for m in metrics if f"metric_{m}" in self.df.columns]
        if len(metric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.df[metric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title("Metric Correlation Matrix")
            plt.tight_layout()
            plt.savefig(plots_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 3. 实验运行时间分析
        if "runtime_seconds" in self.df.columns and "model_name" in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=self.df, x="model_name", y="runtime_seconds")
            plt.title("Experiment Runtime by Model")
            plt.ylabel("Runtime (seconds)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / "runtime_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

        LOGGER.info(f"Created visualizations in {plots_dir}")

    def generate_report(self, metrics: List[str], output_file: Optional[str] = None) -> str:
        """
        生成分析报告

        Args:
            metrics: 要分析的指标列表
            output_file: 输出文件名

        Returns:
            报告内容
        """
        if self.df is None or len(self.df) == 0:
            return "No experiment data available for analysis."

        report_lines = []
        report_lines.append("# 实验分析报告")
        report_lines.append(f"\n生成时间: {pd.Timestamp.now()}")
        report_lines.append(f"实验数量: {len(self.results)}")
        report_lines.append(f"分析指标: {', '.join(metrics)}")
        report_lines.append("\n" + "="*50 + "\n")

        # 总体统计
        report_lines.append("## 总体统计")
        if "model_name" in self.df.columns:
            model_counts = self.df["model_name"].value_counts()
            report_lines.append(f"涉及的模型: {', '.join(model_counts.index.tolist())}")
            report_lines.append(f"各模型实验次数: {dict(model_counts)}")
        report_lines.append(f"总运行时间: {self.df['runtime_seconds'].sum():.2f} 秒")
        report_lines.append(f"平均运行时间: {self.df['runtime_seconds'].mean():.2f} 秒\n")

        # 各指标详细分析
        for metric in metrics:
            report_lines.append(f"## {metric.upper()} 指标分析")

            # 汇总统计
            summary = self.get_summary_statistics(metric)
            if summary:
                report_lines.append("### 汇总统计")
                report_lines.append(f"- 平均值: {summary['mean']:.4f} ± {summary['std']:.4f}")
                report_lines.append(f"- 中位数: {summary['median']:.4f}")
                report_lines.append(f"- 范围: [{summary['min']:.4f}, {summary['max']:.4f}]")

                # 按模型统计
                if "by_model" in summary:
                    report_lines.append("\n### 按模型统计")
                    for model, stats in summary["by_model"].items():
                        report_lines.append(f"- {model}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")

            # 模型排名
            ranking = self.rank_models(metric)
            if not ranking.empty:
                report_lines.append("\n### 模型排名")
                for _, row in ranking.head(5).iterrows():
                    report_lines.append(f"{int(row['rank'])}. {row['model_name']}: "
                                      f"{row['mean_score']:.4f} ± {row['std_score']:.4f}")

            # 统计比较
            comparisons = self.compare_models(metric)
            if comparisons:
                report_lines.append("\n### 显著差异")
                significant_comparisons = [c for c in comparisons if c.p_value < 0.05]
                for comp in significant_comparisons:
                    report_lines.append(f"- {comp.interpretation} (p={comp.p_value:.4f}, "
                                      f"effect_size={comp.effect_size:.3f})")

            report_lines.append("")

        # 失败实验分析
        if "status" in self.df.columns:
            failed_experiments = self.df[self.df["status"] != "completed"]
            if not failed_experiments.empty:
                report_lines.append("## 失败实验分析")
                report_lines.append(f"失败实验数量: {len(failed_experiments)}")
                for _, exp in failed_experiments.iterrows():
                    report_lines.append(f"- {exp.get('experiment_id', 'Unknown')}: {exp.get('status', 'Unknown')}")

        report_content = "\n".join(report_lines)

        # 保存报告
        if output_file is None:
            output_file = "experiment_analysis_report.md"

        output_path = self.results_dir / "analysis" / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        LOGGER.info(f"Generated analysis report: {output_path}")
        return report_content

    def export_to_wandb(self, project_name: str = "meld-analysis") -> None:
        """
        导出结果到Weights & Biases

        Args:
            project_name: W&B项目名称
        """
        if not WANDB_AVAILABLE:
            LOGGER.warning("WandB not available, skipping export")
            return

        if self.df is None or len(self.df) == 0:
            LOGGER.warning("No data available for W&B export")
            return

        # 初始化wandb
        wandb.init(project=project_name, name="experiment-analysis")

        # 创建表格
        table_data = []
        for _, row in self.df.iterrows():
            table_data.append([row.get(col, "") for col in self.df.columns])

        table = wandb.Table(columns=self.df.columns.tolist(), data=table_data)
        wandb.log({"experiment_results": table})

        # 添加汇总统计
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col.startswith("metric_"):
                metric_name = col.replace("metric_", "")
                summary = self.get_summary_statistics(metric_name)
                if summary:
                    wandb.log({
                        f"{metric_name}_mean": summary["mean"],
                        f"{metric_name}_std": summary["std"],
                        f"{metric_name}_median": summary["median"]
                    })

        wandb.finish()
        LOGGER.info("Exported results to Weights & Biases")


def main():
    """主函数示例"""
    analyzer = ExperimentAnalyzer("./experiments/results")

    # 加载结果
    analyzer.load_results()

    # 定义要分析的指标
    metrics = ["accuracy", "macro_f1", "auroc", "aupr"]

    # 生成报告
    analyzer.generate_report(metrics)

    # 创建可视化
    analyzer.create_visualizations(metrics)

    # 导出到W&B（可选）
    # analyzer.export_to_wandb()


if __name__ == "__main__":
    main()