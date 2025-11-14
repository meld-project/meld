"""
自动化实验分析器

集成到训练脚本中的自动化分析功能
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .experiment_analyzer import ExperimentAnalyzer, ExperimentResult
from .result_collector import ResultCollector

LOGGER = logging.getLogger(__name__)


class AutoAnalyzer:
    """自动化实验分析器"""

    def __init__(self,
                 results_dir: str = "./experiments/results",
                 auto_save: bool = True,
                 auto_analyze: bool = True):
        """
        初始化自动化分析器

        Args:
            results_dir: 结果目录
            auto_save: 是否自动保存结果
            auto_analyze: 是否自动进行分析
        """
        self.results_dir = Path(results_dir)
        self.auto_save = auto_save
        self.auto_analyze = auto_analyze

        self.collector = ResultCollector(str(self.results_dir))
        self.current_experiment: Optional[ExperimentResult] = None
        self.start_time: Optional[float] = None

        LOGGER.info("Initialized auto analyzer")

    def start_experiment(self,
                        experiment_id: str,
                        model_name: str,
                        dataset_name: str,
                        config: Dict[str, Any]) -> None:
        """
        开始实验

        Args:
            experiment_id: 实验ID
            model_name: 模型名称
            dataset_name: 数据集名称
            config: 实验配置
        """
        self.start_time = time.time()

        self.current_experiment = ExperimentResult(
            experiment_id=experiment_id,
            model_name=model_name,
            dataset_name=dataset_name,
            config=config,
            metrics={},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            runtime_seconds=0.0,
            status="running"
        )

        LOGGER.info(f"Started experiment: {experiment_id}")

    def log_metric(self, metric_name: str, value: Union[float, int]) -> None:
        """
        记录指标

        Args:
            metric_name: 指标名称
            value: 指标值
        """
        if self.current_experiment is None:
            LOGGER.warning("No active experiment to log metric to")
            return

        self.current_experiment.metrics[metric_name] = float(value)
        LOGGER.debug(f"Logged metric {metric_name}: {value}")

    def log_metrics(self, metrics: Dict[str, Union[float, int]]) -> None:
        """
        批量记录指标

        Args:
            metrics: 指标字典
        """
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value)

    def log_metadata(self, key: str, value: Any) -> None:
        """
        记录元数据

        Args:
            key: 键
            value: 值
        """
        if self.current_experiment is None:
            LOGGER.warning("No active experiment to log metadata to")
            return

        if self.current_experiment.metadata is None:
            self.current_experiment.metadata = {}

        self.current_experiment.metadata[key] = value
        LOGGER.debug(f"Logged metadata {key}: {value}")

    def finish_experiment(self, status: str = "completed", error_message: Optional[str] = None) -> Optional[str]:
        """
        完成实验

        Args:
            status: 实验状态
            error_message: 错误信息（如果有）

        Returns:
            保存的文件路径
        """
        if self.current_experiment is None:
            LOGGER.warning("No active experiment to finish")
            return None

        if self.start_time is not None:
            self.current_experiment.runtime_seconds = time.time() - self.start_time

        self.current_experiment.status = status
        self.current_experiment.error_message = error_message

        if self.auto_save:
            saved_path = self.collector.save_result(self.current_experiment)
            LOGGER.info(f"Finished experiment: {self.current_experiment.experiment_id}")

            if self.auto_analyze:
                self._trigger_analysis()

            # 重置当前实验
            self.current_experiment = None
            self.start_time = None

            return saved_path
        else:
            self.collector.add_result(self.current_experiment)
            return None

    def _trigger_analysis(self) -> None:
        """触发自动分析"""
        try:
            analyzer = ExperimentAnalyzer(str(self.results_dir))
            analyzer.load_results()

            # 获取所有可用的指标
            if analyzer.df is not None:
                metric_columns = [col for col in analyzer.df.columns if col.startswith("metric_")]
                metrics = [col.replace("metric_", "") for col in metric_columns]

                if metrics:
                    # 生成简要报告
                    report = analyzer.generate_report(metrics, "auto_analysis_report.md")
                    LOGGER.info("Auto-analysis completed")

        except Exception as e:
            LOGGER.warning(f"Auto-analysis failed: {e}")

    def create_experiment_comparison(self,
                                  base_experiment_id: str,
                                  compare_experiment_ids: List[str],
                                  metrics: List[str]) -> Dict[str, Any]:
        """
        创建实验比较

        Args:
            base_experiment_id: 基准实验ID
            compare_experiment_ids: 比较实验ID列表
            metrics: 比较的指标

        Returns:
            比较结果
        """
        analyzer = ExperimentAnalyzer(str(self.results_dir))
        analyzer.load_results()

        if analyzer.df is None:
            return {"error": "No data available for comparison"}

        # 获取基准实验
        base_exp = analyzer.df[analyzer.df["experiment_id"] == base_experiment_id]
        if base_exp.empty:
            return {"error": f"Base experiment {base_experiment_id} not found"}

        comparison_results = {"base_experiment": base_experiment_id, "comparisons": []}

        for compare_id in compare_experiment_ids:
            compare_exp = analyzer.df[analyzer.df["experiment_id"] == compare_id]
            if compare_exp.empty:
                continue

            comparison = {"experiment_id": compare_id, "metrics": {}}

            for metric in metrics:
                metric_col = f"metric_{metric}"
                if metric_col in analyzer.df.columns:
                    base_value = base_exp[metric_col].iloc[0]
                    compare_value = compare_exp[metric_col].iloc[0]

                    improvement = ((compare_value - base_value) / base_value * 100) if base_value != 0 else 0

                    comparison["metrics"][metric] = {
                        "base_value": float(base_value),
                        "compare_value": float(compare_value),
                        "improvement_percent": float(improvement),
                        "better": compare_value > base_value
                    }

            comparison_results["comparisons"].append(comparison)

        return comparison_results

    def get_best_models(self, metric: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        获取最佳模型

        Args:
            metric: 评估指标
            top_k: 返回前k个模型

        Returns:
            最佳模型列表
        """
        analyzer = ExperimentAnalyzer(str(self.results_dir))
        analyzer.load_results()

        if analyzer.df is None:
            return []

        ranking = analyzer.rank_models(metric)
        best_models = []

        for _, row in ranking.head(top_k).iterrows():
            model_info = {
                "rank": int(row["rank"]),
                "model_name": row["model_name"],
                "mean_score": float(row["mean_score"]),
                "std_score": float(row["std_score"]),
                "num_experiments": int(row["num_experiments"])
            }
            best_models.append(model_info)

        return best_models

    def export_results_for_paper(self,
                               metrics: List[str],
                               format: str = "latex") -> str:
        """
        导出结果用于论文

        Args:
            metrics: 要导出的指标
            format: 导出格式 (latex, csv, json)

        Returns:
            导出的文件路径
        """
        analyzer = ExperimentAnalyzer(str(self.results_dir))
        analyzer.load_results()

        if analyzer.df is None:
            raise ValueError("No data available for export")

        timestamp = int(time.time())

        if format.lower() == "latex":
            return self._export_latex_table(analyzer, metrics, timestamp)
        elif format.lower() == "csv":
            return self._export_csv(analyzer, metrics, timestamp)
        elif format.lower() == "json":
            return self._export_json(analyzer, metrics, timestamp)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_latex_table(self, analyzer: ExperimentAnalyzer, metrics: List[str], timestamp: int) -> str:
        """导出LaTeX表格"""
        if analyzer.df is None:
            return ""

        # 按模型分组计算平均指标
        model_stats = {}
        for model in analyzer.df["model_name"].unique():
            model_data = analyzer.df[analyzer.df["model_name"] == model]
            model_stats[model] = {}

            for metric in metrics:
                metric_col = f"metric_{metric}"
                if metric_col in model_data.columns:
                    values = model_data[metric_col].dropna()
                    if len(values) > 0:
                        mean_val = float(values.mean())
                        std_val = float(values.std())
                        model_stats[model][metric] = f"{mean_val:.3f} ± {std_val:.3f}"

        # 创建LaTeX表格
        latex_lines = []
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Model Performance Comparison}")

        # 表头
        header = "Model & " + " & ".join([m.replace("_", " ").title() for m in metrics]) + " \\\\"
        latex_lines.append(header)
        latex_lines.append("\\hline")

        # 数据行
        for model, stats in model_stats.items():
            row = model
            for metric in metrics:
                row += f" & {stats.get(metric, 'N/A')}"
            row += " \\\\"
            latex_lines.append(row)

        latex_lines.append("\\end{table}")

        output_path = self.results_dir / "analysis" / f"results_table_{timestamp}.tex"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(latex_lines))

        return str(output_path)

    def _export_csv(self, analyzer: ExperimentAnalyzer, metrics: List[str], timestamp: int) -> str:
        """导出CSV格式"""
        if analyzer.df is None:
            return ""

        # 选择相关列
        columns = ["model_name", "dataset_name", "experiment_id"]
        for metric in metrics:
            metric_col = f"metric_{metric}"
            if metric_col in analyzer.df.columns:
                columns.append(metric_col)

        export_df = analyzer.df[columns].copy()

        output_path = self.results_dir / "analysis" / f"results_{timestamp}.csv"
        export_df.to_csv(output_path, index=False)

        return str(output_path)

    def _export_json(self, analyzer: ExperimentAnalyzer, metrics: List[str], timestamp: int) -> str:
        """导出JSON格式"""
        if analyzer.df is None:
            return ""

        # 转换为JSON友好格式
        results = []
        for _, row in analyzer.df.iterrows():
            result = {
                "model_name": row.get("model_name"),
                "dataset_name": row.get("dataset_name"),
                "experiment_id": row.get("experiment_id")
            }

            for metric in metrics:
                metric_col = f"metric_{metric}"
                if metric_col in row:
                    result[metric] = float(row[metric_col])

            results.append(result)

        output_path = self.results_dir / "analysis" / f"results_{timestamp}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return str(output_path)


# 装饰器用于自动记录实验
def track_experiment(model_name: str,
                    dataset_name: str,
                    analyzer: Optional[AutoAnalyzer] = None,
                    experiment_id: Optional[str] = None):
    """
    实验跟踪装饰器

    Args:
        model_name: 模型名称
        dataset_name: 数据集名称
        analyzer: 分析器实例
        experiment_id: 实验ID
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal analyzer

            if analyzer is None:
                # 创建默认分析器
                analyzer = AutoAnalyzer()

            # 生成实验ID
            if experiment_id is None:
                exp_id = f"{model_name}_{dataset_name}_{int(time.time())}"
            else:
                exp_id = experiment_id

            # 提取配置（如果kwargs中有的话）
            config = kwargs.get("config", {})

            # 开始实验
            analyzer.start_experiment(exp_id, model_name, dataset_name, config)

            try:
                # 执行函数
                result = func(*args, **kwargs)

                # 如果返回结果是指标字典，记录它们
                if isinstance(result, dict):
                    metrics = {k: v for k, v in result.items()
                             if isinstance(v, (int, float))}
                    analyzer.log_metrics(metrics)

                # 完成实验
                analyzer.finish_experiment("completed")
                return result

            except Exception as e:
                # 记录错误并完成实验
                analyzer.finish_experiment("failed", str(e))
                raise

        return wrapper
    return decorator