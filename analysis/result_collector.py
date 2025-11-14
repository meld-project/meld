"""
实验结果收集器

自动收集、整理和存储实验结果
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .experiment_analyzer import ExperimentResult

LOGGER = logging.getLogger(__name__)


class ResultCollector:
    """实验结果收集器"""

    def __init__(self, output_dir: str = "./experiments/results"):
        """
        初始化收集器

        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)

        self.pending_results: List[ExperimentResult] = []
        LOGGER.info(f"Initialized result collector: {self.output_dir}")

    def add_result(self, result: ExperimentResult) -> None:
        """
        添加实验结果

        Args:
            result: 实验结果
        """
        self.pending_results.append(result)
        LOGGER.debug(f"Added result for experiment: {result.experiment_id}")

    def save_result(self, result: ExperimentResult, filename: Optional[str] = None) -> str:
        """
        保存单个实验结果

        Args:
            result: 实验结果
            filename: 可选的文件名

        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"{result.experiment_id}_{timestamp}.json"

        output_path = self.output_dir / "raw" / filename

        # 转换为可序列化的格式
        result_dict = asdict(result)
        # 确保所有值都是JSON可序列化的
        for key, value in result_dict.items():
            if hasattr(value, 'tolist'):  # numpy数组
                result_dict[key] = value.tolist()
            elif hasattr(value, '__dict__'):  # 其他对象
                result_dict[key] = str(value)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"Saved result to {output_path}")
        return str(output_path)

    def save_all_pending(self) -> List[str]:
        """
        保存所有待处理结果

        Returns:
            保存的文件路径列表
        """
        saved_paths = []
        for result in self.pending_results:
            try:
                path = self.save_result(result)
                saved_paths.append(path)
            except Exception as e:
                LOGGER.error(f"Failed to save result {result.experiment_id}: {e}")

        self.pending_results.clear()
        LOGGER.info(f"Saved {len(saved_paths)} pending results")
        return saved_paths

    def load_wandb_results(self, wandb_run_path: str) -> List[ExperimentResult]:
        """
        从W&B加载结果

        Args:
            wandb_run_path: W&B运行路径

        Returns:
            加载的实验结果列表
        """
        try:
            import wandb
            api = wandb.Api()
            run = api.run(wandb_run_path)

            results = []
            # 从W&B历史记录中提取指标
            history = run.history()
            if not history.empty:
                for _, row in history.iterrows():
                    # 提取指标
                    metrics = {}
                    for col in history.columns:
                        if col.startswith('_'):
                            continue
                        try:
                            value = row[col]
                            if pd.notna(value) and isinstance(value, (int, float)):
                                metrics[col] = float(value)
                        except Exception:
                            continue

                    if metrics:
                        result = ExperimentResult(
                            experiment_id=f"wandb_{run.id}_{row.name}",
                            model_name=run.config.get("model_name", "unknown"),
                            dataset_name=run.config.get("dataset_name", "unknown"),
                            config=run.config,
                            metrics=metrics,
                            timestamp=str(run.created_at),
                            runtime_seconds=0.0  # W&B可能不提供此信息
                        )
                        results.append(result)

            LOGGER.info(f"Loaded {len(results)} results from W&B run {wandb_run_path}")
            return results

        except ImportError:
            LOGGER.error("WandB not available")
            return []
        except Exception as e:
            LOGGER.error(f"Failed to load W&B results: {e}")
            return []

    def merge_duplicate_experiments(self) -> int:
        """
        合并重复实验

        Returns:
            合并的实验数量
        """
        raw_dir = self.output_dir / "raw"
        if not raw_dir.exists():
            return 0

        # 按实验ID分组
        experiment_groups = {}
        for file_path in raw_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                exp_id = data.get("experiment_id", file_path.stem)
                if exp_id not in experiment_groups:
                    experiment_groups[exp_id] = []
                experiment_groups[exp_id].append((file_path, data))
            except Exception as e:
                LOGGER.warning(f"Failed to load {file_path}: {e}")

        merged_count = 0
        for exp_id, experiments in experiment_groups.items():
            if len(experiments) > 1:
                # 合并重复实验
                merged_result = self._merge_experiment_results(exp_id, experiments)
                if merged_result:
                    # 保存合并后的结果
                    output_path = self.output_dir / "processed" / f"{exp_id}_merged.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(asdict(merged_result), f, indent=2, ensure_ascii=False)

                    # 移动原始文件到已处理目录
                    for file_path, _ in experiments:
                        processed_dir = self.output_dir / "processed" / "originals"
                        processed_dir.mkdir(exist_ok=True)
                        file_path.rename(processed_dir / file_path.name)

                    merged_count += 1
                    LOGGER.info(f"Merged {len(experiments)} duplicate experiments for {exp_id}")

        return merged_count

    def _merge_experiment_results(self, exp_id: str, experiments: List) -> Optional[ExperimentResult]:
        """
        合并多个实验结果

        Args:
            exp_id: 实验ID
            experiments: 实验列表

        Returns:
            合并后的实验结果
        """
        if not experiments:
            return None

        # 使用最新的完整结果作为基础
        latest_experiment = max(experiments, key=lambda x: x[1].get("timestamp", ""))
        base_data = latest_experiment[1].copy()

        # 合并指标（如果有多个运行的结果）
        all_metrics = []
        for _, data in experiments:
            if "metrics" in data and data["metrics"]:
                all_metrics.append(data["metrics"])

        if len(all_metrics) > 1:
            # 计算指标的平均值和标准差
            merged_metrics = {}
            metric_names = set()
            for metrics in all_metrics:
                metric_names.update(metrics.keys())

            for metric_name in metric_names:
                values = [m.get(metric_name) for m in all_metrics if metric_name in m]
                if values:
                    import numpy as np
                    merged_metrics[f"{metric_name}_mean"] = float(np.mean(values))
                    merged_metrics[f"{metric_name}_std"] = float(np.std(values))
                    merged_metrics[f"{metric_name}_median"] = float(np.median(values))
                    merged_metrics[f"{metric_name}_min"] = float(np.min(values))
                    merged_metrics[f"{metric_name}_max"] = float(np.max(values))

            # 添加原始值
            for i, metrics in enumerate(all_metrics):
                for metric_name, value in metrics.items():
                    merged_metrics[f"{metric_name}_run_{i+1}"] = value

            base_data["metrics"] = merged_metrics

        # 合并元数据
        all_metadata = []
        for _, data in experiments:
            if "metadata" in data and data["metadata"]:
                all_metadata.append(data["metadata"])

        if all_metadata:
            merged_metadata = {
                "num_runs": len(all_metadata),
                "run_ids": [exp_id for exp_id, _ in experiments],
                "merge_timestamp": time.time()
            }
            base_data["metadata"] = merged_metadata

        return ExperimentResult(**base_data)

    def cleanup_old_results(self, days_old: int = 30) -> int:
        """
        清理旧的实验结果

        Args:
            days_old: 保留天数

        Returns:
            删除的文件数量
        """
        import os

        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        deleted_count = 0

        for directory in [self.output_dir / "raw", self.output_dir / "processed"]:
            if not directory.exists():
                continue

            for file_path in directory.glob("*.json"):
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                except Exception as e:
                    LOGGER.warning(f"Failed to delete {file_path}: {e}")

        LOGGER.info(f"Cleaned up {deleted_count} old result files")
        return deleted_count

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取收集统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "output_dir": str(self.output_dir),
            "pending_results": len(self.pending_results),
            "raw_files": len(list((self.output_dir / "raw").glob("*.json"))) if (self.output_dir / "raw").exists() else 0,
            "processed_files": len(list((self.output_dir / "processed").glob("*.json"))) if (self.output_dir / "processed").exists() else 0
        }

        # 计算总大小
        total_size = 0
        for directory in [self.output_dir / "raw", self.output_dir / "processed"]:
            if directory.exists():
                for file_path in directory.glob("*.json"):
                    try:
                        total_size += file_path.stat().st_size
                    except Exception:
                        continue

        stats["total_size_bytes"] = total_size
        stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        return stats