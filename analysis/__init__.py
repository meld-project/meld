"""
实验数据分析模块

提供全面的实验结果收集、分析和可视化功能
"""

from .experiment_analyzer import ExperimentAnalyzer, ExperimentResult, ComparisonResult
from .result_collector import ResultCollector
from .auto_analyzer import AutoAnalyzer, track_experiment

__all__ = [
    "ExperimentAnalyzer",
    "ExperimentResult",
    "ComparisonResult",
    "ResultCollector",
    "AutoAnalyzer",
    "track_experiment"
]