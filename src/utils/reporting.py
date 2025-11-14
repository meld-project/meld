"""Enhanced reporting utilities for MELD experiments.

This module provides utilities for:
- Rich progress reporting
- Time tracking
- Resource monitoring
- Summary generation
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Progress = None
    Table = None
    Panel = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

LOGGER = logging.getLogger("reporting")


class ExperimentReporter:
    """Enhanced experiment reporter with rich formatting."""
    
    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich and RICH_AVAILABLE
        if self.use_rich:
            self.console = Console()
        self.start_time = None
        self.phases: List[Dict[str, Any]] = []
    
    def start_experiment(self, name: str, config: Dict[str, Any]):
        """Start experiment reporting."""
        self.start_time = time.time()
        if self.use_rich:
            self.console.print(Panel.fit(
                f"[bold blue]开始实验: {name}[/bold blue]",
                border_style="blue"
            ))
            # Print config summary
            config_table = Table(show_header=False, box=None)
            for key, value in list(config.items())[:10]:  # Show first 10 items
                config_table.add_row(f"[dim]{key}:[/dim]", str(value))
            self.console.print(config_table)
        else:
            LOGGER.info("=" * 70)
            LOGGER.info(f"开始实验: {name}")
            LOGGER.info("=" * 70)
    
    def report_phase(self, phase_name: str, status: str = "running", **kwargs):
        """Report a phase of the experiment."""
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        phase_info = {
            "phase": phase_name,
            "status": status,
            "elapsed_time": elapsed,
            **kwargs,
        }
        self.phases.append(phase_info)
        
        if self.use_rich:
            status_color = {
                "running": "yellow",
                "completed": "green",
                "failed": "red",
            }.get(status, "white")
            self.console.print(
                f"[{status_color}]{phase_name}[/{status_color}]: {status} "
                f"({elapsed:.2f}s)"
            )
        else:
            LOGGER.info(f"[{phase_name}] {status} ({elapsed:.2f}s)")
    
    def report_metrics(self, metrics: Dict[str, float], title: str = "实验结果"):
        """Report metrics in a formatted table."""
        if self.use_rich:
            table = Table(title=title, show_header=True, header_style="bold")
            table.add_column("指标", style="cyan")
            table.add_column("值", style="green", justify="right")
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))
            
            self.console.print(table)
        else:
            LOGGER.info("=" * 70)
            LOGGER.info(title)
            LOGGER.info("=" * 70)
            for key, value in metrics.items():
                if isinstance(value, float):
                    LOGGER.info(f"{key}: {value:.4f}")
                else:
                    LOGGER.info(f"{key}: {value}")
            LOGGER.info("=" * 70)
    
    def report_summary(self, summary: Dict[str, Any]):
        """Report experiment summary."""
        total_time = time.time() - self.start_time if self.start_time else 0.0
        
        if self.use_rich:
            summary_panel = Panel(
                f"[bold]实验完成[/bold]\n"
                f"总耗时: {total_time:.2f}s ({total_time/60:.2f}分钟)\n"
                f"阶段数: {len(self.phases)}",
                title="实验摘要",
                border_style="green"
            )
            self.console.print(summary_panel)
        else:
            LOGGER.info("=" * 70)
            LOGGER.info("实验摘要")
            LOGGER.info("=" * 70)
            LOGGER.info(f"总耗时: {total_time:.2f}s ({total_time/60:.2f}分钟)")
            LOGGER.info(f"阶段数: {len(self.phases)}")
            LOGGER.info("=" * 70)


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


def format_memory(mb: float) -> str:
    """Format memory in human-readable format."""
    if mb < 1024:
        return f"{mb:.2f}MB"
    elif mb < 1024 * 1024:
        return f"{mb/1024:.2f}GB"
    else:
        return f"{mb/(1024*1024):.2f}TB"


def generate_experiment_report(
    results: List[Dict[str, Any]],
    output_file: Optional[str] = None,
) -> str:
    """Generate a comprehensive experiment report.
    
    Args:
        results: List of experiment result dictionaries
        output_file: Optional output file path
    
    Returns:
        Report text
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MELD 实验报告")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary statistics
    total_experiments = len(results)
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    
    lines.append("实验统计:")
    lines.append(f"  总数: {total_experiments}")
    lines.append(f"  成功: {successful}")
    lines.append(f"  失败: {failed}")
    lines.append(f"  跳过: {skipped}")
    lines.append("")
    
    # Results table
    lines.append("实验结果:")
    lines.append("-" * 80)
    lines.append(f"{'实验名称':<30} {'状态':<10} {'F1':<10} {'AUC':<10}")
    lines.append("-" * 80)
    
    for result in results:
        name = result.get("name", "unknown")
        status = result.get("status", "unknown")
        metrics = result.get("metrics", {})
        f1 = metrics.get("macro_f1", "N/A")
        auc = metrics.get("auroc", "N/A")
        
        if isinstance(f1, float):
            f1 = f"{f1:.4f}"
        if isinstance(auc, float):
            auc = f"{auc:.4f}"
        
        lines.append(f"{name:<30} {status:<10} {f1:<10} {auc:<10}")
    
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        LOGGER.info(f"报告已保存到: {output_file}")
    
    return report_text

