#!/usr/bin/env python3
"""
增强的 W&B 可视化工具
提供 ROC/PR 曲线、混淆矩阵、层性能对比等功能
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    sns.set_style("whitegrid")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None


def log_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str = "ROC Curve") -> None:
    """记录 ROC 曲线到 W&B"""
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return

    try:
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"ROC 曲线生成失败: {exc}")


def log_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str = "PR Curve") -> None:
    """记录 PR 曲线到 W&B"""
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return

    try:
        from sklearn.metrics import precision_recall_curve, auc

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline = {baseline:.4f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"PR 曲线生成失败: {exc}")


def log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix") -> None:
    """记录混淆矩阵到 W&B"""
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return

    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Benign', 'Malicious'],
                    yticklabels=['Benign', 'Malicious'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        plt.tight_layout()

        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"混淆矩阵生成失败: {exc}")


def log_layer_performance_chart(
    results: List[Dict],
    title: str = "Layer Performance",
    include_metrics: Optional[List[str]] = None
) -> None:
    """记录各层性能图表到 W&B（通用版本）"""
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE or not results:
        return

    try:
        layers = [r["layer_index"] for r in results]
        
        if include_metrics is None:
            include_metrics = ["macro_f1", "aupr", "auroc"]
        
        plt.figure(figsize=(12, 6))
        
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        markers = ['o', 's', '^', 'v', 'D', 'p']
        
        for idx, metric in enumerate(include_metrics):
            if metric in results[0]:
                values = [r[metric] for r in results]
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                label = metric.replace('_', ' ').title()
                plt.plot(layers, values, f'{color}-{marker}', label=label, linewidth=2, markersize=6)

        plt.xlabel('Layer Index')
        plt.ylabel('Performance Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        wandb.log({"layer_performance": wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"层性能图表生成失败: {exc}")


def log_score_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Score Distribution"
) -> None:
    """记录预测分数分布到 W&B"""
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return

    try:
        benign_scores = y_prob[y_true == 0]
        malicious_scores = y_prob[y_true == 1]

        plt.figure(figsize=(10, 6))
        plt.hist(benign_scores, bins=50, alpha=0.6, label='Benign', color='blue', density=True)
        plt.hist(malicious_scores, bins=50, alpha=0.6, label='Malicious', color='red', density=True)
        
        if threshold is not None:
            plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
        
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"分数分布图生成失败: {exc}")


def log_experiment_comparison_table(
    experiments: List[Dict],
    title: str = "Experiment Comparison"
) -> None:
    """记录实验对比表格到 W&B"""
    if not WANDB_AVAILABLE or not wandb.run:
        return

    try:
        columns = ["Experiment", "Method", "Macro F1", "AUROC", "AUPR", "TPR", "FPR", "Best Layer"]
        data = []
        
        for exp in experiments:
            best = exp.get("best", {})
            row = [
                exp.get("name", "Unknown"),
                exp.get("method", "Unknown"),
                f"{best.get('macro_f1', 0):.4f}",
                f"{best.get('auroc', 0):.4f}",
                f"{best.get('aupr', 0):.4f}",
                f"{best.get('tpr_id', 0):.4f}",
                f"{best.get('fpr_id', 0):.4f}",
                str(best.get('layer_index', 'N/A')),
            ]
            data.append(row)
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({title.lower().replace(" ", "_"): table})
    except Exception as exc:
        print(f"实验对比表格生成失败: {exc}")


def log_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    title: str = "Threshold Analysis"
) -> None:
    """记录阈值分析（TPR/FPR vs 阈值）"""
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return

    try:
        from sklearn.metrics import confusion_matrix

        tprs = []
        fprs = []
        
        for tau in thresholds:
            y_pred = (y_prob >= tau).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            else:
                tpr = fpr = 0
            tprs.append(tpr)
            fprs.append(fpr)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, tprs, 'b-o', label='TPR', linewidth=2, markersize=4)
        plt.plot(thresholds, fprs, 'r-s', label='FPR', linewidth=2, markersize=4)
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"阈值分析图生成失败: {exc}")


def log_best_layer_comprehensive_panel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    layer_idx: int,
    metrics: Dict,
    prefix: str = "best_layer"
) -> None:
    """记录最佳层的综合性能面板（多子图）"""
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return

    try:
        import matplotlib.gridspec as gridspec
        from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

        y_pred = (y_prob >= threshold).astype(int)
        
        # 计算 ROC 和 PR 曲线
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # 创建综合面板
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 子图1: ROC 曲线
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # 子图2: PR 曲线
        ax2 = fig.add_subplot(gs[0, 1])
        baseline = np.sum(y_true) / len(y_true)
        ax2.plot(recall, precision, color='blue', lw=2, label=f'PR (AUC = {pr_auc:.4f})')
        ax2.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline = {baseline:.4f}')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        # 子图3: 混淆矩阵热力图
        ax3 = fig.add_subplot(gs[0, 2])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Benign', 'Malicious'],
                    yticklabels=['Benign', 'Malicious'],
                    ax=ax3)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Confusion Matrix')

        # 子图4: 分数分布
        ax4 = fig.add_subplot(gs[1, 0])
        benign_scores = y_prob[y_true == 0]
        malicious_scores = y_prob[y_true == 1]
        ax4.hist(benign_scores, bins=50, alpha=0.6, label='Benign', color='blue', density=True)
        ax4.hist(malicious_scores, bins=50, alpha=0.6, label='Malicious', color='red', density=True)
        ax4.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
        ax4.set_xlabel('Prediction Score')
        ax4.set_ylabel('Density')
        ax4.set_title('Score Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 子图5: 关键指标条形图
        ax5 = fig.add_subplot(gs[1, 1])
        metric_names = ['Macro F1', 'AUROC', 'AUPR', 'Accuracy', 'TPR', 'FPR']
        metric_values = [
            metrics.get('macro_f1', 0),
            metrics.get('auroc', 0),
            metrics.get('aupr', 0),
            metrics.get('accuracy', 0),
            metrics.get('tpr_id', 0),
            metrics.get('fpr_id', 0),
        ]
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'red']
        bars = ax5.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax5.set_ylabel('Score')
        ax5.set_title('Key Metrics')
        ax5.set_ylim([0, 1.1])
        ax5.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 子图6: 混淆矩阵统计
        ax6 = fig.add_subplot(gs[1, 2])
        cm_stats = ['TN', 'FP', 'FN', 'TP']
        cm_values = [tn, fp, fn, tp]
        colors_cm = ['blue', 'red', 'orange', 'green']
        bars = ax6.bar(cm_stats, cm_values, color=colors_cm, alpha=0.7)
        ax6.set_ylabel('Count')
        ax6.set_title('Confusion Matrix Elements')
        ax6.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for bar, val in zip(bars, cm_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}', ha='center', va='bottom', fontsize=10)

        plt.suptitle(f'{prefix} - Layer {layer_idx} Comprehensive Analysis', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        wandb.log({f"{prefix}_comprehensive_panel": wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"最佳层综合面板生成失败: {exc}")


def log_experiment_comparison_charts(
    experiments: List[Dict],
    title: str = "Experiment Comparison"
) -> None:
    """记录实验对比图表到 W&B（条形图、雷达图等）"""
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return
    
    if not experiments:
        return
    
    try:
        import matplotlib.gridspec as gridspec
        
        # 提取数据
        exp_names = [exp.get("name", f"Exp_{i}") for i, exp in enumerate(experiments)]
        methods = [exp.get("method", exp.get("baseline", "Unknown")) for exp in experiments]
        
        # 提取指标
        metrics_data = {
            "Macro F1": [exp.get("best", {}).get("macro_f1", 0) for exp in experiments],
            "AUROC": [exp.get("best", {}).get("auroc", 0) for exp in experiments],
            "AUPR": [exp.get("best", {}).get("aupr", 0) for exp in experiments],
            "Accuracy": [exp.get("best", {}).get("accuracy", 0) for exp in experiments],
            "TPR": [exp.get("best", {}).get("tpr_id", exp.get("best", {}).get("tpr_ood", 0)) for exp in experiments],
            "FPR": [exp.get("best", {}).get("fpr_id", exp.get("best", {}).get("fpr_ood", 0)) for exp in experiments],
        }
        
        # 创建多子图对比
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 子图1: 主要指标条形图
        ax1 = fig.add_subplot(gs[0, 0])
        x_pos = np.arange(len(exp_names))
        width = 0.2
        for i, (metric_name, values) in enumerate(list(metrics_data.items())[:4]):
            ax1.bar(x_pos + i * width, values, width, label=metric_name, alpha=0.8)
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Score')
        ax1.set_title('Main Metrics Comparison')
        ax1.set_xticks(x_pos + width * 1.5)
        ax1.set_xticklabels(exp_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.1])
        
        # 子图2: 雷达图（前4个指标）
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for i, exp_name in enumerate(exp_names[:5]):  # 最多5个实验
            values = [
                metrics_data["Macro F1"][i],
                metrics_data["AUROC"][i],
                metrics_data["AUPR"][i],
                metrics_data["Accuracy"][i],
            ]
            values += values[:1]  # 闭合
            ax2.plot(angles, values, 'o-', linewidth=2, label=exp_name[:15])
            ax2.fill(angles, values, alpha=0.15)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(["F1", "AUROC", "AUPR", "Accuracy"])
        ax2.set_ylim([0, 1])
        ax2.set_title('Radar Chart Comparison', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax2.grid(True)
        
        # 子图3: TPR vs FPR 散点图
        ax3 = fig.add_subplot(gs[0, 2])
        colors = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))
        for i, (name, tpr, fpr) in enumerate(zip(exp_names, metrics_data["TPR"], metrics_data["FPR"])):
            ax3.scatter(fpr, tpr, s=100, c=[colors[i]], label=name[:15], alpha=0.7, edgecolors='black')
            ax3.annotate(name[:10], (fpr, tpr), xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('TPR vs FPR')
        ax3.set_xlim([-0.05, max(metrics_data["FPR"]) * 1.2 if max(metrics_data["FPR"]) > 0 else 0.1])
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        
        # 子图4: 热力图对比
        ax4 = fig.add_subplot(gs[1, 0])
        heatmap_data = np.array([
            metrics_data["Macro F1"],
            metrics_data["AUROC"],
            metrics_data["AUPR"],
            metrics_data["Accuracy"],
        ])
        im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax4.set_xticks(np.arange(len(exp_names)))
        ax4.set_xticklabels(exp_names, rotation=45, ha='right')
        ax4.set_yticks(np.arange(len(["F1", "AUROC", "AUPR", "Accuracy"])))
        ax4.set_yticklabels(["F1", "AUROC", "AUPR", "Accuracy"])
        ax4.set_title('Metrics Heatmap')
        plt.colorbar(im, ax=ax4)
        
        # 添加数值标注
        for i in range(len(["F1", "AUROC", "AUPR", "Accuracy"])):
            for j in range(len(exp_names)):
                text = ax4.text(j, i, f'{heatmap_data[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # 子图5: 箱线图对比（如果有多个种子）
        ax5 = fig.add_subplot(gs[1, 1])
        # 检查是否有 per_seed 数据
        has_seeds = any("per_seed" in exp for exp in experiments)
        if has_seeds:
            seed_data = []
            seed_labels = []
            for exp in experiments:
                if "per_seed" in exp:
                    seeds = exp["per_seed"]
                    f1_scores = [s.get("best", {}).get("macro_f1", 0) for s in seeds]
                    seed_data.append(f1_scores)
                    seed_labels.append(exp.get("name", "Unknown")[:15])
            if seed_data:
                bp = ax5.boxplot(seed_data, labels=seed_labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                ax5.set_ylabel('Macro F1 Score')
                ax5.set_title('Performance Distribution (Multiple Seeds)')
                ax5.grid(True, alpha=0.3, axis='y')
                plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            # 如果没有种子数据，显示指标对比条形图
            metric_names_short = ["F1", "AUC", "AUPR", "Acc"]
            x = np.arange(len(metric_names_short))
            for i, exp_name in enumerate(exp_names[:5]):
                values = [
                    metrics_data["Macro F1"][i],
                    metrics_data["AUROC"][i],
                    metrics_data["AUPR"][i],
                    metrics_data["Accuracy"][i],
                ]
                ax5.plot(x, values, marker='o', label=exp_name[:15], linewidth=2, markersize=8)
            ax5.set_xticks(x)
            ax5.set_xticklabels(metric_names_short)
            ax5.set_ylabel('Score')
            ax5.set_title('Metrics Comparison')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim([0, 1.1])
        
        # 子图6: 排名对比
        ax6 = fig.add_subplot(gs[1, 2])
        # 计算每个实验的综合排名（基于多个指标的平均排名）
        rankings = {}
        for metric_name, values in metrics_data.items():
            if metric_name in ["Macro F1", "AUROC", "AUPR", "Accuracy"]:
                sorted_indices = np.argsort(values)[::-1]  # 降序
                for rank, idx in enumerate(sorted_indices):
                    if idx not in rankings:
                        rankings[idx] = []
                    rankings[idx].append(rank + 1)
        
        avg_rankings = [np.mean(rankings.get(i, [len(exp_names)])) for i in range(len(exp_names))]
        sorted_idx = np.argsort(avg_rankings)
        
        colors_rank = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(exp_names)))
        bars = ax6.barh(range(len(exp_names)), [avg_rankings[i] for i in sorted_idx], 
                       color=[colors_rank[i] for i in sorted_idx], alpha=0.7)
        ax6.set_yticks(range(len(exp_names)))
        ax6.set_yticklabels([exp_names[i] for i in sorted_idx])
        ax6.set_xlabel('Average Rank (lower is better)')
        ax6.set_title('Overall Ranking')
        ax6.grid(True, alpha=0.3, axis='x')
        
        # 添加排名数值
        for i, (bar, rank) in enumerate(zip(bars, [avg_rankings[j] for j in sorted_idx])):
            ax6.text(rank, i, f' {rank:.2f}', va='center', fontsize=9)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
        
    except Exception as exc:
        print(f"实验对比图表生成失败: {exc}")


def log_roc_curves_comparison(
    experiments_data: List[Dict[str, any]],
    title: str = "ROC Curves Comparison"
) -> None:
    """在同一图上对比多个实验的 ROC 曲线"""
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))
        
        for i, exp_data in enumerate(experiments_data):
            y_true = exp_data["y_true"]
            y_prob = exp_data["y_prob"]
            exp_name = exp_data.get("name", f"Experiment {i+1}")
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                    label=f'{exp_name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"ROC 曲线对比生成失败: {exc}")


def log_pr_curves_comparison(
    experiments_data: List[Dict[str, any]],
    title: str = "PR Curves Comparison"
) -> None:
    """在同一图上对比多个实验的 PR 曲线"""
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        from sklearn.metrics import precision_recall_curve, auc
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))
        
        for i, exp_data in enumerate(experiments_data):
            y_true = exp_data["y_true"]
            y_prob = exp_data["y_prob"]
            exp_name = exp_data.get("name", f"Experiment {i+1}")
            
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, color=colors[i], lw=2,
                    label=f'{exp_name} (AUC = {pr_auc:.4f})')
        
        # Baseline (random classifier)
        baseline = np.sum(experiments_data[0]["y_true"]) / len(experiments_data[0]["y_true"])
        plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline = {baseline:.4f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"PR 曲线对比生成失败: {exc}")

