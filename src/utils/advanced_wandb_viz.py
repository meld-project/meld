#!/usr/bin/env python3
"""
高级 W&B 可视化工具
包括分数分布分析、时间漂移、族群差异、校准曲线、误报分析等
"""

from __future__ import annotations

import re
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
    from scipy import stats
    from scipy.spatial.distance import jensenshannon
    MATPLOTLIB_AVAILABLE = True
    sns.set_style("whitegrid")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None
    stats = None
    jensenshannon = None


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """计算 KL 散度 KL(P||Q)"""
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    return np.sum(p * np.log(p / q))


def wasserstein_distance(p: np.ndarray, q: np.ndarray) -> float:
    """计算 Wasserstein 距离（1D）"""
    try:
        from scipy.stats import wasserstein_distance as wd
        return wd(p, q)
    except ImportError:
        # 简单实现：排序后的 L1 距离
        p_sorted = np.sort(p)
        q_sorted = np.sort(q)
        return np.mean(np.abs(p_sorted - q_sorted))


def log_score_distribution_with_divergence(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    title: str = "Score Distribution with Divergence",
    use_kde: bool = True,
) -> None:
    """
    分数分布 + 阈值：负样本/正样本的核密度曲线或直方图（两种颜色）
    加上当前阈值 τ 的竖线，旁边标出 KL/JS 或 Wasserstein 距离指标
    """
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return

    try:
        benign_scores = y_prob[y_true == 0]
        malicious_scores = y_prob[y_true == 1]

        if len(benign_scores) == 0 or len(malicious_scores) == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # 计算分布距离
        # 使用直方图估计概率分布
        bins = np.linspace(min(y_prob.min(), threshold - 0.1), 
                          max(y_prob.max(), threshold + 0.1), 50)
        benign_hist, _ = np.histogram(benign_scores, bins=bins, density=True)
        malicious_hist, _ = np.histogram(malicious_scores, bins=bins, density=True)
        
        # 归一化
        benign_hist = benign_hist / (benign_hist.sum() + 1e-10)
        malicious_hist = malicious_hist / (malicious_hist.sum() + 1e-10)
        
        # 计算距离指标
        kl_benign_malicious = kl_divergence(benign_hist, malicious_hist)
        kl_malicious_benign = kl_divergence(malicious_hist, benign_hist)
        js_distance = jensenshannon(benign_hist, malicious_hist) if jensenshannon else 0
        wass_dist = wasserstein_distance(benign_scores, malicious_scores)

        if use_kde:
            # 核密度估计
            from scipy.stats import gaussian_kde
            try:
                kde_benign = gaussian_kde(benign_scores)
                kde_malicious = gaussian_kde(malicious_scores)
                x_range = np.linspace(bins[0], bins[-1], 200)
                ax.plot(x_range, kde_benign(x_range), 'b-', label='Benign (KDE)', linewidth=2, alpha=0.7)
                ax.plot(x_range, kde_malicious(x_range), 'r-', label='Malicious (KDE)', linewidth=2, alpha=0.7)
            except:
                use_kde = False

        if not use_kde:
            # 直方图
            ax.hist(benign_scores, bins=bins, alpha=0.6, label='Benign', color='blue', density=True)
            ax.hist(malicious_scores, bins=bins, alpha=0.6, label='Malicious', color='red', density=True)

        # 添加阈值线
        ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2, 
                   label=f'Threshold τ = {threshold:.4f}')
        
        # 计算阈值处的密度
        threshold_idx = np.argmin(np.abs(bins[:-1] - threshold))
        benign_density_at_threshold = benign_hist[threshold_idx] if threshold_idx < len(benign_hist) else 0
        malicious_density_at_threshold = malicious_hist[threshold_idx] if threshold_idx < len(malicious_hist) else 0
        
        # 添加文本标注
        textstr = f'KL(Benign||Malicious) = {kl_benign_malicious:.4f}\n'
        textstr += f'KL(Malicious||Benign) = {kl_malicious_benign:.4f}\n'
        textstr += f'JS Distance = {js_distance:.4f}\n'
        textstr += f'Wasserstein = {wass_dist:.4f}\n'
        textstr += f'Benign density @τ = {benign_density_at_threshold:.4f}\n'
        textstr += f'Malicious density @τ = {malicious_density_at_threshold:.4f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        ax.set_xlabel('Prediction Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"分数分布图生成失败: {exc}")


def log_temporal_drift_analysis(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: Optional[np.ndarray],
    thresholds: Optional[np.ndarray],
    title: str = "Temporal Drift Analysis",
) -> None:
    """
    时间滑窗漂移图：把每周/每月的 neg_score 与 pos_score 分布用 KL、MMD 或 z-score 画成折线
    再叠加 TPR/FPR 轨迹
    """
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return

    if timestamps is None:
        print("缺少时间戳信息，跳过时间漂移分析")
        return

    try:
        from datetime import datetime
        from sklearn.metrics import confusion_matrix

        # 转换时间戳
        if isinstance(timestamps[0], str):
            dates = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
        else:
            dates = timestamps

        # 按周/月分组
        weeks = []
        week_scores_neg = []
        week_scores_pos = []
        week_tprs = []
        week_fprs = []
        week_kls = []

        # 简化：按月分组
        from collections import defaultdict
        monthly_data = defaultdict(lambda: {'neg': [], 'pos': [], 'labels': [], 'scores': []})
        
        for i, (score, label, date) in enumerate(zip(scores, labels, dates)):
            month_key = f"{date.year}-{date.month:02d}"
            monthly_data[month_key]['scores'].append(score)
            monthly_data[month_key]['labels'].append(label)
            if label == 0:
                monthly_data[month_key]['neg'].append(score)
            else:
                monthly_data[month_key]['pos'].append(score)

        # 计算每月指标
        months = sorted(monthly_data.keys())
        ref_month = months[0] if months else None
        
        for month in months:
            data = monthly_data[month]
            if len(data['neg']) == 0 or len(data['pos']) == 0:
                continue

            neg_scores = np.array(data['neg'])
            pos_scores = np.array(data['pos'])
            all_scores = np.array(data['scores'])
            all_labels = np.array(data['labels'])

            # 计算 KL 散度（相对于第一个月）
            if ref_month and month != ref_month:
                ref_data = monthly_data[ref_month]
                ref_neg = np.array(ref_data['neg'])
                if len(ref_neg) > 0:
                    bins = np.linspace(min(all_scores.min(), ref_neg.min()),
                                     max(all_scores.max(), ref_neg.max()), 20)
                    ref_hist, _ = np.histogram(ref_neg, bins=bins, density=True)
                    curr_hist, _ = np.histogram(neg_scores, bins=bins, density=True)
                    ref_hist = ref_hist / (ref_hist.sum() + 1e-10)
                    curr_hist = curr_hist / (curr_hist.sum() + 1e-10)
                    kl = kl_divergence(curr_hist, ref_hist)
                else:
                    kl = 0
            else:
                kl = 0

            # 计算 TPR/FPR（如果有阈值）
            if thresholds is not None and len(thresholds) > 0:
                threshold = thresholds[0]  # 使用第一个阈值
                y_pred = (all_scores >= threshold).astype(int)
                cm = confusion_matrix(all_labels, y_pred)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                else:
                    tpr = fpr = 0
            else:
                tpr = fpr = 0

            week_scores_neg.append(neg_scores.mean())
            week_scores_pos.append(pos_scores.mean())
            week_tprs.append(tpr)
            week_fprs.append(fpr)
            week_kls.append(kl)

        if len(months) == 0:
            return

        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 子图1: 分数分布和 KL 散度
        ax1_twin = ax1.twinx()
        ax1.plot(months, week_scores_neg, 'b-o', label='Mean Neg Score', linewidth=2, markersize=6)
        ax1.plot(months, week_scores_pos, 'r-s', label='Mean Pos Score', linewidth=2, markersize=6)
        ax1_twin.plot(months, week_kls, 'g--^', label='KL Divergence', linewidth=2, markersize=6, alpha=0.7)
        
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Mean Score', color='black')
        ax1_twin.set_ylabel('KL Divergence', color='green')
        ax1.set_title('Score Distribution Drift Over Time')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 子图2: TPR/FPR 轨迹
        ax2.plot(months, week_tprs, 'g-o', label='TPR', linewidth=2, markersize=6)
        ax2.plot(months, week_fprs, 'r-s', label='FPR', linewidth=2, markersize=6)
        ax2.axhline(y=0.01, color='k', linestyle='--', alpha=0.5, label='Target FPR (1%)')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Rate')
        ax2.set_title('TPR/FPR Trajectory Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"时间漂移分析生成失败: {exc}")


def log_family_difference_heatmap(
    scores_by_family: Dict[str, np.ndarray],
    train_scores: np.ndarray,
    title: str = "Family Difference Heatmap",
) -> None:
    """
    族群/模态差异热力图：held-out 家族 vs. 训练分布的 KL matrix
    """
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return

    try:
        families = list(scores_by_family.keys())
        if len(families) == 0:
            return

        # 计算 KL 散度矩阵
        kl_matrix = np.zeros((len(families), len(families)))
        
        # 统一 bins
        all_scores = np.concatenate([train_scores] + list(scores_by_family.values()))
        bins = np.linspace(all_scores.min(), all_scores.max(), 30)

        # 计算训练集分布
        train_hist, _ = np.histogram(train_scores, bins=bins, density=True)
        train_hist = train_hist / (train_hist.sum() + 1e-10)

        # 计算每个家族的分布
        family_hists = {}
        for family, scores in scores_by_family.items():
            hist, _ = np.histogram(scores, bins=bins, density=True)
            hist = hist / (hist.sum() + 1e-10)
            family_hists[family] = hist

        # 计算 KL 散度
        for i, family1 in enumerate(families):
            for j, family2 in enumerate(families):
                if i == j:
                    kl_matrix[i, j] = 0
                else:
                    hist1 = family_hists[family1]
                    hist2 = family_hists[family2]
                    kl_matrix[i, j] = kl_divergence(hist1, hist2)

        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(kl_matrix, annot=True, fmt='.4f', cmap='YlOrRd', 
                    xticklabels=families, yticklabels=families,
                    cbar_kws={'label': 'KL Divergence'})
        plt.title(title)
        plt.xlabel('Family')
        plt.ylabel('Family')
        plt.tight_layout()

        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"家族差异热力图生成失败: {exc}")


def log_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve (Reliability Diagram)",
) -> None:
    """
    校准曲线/可靠性图：把 LEC 概率输出 vs. 真实频率画成 reliability diagram 或 ECE 柱状图
    """
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return

    try:
        from sklearn.calibration import calibration_curve

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )

        # 计算 ECE (Expected Calibration Error)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 子图1: 可靠性图
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="LEC", linewidth=2, markersize=6)
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'Reliability Diagram (ECE = {ece:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: ECE 柱状图
        bin_counts = []
        bin_accuracies = []
        bin_confidences = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.sum()
            bin_counts.append(prop_in_bin)
            if prop_in_bin > 0:
                bin_accuracies.append(y_true[in_bin].mean())
                bin_confidences.append(y_prob[in_bin].mean())
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2)

        x_pos = np.arange(len(bin_counts))
        width = 0.35
        ax2.bar(x_pos - width/2, bin_accuracies, width, label='Accuracy', alpha=0.7)
        ax2.bar(x_pos + width/2, bin_confidences, width, label='Confidence', alpha=0.7)
        ax2.set_xlabel('Bin')
        ax2.set_ylabel('Value')
        ax2.set_title('ECE Breakdown by Bin')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        wandb.log({"ece": ece})
        plt.close()
    except Exception as exc:
        print(f"校准曲线生成失败: {exc}")


def log_false_positive_analysis(
    false_positives: List[Dict],
    title: str = "False Positive Root Cause Analysis",
) -> None:
    """
    误报根因分布图：对所有 false positives 画出"报告长度/家族/IOC 覆盖率"的堆叠柱
    """
    if not WANDB_AVAILABLE or not wandb.run or not MATPLOTLIB_AVAILABLE:
        return

    if len(false_positives) == 0:
        return

    try:
        # 提取特征
        lengths = [fp.get('length', 0) for fp in false_positives]
        families = [fp.get('family', 'Unknown') for fp in false_positives]
        ioc_coverage = [fp.get('ioc_coverage', 0) for fp in false_positives]

        # 按家族分组
        from collections import Counter
        family_counts = Counter(families)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: 家族分布
        families_sorted = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        families_names = [f[0] for f in families_sorted]
        families_counts = [f[1] for f in families_sorted]
        axes[0, 0].barh(families_names, families_counts, color='coral')
        axes[0, 0].set_xlabel('Count')
        axes[0, 0].set_title('False Positives by Family')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # 子图2: 报告长度分布
        axes[0, 1].hist(lengths, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Report Length (chars)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('False Positive Report Length Distribution')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 子图3: IOC 覆盖率分布
        axes[1, 0].hist(ioc_coverage, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('IOC Coverage')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('False Positive IOC Coverage Distribution')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 子图4: 堆叠柱状图（家族 vs 长度区间）
        length_bins = ['0-500', '500-1000', '1000-2000', '2000+']
        length_ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, float('inf'))]
        
        family_length_matrix = {}
        for family in set(families):
            family_length_matrix[family] = [0] * len(length_bins)
        
        for fp in false_positives:
            family = fp.get('family', 'Unknown')
            length = fp.get('length', 0)
            for idx, (low, high) in enumerate(length_ranges):
                if low <= length < high:
                    if family not in family_length_matrix:
                        family_length_matrix[family] = [0] * len(length_bins)
                    family_length_matrix[family][idx] += 1
                    break

        # 只显示前5个家族
        top_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_family_names = [f[0] for f in top_families]
        
        bottom = np.zeros(len(length_bins))
        for family in top_family_names:
            counts = family_length_matrix.get(family, [0] * len(length_bins))
            axes[1, 1].bar(length_bins, counts, bottom=bottom, label=family, alpha=0.7)
            bottom += counts
        
        axes[1, 1].set_xlabel('Report Length Range')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('False Positives: Family × Length Range')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        wandb.log({title.lower().replace(" ", "_"): wandb.Image(plt)})
        plt.close()
    except Exception as exc:
        print(f"误报分析图生成失败: {exc}")

