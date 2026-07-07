#!/usr/bin/env python3
"""分析训练结果"""
import json
import sys
from typing import Dict, List

def analyze_results(result_file: str = "result.json"):
    """分析训练结果"""
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("训练结果分析报告")
    print("=" * 80)
    print()
    
    # 1. 实验配置信息
    print("【实验配置】")
    print(f"  数据集: {data['config']['md_dir']}")
    print(f"  样本数: {data['num_samples']}")
    print(f"  模型层数: {data['num_layers']}")
    print(f"  隐藏层大小: {data['hidden_size']}")
    print(f"  分类器: {data['clf']}")
    print(f"  交叉验证折数: {data['cv_splits']}")
    print(f"  目标FPR: {data['target_fpr']}")
    print()
    
    # 2. 最佳模型性能
    best = data['best']
    print("【最佳模型性能】")
    print(f"  最佳层: 第 {best['layer_index']} 层")
    print(f"  阈值: {best['threshold']:.4f}")
    print(f"  Macro F1: {best['macro_f1']:.4f} ({best['macro_f1']*100:.2f}%)")
    print(f"  AUPR: {best['aupr']:.4f}")
    print(f"  AUROC: {best['auroc']:.4f}")
    print(f"  Drift Score: {best['drift_score']:.4f}")
    print(f"  TPR (ID): {best['tpr_id']:.4f} ({best['tpr_id']*100:.2f}%)")
    print(f"  TPR (OOD): {best['tpr_ood']:.4f} ({best['tpr_ood']*100:.2f}%)")
    print(f"  FPR (ID/OOD): {best['fpr_id']:.4f} ({best['fpr_id']*100:.2f}%)")
    print(f"  Z-score (ID): {best['z_id']:.4f}")
    print(f"  Z-score (OOD): {best['z_ood']:.4f}")
    print()
    
    # 3. 各层性能对比
    all_layers = data['all_layers']
    print("【各层性能对比】")
    print(f"{'层':<4} {'Macro F1':<10} {'AUPR':<10} {'AUROC':<10} {'TPR':<10} {'FPR':<10} {'Drift Score':<12}")
    print("-" * 80)
    
    for layer in all_layers:
        layer_idx = layer['layer_index']
        macro_f1 = layer['macro_f1']
        aupr = layer['aupr']
        auroc = layer['auroc']
        tpr = layer['tpr_id']
        fpr = layer['fpr_id']
        drift = layer['drift_score']
        
        marker = "★" if layer_idx == best['layer_index'] else " "
        print(f"{marker}{layer_idx:<3} {macro_f1:<10.4f} {aupr:<10.4f} {auroc:<10.4f} "
              f"{tpr:<10.4f} {fpr:<10.4f} {drift:<12.4f}")
    print()
    
    # 4. 性能趋势分析
    print("【性能趋势分析】")
    
    # Macro F1趋势
    macro_f1_values = [l['macro_f1'] for l in all_layers]
    best_f1_layer = max(range(len(macro_f1_values)), key=lambda i: macro_f1_values[i]) + 1
    worst_f1_layer = min(range(len(macro_f1_values)), key=lambda i: macro_f1_values[i]) + 1
    print(f"  Macro F1:")
    print(f"    - 最高: 层 {best_f1_layer} ({max(macro_f1_values):.4f})")
    print(f"    - 最低: 层 {worst_f1_layer} ({min(macro_f1_values):.4f})")
    print(f"    - 平均: {sum(macro_f1_values)/len(macro_f1_values):.4f}")
    
    # AUROC趋势
    auroc_values = [l['auroc'] for l in all_layers]
    best_auroc_layer = max(range(len(auroc_values)), key=lambda i: auroc_values[i]) + 1
    print(f"  AUROC:")
    print(f"    - 最高: 层 {best_auroc_layer} ({max(auroc_values):.4f})")
    print(f"    - 最低: 层 {min(range(len(auroc_values)), key=lambda i: auroc_values[i]) + 1} ({min(auroc_values):.4f})")
    print(f"    - 平均: {sum(auroc_values)/len(auroc_values):.4f}")
    
    # AUPR趋势
    aupr_values = [l['aupr'] for l in all_layers]
    best_aupr_layer = max(range(len(aupr_values)), key=lambda i: aupr_values[i]) + 1
    print(f"  AUPR:")
    print(f"    - 最高: 层 {best_aupr_layer} ({max(aupr_values):.4f})")
    print(f"    - 最低: 层 {min(range(len(aupr_values)), key=lambda i: aupr_values[i]) + 1} ({min(aupr_values):.4f})")
    print(f"    - 平均: {sum(aupr_values)/len(aupr_values):.4f}")
    print()
    
    # 5. 关键发现
    print("【关键发现】")
    
    # 检查ID和OOD的一致性
    id_ood_consistent = all(
        abs(layer['tpr_id'] - layer['tpr_ood']) < 0.001 and
        abs(layer['fpr_id'] - layer['fpr_ood']) < 0.001
        for layer in all_layers
    )
    if id_ood_consistent:
        print("  ✓ ID和OOD数据集表现高度一致，模型泛化能力良好")
    else:
        print("  ⚠ ID和OOD数据集表现存在差异")
    
    # FPR控制
    fpr_values = [l['fpr_id'] for l in all_layers]
    target_fpr = data['target_fpr']
    fpr_control = all(fpr <= target_fpr * 1.5 for fpr in fpr_values)  # 允许50%误差
    if fpr_control:
        print(f"  ✓ FPR控制在目标范围内（目标: {target_fpr:.3f}，实际: {max(fpr_values):.4f}）")
    else:
        print(f"  ⚠ FPR超出目标范围（目标: {target_fpr:.3f}，实际最大: {max(fpr_values):.4f}）")
    
    # 层深度影响
    early_layers = [l['macro_f1'] for l in all_layers[:5]]
    late_layers = [l['macro_f1'] for l in all_layers[-5:]]
    early_avg = sum(early_layers) / len(early_layers)
    late_avg = sum(late_layers) / len(late_layers)
    
    if late_avg > early_avg:
        print(f"  ✓ 深层特征表现更好（前5层平均: {early_avg:.4f}, 后5层平均: {late_avg:.4f}）")
    else:
        print(f"  ⚠ 浅层特征表现更好（前5层平均: {early_avg:.4f}, 后5层平均: {late_avg:.4f}）")
    
    # 最佳层位置
    if best['layer_index'] == len(all_layers):
        print(f"  ✓ 最佳性能出现在最深层（第{best['layer_index']}层）")
    elif best['layer_index'] <= 3:
        print(f"  ✓ 最佳性能出现在浅层（第{best['layer_index']}层）")
    else:
        print(f"  ✓ 最佳性能出现在中间层（第{best['layer_index']}层）")
    
    print()
    
    # 6. 性能评估
    print("【性能评估】")
    if best['macro_f1'] >= 0.95:
        print("  ★★★★★ 优秀：Macro F1 > 0.95")
    elif best['macro_f1'] >= 0.90:
        print("  ★★★★☆ 良好：Macro F1 > 0.90")
    elif best['macro_f1'] >= 0.85:
        print("  ★★★☆☆ 中等：Macro F1 > 0.85")
    else:
        print("  ★★☆☆☆ 需改进：Macro F1 < 0.85")
    
    if best['auroc'] >= 0.98:
        print("  ★★★★★ 优秀：AUROC > 0.98")
    elif best['auroc'] >= 0.95:
        print("  ★★★★☆ 良好：AUROC > 0.95")
    elif best['auroc'] >= 0.90:
        print("  ★★★☆☆ 中等：AUROC > 0.90")
    else:
        print("  ★★☆☆☆ 需改进：AUROC < 0.90")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    result_file = sys.argv[1] if len(sys.argv) > 1 else "result.json"
    analyze_results(result_file)




