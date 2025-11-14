"""
实验分析命令行工具
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis import ExperimentAnalyzer, AutoAnalyzer

LOGGER = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def analyze_command(args):
    """分析命令"""
    print(f"正在分析实验结果...")

    analyzer = ExperimentAnalyzer(args.results_dir)

    # 加载结果
    if args.pattern:
        count = analyzer.load_results(args.pattern)
    else:
        count = analyzer.load_results()

    print(f"加载了 {count} 个实验结果")

    if count == 0:
        print("没有找到实验结果")
        return

    # 定义要分析的指标
    metrics = args.metrics.split(",") if args.metrics else ["accuracy", "macro_f1", "auroc"]

    # 生成报告
    if args.report:
        print("生成分析报告...")
        report = analyzer.generate_report(metrics, args.output)
        print(f"报告已保存到: {args.output}")

    # 创建可视化
    if args.viz:
        print("创建可视化图表...")
        analyzer.create_visualizations(metrics)
        print(f"图表已保存到: {Path(args.results_dir) / 'plots'}")

    # 导出结果
    if args.export:
        print(f"导出结果为 {args.export} 格式...")
        output_path = analyzer.export_to_wandb() if args.export == "wandb" else None
        if output_path:
            print(f"结果已导出")


def compare_command(args):
    """比较命令"""
    print(f"正在比较模型...")

    analyzer = ExperimentAnalyzer(args.results_dir)
    analyzer.load_results()

    if analyzer.df is None or len(analyzer.df) == 0:
        print("没有找到实验结果")
        return

    # 比较模型
    comparisons = analyzer.compare_models(args.metric)

    print(f"\n{args.metric.upper()} 指标模型比较:")
    print("-" * 80)

    for comp in comparisons:
        print(f"{comp.model_a} vs {comp.model_b}:")
        print(f"  平均差异: {comp.mean_diff:.4f}")
        print(f"  P值: {comp.p_value:.4f}")
        print(f"  效应大小: {comp.effect_size:.3f}")
        print(f"  置信区间: [{comp.confidence_interval[0]:.4f}, {comp.confidence_interval[1]:.4f}]")
        print(f"  结论: {comp.interpretation}")
        print()


def rank_command(args):
    """排名命令"""
    print(f"正在为模型排名...")

    analyzer = ExperimentAnalyzer(args.results_dir)
    analyzer.load_results()

    if analyzer.df is None or len(analyzer.df) == 0:
        print("没有找到实验结果")
        return

    # 模型排名
    ranking = analyzer.rank_models(args.metric, args.higher_is_better)

    if ranking.empty:
        print(f"没有找到 {args.metric} 指标的数据")
        return

    print(f"\n{args.metric.upper()} 指标模型排名:")
    print("-" * 60)
    print(f"{'排名':<4} {'模型':<20} {'平均值':<10} {'标准差':<10} {'实验次数':<8}")
    print("-" * 60)

    for _, row in ranking.iterrows():
        print(f"{int(row['rank']):<4} {row['model_name']:<20} "
              f"{row['mean_score']:<10.4f} {row['std_score']:<10.4f} {int(row['num_experiments']):<8}")


def export_command(args):
    """导出命令"""
    print(f"正在导出结果...")

    auto_analyzer = AutoAnalyzer(args.results_dir, auto_save=False, auto_analyze=False)

    metrics = args.metrics.split(",") if args.metrics else ["accuracy", "macro_f1"]

    try:
        output_path = auto_analyzer.export_results_for_paper(metrics, args.format)
        print(f"结果已导出到: {output_path}")
    except Exception as e:
        print(f"导出失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MELD实验分析工具")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--results-dir", "-r", default="./experiments/results",
                       help="实验结果目录")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 分析命令
    analyze_parser = subparsers.add_parser("analyze", help="分析实验结果")
    analyze_parser.add_argument("--pattern", "-p", help="结果文件匹配模式")
    analyze_parser.add_argument("--metrics", "-m", help="要分析的指标，逗号分隔")
    analyze_parser.add_argument("--report", action="store_true", help="生成报告")
    analyze_parser.add_argument("--output", "-o", default="analysis_report.md", help="报告输出文件")
    analyze_parser.add_argument("--viz", action="store_true", help="创建可视化")
    analyze_parser.add_argument("--export", choices=["wandb"], help="导出到外部平台")

    # 比较命令
    compare_parser = subparsers.add_parser("compare", help="比较模型")
    compare_parser.add_argument("metric", help="比较的指标名称")

    # 排名命令
    rank_parser = subparsers.add_parser("rank", help="模型排名")
    rank_parser.add_argument("metric", help="排名的指标名称")
    rank_parser.add_argument("--higher-is-better", action="store_true", default=True,
                           help="指标是否越高越好")

    # 导出命令
    export_parser = subparsers.add_parser("export", help="导出结果")
    export_parser.add_argument("--format", "-f", choices=["latex", "csv", "json"],
                             default="latex", help="导出格式")
    export_parser.add_argument("--metrics", "-m", help="要导出的指标，逗号分隔")

    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "analyze":
            analyze_command(args)
        elif args.command == "compare":
            compare_command(args)
        elif args.command == "rank":
            rank_command(args)
        elif args.command == "export":
            export_command(args)
        else:
            print(f"未知命令: {args.command}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()