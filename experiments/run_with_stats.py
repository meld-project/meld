#!/usr/bin/env python3
"""
Enhanced experiment runner with automatic statistical analysis.

This script extends the existing experiment framework to automatically
perform statistical significance testing on experimental results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import existing functionality
try:
    from .run_enhanced import EnhancedExperimentRunner
    from .statistical_analyzer import ExperimentAnalyzer, analyze_experiment_directory
    ENHANCED_RUNNER_AVAILABLE = True
except ImportError:
    ENHANCED_RUNNER_AVAILABLE = False
    logging.warning("Enhanced runner not available")

try:
    from .run_suite import main as run_suite_main
    RUN_SUITE_AVAILABLE = True
except ImportError:
    RUN_SUITE_AVAILABLE = False
    logging.warning("Run suite not available")

LOGGER = logging.getLogger(__name__)


class StatisticalExperimentRunner:
    """Experiment runner with integrated statistical analysis."""

    def __init__(self, output_dir: str = "./experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if ENHANCED_RUNNER_AVAILABLE:
            self.runner = EnhancedExperimentRunner()
        else:
            self.runner = None

        self.analyzer = ExperimentAnalyzer()

    def run_experiment_with_stats(
        self,
        config_path: str,
        experiment_name: Optional[str] = None,
        auto_analyze: bool = True
    ) -> Dict[str, Any]:
        """
        Run an experiment and automatically perform statistical analysis.

        Args:
            config_path: Path to experiment configuration
            experiment_name: Name for the experiment (optional)
            auto_analyze: Whether to automatically analyze results

        Returns:
            Dictionary with experiment results and statistical analysis
        """
        if not self.runner:
            raise RuntimeError("Enhanced runner not available")

        # Load configuration
        config = self.runner.load_experiment(config_path)

        if experiment_name:
            config.global_config.experiment_name = experiment_name

        LOGGER.info(f"Running experiment: {config.global_config.experiment_name}")

        # Run the experiment
        result = self.runner.run_experiment(config)

        # Analyze results if requested
        analysis = None
        if auto_analyze and result.get("status") == "success":
            try:
                result_path = result.get("output")
                if result_path and Path(result_path).exists():
                    self.analyzer.extract_from_json_result(
                        result_path, config.global_config.experiment_name
                    )

                    # Generate quick analysis
                    if len(self.analyzer.experiments) > 0:
                        analysis = self._generate_quick_analysis()

            except Exception as e:
                LOGGER.warning(f"Failed to analyze results: {e}")

        return {
            "experiment_result": result,
            "statistical_analysis": analysis,
            "experiment_name": config.global_config.experiment_name
        }

    def _generate_quick_analysis(self) -> Dict[str, Any]:
        """Generate quick statistical analysis for the most recent experiment."""
        if not self.analyzer.experiments:
            return {}

        # Get the most recent experiment
        latest_exp = list(self.analyzer.experiments.values())[-1]

        return {
            "experiment_name": latest_exp.experiment_name,
            "metric": latest_exp.metric_name,
            "mean_performance": latest_exp.mean,
            "std_performance": latest_exp.std,
            "confidence_interval": latest_exp.confidence_interval,
            "sample_size": latest_exp.sample_size,
            "interpretation": self._interpret_performance(latest_exp)
        }

    def _interpret_performance(self, summary) -> str:
        """Interpret the performance of an experiment."""
        if summary.metric_name in ["macro_f1", "f1", "accuracy"]:
            if summary.mean >= 0.9:
                return "Excellent performance"
            elif summary.mean >= 0.8:
                return "Good performance"
            elif summary.mean >= 0.7:
                return "Moderate performance"
            else:
                return "Poor performance"
        elif summary.metric_name in ["auc", "auroc"]:
            if summary.mean >= 0.95:
                return "Excellent discrimination"
            elif summary.mean >= 0.85:
                return "Good discrimination"
            elif summary.mean >= 0.75:
                return "Moderate discrimination"
            else:
                return "Poor discrimination"
        else:
            return "Performance interpretation not available for this metric"

    def run_suite_with_stats(
        self,
        suite_config_path: str,
        auto_analyze: bool = True
    ) -> Dict[str, Any]:
        """
        Run a suite of experiments with statistical analysis.

        Args:
            suite_config_path: Path to suite configuration
            auto_analyze: Whether to automatically analyze results

        Returns:
            Dictionary with suite results and statistical analysis
        """
        if not RUN_SUITE_AVAILABLE:
            raise RuntimeError("Run suite not available")

        # Create temporary output directory for this suite
        suite_output = self.output_dir / f"suite_{int(time.time())}"
        suite_output.mkdir(exist_ok=True)

        # Run the suite
        original_args = sys.argv
        try:
            sys.argv = [
                "run_suite.py",
                "--config", suite_config_path,
                "--out", str(suite_output / "suite_summary.json")
            ]

            # Capture the suite results
            suite_result = run_suite_main()

        finally:
            sys.argv = original_args

        # Analyze all results
        analysis = None
        if auto_analyze:
            try:
                analyzer = analyze_experiment_directory(
                    str(suite_output),
                    pattern="*.json",
                    output_dir=str(suite_output / "statistical_analysis")
                )

                analysis = {
                    "total_experiments": len(analyzer.experiments),
                    "comparisons_made": len(analyzer.comparisons),
                    "report_path": str(suite_output / "statistical_analysis" / "statistical_analysis_report.txt")
                }

            except Exception as e:
                LOGGER.warning(f"Failed to analyze suite results: {e}")

        return {
            "suite_output_dir": str(suite_output),
            "statistical_analysis": analysis
        }


def create_statistical_config_template() -> Dict[str, Any]:
    """Create a configuration template for experiments with statistical analysis."""
    return {
        "global_config": {
            "experiment_name": "statistical_analysis_demo",
            "use_wandb": True,
            "wandb_project": "meld-statistical-analysis",
            "output_dir": "./experiments/statistical_results",
            "parallel_jobs": 1,
            "fail_fast": False
        },
        "model_config": {
            "model_type": "lec",
            "model_dir": "/data/models/Qwen3-0.6B",
            "max_tokens": 512,
            "stride": 256,
            "clf": "logreg",
            "target_fpr": 0.01,
            "lambda_penalty": 0.1
        },
        "data_config": {
            "data_type": "markdown",
            "md_dir": "/home/user/rko/dataset/meld-data",
            "split_mode": "cv",
            "n_splits": 5,
            "limit": 1000
        },
        "seed": 42,
        "seeds": [42, 43, 44],  # Multiple seeds for statistical analysis
        "progress": True,
        "save_raw": True
    }


def main() -> None:
    """Main entry point for statistical experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run MELD experiments with statistical significance testing"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run single experiment
    run_parser = subparsers.add_parser("run", help="Run single experiment with stats")
    run_parser.add_argument("--config", required=True, help="Experiment configuration file")
    run_parser.add_argument("--name", help="Experiment name")
    run_parser.add_argument("--output", default="./experiments/statistical_results", help="Output directory")
    run_parser.add_argument("--no-analyze", action="store_true", help="Skip statistical analysis")

    # Run suite
    suite_parser = subparsers.add_parser("suite", help="Run experiment suite with stats")
    suite_parser.add_argument("--config", required=True, help="Suite configuration file")
    suite_parser.add_argument("--output", default="./experiments/statistical_results", help="Output directory")
    suite_parser.add_argument("--no-analyze", action="store_true", help="Skip statistical analysis")

    # Analyze existing results
    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing experimental results")
    analyze_parser.add_argument("--results-dir", required=True, help="Directory containing result files")
    analyze_parser.add_argument("--pattern", default="*.json", help="File pattern to match")
    analyze_parser.add_argument("--output", help="Output directory for analysis")

    # Create template
    template_parser = subparsers.add_parser("template", help="Create statistical analysis template")
    template_parser.add_argument("--output", default="./statistical_template.json", help="Output file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    try:
        if args.command == "run":
            runner = StatisticalExperimentRunner(args.output)

            result = runner.run_experiment_with_stats(
                args.config,
                args.name,
                auto_analyze=not args.no_analyze
            )

            print(f"\nExperiment completed: {result['experiment_name']}")
            if result["statistical_analysis"]:
                analysis = result["statistical_analysis"]
                print(f"Performance: {analysis['mean_performance']:.3f} ± {analysis['std_performance']:.3f}")
                print(f"Interpretation: {analysis['interpretation']}")

        elif args.command == "suite":
            runner = StatisticalExperimentRunner(args.output)

            result = runner.run_suite_with_stats(
                args.config,
                auto_analyze=not args.no_analyze
            )

            print(f"\nSuite completed: {result['suite_output_dir']}")
            if result["statistical_analysis"]:
                analysis = result["statistical_analysis"]
                print(f"Analyzed {analysis['total_experiments']} experiments")
                print(f"Made {analysis['comparisons_made']} comparisons")
                print(f"Report saved to: {analysis['report_path']}")

        elif args.command == "analyze":
            analyzer = analyze_experiment_directory(
                args.results_dir,
                args.pattern,
                args.output
            )

            print(f"\nAnalysis completed:")
            print(f"Experiments analyzed: {len(analyzer.experiments)}")
            print(f"Comparisons made: {len(analyzer.comparisons)}")

            if args.output:
                print(f"Results saved to: {args.output}")

        elif args.command == "template":
            template = create_statistical_config_template()

            with open(args.output, 'w') as f:
                json.dump(template, f, indent=2)

            print(f"Statistical analysis template created: {args.output}")

    except Exception as e:
        LOGGER.exception(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import time
    main()