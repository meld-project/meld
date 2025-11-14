"""
Enhanced experiment runner with professional ML libraries integration.

This demonstrates how to integrate MLflow, Optuna, and other professional tools
into the MELD experiment framework.
"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Professional ML libraries (optional dependencies)
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    import optuna
    from optuna.integration.mlflow import MLflowCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .statistical_analyzer import ExperimentAnalyzer
from ..utils.statistical_tests import SignificanceTester

LOGGER = logging.getLogger(__name__)


class ProfessionalExperimentRunner:
    """Enhanced experiment runner with professional ML tools."""

    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        enable_optuna: bool = True,
        enable_visualization: bool = True
    ):
        """
        Initialize professional experiment runner.

        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            enable_optuna: Enable hyperparameter optimization
            enable_visualization: Enable advanced visualizations
        """
        self.enable_optuna = enable_optuna and OPTUNA_AVAILABLE
        self.enable_visualization = enable_visualization and PLOTLY_AVAILABLE
        self.enable_mlflow = MLFLOW_AVAILABLE and mlflow_tracking_uri is not None

        if self.enable_mlflow:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            LOGGER.info(f"MLflow tracking enabled: {mlflow_tracking_uri}")

        if self.enable_optuna:
            LOGGER.info("Optuna hyperparameter optimization enabled")

        if self.enable_visualization:
            LOGGER.info("Advanced visualizations enabled")

        self.statistical_analyzer = ExperimentAnalyzer()

    def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
        storage: Optional[str] = None
    ):
        """Create Optuna study for hyperparameter optimization."""
        if not self.enable_optuna:
            raise RuntimeError("Optuna not available")

        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=True
        )

        # Add MLflow callback if both are available
        callbacks = []
        if self.enable_mlflow:
            callbacks.append(MLflowCallback())

        return study, callbacks

    def optimize_hyperparameters(
        self,
        objective_func,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        study_name: str = "meld_optimization"
    ):
        """Run hyperparameter optimization."""
        if not self.enable_optuna:
            raise RuntimeError("Optuna not available")

        study, callbacks = self.create_study(study_name)

        # Run optimization
        study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks
        )

        return study

    def log_experiment_to_mlflow(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ):
        """Log experiment to MLflow."""
        if not self.enable_mlflow:
            LOGGER.warning("MLflow not available, skipping logging")
            return

        with mlflow.start_run(
            experiment_name=experiment_name,
            tags=tags or []
        ) as run:
            # Log parameters
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)

            # Log metrics
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            # Log artifacts
            if artifacts:
                for artifact_path in artifacts:
                    if Path(artifact_path).exists():
                        mlflow.log_artifact(artifact_path)

            LOGGER.info(f"Logged experiment to MLflow: {run.info.run_id}")

    def create_results_dashboard(self, results_dir: str) -> str:
        """Create interactive dashboard for experiment results."""
        if not self.enable_visualization:
            raise RuntimeError("Plotly not available")

        # Load results using polars if available
        if POLARS_AVAILABLE:
            df = pl.read_json(results_dir + "/*.json")
        else:
            # Fallback to pandas
            import pandas as pd
            df = pd.read_json(results_dir + "/*.json")

        # Create dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Performance Distribution", "Metric Correlations",
                          "Experiment Timeline", "Model Comparison"),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # Add plots
        # 1. Performance distribution
        fig.add_trace(
            go.Histogram(x=df["macro_f1"], name="F1 Distribution"),
            row=1, col=1
        )

        # 2. Metric correlations
        fig.add_trace(
            go.Scatter(x=df["auc"], y=df["macro_f1"], mode="markers",
                      name="AUC vs F1"),
            row=1, col=2
        )

        # 3. Timeline
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["macro_f1"],
                      mode="lines+markers", name="Performance Over Time"),
            row=2, col=1
        )

        # 4. Model comparison
        model_performance = df.groupby("model_type")["macro_f1"].mean()
        fig.add_trace(
            go.Bar(x=model_performance.index, y=model_performance.values,
                   name="Average F1 by Model"),
            row=2, col=2
        )

        # Save dashboard
        dashboard_path = Path(results_dir) / "dashboard.html"
        fig.write_html(str(dashboard_path))

        LOGGER.info(f"Interactive dashboard saved: {dashboard_path}")
        return str(dashboard_path)

    def run_professional_experiment_pipeline(
        self,
        experiment_config: Dict[str, Any],
        optimize_hyperparams: bool = False,
        create_dashboard: bool = True
    ) -> Dict[str, Any]:
        """Run complete professional experiment pipeline."""
        results = {}

        # 1. Hyperparameter optimization (if requested)
        if optimize_hyperparams and self.enable_optuna:
            LOGGER.info("Starting hyperparameter optimization...")

            def objective(trial):
                # Example objective function
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
                batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
                max_tokens = trial.suggest_int("max_tokens", 256, 1024)

                # Update config with suggested parameters
                config = experiment_config.copy()
                config.update({
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "max_tokens": max_tokens
                })

                # Run experiment (simplified)
                # In real implementation, this would call your actual training function
                import random
                score = random.uniform(0.7, 0.9)  # Mock result

                # Log to MLflow if available
                if self.enable_mlflow:
                    with mlflow.start_run(nested=True):
                        mlflow.log_params({
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "max_tokens": max_tokens
                        })
                        mlflow.log_metric("score", score)

                return score

            study = self.optimize_hyperparameters(
                objective_func=objective,
                n_trials=50,
                study_name=f"{experiment_config.get('experiment_name', 'experiment')}_optimization"
            )

            results["optimization"] = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials)
            }

            # Update config with best parameters
            experiment_config.update(study.best_params)

        # 2. Run main experiment
        LOGGER.info("Running main experiment...")
        # In real implementation, this would call your actual experiment runner
        import random
        mock_results = {
            "macro_f1": random.uniform(0.8, 0.9),
            "auc": random.uniform(0.85, 0.95),
            "accuracy": random.uniform(0.75, 0.85),
            "training_time": random.uniform(100, 1000)
        }

        results["experiment"] = mock_results

        # 3. Log to MLflow
        if self.enable_mlflow:
            self.log_experiment_to_mlflow(
                experiment_name=experiment_config.get("experiment_name", "professional_experiment"),
                config=experiment_config,
                metrics=mock_results,
                tags=["professional", "optimized"] if optimize_hyperparams else ["professional"]
            )

        # 4. Statistical analysis
        LOGGER.info("Performing statistical analysis...")
        # Add mock data for demonstration
        self.statistical_analyzer.add_experiment_results(
            experiment_name=experiment_config.get("experiment_name", "professional_experiment"),
            experiment_id="prof_001",
            model_type=experiment_config.get("model_type", "lec"),
            metric_name="macro_f1",
            scores=[mock_results["macro_f1"] + random.uniform(-0.02, 0.02) for _ in range(5)]
        )

        # 5. Create dashboard
        if create_dashboard and self.enable_visualization:
            LOGGER.info("Creating interactive dashboard...")
            # Mock dashboard creation
            dashboard_path = Path("./professional_dashboard.html")
            results["dashboard"] = str(dashboard_path)

        return results


# Installation and setup utilities
def check_dependencies():
    """Check which professional libraries are available."""
    status = {
        "mlflow": MLFLOW_AVAILABLE,
        "optuna": OPTUNA_AVAILABLE,
        "polars": POLARS_AVAILABLE,
        "plotly": PLOTLY_AVAILABLE
    }

    print("Professional ML Libraries Status:")
    for lib, available in status.items():
        status_str = "✅ Available" if available else "❌ Not available"
        print(f"  {lib}: {status_str}")

    missing = [lib for lib, available in status.items() if not available]
    if missing:
        print(f"\nMissing libraries: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))

    return status


def setup_professional_environment():
    """Setup professional ML environment."""
    print("Setting up professional MELD environment...")

    # Create directories
    directories = [
        "./mlruns",  # MLflow
        "./optuna_studies",  # Optuna
        "./dashboards",  # Visualizations
        "./data/processed",  # Processed data
        "./models/registered"  # Registered models
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create example configuration
    config = {
        "mlflow": {
            "tracking_uri": "./mlruns",
            "experiment_name": "meld_experiments"
        },
        "optuna": {
            "storage": "sqlite:///optuna_studies/studies.db",
            "n_trials": 100,
            "timeout": 3600
        },
        "dashboard": {
            "output_dir": "./dashboards",
            "auto_refresh": True
        }
    }

    config_path = Path("./professional_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created configuration: {config_path}")
    print("Professional environment setup complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Professional MELD Experiment Runner")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    parser.add_argument("--setup", action="store_true", help="Setup professional environment")
    parser.add_argument("--demo", action="store_true", help="Run demo experiment")

    args = parser.parse_args()

    if args.check_deps:
        check_dependencies()
    elif args.setup:
        setup_professional_environment()
    elif args.demo:
        # Run demo
        runner = ProfessionalExperimentRunner(
            mlflow_tracking_uri="./mlruns",
            enable_optuna=True,
            enable_visualization=True
        )

        demo_config = {
            "experiment_name": "professional_demo",
            "model_type": "lec",
            "model_dir": "/data/models/Qwen3-0.6B"
        }

        results = runner.run_professional_experiment_pipeline(
            demo_config,
            optimize_hyperparams=True,
            create_dashboard=True
        )

        print("Demo experiment completed!")
        print(f"Results: {results}")
    else:
        parser.print_help()