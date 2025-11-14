"""
Optuna integration for MELD hyperparameter optimization.

This module provides specialized Optuna optimizers for different MELD models
and experiment types.
"""

from __future__ import annotations

import logging
import json
import time
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path

try:
    import optuna
    from optuna.integration.mlflow import MLflowCallback
    from optuna.visualization import plot_optimization_history, plot_param_importances
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

LOGGER = logging.getLogger(__name__)


class MELDOptunaOptimizer:
    """Specialized Optuna optimizer for MELD experiments."""

    def __init__(
        self,
        study_name: str,
        storage: Optional[str] = None,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None
    ):
        """
        Initialize MELD Optuna optimizer.

        Args:
            study_name: Name of the optimization study
            storage: Database URL for persistence
            direction: Optimization direction ("maximize" or "minimize")
            sampler: Sampling algorithm (default: TPE)
            pruner: Pruning algorithm (default: Median)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        self.study_name = study_name
        self.storage = storage
        self.direction = direction

        # Default sampler and pruner
        if sampler is None:
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                seed=42
            )

        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3,
                interval_steps=1
            )

        self.sampler = sampler
        self.pruner = pruner

        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )

        LOGGER.info(f"Created Optuna study: {study_name}")

    def optimize_lec_hyperparams(
        self,
        objective_func: Callable[[optuna.Trial], float],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        mlflow_callback: bool = True
    ) -> optuna.Study:
        """
        Optimize LEC hyperparameters.

        Args:
            objective_func: Function that takes trial and returns metric
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            mlflow_callback: Whether to use MLflow callback

        Returns:
            Optimized study object
        """
        callbacks = []
        if mlflow_callback:
            try:
                callbacks.append(MLflowCallback(
                    metric_name="objective",
                    mlflow_kwargs={"nested": True}
                ))
                LOGGER.info("MLflow callback enabled")
            except Exception as e:
                LOGGER.warning(f"Failed to create MLflow callback: {e}")

        LOGGER.info(f"Starting LEC optimization: {n_trials} trials")
        start_time = time.time()

        self.study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks
        )

        duration = time.time() - start_time
        LOGGER.info(f"Optimization completed in {duration:.1f}s")

        return self.study

    def suggest_lec_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest LEC-specific hyperparameters.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        # Model parameters
        params["max_tokens"] = trial.suggest_int("max_tokens", 128, 2048, step=128)
        params["stride"] = trial.suggest_int("stride", 32, 512, step=32)
        params["until_layer"] = trial.suggest_int("until_layer", 1, 24)

        # Classifier parameters
        params["clf"] = trial.suggest_categorical("clf", ["logreg", "ridge"])

        if params["clf"] == "logreg":
            params["C"] = trial.suggest_float("C", 0.001, 1000, log=True)
            params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
        else:  # ridge
            params["alpha"] = trial.suggest_float("alpha", 0.001, 100, log=True)

        # Threshold parameters
        params["target_fpr"] = trial.suggest_float("target_fpr", 0.001, 0.1, log=True)
        params["lambda_penalty"] = trial.suggest_float("lambda_penalty", 0.001, 1.0, log=True)

        # Data parameters
        params["n_splits"] = trial.suggest_int("n_splits", 3, 10)
        params["limit"] = trial.suggest_int("limit", 100, 5000, step=100)

        return params

    def suggest_nebula_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest Nebula baseline hyperparameters.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        # Model type
        params["baseline"] = trial.suggest_categorical("baseline", ["neurlux", "quovadis", "dmds"])

        # Common parameters
        params["epochs"] = trial.suggest_int("epochs", 5, 50)
        params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        # Model-specific parameters
        if params["baseline"] == "neurlux":
            params["vocab_size"] = trial.suggest_int("vocab_size", 5000, 50000, step=5000)
            params["max_length"] = trial.suggest_int("max_length", 128, 1024, step=128)

        elif params["baseline"] == "quovadis":
            params["quovadis_seq_len"] = trial.suggest_int("quovadis_seq_len", 50, 500, step=50)
            params["quovadis_top_api"] = trial.suggest_int("quovadis_top_api", 100, 1000, step=100)

        elif params["baseline"] == "dmds":
            params["dmds_seq_len"] = trial.suggest_int("dmds_seq_len", 100, 1000, step=100)

        # Data parameters
        params["limit_per_class"] = trial.suggest_int("limit_per_class", 50, 1000, step=50)

        return params

    def suggest_text_baseline_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest text baseline hyperparameters.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        # Encoder type
        params["encoder"] = trial.suggest_categorical(
            "encoder",
            ["tfidf_word", "tfidf_char", "ngram_word", "ngram_char"]
        )

        # TF-IDF parameters
        if "tfidf" in params["encoder"]:
            params["max_features"] = trial.suggest_int("max_features", 1000, 50000, step=1000)
            params["ngram_range"] = trial.suggest_categorical("ngram_range", [(1,1), (1,2), (2,2)])
            params["min_df"] = trial.suggest_int("min_df", 1, 10)
            params["max_df"] = trial.suggest_float("max_df", 0.5, 1.0)

        # Classifier parameters
        params["clf"] = trial.suggest_categorical("clf", ["logreg", "ridge", "svm"])

        if params["clf"] == "logreg":
            params["C"] = trial.suggest_float("C", 0.001, 1000, log=True)
            params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
        elif params["clf"] == "ridge":
            params["alpha"] = trial.suggest_float("alpha", 0.001, 100, log=True)
        elif params["clf"] == "svm":
            params["C"] = trial.suggest_float("C", 0.001, 1000, log=True)
            params["kernel"] = trial.suggest_categorical("kernel", ["linear", "rbf"])
            if params["kernel"] == "rbf":
                params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)

        # Data parameters
        params["n_splits"] = trial.suggest_int("n_splits", 3, 10)
        params["limit"] = trial.suggest_int("limit", 100, 5000, step=100)

        return params

    def create_multi_objective_study(
        self,
        directions: List[str],
        study_name: Optional[str] = None
    ) -> optuna.Study:
        """
        Create multi-objective optimization study.

        Args:
            directions: List of optimization directions ("maximize" or "minimize")
            study_name: Study name (auto-generated if None)

        Returns:
            Multi-objective study
        """
        if study_name is None:
            study_name = f"{self.study_name}_multi_objective"

        study = optuna.create_study(
            study_name=study_name,
            directions=directions,
            storage=self.storage,
            sampler=self.sampler,
            load_if_exists=True
        )

        return study

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from the study."""
        return self.study.best_params

    def get_best_value(self) -> float:
        """Get best objective value."""
        return self.study.best_value

    def get_trial_results(self) -> List[Dict[str, Any]]:
        """Get all trial results as a list of dictionaries."""
        results = []
        for trial in self.study.trials:
            result = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None
            }
            results.append(result)
        return results

    def save_results(self, output_path: str) -> None:
        """Save optimization results to file."""
        results = {
            "study_name": self.study_name,
            "direction": self.direction,
            "best_params": self.get_best_params(),
            "best_value": self.get_best_value(),
            "n_trials": len(self.study.trials),
            "trials": self.get_trial_results()
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        LOGGER.info(f"Results saved to {output_path}")

    def generate_visualizations(self, output_dir: str) -> Dict[str, str]:
        """
        Generate optimization visualizations.

        Args:
            output_dir: Directory to save visualizations

        Returns:
            Dictionary mapping visualization names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        visualizations = {}

        try:
            # Optimization history
            fig = plot_optimization_history(self.study)
            history_path = output_path / "optimization_history.html"
            fig.write_html(str(history_path))
            visualizations["history"] = str(history_path)

            # Parameter importances
            if len(self.study.trials) > 1:
                fig = plot_param_importances(self.study)
                importance_path = output_path / "param_importances.html"
                fig.write_html(str(importance_path))
                visualizations["importances"] = str(importance_path)

        except Exception as e:
            LOGGER.warning(f"Failed to generate visualizations: {e}")

        return visualizations

    def optimize_with_pruning(
        self,
        objective_func: Callable[[optuna.Trial], float],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        prune_steps: List[int] = [3, 5, 10]
    ) -> optuna.Study:
        """
        Optimize with early pruning based on intermediate results.

        Args:
            objective_func: Function that reports intermediate values
            n_trials: Number of trials
            timeout: Timeout in seconds
            prune_steps: Steps at which to evaluate for pruning

        Returns:
            Optimized study
        """
        def wrapped_objective(trial):
            best_value = float('-inf')

            for step in prune_steps:
                # Get intermediate value from objective function
                try:
                    intermediate_value = objective_func(trial, step=step)

                    # Report intermediate value for pruning
                    trial.report(intermediate_value, step)

                    # Check if should prune
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                    best_value = max(best_value, intermediate_value)

                except optuna.exceptions.TrialPruned:
                    raise

            return best_value

        LOGGER.info(f"Starting optimization with pruning: {n_trials} trials")
        self.study.optimize(wrapped_objective, n_trials=n_trials, timeout=timeout)

        return self.study


# Convenience functions for common optimization tasks
def optimize_lec_model(
    model_dir: str,
    data_dir: str,
    study_name: str = "lec_optimization",
    storage: Optional[str] = None,
    n_trials: int = 100
) -> Dict[str, Any]:
    """
    Convenience function to optimize LEC model hyperparameters.

    Args:
        model_dir: Directory containing the model
        data_dir: Directory containing the data
        study_name: Name of the optimization study
        storage: Database URL for persistence
        n_trials: Number of optimization trials

    Returns:
        Dictionary with optimization results
    """
    optimizer = MELDOptunaOptimizer(study_name, storage)

    def lec_objective(trial):
        # Get suggested parameters
        params = optimizer.suggest_lec_params(trial)

        # Import here to avoid circular imports
        from ..lec.train_lec import run_experiment
        from ..config.adapter import ConfigAdapter
        from ..config import ExperimentConfig, GlobalConfig, ModelConfig, DataConfig

        # Create configuration
        config = ExperimentConfig(
            global_config=GlobalConfig(
                experiment_name=f"LEC_Opt_Trial_{trial.number}",
                output_dir=f"./optuna_results/{study_name}"
            ),
            model_config=ModelConfig(
                model_type="lec",
                model_dir=model_dir,
                max_tokens=params["max_tokens"],
                stride=params["stride"],
                until_layer=params["until_layer"],
                clf=params["clf"],
                target_fpr=params["target_fpr"],
                lambda_penalty=params["lambda_penalty"]
            ),
            data_config=DataConfig(
                data_type="markdown",
                md_dir=data_dir,
                split_mode="cv",
                n_splits=params["n_splits"],
                limit=params["limit"]
            ),
            seed=42
        )

        # Add classifier-specific parameters
        if params["clf"] == "logreg":
            config.model_config.C = params["C"]
            config.model_config.penalty = params["penalty"]
        else:
            config.model_config.alpha = params["alpha"]

        # Run experiment
        try:
            result = run_experiment(config)

            # Extract metric to optimize (e.g., best macro_f1)
            if "best" in result:
                metric_value = result["best"].get("macro_f1", 0.0)
            else:
                metric_value = result.get("macro_f1", 0.0)

            LOGGER.info(f"Trial {trial.number}: {metric_value:.4f}")
            return metric_value

        except Exception as e:
            LOGGER.error(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()

    # Run optimization
    study = optimizer.optimize_lec_hyperparams(lec_objective, n_trials)

    # Generate results
    results = {
        "best_params": optimizer.get_best_params(),
        "best_value": optimizer.get_best_value(),
        "n_trials": len(study.trials),
        "study_name": study_name
    }

    # Save results
    output_path = f"./optuna_results/{study_name}_results.json"
    optimizer.save_results(output_path)

    # Generate visualizations
    viz_dir = f"./optuna_results/{study_name}_visualizations"
    visualizations = optimizer.generate_visualizations(viz_dir)
    results["visualizations"] = visualizations

    return results


if __name__ == "__main__":
    # Example usage
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Install with: pip install optuna")
        exit(1)

    print("MELD Optuna Optimizer Example")
    print("=" * 40)

    # Example 1: Basic LEC optimization
    def dummy_objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2  # Minimize (x-2)^2

    optimizer = MELDOptunaOptimizer("example_study", direction="minimize")
    study = optimizer.optimize_lec_hyperparams(dummy_objective, n_trials=20)

    print(f"Best params: {optimizer.get_best_params()}")
    print(f"Best value: {optimizer.get_best_value()}")

    # Example 2: LEC parameter suggestions
    print("\nExample LEC parameters:")
    import optuna
    trial = optuna.trial.Trial(study, 0)
    lec_params = optimizer.suggest_lec_params(trial)
    for key, value in lec_params.items():
        print(f"  {key}: {value}")