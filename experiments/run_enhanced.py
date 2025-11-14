#!/usr/bin/env python3
"""Enhanced experiment runner with improved configuration management.

This script provides an enhanced version of run_suite.py with:
- Version-controlled configuration management
- Template system for experiment configurations
- Automatic configuration validation
- Experiment tracking and metadata
- Resource management and retry logic
- Integration with existing experiment runners
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ConfigManager, ExperimentConfig, GlobalConfig,
    ModelConfig, DataConfig, ConfigAdapter
)
from src.config.adapter import LegacyConfigConverter

# Import existing runners
try:
    from .run_suite import (
        RUNNERS, load_config_file, resolve_params, extract_highlights,
        plan_output_path, execute_run
    )
    RUN_SUITE_AVAILABLE = True
except ImportError:
    RUN_SUITE_AVAILABLE = False
    RUNNERS = {}
    LOGGER = logging.getLogger(__name__)

LOGGER = logging.getLogger("run_enhanced")


class EnhancedExperimentRunner:
    """Enhanced experiment runner with configuration management."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.experiment_registry: Dict[str, Dict[str, Any]] = {}

    def create_experiment_from_template(
        self,
        template_name: str,
        experiment_name: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> ExperimentConfig:
        """Create experiment from template with optional overrides."""
        try:
            config = self.config_manager.load_template(template_name)

            # Apply overrides
            if overrides:
                self._apply_overrides(config, overrides)

            # Update experiment name and ID
            config.global_config.experiment_name = experiment_name
            config.global_config.experiment_id = f"{experiment_name}_{int(time.time())}"

            # Update metadata
            if config.metadata:
                config.metadata.experiment_id = config.global_config.experiment_id
                config.metadata.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

            config.validate()
            return config

        except Exception as e:
            LOGGER.error(f"Failed to create experiment from template {template_name}: {e}")
            raise

    def create_experiment(
        self,
        name: str,
        model_type: str,
        data_type: str = "markdown",
        **kwargs
    ) -> ExperimentConfig:
        """Create new experiment with minimal parameters."""

        # Set defaults based on model type
        model_defaults = self._get_model_defaults(model_type)
        data_defaults = self._get_data_defaults(data_type)

        # Merge with provided kwargs
        global_config = kwargs.get("global_config", {})
        model_config = {**model_defaults, **kwargs.get("model_config", {})}
        data_config = {**data_defaults, **kwargs.get("data_config", {})}

        config = self.config_manager.create_config(
            name=name,
            global_config=global_config,
            model_config=model_config,
            data_config=data_config,
            **{k: v for k, v in kwargs.items() if k not in ["global_config", "model_config", "data_config"]}
        )

        return config

    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment."""
        start_time = time.time()

        try:
            # Register experiment
            exp_id = config.global_config.experiment_id
            self.experiment_registry[exp_id] = {
                "config": config,
                "status": "running",
                "start_time": start_time,
                "metadata": asdict(config.metadata) if config.metadata else {}
            }

            LOGGER.info(f"Starting experiment: {config.global_config.experiment_name} ({exp_id})")

            # Convert to legacy format and run
            adapter = ConfigAdapter(config)

            if not RUN_SUITE_AVAILABLE:
                raise RuntimeError("Legacy run_suite module not available")

            # Create run suite config
            suite_config = adapter.to_run_suite_config()

            # Execute the run
            if suite_config["runs"]:
                run_spec = suite_config["runs"][0]  # Take first run
                record = execute_run(
                    run_spec,
                    base_dir=Path.cwd(),
                    default_output_dir=Path(config.global_config.output_dir),
                    dry_run=config.global_config.dry_run
                )

                # Update registry
                self.experiment_registry[exp_id].update({
                    "status": record.get("status", "unknown"),
                    "result": record,
                    "end_time": time.time(),
                    "duration": time.time() - start_time
                })

                return record
            else:
                raise ValueError("No runs to execute")

        except Exception as e:
            LOGGER.exception(f"Experiment {config.global_config.experiment_name} failed: {e}")

            # Update registry with failure
            if exp_id in self.experiment_registry:
                self.experiment_registry[exp_id].update({
                    "status": "failed",
                    "error": str(e),
                    "end_time": time.time(),
                    "duration": time.time() - start_time
                })

            raise

    def run_multiple_experiments(
        self,
        configs: List[ExperimentConfig],
        parallel_jobs: int = 1
    ) -> List[Dict[str, Any]]:
        """Run multiple experiments."""
        results = []

        if parallel_jobs == 1:
            # Sequential execution
            for config in configs:
                try:
                    result = self.run_experiment(config)
                    results.append(result)
                except Exception as e:
                    LOGGER.error(f"Failed to run experiment {config.global_config.experiment_name}: {e}")
                    results.append({
                        "status": "failed",
                        "error": str(e),
                        "experiment_name": config.global_config.experiment_name
                    })
        else:
            # Parallel execution (placeholder for future implementation)
            LOGGER.warning(f"Parallel execution with {parallel_jobs} jobs not yet implemented, running sequentially")
            return self.run_multiple_experiments(configs, parallel_jobs=1)

        return results

    def save_experiment(self, config: ExperimentConfig, filename: Optional[str] = None) -> str:
        """Save experiment configuration."""
        return self.config_manager.save_config(config, filename)

    def load_experiment(self, path: str) -> ExperimentConfig:
        """Load experiment configuration."""
        return self.config_manager.load_config(path)

    def list_experiments(self, filter_tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments."""
        return self.config_manager.list_configs(filter_tag)

    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment status."""
        return self.experiment_registry.get(experiment_id)

    def _apply_overrides(self, config: ExperimentConfig, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides."""
        for key, value in overrides.items():
            if hasattr(config.global_config, key):
                setattr(config.global_config, key, value)
            elif hasattr(config.model_config, key):
                setattr(config.model_config, key, value)
            elif hasattr(config.data_config, key):
                setattr(config.data_config, key, value)
            elif hasattr(config, key):
                setattr(config, key, value)

    def _get_model_defaults(self, model_type: str) -> Dict[str, Any]:
        """Get default configuration for model type."""
        defaults = {
            "model_type": model_type,
            "max_tokens": 1024,
            "stride": 256,
            "batch_size": 64,
            "target_fpr": 0.01,
            "clf": "logreg",
        }

        if model_type == "lec":
            defaults.update({
                "lambda_penalty": 0.1,
                "bootstrap_samples": 0,
            })
        elif model_type in ["neurlux", "quovadis", "dmds"]:
            defaults.update({
                "epochs": 10,
                "learning_rate": 2.5e-4,
                "weight_decay": 1e-2,
                "vocab_size": 10000,
                "max_length": 512,
            })
        elif model_type == "text_baseline":
            defaults.update({
                "encoder": "tfidf_word",
            })
        elif model_type == "embedding_baseline":
            defaults.update({
                "encoder": "bert",
            })

        return defaults

    def _get_data_defaults(self, data_type: str) -> Dict[str, Any]:
        """Get default configuration for data type."""
        defaults = {
            "data_type": data_type,
            "split_mode": "cv",
            "val_ratio": 0.1,
            "test_ratio": 0.2,
            "n_splits": 5,
            "cache_features": True,
            "cache_preprocessed": True,
        }

        if data_type == "markdown":
            defaults["split_mode"] = "cv"
            defaults["n_splits"] = 10
        elif data_type == "cape_json":
            defaults["split_mode"] = "holdout"
            defaults["val_ratio"] = 0.1
            defaults["test_ratio"] = 0.2

        return defaults


def setup_default_templates(config_manager: ConfigManager) -> None:
    """Setup default experiment templates."""

    # LEC template
    lec_config = ExperimentConfig(
        global_config=GlobalConfig(
            experiment_name="lec_template",
            use_wandb=True,
            wandb_project="lec-experiments"
        ),
        model_config=ModelConfig(
            model_type="lec",
            max_tokens=1024,
            stride=256,
            clf="logreg",
            target_fpr=0.01,
            lambda_penalty=0.1,
            bootstrap_samples=0
        ),
        data_config=DataConfig(
            data_type="markdown",
            split_mode="cv",
            n_splits=10,
            cache_features=True
        ),
        seed=42,
        progress=True,
    )
    config_manager.create_template("lec", lec_config)

    # Nebula template
    nebula_config = ExperimentConfig(
        global_config=GlobalConfig(
            experiment_name="nebula_template",
            use_wandb=True,
            wandb_project="nebula-baselines"
        ),
        model_config=ModelConfig(
            model_type="neurlux",
            baseline="neurlux",
            epochs=10,
            batch_size=64,
            learning_rate=2.5e-4,
            weight_decay=1e-2,
            vocab_size=10000,
            max_length=512,
            target_fpr=0.01
        ),
        data_config=DataConfig(
            data_type="cape_json",
            split_mode="holdout",
            val_ratio=0.1,
            test_ratio=0.2,
            n_splits=5
        ),
        seed=42,
        progress=True,
    )
    config_manager.create_template("nebula", nebula_config)

    # Text baseline template
    text_config = ExperimentConfig(
        global_config=GlobalConfig(
            experiment_name="text_baseline_template",
            use_wandb=True,
            wandb_project="text-baselines"
        ),
        model_config=ModelConfig(
            model_type="text_baseline",
            encoder="tfidf_word",
            target_fpr=0.01
        ),
        data_config=DataConfig(
            data_type="markdown",
            split_mode="cv",
            n_splits=5
        ),
        seed=42,
        progress=True,
    )
    config_manager.create_template("text_baseline", text_config)

    LOGGER.info("Default templates created")


def migrate_legacy_config(config_path: str, output_dir: str = "./configs") -> None:
    """Migrate legacy configuration to enhanced format."""
    try:
        # Load legacy config
        with open(config_path, 'r') as f:
            legacy_data = json.load(f)

        config_manager = ConfigManager(output_dir)

        # Convert based on format
        if "runs" in legacy_data:
            # Run suite format
            enhanced_config = LegacyConfigConverter.from_run_suite_config(legacy_data)
        else:
            # Single experiment format - try to auto-detect
            if "model_dir" in legacy_data:
                enhanced_config = LegacyConfigConverter.from_legacy_lec_config(legacy_data)
            elif "mal_dir" in legacy_data and "benign_dir" in legacy_data:
                enhanced_config = LegacyConfigConverter.from_legacy_nebula_config(legacy_data)
            else:
                raise ValueError("Could not determine legacy config format")

        # Save enhanced config
        output_path = config_manager.save_config(enhanced_config)
        LOGGER.info(f"Migrated legacy config to: {output_path}")

    except Exception as e:
        LOGGER.error(f"Failed to migrate legacy config: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced MELD experiment runner")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("--config", help="Enhanced config file path")
    run_parser.add_argument("--template", help="Template name to use")
    run_parser.add_argument("--name", required=True, help="Experiment name")
    run_parser.add_argument("--overrides", help="JSON overrides for template")
    run_parser.add_argument("--create-only", action="store_true", help="Create config only, don't run")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create new experiment")
    create_parser.add_argument("--name", required=True, help="Experiment name")
    create_parser.add_argument("--model-type", required=True, choices=["lec", "neurlux", "quovadis", "dmds", "text_baseline", "embedding_baseline"])
    create_parser.add_argument("--data-type", default="markdown", choices=["markdown", "cape_json"])
    create_parser.add_argument("--model-dir", help="Model directory")
    create_parser.add_argument("--data-dir", help="Data directory")
    create_parser.add_argument("--output", help="Output config file")

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--tag", help="Filter by tag")

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate legacy config")
    migrate_parser.add_argument("--config", required=True, help="Legacy config path")
    migrate_parser.add_argument("--output", default="./configs", help="Output directory")

    # Template command
    template_parser = subparsers.add_parser("template", help="Template management")
    template_parser.add_argument("--action", choices=["list", "create", "setup"], required=True)
    template_parser.add_argument("--name", help="Template name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # Initialize config manager
    config_manager = ConfigManager()
    runner = EnhancedExperimentRunner(config_manager)

    try:
        if args.command == "run":
            if args.config:
                config = runner.load_experiment(args.config)
            elif args.template:
                overrides = json.loads(args.overrides) if args.overrides else {}
                config = runner.create_experiment_from_template(args.template, args.name, overrides)
            else:
                raise ValueError("Must provide either --config or --template")

            if args.create_only:
                path = runner.save_experiment(config)
                print(f"Created experiment config: {path}")
            else:
                result = runner.run_experiment(config)
                print(f"Experiment completed: {result.get('status')}")

        elif args.command == "create":
            kwargs = {}
            if args.model_dir:
                kwargs["model_config"] = {"model_dir": args.model_dir}
            if args.data_dir:
                if args.data_type == "markdown":
                    kwargs["data_config"] = {"md_dir": args.data_dir}
                else:
                    kwargs["data_config"] = {"mal_dir": f"{args.data_dir}/mal", "benign_dir": f"{args.data_dir}/benign"}

            config = runner.create_experiment(args.name, args.model_type, args.data_type, **kwargs)
            path = runner.save_experiment(config, args.output)
            print(f"Created experiment config: {path}")

        elif args.command == "list":
            experiments = runner.list_experiments(args.tag)
            for exp in experiments:
                print(f"{exp['id']}: {exp['name']} ({exp['updated_at']})")

        elif args.command == "migrate":
            migrate_legacy_config(args.config, args.output)

        elif args.command == "template":
            if args.action == "list":
                templates = list(config_manager._registry.get("templates", {}).keys())
                print("Available templates:", templates)
            elif args.action == "setup":
                setup_default_templates(config_manager)
                print("Default templates created")
            elif args.action == "create" and args.name:
                # TODO: Implement template creation from existing config
                print(f"Template creation for '{args.name}' not yet implemented")

    except Exception as e:
        LOGGER.exception(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()