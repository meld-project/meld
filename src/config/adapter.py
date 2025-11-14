"""
Configuration adapter to bridge enhanced config system with existing experiment framework.

This module provides compatibility layers to convert between the new enhanced
configuration system and the legacy configuration classes used in individual
experiment modules.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from pathlib import Path

from . import (
    ExperimentConfig, GlobalConfig, ModelConfig, DataConfig,
    ExperimentMetadata, ResourceLimits, RetryConfig, NotificationConfig
)

LOGGER = logging.getLogger(__name__)


class ConfigAdapter:
    """Adapter to convert enhanced configs to legacy configs."""

    def __init__(self, enhanced_config: ExperimentConfig):
        self.config = enhanced_config

    def to_lec_config(self) -> Dict[str, Any]:
        """Convert to LEC TrainConfig format."""
        model = self.config.model_config
        data = self.config.data_config
        global_conf = self.config.global_config

        return {
            "model_dir": model.model_dir,
            "split_mode": data.split_mode,
            "md_dir": data.md_dir,
            "train_dir": data.train_dir,
            "val_dir": data.val_dir,
            "test_dir": data.test_dir,
            "val_ratio": data.val_ratio,
            "test_ratio": data.test_ratio,
            "seed": self.config.seed,
            "seeds": self.config.seeds,
            "limit": data.limit,
            "max_tokens": model.max_tokens,
            "stride": model.stride,
            "clf": model.clf,
            "n_splits": data.n_splits,
            "until_layer": model.until_layer,
            "out": self.config.out,
            "gpu": self.config.gpu,
            "progress": self.config.progress,
            "save_raw": self.config.save_raw,
            "run_tag": self.config.run_tag,
            "cache_dir": global_conf.cache_dir,
            "target_fpr": model.target_fpr,
            "lambda_penalty": model.lambda_penalty,
            "bootstrap_samples": model.bootstrap_samples,
            "use_wandb": global_conf.use_wandb,
            "wandb_project": global_conf.wandb_project,
            "wandb_entity": global_conf.wandb_entity,
            "wandb_api_key": global_conf.wandb_api_key,
        }

    def to_nebula_config(self) -> Dict[str, Any]:
        """Convert to Nebula baseline config format."""
        model = self.config.model_config
        data = self.config.data_config
        global_conf = self.config.global_config

        return {
            "mal_dir": data.mal_dir,
            "benign_dir": data.benign_dir,
            "baseline": model.baseline,
            "split_mode": data.split_mode,
            "val_ratio": data.val_ratio,
            "test_ratio": data.test_ratio,
            "seed": self.config.seed,
            "seeds": self.config.seeds,
            "limit_per_class": data.limit_per_class,
            "n_splits": data.n_splits,
            "target_fpr": model.target_fpr,
            "vocab_size": model.vocab_size,
            "max_length": model.max_length,
            "quovadis_seq_len": model.max_length,
            "quovadis_top_api": min(model.vocab_size, 600),
            "dmds_seq_len": model.max_length,
            "epochs": model.epochs,
            "batch_size": model.batch_size,
            "learning_rate": model.learning_rate,
            "weight_decay": model.weight_decay,
            "device": f"cuda:{self.config.gpu}" if self.config.gpu >= 0 else "cpu",
            "out": self.config.out,
            "save_raw": self.config.save_raw,
            "progress": self.config.progress,
            "use_wandb": global_conf.use_wandb,
            "wandb_project": global_conf.wandb_project,
            "wandb_entity": global_conf.wandb_entity,
            "wandb_api_key": global_conf.wandb_api_key,
            "run_tag": self.config.run_tag,
        }

    def to_text_baseline_config(self) -> Dict[str, Any]:
        """Convert to text baseline config format."""
        model = self.config.model_config
        data = self.config.data_config
        global_conf = self.config.global_config

        return {
            "md_dir": data.md_dir,
            "encoder": model.encoder,
            "split_mode": data.split_mode,
            "n_splits": data.n_splits,
            "seed": self.config.seed,
            "seeds": self.config.seeds,
            "target_fpr": model.target_fpr,
            "limit": data.limit,
            "progress": self.config.progress,
            "use_wandb": global_conf.use_wandb,
            "wandb_project": global_conf.wandb_project,
            "wandb_entity": global_conf.wandb_entity,
            "wandb_api_key": global_conf.wandb_api_key,
            "out": self.config.out,
            "save_raw": self.config.save_raw,
            "run_tag": self.config.run_tag,
        }

    def to_embedding_baseline_config(self) -> Dict[str, Any]:
        """Convert to embedding baseline config format."""
        model = self.config.model_config
        data = self.config.data_config
        global_conf = self.config.global_config

        return {
            "model_dir": model.model_dir,
            "md_dir": data.md_dir,
            "encoder": "bert",  # Embedding baseline uses BERT
            "split_mode": data.split_mode,
            "n_splits": data.n_splits,
            "seed": self.config.seed,
            "seeds": self.config.seeds,
            "target_fpr": model.target_fpr,
            "limit": data.limit,
            "max_tokens": model.max_tokens,
            "stride": model.stride,
            "progress": self.config.progress,
            "use_wandb": global_conf.use_wandb,
            "wandb_project": global_conf.wandb_project,
            "wandb_entity": global_conf.wandb_entity,
            "wandb_api_key": global_conf.wandb_api_key,
            "out": self.config.out,
            "save_raw": self.config.save_raw,
            "run_tag": self.config.run_tag,
            "gpu": self.config.gpu,
        }

    def to_run_suite_config(self) -> Dict[str, Any]:
        """Convert to run suite configuration format."""
        global_conf = self.config.global_config

        runs = []

        # Determine which experiment to run based on model type
        if self.config.model_config.model_type == "lec":
            runs.append({
                "name": self.config.global_config.experiment_name,
                "runner": "lec",
                "enabled": True,
                "notes": self.config.global_config.notes,
                "params": self.to_lec_config()
            })
        elif self.config.model_config.model_type == "neurlux":
            runs.append({
                "name": self.config.global_config.experiment_name,
                "runner": "nebula_baselines",
                "enabled": True,
                "notes": self.config.global_config.notes,
                "params": self.to_nebula_config()
            })
        elif self.config.model_config.model_type == "text_baseline":
            runs.append({
                "name": self.config.global_config.experiment_name,
                "runner": "text_baseline",
                "enabled": True,
                "notes": self.config.global_config.notes,
                "params": self.to_text_baseline_config()
            })
        elif self.config.model_config.model_type == "embedding_baseline":
            runs.append({
                "name": self.config.global_config.experiment_name,
                "runner": "embedding_baseline",
                "enabled": True,
                "notes": self.config.global_config.notes,
                "params": self.to_embedding_baseline_config()
            })

        return {
            "output_dir": global_conf.output_dir,
            "runs": runs,
            "summary_path": global_conf.output_dir + f"/{self.config.global_config.experiment_name}_summary.json"
        }


class LegacyConfigConverter:
    """Convert legacy configs to enhanced format."""

    @staticmethod
    def from_legacy_lec_config(legacy_config: Dict[str, Any]) -> ExperimentConfig:
        """Convert from LEC TrainConfig to enhanced format."""
        return ExperimentConfig(
            global_config=GlobalConfig(
                experiment_name=legacy_config.get("run_tag", "lec_experiment"),
                output_dir=str(Path(legacy_config.get("out", "./results")).parent),
                cache_dir=legacy_config.get("cache_dir", "./cache"),
                use_wandb=legacy_config.get("use_wandb", False),
                wandb_project=legacy_config.get("wandb_project", "lec-experiments"),
                wandb_entity=legacy_config.get("wandb_entity"),
                wandb_api_key=legacy_config.get("wandb_api_key"),
                log_level="INFO",
            ),
            model_config=ModelConfig(
                model_type="lec",
                model_dir=legacy_config["model_dir"],
                max_tokens=legacy_config.get("max_tokens", 1024),
                stride=legacy_config.get("stride", 256),
                until_layer=legacy_config.get("until_layer"),
                clf=legacy_config.get("clf", "logreg"),
                target_fpr=legacy_config.get("target_fpr", 0.01),
                lambda_penalty=legacy_config.get("lambda_penalty", 0.1),
                bootstrap_samples=legacy_config.get("bootstrap_samples", 0),
            ),
            data_config=DataConfig(
                data_type="markdown",
                md_dir=legacy_config.get("md_dir"),
                train_dir=legacy_config.get("train_dir"),
                val_dir=legacy_config.get("val_dir"),
                test_dir=legacy_config.get("test_dir"),
                split_mode=legacy_config.get("split_mode", "cv"),
                val_ratio=legacy_config.get("val_ratio", 0.1),
                test_ratio=legacy_config.get("test_ratio", 0.2),
                n_splits=legacy_config.get("n_splits", 10),
                limit=legacy_config.get("limit"),
            ),
            seed=legacy_config.get("seed", 42),
            seeds=legacy_config.get("seeds"),
            gpu=legacy_config.get("gpu", 0),
            out=legacy_config.get("out"),
            save_raw=legacy_config.get("save_raw", False),
            progress=legacy_config.get("progress", False),
            run_tag=legacy_config.get("run_tag", ""),
        )

    @staticmethod
    def from_legacy_nebula_config(legacy_config: Dict[str, Any]) -> ExperimentConfig:
        """Convert from Nebula baseline config to enhanced format."""
        return ExperimentConfig(
            global_config=GlobalConfig(
                experiment_name=legacy_config.get("run_tag", "nebula_experiment"),
                output_dir=str(Path(legacy_config.get("out", "./results")).parent),
                use_wandb=legacy_config.get("use_wandb", False),
                wandb_project=legacy_config.get("wandb_project", "nebula-baselines"),
                wandb_entity=legacy_config.get("wandb_entity"),
                wandb_api_key=legacy_config.get("wandb_api_key"),
            ),
            model_config=ModelConfig(
                model_type="neurlux",  # Will be overridden by baseline field
                baseline=legacy_config.get("baseline", "neurlux"),
                epochs=legacy_config.get("epochs", 10),
                batch_size=legacy_config.get("batch_size", 64),
                learning_rate=legacy_config.get("learning_rate", 2.5e-4),
                weight_decay=legacy_config.get("weight_decay", 1e-2),
                vocab_size=legacy_config.get("vocab_size", 10000),
                max_length=legacy_config.get("max_length", 512),
                target_fpr=legacy_config.get("target_fpr", 0.01),
            ),
            data_config=DataConfig(
                data_type="cape_json",
                mal_dir=legacy_config["mal_dir"],
                benign_dir=legacy_config["benign_dir"],
                split_mode=legacy_config.get("split_mode", "holdout"),
                val_ratio=legacy_config.get("val_ratio", 0.1),
                test_ratio=legacy_config.get("test_ratio", 0.2),
                n_splits=legacy_config.get("n_splits", 5),
                limit_per_class=legacy_config.get("limit_per_class"),
            ),
            seed=legacy_config.get("seed", 42),
            seeds=legacy_config.get("seeds"),
            out=legacy_config.get("out"),
            save_raw=legacy_config.get("save_raw", False),
            progress=legacy_config.get("progress", False),
            run_tag=legacy_config.get("run_tag", ""),
        )

    @staticmethod
    def from_run_suite_config(suite_config: Dict[str, Any]) -> ExperimentConfig:
        """Convert from run suite config to enhanced format."""
        runs = suite_config.get("runs", [])
        if not runs:
            raise ValueError("No runs found in suite config")

        # For now, just convert the first run
        # TODO: Handle multiple runs by creating multiple ExperimentConfigs
        first_run = runs[0]
        runner = first_run.get("runner")
        params = first_run.get("params", {})

        if runner == "lec":
            return LegacyConfigConverter.from_legacy_lec_config(params)
        elif runner in ["nebula_baseline", "nebula_baselines"]:
            return LegacyConfigConverter.from_legacy_nebula_config(params)
        elif runner == "text_baseline":
            # Convert text baseline config
            return ExperimentConfig(
                global_config=GlobalConfig(
                    experiment_name=first_run.get("name", "text_baseline_experiment"),
                    use_wandb=params.get("use_wandb", False),
                    wandb_project=params.get("wandb_project", "text-baselines"),
                ),
                model_config=ModelConfig(
                    model_type="text_baseline",
                    encoder=params.get("encoder", "tfidf_word"),
                    target_fpr=params.get("target_fpr", 0.01),
                ),
                data_config=DataConfig(
                    data_type="markdown",
                    md_dir=params.get("md_dir"),
                    split_mode=params.get("split_mode", "cv"),
                    n_splits=params.get("n_splits", 5),
                    limit=params.get("limit"),
                ),
                seed=params.get("seed", 42),
                seeds=params.get("seeds"),
                out=params.get("out"),
                progress=params.get("progress", False),
            )
        else:
            raise ValueError(f"Unsupported runner type: {runner}")