"""
Enhanced Configuration Management System for MELD

This module provides a centralized, version-controlled configuration management
system with validation, templating, and experiment tracking capabilities.
"""

from __future__ import annotations

import json
import logging
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from datetime import datetime
import yaml

try:
    import jsonschema
    SCHEMA_AVAILABLE = True
except ImportError:
    SCHEMA_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseConfig')


@dataclass
class ExperimentMetadata:
    """Metadata for experiment tracking."""
    experiment_id: str
    created_at: str
    updated_at: str
    version: str
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    parent_experiment: Optional[str] = None
    git_commit: Optional[str] = None
    environment_hash: Optional[str] = None


@dataclass
class ResourceLimits:
    """Resource limits for experiments."""
    max_memory_gb: Optional[float] = None
    max_gpu_memory_gb: Optional[float] = None
    max_runtime_hours: Optional[float] = None
    cpu_cores: Optional[int] = None
    gpu_count: Optional[int] = None


@dataclass
class RetryConfig:
    """Retry configuration for failed experiments."""
    max_retries: int = 3
    retry_delay_seconds: int = 60
    retry_on_timeout: bool = True
    retry_on_memory_error: bool = True


@dataclass
class NotificationConfig:
    """Notification configuration."""
    email_on_completion: bool = False
    email_on_failure: bool = True
    webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None


class BaseConfig:
    """Base configuration class with common functionality."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._schema_cache = {}

    def validate(self) -> None:
        """Validate configuration values."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create config from dictionary."""
        return cls(**data)

    def get_hash(self) -> str:
        """Get unique hash of configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]


@dataclass
class GlobalConfig(BaseConfig):
    """Global experiment configuration."""

    # Experiment metadata
    experiment_name: str = ""
    experiment_id: Optional[str] = None
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    # Environment
    environment: str = "development"  # development, staging, production
    log_level: str = "INFO"
    output_dir: str = "./experiments/results"
    cache_dir: str = "./cache"
    temp_dir: str = "./temp"

    # Execution
    parallel_jobs: int = 1
    fail_fast: bool = False
    dry_run: bool = False
    resume: bool = False

    # Resources
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    retry_config: RetryConfig = field(default_factory=RetryConfig)

    # Notifications
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    # Tracking
    use_wandb: bool = False
    wandb_project: str = "meld-experiments"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None

    def validate(self) -> None:
        """Validate global configuration."""
        if self.experiment_name == "":
            raise ValueError("experiment_name cannot be empty")

        if self.environment not in ["development", "staging", "production"]:
            raise ValueError(f"Invalid environment: {self.environment}")

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")

        if self.parallel_jobs < 1:
            raise ValueError("parallel_jobs must be >= 1")


@dataclass
class ModelConfig(BaseConfig):
    """Model-specific configuration."""

    # Model type and path
    model_type: str = "lec"  # lec, neurlux, quovadis, dmds, text_baseline, embedding_baseline
    model_dir: Optional[str] = None
    model_name: str = ""

    # Model parameters
    max_tokens: int = 1024
    stride: int = 256
    until_layer: Optional[int] = None
    batch_size: int = 64

    # LEC specific
    clf: str = "logreg"  # logreg, ridge
    target_fpr: float = 0.01
    lambda_penalty: float = 0.1
    bootstrap_samples: int = 0

    # Text baseline specific
    encoder: str = "tfidf_word"  # tfidf_word, tfidf_char, bert

    # Nebula specific
    baseline: str = "neurlux"  # neurlux, quovadis, dmds
    vocab_size: int = 10000
    max_length: int = 512
    epochs: int = 10
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-2

    def validate(self) -> None:
        """Validate model configuration."""
        valid_model_types = ["lec", "neurlux", "quovadis", "dmds", "text_baseline", "embedding_baseline"]
        if self.model_type not in valid_model_types:
            raise ValueError(f"Invalid model_type: {self.model_type}")

        if self.model_type in ["lec", "neurlux", "quovadis", "embedding_baseline"] and not self.model_dir:
            raise ValueError(f"{self.model_type} requires model_dir")

        if self.clf not in ["logreg", "ridge"]:
            raise ValueError(f"Invalid classifier: {self.clf}")

        if not (0.0 < self.target_fpr < 1.0):
            raise ValueError("target_fpr must be between 0 and 1")


@dataclass
class DataConfig(BaseConfig):
    """Data configuration."""

    # Data paths
    data_type: str = "markdown"  # markdown, cape_json
    md_dir: Optional[str] = None
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None
    mal_dir: Optional[str] = None
    benign_dir: Optional[str] = None

    # Split configuration
    split_mode: str = "cv"  # cv, holdout, dir, time_ood
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    n_splits: int = 5
    
    # Time-OOD specific configuration
    train_end_date: Optional[str] = None  # Format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
    val_end_date: Optional[str] = None     # Format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
    malicious_manifest_path: Optional[str] = None  # Path to malicious manifest CSV
    benign_manifest_path: Optional[str] = None    # Path to benign manifest CSV

    # Data limits
    limit: Optional[int] = None
    limit_per_class: Optional[int] = None

    # Data processing
    cache_features: bool = True
    cache_preprocessed: bool = True

    def validate(self) -> None:
        """Validate data configuration."""
        valid_data_types = ["markdown", "cape_json"]
        if self.data_type not in valid_data_types:
            raise ValueError(f"Invalid data_type: {self.data_type}")

        valid_split_modes = ["cv", "holdout", "dir", "time_ood"]
        if self.split_mode not in valid_split_modes:
            raise ValueError(f"Invalid split_mode: {self.split_mode}")
        
        if self.split_mode == "time_ood":
            if not self.malicious_manifest_path:
                raise ValueError("malicious_manifest_path required for time_ood split mode")
            if not self.benign_manifest_path:
                raise ValueError("benign_manifest_path required for time_ood split mode")
            # If dates not provided, use default optimal dates
            if not self.train_end_date:
                self.train_end_date = "2025-04-23 08:08:46"
            if not self.val_end_date:
                self.val_end_date = "2025-06-14 11:39:39"

        if self.split_mode == "dir":
            required_dirs = ["train_dir", "val_dir", "test_dir"]
            for dir_name in required_dirs:
                if not getattr(self, dir_name):
                    raise ValueError(f"{dir_name} required for dir split mode")
        elif self.data_type == "markdown" and not self.md_dir:
            raise ValueError("md_dir required for markdown data")
        elif self.data_type == "cape_json":
            if not self.mal_dir or not self.benign_dir:
                raise ValueError("mal_dir and benign_dir required for CAPE JSON data")

        if self.val_ratio + self.test_ratio >= 0.95:
            raise ValueError("val_ratio + test_ratio too large")

        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Core components
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)

    # Execution
    seed: int = 42
    seeds: Optional[List[int]] = None
    gpu: int = 0

    # Output
    out: Optional[str] = None
    save_raw: bool = False
    progress: bool = False
    run_tag: str = ""

    # Metadata
    metadata: Optional[ExperimentMetadata] = None

    def validate(self) -> None:
        """Validate complete configuration."""
        self.global_config.validate()
        self.model_config.validate()
        self.data_config.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with nested structure."""
        return {
            "global_config": self.global_config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "data_config": self.data_config.to_dict(),
            "seed": self.seed,
            "seeds": self.seeds,
            "gpu": self.gpu,
            "out": self.out,
            "save_raw": self.save_raw,
            "progress": self.progress,
            "run_tag": self.run_tag,
            "metadata": asdict(self.metadata) if self.metadata else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary with nested structure."""
        return cls(
            global_config=GlobalConfig.from_dict(data.get("global_config", {})),
            model_config=ModelConfig.from_dict(data.get("model_config", {})),
            data_config=DataConfig.from_dict(data.get("data_config", {})),
            seed=data.get("seed", 42),
            seeds=data.get("seeds"),
            gpu=data.get("gpu", 0),
            out=data.get("out"),
            save_raw=data.get("save_raw", False),
            progress=data.get("progress", False),
            run_tag=data.get("run_tag", ""),
            metadata=ExperimentMetadata(**data["metadata"]) if data.get("metadata") else None,
        )

    def get_hash(self) -> str:
        """Get unique hash of configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]


class ConfigManager:
    """Enhanced configuration manager with versioning and validation."""

    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.config_dir / "registry.json"
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load configuration registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"experiments": {}, "templates": {}, "versions": {}}

    def _save_registry(self) -> None:
        """Save configuration registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(self._registry, f, indent=2)

    def create_config(
        self,
        name: str,
        global_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        data_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ExperimentConfig:
        """Create new experiment configuration."""
        config = ExperimentConfig(
            global_config=GlobalConfig.from_dict(global_config or {}),
            model_config=ModelConfig.from_dict(model_config or {}),
            data_config=DataConfig.from_dict(data_config or {}),
            **kwargs
        )

        # Set experiment ID and metadata
        config.global_config.experiment_name = name
        config.global_config.experiment_id = f"{name}_{int(time.time())}"

        config.metadata = ExperimentMetadata(
            experiment_id=config.global_config.experiment_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            version="1.0.0",
            tags=config.global_config.tags,
            environment_hash=self._get_environment_hash()
        )

        config.validate()
        return config

    def save_config(self, config: ExperimentConfig, filename: Optional[str] = None) -> str:
        """Save configuration to file."""
        if filename is None:
            filename = f"{config.global_config.experiment_name}_{config.get_hash()}.json"

        config_path = self.config_dir / filename

        # Update metadata
        if config.metadata:
            config.metadata.updated_at = datetime.now().isoformat()

        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Update registry
        self._registry["experiments"][config.global_config.experiment_id] = {
            "name": config.global_config.experiment_name,
            "filename": filename,
            "hash": config.get_hash(),
            "created_at": config.metadata.created_at if config.metadata else datetime.now().isoformat(),
            "updated_at": config.metadata.updated_at if config.metadata else datetime.now().isoformat(),
        }

        self._save_registry()
        LOGGER.info(f"Configuration saved to {config_path}")
        return str(config_path)

    def load_config(self, path: str) -> ExperimentConfig:
        """Load configuration from file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(config_path, 'r') as f:
            data = json.load(f)

        config = ExperimentConfig.from_dict(data)
        config.validate()
        return config

    def list_configs(self, filter_tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all configurations."""
        configs = []
        for exp_id, exp_data in self._registry["experiments"].items():
            if filter_tag and filter_tag not in exp_data.get("tags", []):
                continue
            configs.append({"id": exp_id, **exp_data})
        return sorted(configs, key=lambda x: x["updated_at"], reverse=True)

    def create_template(self, name: str, config: ExperimentConfig) -> None:
        """Save configuration as template."""
        template_data = config.to_dict()
        template_path = self.config_dir / "templates" / f"{name}.json"
        template_path.parent.mkdir(parents=True, exist_ok=True)

        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2)

        self._registry["templates"][name] = {
            "filename": f"templates/{name}.json",
            "created_at": datetime.now().isoformat(),
        }
        self._save_registry()

    def load_template(self, name: str) -> ExperimentConfig:
        """Load configuration template."""
        if name not in self._registry["templates"]:
            raise ValueError(f"Template not found: {name}")

        template_path = self.config_dir / self._registry["templates"][name]["filename"]
        return self.load_config(template_path)

    def _get_environment_hash(self) -> str:
        """Get hash of current environment."""
        env_vars = ["PYTHONPATH", "CUDA_VISIBLE_DEVICES", "WANDB_API_KEY"]
        env_str = "|".join([f"{k}={os.environ.get(k, '')}" for k in env_vars])
        return hashlib.md5(env_str.encode()).hexdigest()[:8]

    def validate_config_schema(self, config: ExperimentConfig) -> bool:
        """Validate configuration against schema (if available)."""
        if not SCHEMA_AVAILABLE:
            LOGGER.warning("jsonschema not available, skipping schema validation")
            return True

        # TODO: Add JSON schema validation
        return True