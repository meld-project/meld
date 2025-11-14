"""统一的可选依赖处理工具

提供一致的接口来处理可选依赖，避免重复的 try-except 代码。
"""

from __future__ import annotations

import importlib
import logging
import warnings
from typing import Any, Dict, Optional, TypeVar, Union, Callable

LOGGER = logging.getLogger(__name__)

T = TypeVar('T')


class OptionalImportError(Exception):
    """可选依赖导入失败时抛出的异常"""
    pass


class OptionalDependency:
    """可选依赖管理器"""

    def __init__(self, module_name: str, install_hint: Optional[str] = None,
                 warn_on_import: bool = True, required_version: Optional[str] = None):
        """
        初始化可选依赖管理器

        Args:
            module_name: 模块名称
            install_hint: 安装提示信息
            warn_on_import: 是否在导入失败时发出警告
            required_version: 要求的最低版本
        """
        self.module_name = module_name
        self.install_hint = install_hint or f"pip install {module_name}"
        self.warn_on_import = warn_on_import
        self.required_version = required_version
        self._module: Optional[Any] = None
        self._import_attempted = False

    def is_available(self) -> bool:
        """检查依赖是否可用"""
        if not self._import_attempted:
            try:
                self._module = importlib.import_module(self.module_name)
                self._import_attempted = True

                # 检查版本要求
                if self.required_version and hasattr(self._module, '__version__'):
                    from packaging import version
                    if version.parse(self._module.__version__) < version.parse(self.required_version):
                        if self.warn_on_import:
                            warnings.warn(
                                f"{self.module_name} version {self._module.__version__} is below "
                                f"required version {self.required_version}. "
                                f"Please upgrade: pip install --upgrade {self.module_name}>={self.required_version}"
                            )
                        self._module = None

            except ImportError:
                self._module = None
                self._import_attempted = True

                if self.warn_on_import:
                    LOGGER.debug(f"Optional dependency {self.module_name} not available. "
                               f"Install with: {self.install_hint}")

        return self._module is not None

    def import_module(self) -> Any:
        """导入模块，如果不可用则抛出异常"""
        if not self.is_available():
            raise OptionalImportError(
                f"Optional dependency '{self.module_name}' is not available. "
                f"Install with: {self.install_hint}"
            )
        return self._module

    def import_attribute(self, attribute_name: str) -> Any:
        """导入模块的特定属性"""
        module = self.import_module()
        try:
            return getattr(module, attribute_name)
        except AttributeError as e:
            raise OptionalImportError(
                f"Attribute '{attribute_name}' not found in module '{self.module_name}'"
            ) from e

    def safe_import(self, default: Optional[T] = None) -> Union[Any, Optional[T]]:
        """安全导入，失败时返回默认值"""
        try:
            return self.import_module()
        except OptionalImportError:
            return default


# 预定义的常用可选依赖
WANDB = OptionalDependency(
    "wandb",
    install_hint="pip install wandb",
    warn_on_import=True
)

YAML = OptionalDependency(
    "yaml",
    install_hint="pip install PyYAML",
    warn_on_import=False
)

TQDM = OptionalDependency(
    "tqdm",
    install_hint="pip install tqdm",
    warn_on_import=False
)

MATPLOTLIB = OptionalDependency(
    "matplotlib",
    install_hint="pip install matplotlib",
    warn_on_import=False
)

SEABORN = OptionalDependency(
    "seaborn",
    install_hint="pip install seaborn",
    warn_on_import=False
)

PLOTLY = OptionalDependency(
    "plotly",
    install_hint="pip install plotly",
    warn_on_import=False
)

POLARS = OptionalDependency(
    "polars",
    install_hint="pip install polars",
    warn_on_import=False
)

OPTUNA = OptionalDependency(
    "optuna",
    install_hint="pip install optuna",
    warn_on_import=False
)

RAY = OptionalDependency(
    "ray",
    install_hint="pip install ray[tune]",
    warn_on_import=False
)

GENSIM = OptionalDependency(
    "gensim",
    install_hint="pip install gensim",
    warn_on_import=False
)

ORJSON = OptionalDependency(
    "orjson",
    install_hint="pip install orjson",
    warn_on_import=False
)

MLFLOW = OptionalDependency(
    "mlflow",
    install_hint="pip install mlflow",
    warn_on_import=False
)

FASTAPI = OptionalDependency(
    "fastapi",
    install_hint="pip install fastapi",
    warn_on_import=False
)

REDIS = OptionalDependency(
    "redis",
    install_hint="pip install redis",
    warn_on_import=False
)

ELASTICSEARCH = OptionalDependency(
    "elasticsearch",
    install_hint="pip install elasticsearch",
    warn_on_import=False
)

DVC = OptionalDependency(
    "dvc",
    install_hint="pip install dvc",
    warn_on_import=False
)


def check_optional_dependencies() -> Dict[str, bool]:
    """检查所有常用可选依赖的可用性"""
    dependencies = {
        'wandb': WANDB,
        'yaml': YAML,
        'tqdm': TQDM,
        'matplotlib': MATPLOTLIB,
        'seaborn': SEABORN,
        'plotly': PLOTLY,
        'polars': POLARS,
        'optuna': OPTUNA,
        'ray': RAY,
        'gensim': GENSIM,
        'orjson': ORJSON,
        'mlflow': MLFLOW,
        'fastapi': FASTAPI,
        'redis': REDIS,
        'elasticsearch': ELASTICSEARCH,
        'dvc': DVC,
    }

    return {name: dep.is_available() for name, dep in dependencies.items()}


def require_dependencies(dependencies: list[str]) -> None:
    """要求指定的依赖必须可用，否则抛出异常"""
    available = check_optional_dependencies()
    missing = [dep for dep in dependencies if not available.get(dep, False)]

    if missing:
        raise OptionalImportError(
            f"Required optional dependencies are missing: {', '.join(missing)}. "
            "Please install them using pip install -r requirements-optional.txt"
        )


# 装饰器：要求特定依赖
def requires_dependency(*dependencies: str):
    """装饰器：要求指定的可选依赖"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            require_dependencies(list(dependencies))
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 便捷函数
def get_wandb():
    """获取 wandb 模块"""
    return WANDB.import_module()


def get_yaml():
    """获取 yaml 模块"""
    return YAML.import_module()


def get_tqdm():
    """获取 tqdm 模块"""
    return TQDM.import_module()


def get_matplotlib():
    """获取 matplotlib 模块"""
    return MATPLOTLIB.import_module()


def get_seaborn():
    """获取 seaborn 模块"""
    return SEABORN.import_module()


def get_polars():
    """获取 polars 模块"""
    return POLARS.import_module()


def get_orjson():
    """获取 orjson 模块"""
    return ORJSON.import_module()


def get_optuna():
    """获取 optuna 模块"""
    return OPTUNA.import_module()


def get_ray():
    """获取 ray 模块"""
    return RAY.import_module()