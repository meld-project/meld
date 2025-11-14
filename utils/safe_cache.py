"""
安全的JSON缓存系统，替代pickle缓存
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List, Union

LOGGER = logging.getLogger(__name__)


class SafeJSONCache:
    """安全的JSON缓存类，替代不安全的pickle缓存"""

    def __init__(self, cache_dir: str = "./cache"):
        """
        初始化缓存

        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Initialized cache directory: {self.cache_dir}")

    def _get_cache_path(self, key: str) -> Path:
        """
        获取缓存文件路径

        Args:
            key: 缓存键

        Returns:
            缓存文件路径
        """
        # 使用MD5哈希确保文件名安全
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def _is_json_serializable(self, obj: Any) -> bool:
        """
        检查对象是否可以JSON序列化

        Args:
            obj: 要检查的对象

        Returns:
            是否可序列化
        """
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    def _make_serializable(self, obj: Any) -> Any:
        """
        尝试将对象转换为可序列化的格式

        Args:
            obj: 要转换的对象

        Returns:
            可序列化的对象
        """
        if self._is_json_serializable(obj):
            return obj

        # 处理numpy数组
        if hasattr(obj, 'tolist'):
            return {
                '_type': 'numpy_array',
                'data': obj.tolist(),
                'shape': obj.shape if hasattr(obj, 'shape') else None,
                'dtype': str(obj.dtype) if hasattr(obj, 'dtype') else None
            }

        # 处理pandas DataFrame
        if hasattr(obj, 'to_dict'):
            try:
                return {
                    '_type': 'dataframe',
                    'data': obj.to_dict('records')
                }
            except Exception:
                pass

        # 处理其他不可序列化对象
        if hasattr(obj, '__dict__'):
            return {
                '_type': 'object',
                'class': obj.__class__.__name__,
                'data': obj.__dict__
            }

        # 最后尝试字符串转换
        try:
            return {
                '_type': 'string',
                'data': str(obj)
            }
        except Exception:
            return {
                '_type': 'unserializable',
                'data': None
            }

    def _deserialize_object(self, obj: Any) -> Any:
        """
        反序列化特殊对象格式

        Args:
            obj: 序列化的对象

        Returns:
            原始对象
        """
        if isinstance(obj, dict) and '_type' in obj:
            obj_type = obj['_type']
            data = obj['data']

            if obj_type == 'numpy_array':
                import numpy as np
                return np.array(data, dtype=obj.get('dtype'))

            elif obj_type == 'dataframe':
                import pandas as pd
                return pd.DataFrame(data)

            # 对于其他类型，返回数据部分
            elif obj_type in ['object', 'string']:
                return data

            elif obj_type == 'unserializable':
                return None

        return obj

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存的值，如果不存在则返回None
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                return self._deserialize_object(loaded_data)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                LOGGER.warning(f"Cache JSON decode failed for {key}: {e}")
                # 删除损坏的缓存文件
                cache_path.unlink(missing_ok=True)
            except Exception as e:
                LOGGER.warning(f"Cache load failed for {key}: {e}")
                cache_path.unlink(missing_ok=True)
        return None

    def set(self, key: str, value: Any) -> None:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 要缓存的值
        """
        cache_path = self._get_cache_path(key)
        try:
            # 转换为可序列化格式
            serializable_value = self._make_serializable(value)

            # 写入JSON文件
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_value, f, indent=2, ensure_ascii=False)

        except Exception as e:
            LOGGER.warning(f"Cache save failed for {key}: {e}")

    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            缓存是否存在
        """
        cache_path = self._get_cache_path(key)
        return cache_path.exists()

    def delete(self, key: str) -> bool:
        """
        删除特定缓存

        Args:
            key: 缓存键

        Returns:
            是否成功删除
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear(self, pattern: Optional[str] = None) -> int:
        """
        清除缓存文件

        Args:
            pattern: 可选的模式匹配

        Returns:
            删除的文件数量
        """
        if pattern is None:
            # 清除所有缓存
            cache_files = list(self.cache_dir.glob("*.json"))
        else:
            # 使用简单的模式匹配，检查文件内容中的键
            cache_files = []
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        # 简单检查文件内容是否包含模式
                        content = f.read()
                        if pattern in content:
                            cache_files.append(cache_file)
                except Exception:
                    # 如果无法读取，跳过该文件
                    continue

        deleted_count = 0
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                deleted_count += 1
            except Exception as e:
                LOGGER.warning(f"Failed to delete cache file {cache_file}: {e}")

        LOGGER.info(f"Cleared {deleted_count} cache files")
        return deleted_count

    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息

        Returns:
            缓存统计信息
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "num_files": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_file": min(cache_files, key=lambda f: f.stat().st_mtime, default=None),
            "newest_file": max(cache_files, key=lambda f: f.stat().st_mtime, default=None)
        }

    def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """
        清理过期缓存

        Args:
            max_age_hours: 最大缓存时间（小时）

        Returns:
            删除的文件数量
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cache_files = list(self.cache_dir.glob("*.json"))

        deleted_count = 0
        for cache_file in cache_files:
            file_age = current_time - cache_file.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    cache_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    LOGGER.warning(f"Failed to delete expired cache {cache_file}: {e}")

        LOGGER.info(f"Cleaned up {deleted_count} expired cache files")
        return deleted_count


# 创建全局缓存实例
_global_cache = None


def get_cache(cache_dir: str = "./cache") -> SafeJSONCache:
    """
    获取全局缓存实例

    Args:
        cache_dir: 缓存目录

    Returns:
        缓存实例
    """
    global _global_cache
    if _global_cache is None or _global_cache.cache_dir != Path(cache_dir):
        _global_cache = SafeJSONCache(cache_dir)
    return _global_cache