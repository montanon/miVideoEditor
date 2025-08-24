"""In-memory caching system for improved storage performance."""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Configuration for the caching system."""

    max_memory_items: int = Field(
        default=1000, gt=0, description="Maximum items in memory cache"
    )
    max_memory_bytes: int = Field(
        default=512 * 1024**2, gt=0, description="Maximum memory usage in bytes"
    )
    default_ttl_seconds: int = Field(
        default=3600, gt=0, description="Default TTL for cache entries"
    )
    enable_disk_cache: bool = Field(
        default=False, description="Enable disk-based caching"
    )
    disk_cache_dir: Path | None = Field(
        default=None, description="Directory for disk cache"
    )

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }


class CacheEntry:
    """Individual cache entry with metadata."""

    def __init__(self, value: Any, ttl_seconds: int = 3600):
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds
        self.last_accessed = self.created_at
        self.access_count = 0
        self.size_bytes = self._estimate_size(value)

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of cached object."""
        try:
            # Use pickle to estimate serialized size
            return len(pickle.dumps(obj))
        except (pickle.PicklingError, TypeError, ValueError):
            # Fallback for unpicklable objects
            return 1024  # Assume 1KB


class CacheManager:
    """In-memory LRU cache with optional disk persistence."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: dict[str, CacheEntry] = {}
        self._lock = Lock()
        self._current_memory_bytes = 0

        # Setup disk cache if enabled
        if config.enable_disk_cache and config.disk_cache_dir:
            self.disk_cache_dir = Path(config.disk_cache_dir)
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.disk_cache_dir = None

    def get(self, key: str) -> Any | None:
        """Retrieve value from cache."""
        with self._lock:
            # Check memory cache first
            if key in self._cache:
                entry = self._cache[key]

                # Check if expired
                if entry.is_expired():
                    self._remove_entry(key)
                    return None

                # Update access info
                entry.touch()
                logger.debug("Cache hit (memory): %s", key)
                return entry.value

            # Check disk cache if enabled
            if self.disk_cache_dir:
                disk_value = self._get_from_disk(key)
                if disk_value is not None:
                    # Promote to memory cache
                    self.set(key, disk_value, promote_only=True)
                    logger.debug("Cache hit (disk): %s", key)
                    return disk_value

            logger.debug("Cache miss: %s", key)
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
        *,
        promote_only: bool = False,
    ) -> bool:
        """Store value in cache."""
        if ttl_seconds is None:
            ttl_seconds = self.config.default_ttl_seconds

        entry = CacheEntry(value, ttl_seconds)

        with self._lock:
            # Check if we need to make room
            while (
                len(self._cache) >= self.config.max_memory_items
                or self._current_memory_bytes + entry.size_bytes
                > self.config.max_memory_bytes
            ):
                if not self._evict_lru():
                    # Unable to evict, can't store this entry
                    logger.warning("Cache full, unable to store: %s", key)
                    return False

            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Add new entry
            self._cache[key] = entry
            self._current_memory_bytes += entry.size_bytes

            # Store to disk cache if enabled and not just promoting
            if self.disk_cache_dir and not promote_only:
                self._set_to_disk(key, value, ttl_seconds)

            logger.debug(
                "Cache set: %s (size: %s bytes, TTL: %s seconds)",
                key,
                entry.size_bytes,
                ttl_seconds,
            )
            return True

    def delete(self, key: str) -> bool:
        """Remove specific cache entry."""
        with self._lock:
            # Remove from memory
            if key in self._cache:
                self._remove_entry(key)
                removed = True
            else:
                removed = False

            # Remove from disk
            if self.disk_cache_dir:
                disk_path = self.disk_cache_dir / f"{self._safe_filename(key)}.cache"
                if disk_path.exists():
                    disk_path.unlink()
                    removed = True

            if removed:
                logger.debug("Cache entry deleted: %s", key)

            return removed

    def clear(self) -> int:
        """Clear all cache entries."""
        with self._lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            self._current_memory_bytes = 0

            # Clear disk cache
            if self.disk_cache_dir:
                for cache_file in self.disk_cache_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except Exception:
                        logger.exception(
                            "Failed to delete disk cache file %s", cache_file
                        )

            logger.info("Cache cleared: %s entries", cleared_count)
            return cleared_count

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.get(key) is not None

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern (supports * and ? wildcards)."""
        # Convert shell-style pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        regex = re.compile(regex_pattern)

        invalidated_count = 0

        with self._lock:
            # Find matching keys
            keys_to_remove = [key for key in self._cache if regex.match(key)]

            # Remove matching entries
            for key in keys_to_remove:
                self._remove_entry(key)
                invalidated_count += 1

            # Remove from disk cache
            if self.disk_cache_dir:
                for cache_file in self.disk_cache_dir.glob("*.cache"):
                    cache_key = self._filename_to_key(cache_file.stem)
                    if regex.match(cache_key):
                        cache_file.unlink()
                        invalidated_count += 1

        if invalidated_count > 0:
            logger.debug(
                "Pattern invalidation: %s -> %s entries", pattern, invalidated_count
            )

        return invalidated_count

    def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        expired_keys = []

        with self._lock:
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_entry(key)

        if expired_keys:
            logger.debug("Cleaned up %s expired entries", len(expired_keys))

        return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = sum(entry.access_count for entry in self._cache.values())

            stats = {
                "memory_entries": len(self._cache),
                "memory_bytes": self._current_memory_bytes,
                "memory_mb": round(self._current_memory_bytes / (1024 * 1024), 2),
                "max_memory_items": self.config.max_memory_items,
                "max_memory_mb": round(self.config.max_memory_bytes / (1024 * 1024), 2),
                "memory_utilization_percent": round(
                    (self._current_memory_bytes / self.config.max_memory_bytes) * 100, 1
                ),
                "total_accesses": total_accesses,
            }

            # Add disk cache stats if enabled
            if self.disk_cache_dir:
                disk_files = list(self.disk_cache_dir.glob("*.cache"))
                disk_size = sum(f.stat().st_size for f in disk_files)
                stats.update(
                    {
                        "disk_entries": len(disk_files),
                        "disk_bytes": disk_size,
                        "disk_mb": round(disk_size / (1024 * 1024), 2),
                    }
                )

            return stats

    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self._cache:
            return False

        # Find LRU entry
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        self._remove_entry(lru_key)

        logger.debug("LRU eviction: %s", lru_key)
        return True

    def _remove_entry(self, key: str) -> None:
        """Remove entry from memory cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory_bytes -= entry.size_bytes

    def _get_from_disk(self, key: str) -> Any | None:
        """Retrieve value from disk cache."""
        if not self.disk_cache_dir:
            return None

        disk_path = self.disk_cache_dir / f"{self._safe_filename(key)}.cache"

        try:
            if disk_path.exists():
                with disk_path.open("rb") as f:
                    cache_data = pickle.load(f)

                # Check if expired
                if time.time() > cache_data["expires_at"]:
                    disk_path.unlink()
                    return None

                return cache_data["value"]
        except Exception:
            logger.exception("Error reading disk cache %s", disk_path)
            # Remove corrupted cache file
            if disk_path.exists():
                disk_path.unlink()

        return None

    def _set_to_disk(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store value to disk cache."""
        if not self.disk_cache_dir:
            return

        disk_path = self.disk_cache_dir / f"{self._safe_filename(key)}.cache"

        try:
            cache_data = {
                "value": value,
                "expires_at": time.time() + ttl_seconds,
                "created_at": time.time(),
            }

            with disk_path.open("wb") as f:
                pickle.dump(cache_data, f)

        except Exception:
            logger.exception("Error writing disk cache %s", disk_path)

    def _safe_filename(self, key: str) -> str:
        """Convert cache key to safe filename."""
        # Replace unsafe characters with underscores
        safe = re.sub(r'[<>:"/\\|?*]', "_", key)
        # Limit length
        if len(safe) > 200:
            # Use hash for very long keys

            safe = hashlib.md5(key.encode()).hexdigest()
        return safe

    def _filename_to_key(self, filename: str) -> str:
        """Convert safe filename back to cache key (best effort)."""
        # This is a simplified reverse mapping - in practice you'd need
        # to store the original key in the cache file metadata
        return filename.replace("_", ":")  # Simple example
