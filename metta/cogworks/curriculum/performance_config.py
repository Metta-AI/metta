"""Performance optimization configuration for Learning Progress curriculum.

This module provides configuration options to test different performance optimizations
for the LP curriculum algorithm, helping identify the best combination of settings.
"""

from __future__ import annotations

from pydantic import Field

from mettagrid.base_config import Config


class LPPerformanceConfig(Config):
    """Configuration for LP curriculum performance optimizations.

    Use this to test different optimization strategies and find the best combination.
    """

    # Solution 1: Batch cache invalidation
    invalidation_batch_size: int = Field(
        default=100,
        ge=1,
        description="Invalidate LP cache every N task updates (1=every update, 100=every 100 updates)",
    )

    # Solution 2: Task list caching
    cache_task_list: bool = Field(
        default=True,
        description="Cache the result of get_all_tracked_tasks() to avoid repeated scans",
    )

    task_list_cache_ttl: float = Field(
        default=1.0,
        ge=0.0,
        description="Time-to-live for task list cache in seconds (0=no TTL, always use cache)",
    )

    # Solution 3: Reduced task pool size
    use_reduced_pool: bool = Field(
        default=False,
        description="Use a smaller task pool for faster operations",
    )

    reduced_pool_size: int = Field(
        default=100,
        ge=10,
        description="Size of reduced task pool (only used if use_reduced_pool=True)",
    )

    # Solution 4: Score computation optimization
    cache_numpy_arrays: bool = Field(
        default=True,
        description="Cache numpy arrays to avoid rebuilding on each score computation",
    )

    # Monitoring
    log_performance_metrics: bool = Field(
        default=False,
        description="Log detailed performance metrics (call counts, timings, cache hit rates)",
    )

    log_interval_seconds: float = Field(
        default=10.0,
        ge=1.0,
        description="How often to log performance metrics (in seconds)",
    )

    @classmethod
    def baseline(cls) -> "LPPerformanceConfig":
        """Original behavior (no optimizations) - for baseline comparison."""
        return cls(
            invalidation_batch_size=1,  # Invalidate every update (original behavior)
            cache_task_list=False,
            cache_numpy_arrays=False,
            log_performance_metrics=True,
        )

    @classmethod
    def batch_10(cls) -> "LPPerformanceConfig":
        """Light batching - invalidate every 10 updates."""
        return cls(
            invalidation_batch_size=10,
            cache_task_list=True,
            cache_numpy_arrays=True,
            log_performance_metrics=True,
        )

    @classmethod
    def batch_100(cls) -> "LPPerformanceConfig":
        """Medium batching - invalidate every 100 updates (recommended)."""
        return cls(
            invalidation_batch_size=100,
            cache_task_list=True,
            cache_numpy_arrays=True,
            log_performance_metrics=True,
        )

    @classmethod
    def batch_1000(cls) -> "LPPerformanceConfig":
        """Heavy batching - invalidate every 1000 updates."""
        return cls(
            invalidation_batch_size=1000,
            cache_task_list=True,
            cache_numpy_arrays=True,
            log_performance_metrics=True,
        )

    @classmethod
    def small_pool(cls) -> "LPPerformanceConfig":
        """Small task pool (100 tasks) with medium batching."""
        return cls(
            invalidation_batch_size=100,
            cache_task_list=True,
            cache_numpy_arrays=True,
            use_reduced_pool=True,
            reduced_pool_size=100,
            log_performance_metrics=True,
        )

    @classmethod
    def aggressive(cls) -> "LPPerformanceConfig":
        """All optimizations enabled aggressively."""
        return cls(
            invalidation_batch_size=1000,
            cache_task_list=True,
            task_list_cache_ttl=5.0,  # 5 second cache
            cache_numpy_arrays=True,
            log_performance_metrics=True,
        )
