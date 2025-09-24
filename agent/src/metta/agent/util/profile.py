from __future__ import annotations

import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator

import torch


@dataclass
class SectionStats:
    total_time: float = 0.0
    calls: int = 0

    def record(self, elapsed: float) -> None:
        self.total_time += elapsed
        self.calls += 1

    @property
    def avg_time(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.total_time / self.calls


class Profiler:
    """Minimal in-process profiler for instrumenting model sections."""

    def __init__(self) -> None:
        self._sections: Dict[str, SectionStats] = {}

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            stats = self._sections.setdefault(name, SectionStats())
            stats.record(elapsed)

    def summary(self, include_memory: bool = False) -> str:
        if not self._sections:
            return "Profiler: no sections recorded"
        lines = ["Profiler summary (seconds):"]
        for name, stats in sorted(self._sections.items(), key=lambda item: item[1].total_time, reverse=True):
            lines.append(
                f"  {name:<20s} total={stats.total_time:6.3f} avg={stats.avg_time:6.4f} calls={stats.calls:4d}"
            )

        if include_memory:
            rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss_raw:
                import sys

                if sys.platform == "darwin":
                    rss_mb = rss_raw / (1024 * 1024)
                else:
                    rss_mb = rss_raw / 1024
            else:
                rss_mb = 0.0
            lines.append(f"  CPU max RSS            {rss_mb:6.2f} MB")

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                peak = torch.cuda.max_memory_allocated() / 1024**2
                lines.append(
                    f"  GPU memory (MB)        alloc={allocated:6.1f} reserved={reserved:6.1f} peak={peak:6.1f}"
                )
                torch.cuda.reset_peak_memory_stats()

        return "\n".join(lines)

    def reset(self) -> None:
        self._sections.clear()


PROFILER = Profiler()
