from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator


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

    def summary(self) -> str:
        if not self._sections:
            return "Profiler: no sections recorded"
        lines = ["Profiler summary (seconds):"]
        for name, stats in sorted(self._sections.items(), key=lambda item: item[1].total_time, reverse=True):
            lines.append(
                f"  {name:<20s} total={stats.total_time:6.3f} avg={stats.avg_time:6.4f} calls={stats.calls:4d}"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        self._sections.clear()


PROFILER = Profiler()
