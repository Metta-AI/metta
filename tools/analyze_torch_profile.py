import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def load_events(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    if not isinstance(events, list):
        raise ValueError(f"Unexpected traceEvents format in {path}")
    return events


def summarize_events(
    events: Iterable[dict],
    *,
    filter_substr: str | None,
) -> tuple[dict[str, int], dict[str, int], int]:
    durations: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    total_us = 0
    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        if filter_substr and filter_substr not in name:
            continue
        dur = int(ev.get("dur", 0))
        durations[name] += dur
        counts[name] += 1
        total_us += dur
    return durations, counts, total_us


def format_ms(value_us: int) -> str:
    return f"{value_us / 1000:.2f} ms"


def print_summary(
    path: Path,
    durations: dict[str, int],
    counts: dict[str, int],
    total_us: int,
    *,
    top_n: int,
    names: list[str],
) -> None:
    print(f"\n{path.name}: total {format_ms(total_us)} from {sum(counts.values())} events")
    if names:
        for name in names:
            dur = durations.get(name, 0)
            cnt = counts.get(name, 0)
            print(f"  {name:30s} {format_ms(dur)} ({cnt} events)")
        return

    top_items = Counter(durations).most_common(top_n)
    for name, dur in top_items:
        print(f"  {name:30s} {format_ms(dur)} ({counts[name]} events)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize torch profiler Chrome traces (.json.gz).")
    parser.add_argument("paths", nargs="+", type=Path, help="Profiler trace files (.json.gz)")
    parser.add_argument("--top", type=int, default=20, help="Show top N events by total duration")
    parser.add_argument(
        "--filter",
        dest="filter_substr",
        default=None,
        help="Only include events whose name contains this substring",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=[],
        help="Specific event names to report (skips top-N if provided)",
    )
    args = parser.parse_args()

    for path in args.paths:
        events = load_events(path)
        durations, counts, total_us = summarize_events(events, filter_substr=args.filter_substr)
        print_summary(
            path,
            durations,
            counts,
            total_us,
            top_n=args.top,
            names=args.names,
        )


if __name__ == "__main__":
    main()
