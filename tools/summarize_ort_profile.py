#!/usr/bin/env python3
"""Summarize ONNX Runtime profiling JSON files.

The ORT trace format is a Chrome trace JSON list. Kernel execution events usually
carry timing in microseconds in the `dur` field and node metadata in `args`.
This script groups those events by op type and by node name so allocator/no-
allocator runs can be compared quickly on the board.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Stats:
    count: int = 0
    total_us: float = 0.0
    max_us: float = 0.0

    def add(self, dur_us: float) -> None:
        self.count += 1
        self.total_us += dur_us
        self.max_us = max(self.max_us, dur_us)


def event_op(event: dict[str, Any]) -> str:
    args = event.get("args") or {}
    for key in ("op_name", "op_name", "op", "provider"):
        value = args.get(key)
        if value:
            return str(value)

    name = str(event.get("name", "<unknown>"))
    if "_kernel_time" in name:
        return name.split("_kernel_time", 1)[0]
    return name


def event_node(event: dict[str, Any]) -> str:
    args = event.get("args") or {}
    for key in ("node_name", "name"):
        value = args.get(key)
        if value:
            return str(value)
    return str(event.get("name", "<unknown>"))


def is_timed_kernel_event(event: dict[str, Any]) -> bool:
    if "dur" not in event:
        return False
    name = str(event.get("name", ""))
    category = str(event.get("cat", ""))
    if "kernel_time" in name:
        return True
    if category.lower() in {"node", "kernel"}:
        return True
    args = event.get("args") or {}
    return "op_name" in args or "node_name" in args


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "traceEvents" in data:
        data = data["traceEvents"]
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a recognized ORT trace JSON file")
    return [event for event in data if isinstance(event, dict)]


def summarize(path: Path, top: int) -> None:
    by_op: dict[str, Stats] = defaultdict(Stats)
    by_node: dict[str, Stats] = defaultdict(Stats)
    total_us = 0.0
    event_count = 0

    for event in load_events(path):
        if not is_timed_kernel_event(event):
            continue
        dur_us = float(event.get("dur", 0.0))
        if dur_us <= 0:
            continue
        event_count += 1
        total_us += dur_us
        by_op[event_op(event)].add(dur_us)
        by_node[event_node(event)].add(dur_us)

    print(f"Profile: {path}")
    print(f"Timed events: {event_count}")
    print(f"Total timed kernel/node duration: {total_us / 1000.0:.3f} ms")
    print()

    print(f"Top {top} op types by total duration:")
    print(f"{'op':48s} {'count':>8s} {'total_ms':>12s} {'avg_us':>12s} {'max_us':>12s}")
    for op, stats in sorted(by_op.items(), key=lambda item: item[1].total_us, reverse=True)[:top]:
        avg_us = stats.total_us / stats.count if stats.count else 0.0
        print(
            f"{op[:48]:48s} {stats.count:8d} {stats.total_us / 1000.0:12.3f} "
            f"{avg_us:12.3f} {stats.max_us:12.3f}"
        )
    print()

    print(f"Top {top} nodes by total duration:")
    print(f"{'node':72s} {'count':>8s} {'total_ms':>12s} {'avg_us':>12s} {'max_us':>12s}")
    for node, stats in sorted(by_node.items(), key=lambda item: item[1].total_us, reverse=True)[:top]:
        avg_us = stats.total_us / stats.count if stats.count else 0.0
        print(
            f"{node[:72]:72s} {stats.count:8d} {stats.total_us / 1000.0:12.3f} "
            f"{avg_us:12.3f} {stats.max_us:12.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    for index, profile in enumerate(args.profiles):
        if index:
            print("\n" + "=" * 100 + "\n")
        summarize(profile, args.top)


if __name__ == "__main__":
    main()
