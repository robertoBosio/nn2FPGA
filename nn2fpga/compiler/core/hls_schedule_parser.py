#!/usr/bin/env python3
"""
Vitis HLS report parser (class-based).

Usage (library):
    from hls_report import VitisHlsReportParser

    p = VitisHlsReportParser("/path/to/report.txt")
    # Query by stream name (preserves multiple ops per state/predicate)
    rows = p.query_fifo("StreamingConv_0_out0_stream_1")           # all reads+writes
    reads = p.query_fifo("StreamingConv_0_out0_stream_1", "read")  # only reads
    writes = p.query_fifo("..._stream_1", "write")                 # only writes
    # Grouped by state (helpful when same state has multiple predicates)
    grouped = p.query_fifo_grouped("StreamingConv_0_out0_stream_1")

    # Pipeline/latency
    print(p.pipeline_ii, p.pipeline_depth, p.pipeline_states)
    print(p.get_latency_summary())      # {'latency_cycles_min': ..., 'latency_cycles_max': ...}
    print(p.get_loops())                # loop rows including iteration latency and achieved II

Optional CLI:
    python hls_report.py --file report.txt --stream StreamingConv_0_out0_stream_1
    python hls_report.py --file report.txt --json out.json
"""
from __future__ import annotations
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

class VitisHlsReportParser:
    def __init__(self, source: str) -> None:
        self.text = Path(source).read_text(encoding="utf-8", errors="ignore")
        self._parsed: Dict[str, Any] = {}
        self._parse_all()

    def _parse_all(self) -> None:
        lat_block = self._find_latency_block(self.text)
        self._parsed.update(self._parse_latency_min_max(lat_block or ""))
        self._parsed["loops"] = self._parse_loop_rows(lat_block or "")

        if self._parsed.get("loops") is not None and len(self._parsed["loops"]) == 1:
            # If there is exactly one loop, we assume it's the main pipeline
            self._parsed.update(self._parse_pipeline_summary(self.text))
            fifo_ops = []
            for sid, block in self._iter_state_blocks(self.text):
                fifo_ops.extend(self._parse_fifo_ops_from_state(sid, block))
            self._parsed["fifo_ops"] = fifo_ops
            self._parsed["single_loop_function"] = True

        else:
            # If there are multiple loops, we cannot reliably determine FIFO ops
            self._parsed["fifo_ops"] = []
            self._parsed["pipeline_ii"] = None
            self._parsed["pipeline_depth"] = None
            self._parsed["pipeline_states"] = None
            self._parsed["single_loop_function"] = False

    @staticmethod
    def _parse_pipeline_summary(text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        m = re.search(r"Pipeline-0\s*:\s*II\s*=\s*(\d+)\s*,\s*D\s*=\s*(\d+)\s*,\s*States\s*=\s*\{\s*([^\}]+)\s*\}", text)
        if m:
            out["pipeline_ii"] = int(m.group(1))
            out["pipeline_depth"] = int(m.group(2))
            out["pipeline_states"] = [int(s) for s in re.findall(r"\d+", m.group(3))]
        else:
            out["pipeline_ii"] = None
            out["pipeline_depth"] = None
            out["pipeline_states"] = None
        return out

    @staticmethod
    def _find_latency_block(text: str) -> Optional[str]:
        m = re.search(r"\+\s*Latency:\s*(.*?)(?:^\s*={5,}|\Z)", text, re.S | re.M)
        return m.group(1) if m else None

    @staticmethod
    def _parse_latency_min_max(lat_block: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        if not lat_block:
            return out
        # first numeric data row in the "Summary" table
        for line in lat_block.splitlines():
            s = line.strip()
            if not s.startswith("|"):
                continue
            m = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|", s)
            if m:
                out["latency_cycles_min"] = int(m.group(1))
                out["latency_cycles_max"] = int(m.group(2))
                break
        return out

    @staticmethod
    def _parse_loop_rows(lat_block: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not lat_block:
            return rows
        in_loop = False
        for line in lat_block.splitlines():
            if " * Loop:" in line:
                in_loop = True
            if not in_loop:
                continue
            if line.strip().startswith("|-"):
                parts = [p.strip() for p in line.strip().strip("|").split("|")]
                if len(parts) < 8:
                    continue
                name = parts[0].lstrip("- ").strip()
                try:
                    lat_min = int(parts[1]); lat_max = int(parts[2])
                    iter_lat = int(parts[3]) if parts[3] else None
                    achieved = int(parts[4]) if parts[4] else None
                    target = int(parts[5]) if parts[5] else None
                    trip = int(parts[6]) if parts[6] else None
                except ValueError:
                    continue
                rows.append({
                    "loop_name": name,
                    "latency_cycles_min": lat_min,
                    "latency_cycles_max": lat_max,
                    "iteration_latency": iter_lat,
                    "achieved_ii": achieved,
                    "target_ii": target,
                    "trip_count": trip,
                    "pipelined": parts[7] if len(parts) > 7 else None
                })
        return rows

    @staticmethod
    def _iter_state_blocks(text: str) -> Iterable[tuple[int, str]]:
        # Yields (state_id, block_text)
        for m in re.finditer(r"^\s*State\s+(\d+)\b.*?$([\s\S]*?)(?=^\s*State\s+\d+\b|^\s*={5,}|\Z)", text, re.M):
            yield int(m.group(1)), m.group(2)

    @staticmethod
    def _parse_fifo_ops_from_state(state_id: int, block: str) -> List[Dict[str, Any]]:
        ops: List[Dict[str, Any]] = []
        for line in block.splitlines():
            op_m = re.search(
                r"Operation\s+\d+.*?--->\s+\"%.*?=\s*(read|write)\b(.*?)"
                r"<Predicate\s*=\s*([^>]+)>\s*<Delay\s*=\s*([0-9.]+)>\s*"
                r"<CoreInst\s*=\s*\"([^\"]+)\".*?<Width\s*=\s*(\d+)>\s*<Depth\s*=\s*(\d+)>",
                line)
            if not op_m:
                continue
            op_type = op_m.group(1)
            tail = op_m.group(2)
            predicate = op_m.group(3).strip()
            delay = float(op_m.group(4))
            core = op_m.group(5)
            width = int(op_m.group(6))
            depth = int(op_m.group(7))
            # Port name: for reads it's the last “, iXX %PORT”, for writes it’s “, iXX %PORT, …”
            port = None
            if op_type == "read":
                pm = re.search(r",\s*i\d+\s%([A-Za-z0-9_]+)\b", tail)
                port = pm.group(1) if pm else None
            else:
                pm = re.search(r",\s*i\d+\s%([A-Za-z0-9_]+)\s*,", tail)
                port = pm.group(1) if pm else None
            ops.append({
                "state": state_id,
                "op_type": op_type,          # 'read' or 'write'
                "port": port,                # stream/fifo name
                "predicate": predicate,      # may differ within same state
                "delay_ns": delay,
                "core": core,
                "width_bits": width,
                "fifo_depth": depth,
            })
        return ops

    @property
    def fifo_ops(self) -> List[Dict[str, Any]]:
        return list(self._parsed.get("fifo_ops", []))

    @property
    def pipeline_ii(self) -> Optional[int]:
        return self._parsed.get("pipeline_ii")

    @property
    def pipeline_depth(self) -> Optional[int]:
        return self._parsed.get("pipeline_depth")

    @property
    def pipeline_states(self) -> Optional[list[int]]:
        return self._parsed.get("pipeline_states")

    @property
    def single_loop_function(self) -> bool:
        return self._parsed.get("single_loop_function", False)

    def get_latency_summary(self) -> Dict[str, Any]:
        return {
            "latency_cycles_min": self._parsed.get("latency_cycles_min"),
            "latency_cycles_max": self._parsed.get("latency_cycles_max"),
        }

    def get_loops(self) -> List[Dict[str, Any]]:
        return list(self._parsed.get("loops", []))

    def get_stream_names(self) -> List[str]:
        return sorted({op["port"] for op in self.fifo_ops if op.get("port")})

    def query_fifo(self, name: str, op_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return all occurrences where FIFO/stream `name` is read/written.
        Preserves duplicates (e.g., same state but different predicate).
        Adds 'pipeline_stage' if pipeline state ordering is known.
        """
        rows: List[Dict[str, Any]] = []
        states_order = self.pipeline_states or []
        state_to_stage = {sid: idx for idx, sid in enumerate(states_order)} if states_order else {}
        for op in self.fifo_ops:
            if op.get("port") == name and (op_type is None or op["op_type"] == op_type):
                rec = dict(op)
                rec["pipeline_stage"] = state_to_stage.get(op["state"])
                rows.append(rec)
        # Stable sort for readability: by state, op_type, predicate
        rows.sort(key=lambda r: (r["state"], r["op_type"], r.get("predicate", "")))
        return rows

    def query_fifo_grouped(self, name: str, op_type: Optional[str] = None) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group results by FSM state -> list of ops (useful when the same
        state has multiple predicated reads/writes).
        """
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for rec in self.query_fifo(name, op_type):
            grouped.setdefault(rec["state"], []).append(rec)
        return grouped

    def to_json(self) -> str:
        return json.dumps(self._parsed, indent=2)

# -------- Optional CLI --------
def _cli():
    ap = argparse.ArgumentParser(description="Class-based Vitis HLS report parser.")
    ap.add_argument("--file", required=True, help="Path to Vitis HLS report text file")
    ap.add_argument("--stream", help="Stream/port name to query (e.g., StreamingConv_0_out0_stream_1)")
    ap.add_argument("--type", choices=["read", "write"], help="Filter by op type for --stream")
    ap.add_argument("--json", help="Save full parsed JSON to this path")
    args = ap.parse_args()

    p = VitisHlsReportParser(args.file)
    print("== Pipeline ==")
    print("  II:", p.pipeline_ii)
    print("  Depth:", p.pipeline_depth)
    if p.pipeline_states:
        print("  States order:", p.pipeline_states)

    print("\n== Latency (cycles) ==")
    lat = p.get_latency_summary()
    print("  min:", lat.get("latency_cycles_min"), "max:", lat.get("latency_cycles_max"))

    if p.get_loops():
        print("\n== Loop Rows ==")
        for L in p.get_loops():
            print(f"  - {L['loop_name']}: iter_latency={L['iteration_latency']}, "
                  f"achieved_ii={L['achieved_ii']}, trip={L['trip_count']}")

    print("\n== FIFO Ops (summary) ==")
    for op in p.fifo_ops:
        print(f"  State {op['state']}: {op['op_type']} {op['port']} "
              f"pred={op['predicate']} width={op['width_bits']} "
              f"depth={op['fifo_depth']} delay={op['delay_ns']}ns")

    if args.stream:
        print(f"\n== Query: stream '{args.stream}' type={args.type or 'any'} ==")
        rows = p.query_fifo(args.stream, op_type=args.type)
        if not rows:
            print("  (no matches)")
        else:
            for r in rows:
                stage = r.get("pipeline_stage")
                stage_str = f" (stage {stage})" if stage is not None else ""
                print(f"  State {r['state']}{stage_str}: {r['op_type']} "
                      f"pred={r['predicate']} width={r['width_bits']} "
                      f"depth={r['fifo_depth']} delay={r['delay_ns']}ns")

    if args.json:
        Path(args.json).write_text(p.to_json(), encoding="utf-8")
        print(f"\nSaved JSON to {args.json}")

if __name__ == "__main__":
    _cli()
