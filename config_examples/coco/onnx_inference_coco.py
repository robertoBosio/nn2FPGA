import os
import time
import json
import argparse
import sys
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
import onnxruntime as ort
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO


# -----------------------------
# Dataset / DataLoader
# -----------------------------
class COCODataset(Dataset):
    def __init__(self, root: str, annFile: str, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.ids)


def coco_dataloader(batch_size: int, sample_size: int | None = None, num_workers: int = 0):
    root = "/home/datasets/coco/images/val2017"
    annFile = "/home/datasets/coco/annotations/instances_val2017.json"
    transform = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ]
    )

    dataset = COCODataset(root, annFile, transform)

    if sample_size is not None:
        sample_size = min(sample_size, len(dataset))
        dataset = Subset(dataset, list(range(sample_size)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,   # fair comparison: same order for both sessions
        num_workers=num_workers,
        drop_last=False,
    )


# -----------------------------
# Correctness helpers
# -----------------------------
def report_error_stats(output_name: str, expected: np.ndarray, produced: np.ndarray, top_k: int = 10):
    error = np.abs(expected - produced)

    max_error = float(np.max(error))
    mean_error = float(np.mean(error))
    min_error = float(np.min(error))
    std_error = float(np.std(error))

    flat_error = error.flatten()
    flat_idx = np.argsort(-flat_error)  # descending
    topk_idx = flat_idx[:top_k]
    unraveled_idx = [np.unravel_index(i, error.shape) for i in topk_idx]

    print("=" * 60)
    print(f"Output: {output_name}")
    print(f"Expected (first 10): {expected.flatten()[:10]}")
    print(f"Produced (first 10): {produced.flatten()[:10]}")
    print(f"Max Error:  {max_error}")
    print(f"Min Error:  {min_error}")
    print(f"Mean Error: {mean_error}")
    print(f"Std Error:  {std_error}")

    if max_error == 0.0:
        print("No errors detected.")
        print("=" * 60)
        return

    print(f"Top {top_k} errors:")
    for rank, (idx, val) in enumerate(zip(unraveled_idx, flat_error[topk_idx]), 1):
        exp_val = expected[idx]
        prod_val = produced[idx]
        print(f" {rank:2d}. idx={idx}, error={val}, expected={exp_val}, produced={prod_val}")
    print("=" * 60)


def check_correctness_single_output(
    sess_opt: ort.InferenceSession,
    sess_orig: ort.InferenceSession,
    input_name: str,
    x: np.ndarray,
):
    print("Checking correctness (single output)...")
    actual = sess_opt.run(None, {input_name: x})
    expected = sess_orig.run(None, {input_name: x})

    if len(expected) < 1 or len(actual) < 1:
        raise RuntimeError(
            f"Expected at least 1 output from both models, got "
            f"orig={len(expected)}, opt={len(actual)}"
        )

    report_error_stats(
        "global_out_0",
        np.asarray(expected[0]).flatten(),
        np.asarray(actual[0]).flatten(),
    )


def make_session_options(
    custom_op_so: str,
    enable_profiling: bool,
    graph_optimization_level: ort.GraphOptimizationLevel,
) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.register_custom_ops_library(custom_op_so)
    so.graph_optimization_level = graph_optimization_level
    so.enable_profiling = enable_profiling
    return so


# -----------------------------
# Power monitoring
# -----------------------------
RAILNAME_MAP = {
    "VCCPSINTFP": "u76",
    "VCCINTLP": "u77",
    "VCCPSAUX": "u78",
    "VCCPSPLL": "u87",
    "MGTRAVCC": "u85",
    "MGTRAVTT": "u86",
    "VCCPSDDR": "u93",
    "VCCOPS": "u88",
    "VCCOPS3": "u15",
    "VCCPSDDRPLL": "u92",
    "VCCINT": "u79",
    "VCCBRAM": "u81",
    "VCCAUX": "u80",
    "VCC1V2": "u84",
    "VCC3V3": "u16",
    "VADJ_FMC": "u65",
    "MGTAVCC": "u74",
    "MGTAVTT": "u75",
}


@dataclass
class INARail:
    rail: str
    current_path: str
    voltage_path: str


class ZCU102PowerMonitor:
    def __init__(self):
        self.rails = self._discover_rails()

    def _discover_rails(self) -> list[INARail]:
        rails: list[INARail] = []
        directory = "/sys/class/hwmon"

        if not os.path.isdir(directory):
            raise RuntimeError(f"{directory} does not exist")

        for entry in os.listdir(directory):
            if entry.startswith("."):
                continue

            base = os.path.join(directory, entry)
            name_file = os.path.join(base, "name")
            if not os.path.isfile(name_file):
                continue

            try:
                with open(name_file, "r", encoding="utf-8") as f:
                    sensor_name = f.read().strip()
            except OSError:
                continue

            if not sensor_name.startswith("ina"):
                continue

            matched_rail = None
            for rail, chip_name in RAILNAME_MAP.items():
                if chip_name in sensor_name:
                    matched_rail = rail
                    break

            if matched_rail is None:
                continue

            current_path = os.path.join(base, "curr1_input")
            voltage_path = os.path.join(base, "in2_input")

            if os.path.isfile(current_path) and os.path.isfile(voltage_path):
                rails.append(
                    INARail(
                        rail=matched_rail,
                        current_path=current_path,
                        voltage_path=voltage_path,
                    )
                )

        if not rails:
            raise RuntimeError("No ZCU102 INA rails found under /sys/class/hwmon")

        rails.sort(key=lambda x: x.rail)
        return rails

    @staticmethod
    def _read_float(path: str) -> float:
        with open(path, "r", encoding="utf-8") as f:
            return float(f.read().strip())

    def read_total_power_w(self) -> float:
        total_w = 0.0
        for rail in self.rails:
            curr_ma = self._read_float(rail.current_path)
            volt_mv = self._read_float(rail.voltage_path)
            total_w += (curr_ma * volt_mv) / 1_000_000.0
        return total_w

    def read_rail_powers_w(self) -> dict[str, float]:
        out = {}
        for rail in self.rails:
            curr_ma = self._read_float(rail.current_path)
            volt_mv = self._read_float(rail.voltage_path)
            out[rail.rail] = (curr_ma * volt_mv) / 1_000_000.0
        return out


class PowerRecorder:
    def __init__(self, monitor: ZCU102PowerMonitor, sample_period_s: float = 0.01):
        self.monitor = monitor
        self.sample_period_s = sample_period_s
        self.samples: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        next_t = time.perf_counter()
        while not self._stop_event.is_set():
            ts_perf_ns = time.perf_counter_ns()

            try:
                total_power_w = self.monitor.read_total_power_w()
                rail_powers = self.monitor.read_rail_powers_w()
            except Exception as e:
                self.samples.append({"timestamp_perf_ns": ts_perf_ns, "error": repr(e)})
            else:
                row = {
                    "timestamp_perf_ns": ts_perf_ns,
                    "total_power_w": total_power_w,
                }
                row.update({f"rail_{k}_w": v for k, v in rail_powers.items()})
                self.samples.append(row)

            next_t += self.sample_period_s
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("PowerRecorder already started")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def write_csv(self, filepath: str):
        if not self.samples:
            raise RuntimeError("No power samples recorded")

        fieldnames = sorted({k for row in self.samples for k in row.keys()})
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)

    def to_trace_events(self, timeline_origin_ns: int) -> list[dict]:
        events = []
        counter_keys: set[str] = set()
        for sample in self.samples:
            if "error" in sample:
                continue
            if "total_power_w" in sample:
                counter_keys.add("total_power_w")
            for key in sample:
                if key.startswith("rail_") and key.endswith("_w"):
                    counter_keys.add(key)

        key_to_pid = {key: 7000 + i for i, key in enumerate(sorted(counter_keys))}
        for sample in self.samples:
            if "error" in sample:
                continue
            ts_us = (sample["timestamp_perf_ns"] - timeline_origin_ns) / 1e3
            for key, pid in key_to_pid.items():
                val = sample.get(key)
                if val is not None:
                    events.append(
                        {
                            "name": key,
                            "cat": "power",
                            "ph": "C",
                            "ts": ts_us,
                            "pid": pid,
                            "tid": 1,
                            "args": {"W": val},
                        }
                    )
        return events


def outputs_match(
    expected: list[np.ndarray],
    produced: list[np.ndarray],
    image_idx: int,
    atol: float,
    rtol: float,
) -> bool:
    if len(expected) != len(produced):
        print(
            f"Image {image_idx}: FAIL - output count mismatch "
            f"expected={len(expected)}, produced={len(produced)}"
        )
        return False

    image_ok = True
    for output_idx, (expected_out, produced_out) in enumerate(zip(expected, produced)):
        expected_arr = np.asarray(expected_out)
        produced_arr = np.asarray(produced_out)

        if expected_arr.shape != produced_arr.shape:
            print(
                f"Image {image_idx}, output {output_idx}: FAIL - shape mismatch "
                f"expected={expected_arr.shape}, produced={produced_arr.shape}"
            )
            image_ok = False
            continue

        close = np.isclose(expected_arr, produced_arr, atol=atol, rtol=rtol)
        if not np.all(close):
            bad_count = int(np.size(close) - np.count_nonzero(close))
            print(
                f"Image {image_idx}, output {output_idx}: FAIL - values differ "
                f"bad_elements={bad_count}/{close.size} atol={atol} rtol={rtol}"
            )
            report_error_stats(
                f"image_{image_idx}_out_{output_idx}",
                expected_arr.flatten(),
                produced_arr.flatten(),
            )
            image_ok = False

    if image_ok:
        print(f"Image {image_idx}: PASS")

    return image_ok


def run_outputs_concurrent(
    sess: ort.InferenceSession,
    batches: list[np.ndarray],
    input_name: str,
    workers: int,
) -> list[list[np.ndarray]]:
    if workers <= 0:
        raise ValueError("workers must be > 0")

    outputs: list[list[np.ndarray] | None] = [None] * len(batches)

    def run_one(image_idx: int, np_features: np.ndarray):
        return image_idx, sess.run(None, {input_name: np_features})

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_one, image_idx, np_features)
            for image_idx, np_features in enumerate(batches)
        ]
        for future in as_completed(futures):
            image_idx, produced = future.result()
            outputs[image_idx] = produced

    missing = [i for i, item in enumerate(outputs) if item is None]
    if missing:
        raise RuntimeError(f"Missing optimized outputs for image indices: {missing}")

    return [item for item in outputs if item is not None]


# -----------------------------
# Benchmark helpers
# -----------------------------
def percentile_ms(values_s: list[float], p: float) -> float:
    if not values_s:
        return float("nan")
    return float(np.percentile(np.array(values_s, dtype=np.float64) * 1e3, p))


def pct_reduction(old: float, new: float) -> float:
    if old == 0:
        return float("nan")
    return 100.0 * (old - new) / old


def speedup(old: float, new: float) -> float:
    if new == 0:
        return float("inf")
    return old / new


def preload_batches(dataloader: DataLoader, measure_batches: int | None = None) -> list[np.ndarray]:
    batches: list[np.ndarray] = []

    for b, features in enumerate(dataloader):
        if measure_batches is not None and b >= measure_batches:
            break
        batches.append(features.numpy().astype(np.float32))

    if not batches:
        raise RuntimeError("No batches loaded. Check num_images / dataloader.")

    return batches


def warmup_session(
    sess: ort.InferenceSession,
    input_name: str,
    batches: list[np.ndarray],
    warmup_batches: int = 5,
):
    if not batches:
        raise RuntimeError("No batches available for warmup.")

    n = min(warmup_batches, len(batches))
    for i in range(n):
        _ = sess.run(None, {input_name: batches[i]})


def benchmark_preloaded(
    sess: ort.InferenceSession,
    batches: list[np.ndarray],
    input_name: str,
    label: str,
    run_records: list[dict] | None = None,
):
    batch_lat_s: list[float] = []
    img_lat_s: list[float] = []
    total_images = 0
    total_time_s = 0.0

    for batch_idx, np_features in enumerate(batches):
        bs = int(np_features.shape[0])

        if run_records is None:
            t0 = time.perf_counter()
            _ = sess.run(None, {input_name: np_features})
            t1 = time.perf_counter()
            dt = t1 - t0
        else:
            start_perf_ns = time.perf_counter_ns()
            _ = sess.run(None, {input_name: np_features})
            end_perf_ns = time.perf_counter_ns()
            dt = (end_perf_ns - start_perf_ns) / 1e9

            run_records.append(
                {
                    "label": label,
                    "batch_idx": batch_idx,
                    "batch_size": bs,
                    "run_tag": f"{label}_batch_{batch_idx}",
                    "start_perf_ns": start_perf_ns,
                    "end_perf_ns": end_perf_ns,
                    "duration_ms": dt * 1e3,
                }
            )

        batch_lat_s.append(dt)
        img_lat_s.append(dt / bs)

        total_images += bs
        total_time_s += dt

    throughput = total_images / total_time_s
    avg_batch_ms = (sum(batch_lat_s) / len(batch_lat_s)) * 1e3
    avg_img_ms = (sum(img_lat_s) / len(img_lat_s)) * 1e3

    results = {
        "label": label,
        "measured_batches": len(batch_lat_s),
        "measured_images": total_images,
        "total_time_s": total_time_s,
        "throughput_img_s": throughput,
        "avg_batch_ms": avg_batch_ms,
        "avg_img_ms": avg_img_ms,
        "p50_batch_ms": percentile_ms(batch_lat_s, 50),
        "p90_batch_ms": percentile_ms(batch_lat_s, 90),
        "p95_batch_ms": percentile_ms(batch_lat_s, 95),
        "p50_img_ms": percentile_ms(img_lat_s, 50),
        "p90_img_ms": percentile_ms(img_lat_s, 90),
        "p95_img_ms": percentile_ms(img_lat_s, 95),
    }

    print(f"\n===== Benchmark results: {label} =====")
    print(f"Measured batches:        {results['measured_batches']}")
    print(f"Measured images:         {results['measured_images']}")
    print(f"Total measured time (s): {results['total_time_s']:.6f}")
    print(f"Throughput (img/s):      {results['throughput_img_s']:.2f}")
    print(f"Avg batch latency (ms):  {results['avg_batch_ms']:.3f}")
    print(f"Avg img latency (ms):    {results['avg_img_ms']:.3f}")

    print("\nLatency percentiles (per-batch, ms):")
    print(f"  p50: {results['p50_batch_ms']:.3f}")
    print(f"  p90: {results['p90_batch_ms']:.3f}")
    print(f"  p95: {results['p95_batch_ms']:.3f}")

    print("\nLatency percentiles (per-image, ms):")
    print(f"  p50: {results['p50_img_ms']:.3f}")
    print(f"  p90: {results['p90_img_ms']:.3f}")
    print(f"  p95: {results['p95_img_ms']:.3f}")
    print("=" * 40)

    return results


def benchmark_preloaded_concurrent(
    sess: ort.InferenceSession,
    batches: list[np.ndarray],
    input_name: str,
    label: str,
    workers: int,
    run_records: list[dict] | None = None,
):
    if workers <= 0:
        raise ValueError("workers must be > 0")

    for batch_idx, np_features in enumerate(batches):
        if int(np_features.shape[0]) != 1:
            raise ValueError(
                f"Concurrent nn2FPGA benchmark requires batch size 1, "
                f"got batch {batch_idx} with shape {np_features.shape}."
            )

    def run_one(image_idx: int, np_features: np.ndarray):
        if run_records is None:
            t0 = time.perf_counter()
            _ = sess.run(None, {input_name: np_features})
            t1 = time.perf_counter()
            return image_idx, t0, t1

        start_perf_ns = time.perf_counter_ns()
        _ = sess.run(None, {input_name: np_features})
        end_perf_ns = time.perf_counter_ns()
        return image_idx, start_perf_ns, end_perf_ns

    lat_s: list[float] = []
    start_s = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_one, image_idx, np_features)
            for image_idx, np_features in enumerate(batches)
        ]
        for future in as_completed(futures):
            image_idx, start_time, end_time = future.result()
            if run_records is None:
                dt = end_time - start_time
            else:
                dt = (end_time - start_time) / 1e9
                run_records.append(
                    {
                        "label": label,
                        "batch_idx": image_idx,
                        "batch_size": 1,
                        "run_tag": f"{label}_image_{image_idx}",
                        "start_perf_ns": start_time,
                        "end_perf_ns": end_time,
                        "duration_ms": dt * 1e3,
                    }
                )
            lat_s.append(dt)

    total_time_s = time.perf_counter() - start_s
    total_images = len(lat_s)
    throughput = total_images / total_time_s
    avg_img_ms = (sum(lat_s) / len(lat_s)) * 1e3

    results = {
        "label": label,
        "measured_batches": total_images,
        "measured_images": total_images,
        "total_time_s": total_time_s,
        "throughput_img_s": throughput,
        "avg_batch_ms": avg_img_ms,
        "avg_img_ms": avg_img_ms,
        "p50_batch_ms": percentile_ms(lat_s, 50),
        "p90_batch_ms": percentile_ms(lat_s, 90),
        "p95_batch_ms": percentile_ms(lat_s, 95),
        "p50_img_ms": percentile_ms(lat_s, 50),
        "p90_img_ms": percentile_ms(lat_s, 90),
        "p95_img_ms": percentile_ms(lat_s, 95),
        "inflight_runs": workers,
    }

    print(f"\n===== Benchmark results: {label} =====")
    print(f"Measured images:         {results['measured_images']}")
    print(f"Concurrent runs:         {workers}")
    print(f"Wall time (s):           {results['total_time_s']:.6f}")
    print(f"Throughput (img/s):      {results['throughput_img_s']:.2f}")
    print(f"Avg session.run ms:      {results['avg_img_ms']:.3f}")

    print("\nLatency percentiles (session.run, ms):")
    print(f"  p50: {results['p50_img_ms']:.3f}")
    print(f"  p90: {results['p90_img_ms']:.3f}")
    print(f"  p95: {results['p95_img_ms']:.3f}")
    print("=" * 40)

    return results


def build_comparison(orig: dict, opt: dict) -> dict:
    thr_speedup = opt["throughput_img_s"] / orig["throughput_img_s"] if orig["throughput_img_s"] != 0 else float("inf")
    thr_gain_pct = 100.0 * (opt["throughput_img_s"] - orig["throughput_img_s"]) / orig["throughput_img_s"]

    comparison = {
        "throughput": {
            "original_img_s": orig["throughput_img_s"],
            "optimized_img_s": opt["throughput_img_s"],
            "speedup_x": thr_speedup,
            "gain_pct": thr_gain_pct,
        },
        "avg_latency": {
            "batch_ms": {
                "original": orig["avg_batch_ms"],
                "optimized": opt["avg_batch_ms"],
                "speedup_x": speedup(orig["avg_batch_ms"], opt["avg_batch_ms"]),
                "reduction_pct": pct_reduction(orig["avg_batch_ms"], opt["avg_batch_ms"]),
            },
            "image_ms": {
                "original": orig["avg_img_ms"],
                "optimized": opt["avg_img_ms"],
                "speedup_x": speedup(orig["avg_img_ms"], opt["avg_img_ms"]),
                "reduction_pct": pct_reduction(orig["avg_img_ms"], opt["avg_img_ms"]),
            },
        },
        "percentiles_batch_ms": {},
        "percentiles_image_ms": {},
    }

    for key in ["p50_batch_ms", "p90_batch_ms", "p95_batch_ms"]:
        comparison["percentiles_batch_ms"][key] = {
            "original": orig[key],
            "optimized": opt[key],
            "speedup_x": speedup(orig[key], opt[key]),
            "reduction_pct": pct_reduction(orig[key], opt[key]),
        }

    for key in ["p50_img_ms", "p90_img_ms", "p95_img_ms"]:
        comparison["percentiles_image_ms"][key] = {
            "original": orig[key],
            "optimized": opt[key],
            "speedup_x": speedup(orig[key], opt[key]),
            "reduction_pct": pct_reduction(orig[key], opt[key]),
        }

    return comparison


def format_comparison_text(orig: dict, opt: dict, comparison: dict) -> str:
    lines = []
    lines.append("================ Performance improvement vs original ================")
    lines.append("Throughput:")
    lines.append(f"  Original:  {orig['throughput_img_s']:.2f} img/s")
    lines.append(f"  Optimized: {opt['throughput_img_s']:.2f} img/s")
    lines.append(f"  Speedup:   {comparison['throughput']['speedup_x']:.3f}x")
    lines.append(f"  Gain:      {comparison['throughput']['gain_pct']:.2f}%")
    lines.append("")
    lines.append("Average latency:")
    lines.append(f"  Batch: {orig['avg_batch_ms']:.3f} ms -> {opt['avg_batch_ms']:.3f} ms")
    lines.append(
        f"         speedup={comparison['avg_latency']['batch_ms']['speedup_x']:.3f}x, "
        f"reduction={comparison['avg_latency']['batch_ms']['reduction_pct']:.2f}%"
    )
    lines.append(f"  Image: {orig['avg_img_ms']:.3f} ms -> {opt['avg_img_ms']:.3f} ms")
    lines.append(
        f"         speedup={comparison['avg_latency']['image_ms']['speedup_x']:.3f}x, "
        f"reduction={comparison['avg_latency']['image_ms']['reduction_pct']:.2f}%"
    )
    lines.append("")
    lines.append("Per-batch percentiles:")
    for key in ["p50_batch_ms", "p90_batch_ms", "p95_batch_ms"]:
        item = comparison["percentiles_batch_ms"][key]
        lines.append(
            f"  {key}: {item['original']:.3f} ms -> {item['optimized']:.3f} ms | "
            f"speedup={item['speedup_x']:.3f}x | reduction={item['reduction_pct']:.2f}%"
        )
    lines.append("")
    lines.append("Per-image percentiles:")
    for key in ["p50_img_ms", "p90_img_ms", "p95_img_ms"]:
        item = comparison["percentiles_image_ms"][key]
        lines.append(
            f"  {key}: {item['original']:.3f} ms -> {item['optimized']:.3f} ms | "
            f"speedup={item['speedup_x']:.3f}x | reduction={item['reduction_pct']:.2f}%"
        )
    lines.append("====================================================================")
    return "\n".join(lines)


def write_results_file(filepath: str, orig: dict, opt: dict, comparison: dict):
    text_report = format_comparison_text(orig, opt, comparison)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text_report)
        f.write("\n\n")
        f.write("Raw metrics (JSON):\n")
        json.dump(
            {
                "original": orig,
                "optimized": opt,
                "comparison": comparison,
            },
            f,
            indent=2,
        )
        f.write("\n")


def write_json(filepath: str, obj):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _load_ort_events(profile_file: str) -> list[dict]:
    with open(profile_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    return raw.get("traceEvents", [])


def _find_first_session_run_us(events: list[dict], names: set[str]) -> float | None:
    for event in events:
        if event.get("name") in names and "ts" in event:
            return float(event["ts"])
    return None


def _ort_events_to_trace(
    events: list[dict],
    sync_ort_us: float,
    sync_perf_ns: int,
    timeline_origin_ns: int,
    pid_offset: int,
) -> list[dict]:
    sync_wall_us = (sync_perf_ns - timeline_origin_ns) / 1e3
    ort_offset_us = sync_wall_us - sync_ort_us

    out = []
    for event in events:
        item = dict(event)
        if "ts" in item:
            item["ts"] = float(item["ts"]) + ort_offset_us
        if "pid" in item:
            item["pid"] = item["pid"] + pid_offset
        if "tid" in item:
            item["tid"] = item["tid"] + pid_offset
        out.append(item)
    return out


def _run_records_to_trace(run_records: list[dict], timeline_origin_ns: int, pid: int = 8000) -> list[dict]:
    events = []
    label_to_tid = {}
    next_tid = 1

    for record in run_records:
        label = record["label"]
        if label not in label_to_tid:
            label_to_tid[label] = next_tid
            next_tid += 1

        events.append(
            {
                "name": record["run_tag"],
                "cat": "benchmark",
                "ph": "X",
                "ts": (record["start_perf_ns"] - timeline_origin_ns) / 1e3,
                "dur": (record["end_perf_ns"] - record["start_perf_ns"]) / 1e3,
                "pid": pid,
                "tid": label_to_tid[label],
                "args": {
                    "batch_idx": record["batch_idx"],
                    "batch_size": record["batch_size"],
                    "duration_ms": record["duration_ms"],
                },
            }
        )

    return events


def build_merged_trace(
    profile_file_orig: str,
    profile_file_opt: str,
    sync_perf_ns_orig: int,
    sync_perf_ns_opt: int,
    run_records: list[dict],
    power_recorder: PowerRecorder,
    timeline_origin_ns: int,
    output_file: str,
) -> str:
    sync_event_names = {"model_run", "SequentialExecutor::Execute", "session.run"}

    orig_events = [event for event in _load_ort_events(profile_file_orig) if "ts" in event]
    opt_events = [event for event in _load_ort_events(profile_file_opt) if "ts" in event]

    sync_ort_us_orig = _find_first_session_run_us(orig_events, sync_event_names)
    if sync_ort_us_orig is None:
        sync_ort_us_orig = min(float(event["ts"]) for event in orig_events)

    sync_ort_us_opt = _find_first_session_run_us(opt_events, sync_event_names)
    if sync_ort_us_opt is None:
        sync_ort_us_opt = min(float(event["ts"]) for event in opt_events)

    merged: list[dict] = []
    merged.extend(
        _ort_events_to_trace(
            orig_events,
            sync_ort_us=sync_ort_us_orig,
            sync_perf_ns=sync_perf_ns_orig,
            timeline_origin_ns=timeline_origin_ns,
            pid_offset=5000,
        )
    )
    merged.extend(
        _ort_events_to_trace(
            opt_events,
            sync_ort_us=sync_ort_us_opt,
            sync_perf_ns=sync_perf_ns_opt,
            timeline_origin_ns=timeline_origin_ns,
            pid_offset=6000,
        )
    )
    merged.extend(power_recorder.to_trace_events(timeline_origin_ns))
    merged.extend(_run_records_to_trace(run_records, timeline_origin_ns))

    for label, perf_ns in [("sync_orig", sync_perf_ns_orig), ("sync_opt", sync_perf_ns_opt)]:
        merged.append(
            {
                "name": label,
                "cat": "anchor",
                "ph": "i",
                "s": "g",
                "ts": (perf_ns - timeline_origin_ns) / 1e3,
                "pid": 9000,
                "tid": 1,
                "args": {},
            }
        )

    merged.sort(key=lambda event: event.get("ts", 0.0))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"traceEvents": merged}, f)

    return output_file


# -----------------------------
# Modes / CLI
# -----------------------------
def run_correctness(args) -> int:
    custom_op_so = os.path.abspath(args.custom_op)
    print("Loading the operator:", custom_op_so)
    print("Starting correctness sessions with ORT optimizations disabled...")

    sess_orig = ort.InferenceSession(
        args.original_model,
        sess_options=make_session_options(
            custom_op_so,
            enable_profiling=False,
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ),
        providers=["CPUExecutionProvider"],
    )
    sess_opt = ort.InferenceSession(
        args.model,
        sess_options=make_session_options(
            custom_op_so,
            enable_profiling=False,
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ),
        providers=["CPUExecutionProvider"],
    )

    input_name = sess_orig.get_inputs()[0].name
    dataloader = coco_dataloader(batch_size=1, sample_size=args.num_images, num_workers=args.num_workers)
    batches = preload_batches(dataloader)

    checked_images = 0
    failed_images = 0
    expected_outputs: list[list[np.ndarray]] = []

    for x in batches:
        expected_outputs.append(sess_orig.run(None, {input_name: x}))

    if args.inflight_runs == 1:
        produced_outputs = [sess_opt.run(None, {input_name: x}) for x in batches]
    else:
        produced_outputs = run_outputs_concurrent(
            sess_opt,
            batches,
            input_name,
            workers=args.inflight_runs,
        )

    for image_idx, (expected, produced) in enumerate(zip(expected_outputs, produced_outputs)):
        checked_images += 1
        if not outputs_match(expected, produced, image_idx, atol=args.atol, rtol=args.rtol):
            failed_images += 1

    passed_images = checked_images - failed_images
    print("\n===== Correctness summary =====")
    print(f"Checked images:           {checked_images}")
    print(f"Optimized inflight runs:  {args.inflight_runs}")
    print(f"Tolerance:                atol={args.atol} rtol={args.rtol}")
    print(f"Passed images:            {passed_images}")
    print(f"Failed images:            {failed_images}")
    print("=================================")

    return 0 if failed_images == 0 else 1


def run_speed(args) -> int:
    custom_op_so = os.path.abspath(args.custom_op)
    print("Loading the operator:", custom_op_so)

    dataloader = coco_dataloader(
        batch_size=1,
        sample_size=args.num_images,
        num_workers=args.num_workers,
    )
    batches = preload_batches(dataloader, measure_batches=args.measure_batches)

    if args.power_record:
        return run_speed_with_power(args, custom_op_so, batches)

    print("Starting optimized benchmark sessions...")
    sess_orig = ort.InferenceSession(
        args.original_model,
        sess_options=make_session_options(
            custom_op_so,
            enable_profiling=True,
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ),
        providers=["CPUExecutionProvider"],
    )
    sess_opt = ort.InferenceSession(
        args.model,
        sess_options=make_session_options(
            custom_op_so,
            enable_profiling=True,
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ),
        providers=["CPUExecutionProvider"],
    )

    input_name = sess_orig.get_inputs()[0].name

    warmup_session(sess_orig, input_name, batches, warmup_batches=args.warmup_batches)
    warmup_session(sess_opt, input_name, batches, warmup_batches=args.warmup_batches)

    orig_results = benchmark_preloaded(sess_orig, batches, input_name, label="original")
    opt_results = benchmark_preloaded_concurrent(
        sess_opt,
        batches,
        input_name,
        label=f"optimized_concurrent_{args.inflight_runs}",
        workers=args.inflight_runs,
    )

    comparison = build_comparison(orig_results, opt_results)
    report_text = format_comparison_text(orig_results, opt_results, comparison)
    print("\n" + report_text + "\n")

    write_results_file(args.results_file, orig_results, opt_results, comparison)
    print(f"Performance report written to: {os.path.abspath(args.results_file)}")

    prof_file_orig = sess_orig.end_profiling()
    print(f"Original profiling trace written to: {prof_file_orig}")

    prof_file_opt = sess_opt.end_profiling()
    print(f"Optimized profiling trace written to: {prof_file_opt}")

    return 0


def run_speed_with_power(args, custom_op_so: str, batches: list[np.ndarray]) -> int:
    print("Running warmup (unprofiled sessions)...")
    sess_orig_warmup = ort.InferenceSession(
        args.original_model,
        sess_options=make_session_options(
            custom_op_so,
            enable_profiling=False,
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ),
        providers=["CPUExecutionProvider"],
    )
    sess_opt_warmup = ort.InferenceSession(
        args.model,
        sess_options=make_session_options(
            custom_op_so,
            enable_profiling=False,
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ),
        providers=["CPUExecutionProvider"],
    )

    input_name = sess_orig_warmup.get_inputs()[0].name
    warmup_session(sess_orig_warmup, input_name, batches, warmup_batches=args.warmup_batches)
    warmup_session(sess_opt_warmup, input_name, batches, warmup_batches=args.warmup_batches)
    del sess_orig_warmup, sess_opt_warmup

    print("Starting profiled benchmark sessions...")
    sess_orig = ort.InferenceSession(
        args.original_model,
        sess_options=make_session_options(
            custom_op_so,
            enable_profiling=True,
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ),
        providers=["CPUExecutionProvider"],
    )
    sess_opt = ort.InferenceSession(
        args.model,
        sess_options=make_session_options(
            custom_op_so,
            enable_profiling=True,
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ),
        providers=["CPUExecutionProvider"],
    )

    power_monitor = ZCU102PowerMonitor()
    power_recorder = PowerRecorder(power_monitor, sample_period_s=args.power_sample_period)
    inference_runs: list[dict] = []
    timeline_origin_ns = time.perf_counter_ns()
    profile_file_orig: str | None = None
    profile_file_opt: str | None = None

    print("Starting power recording and benchmark...")
    power_recorder.start()
    try:
        sync_perf_ns_orig = time.perf_counter_ns()
        sess_orig.run(None, {input_name: batches[0]})
        orig_results = benchmark_preloaded(
            sess_orig,
            batches,
            input_name,
            label="original",
            run_records=inference_runs,
        )

        sync_perf_ns_opt = time.perf_counter_ns()
        sess_opt.run(None, {input_name: batches[0]})
        opt_results = benchmark_preloaded_concurrent(
            sess_opt,
            batches,
            input_name,
            label=f"optimized_concurrent_{args.inflight_runs}",
            workers=args.inflight_runs,
            run_records=inference_runs,
        )
    finally:
        power_recorder.stop()
        print("Power recording stopped.")

        try:
            profile_file_orig = sess_orig.end_profiling()
            print(f"Original profiling trace written to: {profile_file_orig}")
        except Exception as e:
            print(f"Warning: end_profiling (orig) failed: {e}")

        try:
            profile_file_opt = sess_opt.end_profiling()
            print(f"Optimized profiling trace written to: {profile_file_opt}")
        except Exception as e:
            print(f"Warning: end_profiling (opt) failed: {e}")

    comparison = build_comparison(orig_results, opt_results)
    report_text = format_comparison_text(orig_results, opt_results, comparison)
    print("\n" + report_text + "\n")

    write_results_file(args.results_file, orig_results, opt_results, comparison)
    print(f"Performance report written to: {os.path.abspath(args.results_file)}")

    power_recorder.write_csv(args.power_file)
    print(f"Power samples written to: {os.path.abspath(args.power_file)}")

    write_json(args.runs_file, inference_runs)
    print(f"Inference runs written to: {os.path.abspath(args.runs_file)}")

    if profile_file_orig and profile_file_opt:
        build_merged_trace(
            profile_file_orig=profile_file_orig,
            profile_file_opt=profile_file_opt,
            sync_perf_ns_orig=sync_perf_ns_orig,
            sync_perf_ns_opt=sync_perf_ns_opt,
            run_records=inference_runs,
            power_recorder=power_recorder,
            timeline_origin_ns=timeline_origin_ns,
            output_file=args.merged_trace_file,
        )
        print(f"Merged Perfetto trace written to: {os.path.abspath(args.merged_trace_file)}")
    else:
        print("Warning: one or both profile files missing; merged trace not written.")

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run exact COCO correctness checks or optimized ONNX throughput benchmarks."
    )
    parser.add_argument("--mode", choices=["correctness", "speed"], required=True)
    parser.add_argument("--model", default="nn2FPGA_yolov5nu.onnx")
    parser.add_argument("--original-model", default="original_model_qcdq.onnx")
    parser.add_argument("--custom-op", default="libnn2fpga_customop.so")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of COCO images to use. Use -1 for all images in speed mode.",
    )

    parser.add_argument(
        "--inflight-runs",
        type=int,
        default=4,
        help="Number of concurrent optimized session.run calls.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Number of warmup batches for speed mode.",
    )
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness mode.")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness mode.")
    parser.add_argument(
        "--measure-batches",
        type=int,
        default=None,
        help="Number of preloaded batches to benchmark in speed mode.",
    )
    parser.add_argument("--results-file", default="performance_improvement.txt")
    parser.add_argument(
        "--power-record",
        action="store_true",
        help="Record ZCU102 power samples during speed mode and emit a merged Perfetto trace.",
    )
    parser.add_argument(
        "--power-sample-period",
        type=float,
        default=0.01,
        help="Power sampling period in seconds when --power-record is enabled.",
    )
    parser.add_argument("--power-file", default="power_samples.csv")
    parser.add_argument("--runs-file", default="inference_runs.json")
    parser.add_argument("--merged-trace-file", default="merged_trace.json")

    args = parser.parse_args()
    if args.power_record and args.mode != "speed":
        parser.error("--power-record is only supported in speed mode")
    if args.num_images == -1:
        if args.mode == "correctness":
            parser.error("--num-images must be > 0 in correctness mode")
        args.num_images = None
    elif args.num_images <= 0:
        parser.error("--num-images must be > 0, or -1 in speed mode")
    if args.inflight_runs <= 0:
        parser.error("--inflight-runs must be > 0")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must be >= 0")
    if args.atol < 0:
        parser.error("--atol must be >= 0")
    if args.rtol < 0:
        parser.error("--rtol must be >= 0")
    if args.measure_batches is not None and args.measure_batches <= 0:
        parser.error("--measure-batches must be > 0")
    if args.power_sample_period <= 0:
        parser.error("--power-sample-period must be > 0")

    return args


def main():
    args = parse_args()
    if args.mode == "correctness":
        return run_correctness(args)
    if args.mode == "speed":
        return run_speed(args)
    raise RuntimeError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
