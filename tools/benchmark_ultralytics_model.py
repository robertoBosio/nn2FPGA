#!/usr/bin/env python3
"""
Benchmark + compare two ONNX models inside the Ultralytics pipeline.

What it does
------------
1) Benchmarks ORIGINAL vs ACCELERATED model using Ultralytics' pipeline:
   preprocess + ONNXRuntime inference + postprocess

2) Uses the exact same sampled batches for both models

3) Prints and saves performance comparison to a text file

4) Optionally runs COCO accuracy afterward via Ultralytics model.val()

5) Optionally enables ONNXRuntime profiling only for the timed benchmark section

Notes
-----
- Profiling excludes model.val()
- Profiling also excludes correctness/accuracy steps because benchmark runs happen first
- This compares full Ultralytics pipeline cost, not just raw ORT inference
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import statistics
import time
from contextlib import redirect_stdout
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.loaders import LoadPilAndNumpy


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
STAGE_NAMES = ("source_misc", "preprocess", "inference", "postprocess")


# ----------------------------
# Utility
# ----------------------------
def list_images(root: str) -> List[str]:
    root = os.path.expanduser(root)
    if os.path.isdir(root):
        paths: List[str] = []
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
        return sorted(paths)
    if os.path.isfile(root) and root.lower().endswith(".txt"):
        with open(root, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    raise FileNotFoundError(f"Could not find directory or .txt list: {root}")


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def speedup(old: float, new: float) -> float:
    if new == 0:
        return float("inf")
    return old / new


def pct_reduction(old: float, new: float) -> float:
    if old == 0:
        return float("nan")
    return 100.0 * (old - new) / old


def pct_gain(old: float, new: float) -> float:
    if old == 0:
        return float("nan")
    return 100.0 * (new - old) / old


class PredictorStageTimer:
    """
    Time predictor stages during model.predict().

    The residual time after explicit predictor methods is reported as
    source_misc, which mostly captures file loading/decoding plus framework
    overhead around the predictor calls.
    """

    def __init__(self, task: str):
        self.task = task
        self._patches: List[Tuple[type, str, Callable]] = []
        self._current: Optional[dict[str, float]] = None
        self._current_total_start: Optional[float] = None
        self._batch_records: List[dict] = []

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disable()

    def enable(self) -> None:
        import ultralytics.engine.predictor as predictor_mod

        self._patch_method(predictor_mod.BasePredictor, "preprocess", "preprocess")
        self._patch_method(predictor_mod.BasePredictor, "inference", "inference")
        self._patch_callbacks(predictor_mod.BasePredictor)

        for cls in self._predictor_classes_for_task():
            if hasattr(cls, "postprocess"):
                self._patch_method(cls, "postprocess", "postprocess")

    def disable(self) -> None:
        for cls, method_name, original in reversed(self._patches):
            setattr(cls, method_name, original)
        self._patches.clear()

    def start_batch(self) -> None:
        self._current = {f"{stage}_s": 0.0 for stage in STAGE_NAMES}

    def finish_batch(self, total_s: float) -> dict[str, float]:
        if self._current is None:
            raise RuntimeError("Predictor stage timer batch was not started.")

        accounted_s = sum(self._current[f"{stage}_s"] for stage in STAGE_NAMES if stage != "source_misc")
        self._current["source_misc_s"] = max(0.0, total_s - accounted_s)
        out = dict(self._current)
        self._current = None
        return out

    def reset_records(self) -> None:
        self._batch_records = []

    def consume_records(self) -> List[dict]:
        out = self._batch_records
        self._batch_records = []
        return out

    def _patch_method(self, cls: type, method_name: str, stage_name: str) -> None:
        original = getattr(cls, method_name, None)
        if original is None:
            return

        timer = self

        def wrapped(instance, *args, **kwargs):
            if timer._current is None:
                return original(instance, *args, **kwargs)

            t0 = time.perf_counter()
            try:
                return original(instance, *args, **kwargs)
            finally:
                timer._current[f"{stage_name}_s"] += time.perf_counter() - t0

        setattr(cls, method_name, wrapped)
        self._patches.append((cls, method_name, original))

    def _patch_callbacks(self, cls: type) -> None:
        original = getattr(cls, "run_callbacks", None)
        if original is None:
            return

        timer = self

        def wrapped(instance, event: str):
            if event == "on_predict_batch_start":
                timer._current_total_start = time.perf_counter()
                timer.start_batch()

            result = original(instance, event)

            if event == "on_predict_batch_end" and timer._current_total_start is not None and timer._current is not None:
                total_s = time.perf_counter() - timer._current_total_start
                record = timer.finish_batch(total_s)
                record["batch_size"] = len(getattr(instance, "results", []) or [])
                record["total_s"] = total_s
                timer._batch_records.append(record)
                timer._current_total_start = None

            return result

        setattr(cls, "run_callbacks", wrapped)
        self._patches.append((cls, "run_callbacks", original))

    def _predictor_classes_for_task(self) -> List[type]:
        classes: List[type] = []
        module_map = {
            "detect": ("ultralytics.models.yolo.detect.predict", "DetectionPredictor"),
            "segment": ("ultralytics.models.yolo.segment.predict", "SegmentationPredictor"),
            "classify": ("ultralytics.models.yolo.classify.predict", "ClassificationPredictor"),
            "pose": ("ultralytics.models.yolo.pose.predict", "PosePredictor"),
            "obb": ("ultralytics.models.yolo.obb.predict", "OBBPredictor"),
        }
        target = module_map.get(self.task)
        if target is None:
            return classes

        module_name, class_name = target
        try:
            module = __import__(module_name, fromlist=[class_name])
            classes.append(getattr(module, class_name))
        except Exception:
            pass
        return classes


# ----------------------------
# ONNXRuntime monkeypatch (optional)
# ----------------------------
class OrtSessionPatcher:
    """
    Monkeypatch onnxruntime.InferenceSession so Ultralytics will create sessions
    with custom SessionOptions (custom ops + profiling).
    """

    def __init__(
        self,
        custom_op_so: Optional[str],
        enable_profiling: bool,
        profile_dir: str,
        profile_prefix: str = "ultra_ort_profile",
    ):
        self.custom_op_so = os.path.abspath(custom_op_so) if custom_op_so else None
        self.enable_profiling = enable_profiling
        self.profile_dir = os.path.abspath(profile_dir)
        self.profile_prefix = profile_prefix
        self._ort = None
        self._orig_ctor = None
        self._sessions = []

    def enable(self) -> None:
        if not (self.custom_op_so or self.enable_profiling):
            return

        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError("onnxruntime is required for custom-op/profiling features.") from e

        self._ort = ort

        if self.custom_op_so and not os.path.isfile(self.custom_op_so):
            raise FileNotFoundError(f"Custom op library not found: {self.custom_op_so}")

        os.makedirs(self.profile_dir, exist_ok=True)
        self._orig_ctor = ort.InferenceSession

        def patched_inference_session(path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
            so = sess_options or ort.SessionOptions()

            if self.custom_op_so:
                so.register_custom_ops_library(self.custom_op_so)

            if self.enable_profiling:
                so.enable_profiling = True
                so.profile_file_prefix = os.path.join(self.profile_dir, self.profile_prefix)

            sess = self._orig_ctor(
                path_or_bytes,
                sess_options=so,
                providers=providers,
                provider_options=provider_options,
                **kwargs,
            )
            self._sessions.append(sess)
            return sess

        ort.InferenceSession = patched_inference_session  # type: ignore

    def end_profiling(self) -> List[str]:
        paths: List[str] = []
        if not self.enable_profiling or not self._sessions:
            return paths

        for i, s in enumerate(self._sessions):
            try:
                path = s.end_profiling()
                print(f"[ORT] Profile #{i}: {path}")
                paths.append(path)
            except Exception as e:
                print(f"[ORT] Could not end profiling for session #{i}: {e}")
        return paths

    def disable(self) -> None:
        if self._ort is not None and self._orig_ctor is not None:
            self._ort.InferenceSession = self._orig_ctor  # type: ignore


# ----------------------------
# Benchmark batch preparation
# ----------------------------
def sample_batches(
    images: List[str],
    batch: int,
    warmup: int,
    iters: int,
    seed: int,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Create deterministic warmup and timed batches.
    Both original and accelerated models will use exactly these same batches.
    """
    rng = random.Random(seed)
    need = max(1, (warmup + iters) * batch)

    if len(images) >= need:
        sample = rng.sample(images, need)
    else:
        sample = [rng.choice(images) for _ in range(need)]

    warmup_batches: List[List[str]] = []
    timed_batches: List[List[str]] = []

    idx = 0
    for _ in range(warmup):
        warmup_batches.append(sample[idx: idx + batch])
        idx += batch

    for _ in range(iters):
        timed_batches.append(sample[idx: idx + batch])
        idx += batch

    return warmup_batches, timed_batches


def flatten_batches(batches: List[List[object]]) -> List[object]:
    return [item for batch in batches for item in batch]


def load_image_bgr(path: str) -> np.ndarray:
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"))
    return np.ascontiguousarray(rgb[..., ::-1])


def load_image_batches(path_batches: List[List[str]]) -> List[List[np.ndarray]]:
    return [[load_image_bgr(path) for path in batch] for batch in path_batches]


def make_tensor_batches(image_batches: List[List[np.ndarray]], imgsz: int) -> List[torch.Tensor]:
    from ultralytics.data.augment import LetterBox

    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
    tensor_batches: List[torch.Tensor] = []

    for batch in image_batches:
        stacked = []
        for image in batch:
            resized = letterbox(image=image)
            rgb = resized[..., ::-1].transpose((2, 0, 1))
            stacked.append(np.ascontiguousarray(rgb).astype(np.float32) / 255.0)
        tensor_batches.append(torch.from_numpy(np.stack(stacked, axis=0)))

    return tensor_batches


def prepare_sources(
    warmup_path_batches: List[List[str]],
    timed_path_batches: List[List[str]],
    source_mode: str,
    imgsz: int,
) -> Tuple[List[object], List[object]]:
    if source_mode == "paths":
        return warmup_path_batches, timed_path_batches

    warmup_images = load_image_batches(warmup_path_batches)
    timed_images = load_image_batches(timed_path_batches)

    if source_mode == "ndarray":
        return warmup_images, timed_images
    if source_mode == "tensor":
        return make_tensor_batches(warmup_images, imgsz), make_tensor_batches(timed_images, imgsz)

    raise ValueError(f"Unsupported source mode: {source_mode}")


def count_total_images(batches: List[object]) -> int:
    total = 0
    for batch in batches:
        if isinstance(batch, torch.Tensor):
            total += int(batch.shape[0])
        else:
            total += len(batch)
    return total


class BatchedMemorySource(LoadPilAndNumpy):
    def __init__(self, batches: List[object]):
        self.batches = batches
        self.count = 0
        self.mode = "image"
        self.bs = self._batch_size(batches[0]) if batches else 0
        self.source_type = type(
            "SourceType",
            (),
            {
                "stream": False,
                "screenshot": False,
                "from_img": not bool(batches and isinstance(batches[0], torch.Tensor)),
                "tensor": bool(batches and isinstance(batches[0], torch.Tensor)),
            },
        )()

    def _batch_size(self, batch: object) -> int:
        if isinstance(batch, torch.Tensor):
            return int(batch.shape[0])
        return len(batch)

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return len(self.batches)

    def __next__(self):
        if self.count >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.count]
        self.count += 1

        if isinstance(batch, torch.Tensor):
            paths = [f"tensor_{self.count}_{i}.jpg" for i in range(int(batch.shape[0]))]
            return paths, batch, [""] * int(batch.shape[0])

        paths = [f"memory_{self.count}_{i}.jpg" for i in range(len(batch))]
        return paths, batch, [""] * len(batch)


# ----------------------------
# Benchmark
# ----------------------------
def run_predict_batch(
    model: YOLO,
    batch_source: object,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    half: bool,
    batch: int,
    stream: bool,
):
    return model.predict(
        source=batch_source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        half=half,
        verbose=False,
        batch=batch,
        stream=stream,
    )


def summarize_stage(stage_values_ms: List[float]) -> dict:
    sorted_vals = sorted(stage_values_ms)
    total_ms = sum(stage_values_ms)
    return {
        "total_ms": total_ms,
        "mean_ms": statistics.mean(sorted_vals) if sorted_vals else float("nan"),
        "p50_ms": percentile(sorted_vals, 50),
        "p90_ms": percentile(sorted_vals, 90),
        "p95_ms": percentile(sorted_vals, 95),
    }


def build_stage_breakdown(stage_samples_s: dict[str, List[float]], total_images: int, total_time_s: float) -> dict:
    stages: dict[str, dict] = {}
    total_time_ms = total_time_s * 1000.0

    for stage in STAGE_NAMES:
        values_ms = [v * 1000.0 for v in stage_samples_s[f"{stage}_s"]]
        item = summarize_stage(values_ms)
        item["share_pct"] = (item["total_ms"] / total_time_ms * 100.0) if total_time_ms > 0 else float("nan")
        item["per_image_ms"] = (item["total_ms"] / total_images) if total_images > 0 else float("nan")
        stages[stage] = item

    return stages


def build_results_from_batch_records(
    model_path: str,
    batch_records: List[dict],
    total_time_s: float,
    benchmark_mode: str,
    source_mode: str,
) -> dict:
    per_batch_ms = [record["total_s"] * 1000.0 for record in batch_records]
    per_image_ms = [record["total_s"] * 1000.0 / max(1, record["batch_size"]) for record in batch_records]
    total_images = sum(record["batch_size"] for record in batch_records)
    stage_samples_s = {f"{stage}_s": [record[f"{stage}_s"] for record in batch_records] for stage in STAGE_NAMES}

    batch_sorted = sorted(per_batch_ms)
    image_sorted = sorted(per_image_ms)
    throughput = total_images / total_time_s if total_time_s > 0 else float("inf")

    return {
        "model_path": model_path,
        "benchmark_mode": benchmark_mode,
        "source_mode": source_mode,
        "timed_batches": len(batch_records),
        "timed_images": total_images,
        "total_time_s": total_time_s,
        "throughput_img_s": throughput,
        "mean_batch_ms": statistics.mean(batch_sorted) if batch_sorted else float("nan"),
        "p50_batch_ms": percentile(batch_sorted, 50),
        "p90_batch_ms": percentile(batch_sorted, 90),
        "p95_batch_ms": percentile(batch_sorted, 95),
        "p99_batch_ms": percentile(batch_sorted, 99),
        "mean_img_ms": statistics.mean(image_sorted) if image_sorted else float("nan"),
        "p50_img_ms": percentile(image_sorted, 50),
        "p90_img_ms": percentile(image_sorted, 90),
        "p95_img_ms": percentile(image_sorted, 95),
        "p99_img_ms": percentile(image_sorted, 99),
        "stage_breakdown": build_stage_breakdown(stage_samples_s, total_images, total_time_s),
    }


def benchmark_predict_with_fixed_batches(
    model_path: str,
    warmup_batches: List[object],
    timed_batches: List[object],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    half: bool,
    task: str,
    batch: int,
    benchmark_mode: str,
    source_mode: str,
) -> Tuple[dict, YOLO]:
    """
    Bench Ultralytics pipeline on fixed batches.
    Returns metrics dict and model instance.
    """
    model = YOLO(model_path, task=task)

    with PredictorStageTimer(task) as stage_timer:
        stage_timer.reset_records()
        if benchmark_mode == "single_call":
            warmup_source = flatten_batches(warmup_batches) if source_mode == "paths" else BatchedMemorySource(warmup_batches)
            timed_source = flatten_batches(timed_batches) if source_mode == "paths" else BatchedMemorySource(timed_batches)
            if warmup_batches:
                warmup_iter = run_predict_batch(model, warmup_source, imgsz, conf, iou, device, half, batch=batch, stream=True)
                for _ in warmup_iter:
                    pass
                stage_timer.reset_records()

            t0 = time.perf_counter()
            pred_iter = run_predict_batch(model, timed_source, imgsz, conf, iou, device, half, batch=batch, stream=True)
            for _ in pred_iter:
                pass
            total_time_s = time.perf_counter() - t0
            batch_records = stage_timer.consume_records()
        else:
            for batch_source in warmup_batches:
                _ = run_predict_batch(model, batch_source, imgsz, conf, iou, device, half, batch=batch, stream=False)

            stage_timer.reset_records()
            t0 = time.perf_counter()
            for batch_source in timed_batches:
                _ = run_predict_batch(model, batch_source, imgsz, conf, iou, device, half, batch=batch, stream=False)
            total_time_s = time.perf_counter() - t0
            batch_records = stage_timer.consume_records()

    results = build_results_from_batch_records(model_path, batch_records, total_time_s, benchmark_mode, source_mode)

    return results, model


def print_single_results(label: str, r: dict) -> None:
    print(f"\n=== Ultralytics ONNX Pipeline Benchmark: {label} ===")
    print(f"Model:         {r['model_path']}")
    print(f"Benchmark:     {r['benchmark_mode']} | source={r['source_mode']}")
    print(f"Timed batches: {r['timed_batches']}")
    print(f"Timed images:  {r['timed_images']}")
    print(f"Total time:    {r['total_time_s']:.3f} s")
    print(f"Throughput:    {r['throughput_img_s']:.2f} images/s")

    print("\n--- Per-batch latency (pre + ORT inference + post) ---")
    print(f"mean: {r['mean_batch_ms']:8.3f} ms")
    print(f"p50:  {r['p50_batch_ms']:8.3f} ms")
    print(f"p90:  {r['p90_batch_ms']:8.3f} ms")
    print(f"p95:  {r['p95_batch_ms']:8.3f} ms")
    print(f"p99:  {r['p99_batch_ms']:8.3f} ms")

    print("\n--- Per-image latency (pre + ORT inference + post) ---")
    print(f"mean: {r['mean_img_ms']:8.3f} ms")
    print(f"p50:  {r['p50_img_ms']:8.3f} ms")
    print(f"p90:  {r['p90_img_ms']:8.3f} ms")
    print(f"p95:  {r['p95_img_ms']:8.3f} ms")
    print(f"p99:  {r['p99_img_ms']:8.3f} ms")


def print_timing_breakdown(label: str, r: dict) -> None:
    print(f"\n--- Timing breakdown: {label} ---")
    print("source_misc includes image loading/decoding and framework overhead outside explicit predictor stages.")
    for stage in STAGE_NAMES:
        item = r["stage_breakdown"][stage]
        print(
            f"{stage:>12}: total={item['total_ms']:9.3f} ms | share={item['share_pct']:6.2f}% | "
            f"per_img={item['per_image_ms']:8.3f} ms | mean_batch={item['mean_ms']:8.3f} ms | "
            f"p50={item['p50_ms']:8.3f} ms | p90={item['p90_ms']:8.3f} ms | p95={item['p95_ms']:8.3f} ms"
        )


def build_comparison(orig: dict, accel: dict) -> dict:
    return {
        "throughput": {
            "original_img_s": orig["throughput_img_s"],
            "accelerated_img_s": accel["throughput_img_s"],
            "speedup_x": accel["throughput_img_s"] / orig["throughput_img_s"] if orig["throughput_img_s"] != 0 else float("inf"),
            "gain_pct": pct_gain(orig["throughput_img_s"], accel["throughput_img_s"]),
        },
        "batch_latency_ms": {
            "mean": {
                "original": orig["mean_batch_ms"],
                "accelerated": accel["mean_batch_ms"],
                "speedup_x": speedup(orig["mean_batch_ms"], accel["mean_batch_ms"]),
                "reduction_pct": pct_reduction(orig["mean_batch_ms"], accel["mean_batch_ms"]),
            },
            "p50": {
                "original": orig["p50_batch_ms"],
                "accelerated": accel["p50_batch_ms"],
                "speedup_x": speedup(orig["p50_batch_ms"], accel["p50_batch_ms"]),
                "reduction_pct": pct_reduction(orig["p50_batch_ms"], accel["p50_batch_ms"]),
            },
            "p90": {
                "original": orig["p90_batch_ms"],
                "accelerated": accel["p90_batch_ms"],
                "speedup_x": speedup(orig["p90_batch_ms"], accel["p90_batch_ms"]),
                "reduction_pct": pct_reduction(orig["p90_batch_ms"], accel["p90_batch_ms"]),
            },
            "p95": {
                "original": orig["p95_batch_ms"],
                "accelerated": accel["p95_batch_ms"],
                "speedup_x": speedup(orig["p95_batch_ms"], accel["p95_batch_ms"]),
                "reduction_pct": pct_reduction(orig["p95_batch_ms"], accel["p95_batch_ms"]),
            },
            "p99": {
                "original": orig["p99_batch_ms"],
                "accelerated": accel["p99_batch_ms"],
                "speedup_x": speedup(orig["p99_batch_ms"], accel["p99_batch_ms"]),
                "reduction_pct": pct_reduction(orig["p99_batch_ms"], accel["p99_batch_ms"]),
            },
        },
        "image_latency_ms": {
            "mean": {
                "original": orig["mean_img_ms"],
                "accelerated": accel["mean_img_ms"],
                "speedup_x": speedup(orig["mean_img_ms"], accel["mean_img_ms"]),
                "reduction_pct": pct_reduction(orig["mean_img_ms"], accel["mean_img_ms"]),
            },
            "p50": {
                "original": orig["p50_img_ms"],
                "accelerated": accel["p50_img_ms"],
                "speedup_x": speedup(orig["p50_img_ms"], accel["p50_img_ms"]),
                "reduction_pct": pct_reduction(orig["p50_img_ms"], accel["p50_img_ms"]),
            },
            "p90": {
                "original": orig["p90_img_ms"],
                "accelerated": accel["p90_img_ms"],
                "speedup_x": speedup(orig["p90_img_ms"], accel["p90_img_ms"]),
                "reduction_pct": pct_reduction(orig["p90_img_ms"], accel["p90_img_ms"]),
            },
            "p95": {
                "original": orig["p95_img_ms"],
                "accelerated": accel["p95_img_ms"],
                "speedup_x": speedup(orig["p95_img_ms"], accel["p95_img_ms"]),
                "reduction_pct": pct_reduction(orig["p95_img_ms"], accel["p95_img_ms"]),
            },
            "p99": {
                "original": orig["p99_img_ms"],
                "accelerated": accel["p99_img_ms"],
                "speedup_x": speedup(orig["p99_img_ms"], accel["p99_img_ms"]),
                "reduction_pct": pct_reduction(orig["p99_img_ms"], accel["p99_img_ms"]),
            },
        },
    }


def format_comparison_text(orig: dict, accel: dict, comp: dict) -> str:
    lines = []
    lines.append("================ Performance improvement vs original ================")
    lines.append("Throughput:")
    lines.append(f"  Original:    {orig['throughput_img_s']:.2f} img/s")
    lines.append(f"  Accelerated: {accel['throughput_img_s']:.2f} img/s")
    lines.append(f"  Speedup:     {comp['throughput']['speedup_x']:.3f}x")
    lines.append(f"  Gain:        {comp['throughput']['gain_pct']:.2f}%")
    lines.append("")
    lines.append("Per-batch latency:")
    for k in ["mean", "p50", "p90", "p95", "p99"]:
        item = comp["batch_latency_ms"][k]
        lines.append(
            f"  {k}: {item['original']:.3f} ms -> {item['accelerated']:.3f} ms | "
            f"speedup={item['speedup_x']:.3f}x | reduction={item['reduction_pct']:.2f}%"
        )
    lines.append("")
    lines.append("Per-image latency:")
    for k in ["mean", "p50", "p90", "p95", "p99"]:
        item = comp["image_latency_ms"][k]
        lines.append(
            f"  {k}: {item['original']:.3f} ms -> {item['accelerated']:.3f} ms | "
            f"speedup={item['speedup_x']:.3f}x | reduction={item['reduction_pct']:.2f}%"
        )
    lines.append("====================================================================")
    return "\n".join(lines)


def write_results_file(path: str, orig: dict, accel: dict, comp: dict) -> None:
    report = format_comparison_text(orig, accel, comp)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\n")
        json.dump(
            {
                "original": orig,
                "accelerated": accel,
                "comparison": comp,
            },
            f,
            indent=2,
        )
        f.write("\n")


# ----------------------------
# Accuracy (Ultralytics val)
# ----------------------------
def run_ultralytics_val(
    model_path: str,
    data: Optional[str],
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    task: str,
) -> Tuple[object, str]:
    model = YOLO(model_path, task=task)
    kwargs = dict(imgsz=imgsz, batch=batch, device=device, conf=conf)
    if data:
        kwargs["data"] = data

    buf = io.StringIO()
    with redirect_stdout(buf):
        out = model.val(**kwargs)
    return out, buf.getvalue()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-accel", required=True, help="Path to accelerated ONNX model")
    ap.add_argument("--model-orig", required=True, help="Path to original ONNX model")
    ap.add_argument("--coco-images", required=True, help="COCO images directory or .txt list")
    ap.add_argument("--data", default=None, help="Ultralytics dataset config for val()")
    ap.add_argument("--task", default="detect", choices=["detect", "segment", "classify", "pose", "obb"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source-mode", default="paths", choices=["paths", "ndarray", "tensor"])
    ap.add_argument("--benchmark-mode", default="per_batch", choices=["per_batch", "single_call"])

    # Accuracy stage
    ap.add_argument(
        "--val-model",
        default="accel",
        choices=["orig", "accel", "both", "none"],
        help="Which model to run Ultralytics val() on after benchmark comparison",
    )

    # ORT options
    ap.add_argument("--custom-op", default=None, help="Path to ORT custom op .so to register")
    ap.add_argument("--enable-ort-profiling", action="store_true", help="Enable ONNXRuntime profiling for benchmark only")
    ap.add_argument("--ort-profile-dir", default="./ort_profiles", help="Where to write ORT profiling traces")
    ap.add_argument("--results-file", default="ultralytics_pipeline_performance_improvement.txt")

    args = ap.parse_args()

    images = list_images(args.coco_images)
    if not images:
        raise RuntimeError(f"No images found in: {args.coco_images}")

    warmup_path_batches, timed_path_batches = sample_batches(
        images=images,
        batch=args.batch,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
    )
    warmup_batches, timed_batches = prepare_sources(
        warmup_path_batches=warmup_path_batches,
        timed_path_batches=timed_path_batches,
        source_mode=args.source_mode,
        imgsz=args.imgsz,
    )

    print("Prepared shared benchmark batches:")
    print(f"  warmup batches: {len(warmup_batches)}")
    print(f"  timed batches:  {len(timed_batches)}")
    print(f"  timed images:   {count_total_images(timed_batches)}")
    print(f"  benchmark mode: {args.benchmark_mode}")
    print(f"  source mode:    {args.source_mode}")

    # ----------------------------
    # Benchmark original
    # ----------------------------
    patcher_orig = OrtSessionPatcher(
        custom_op_so=None,
        enable_profiling=args.enable_ort_profiling,
        profile_dir=args.ort_profile_dir,
        profile_prefix="orig_ultra_ort_profile",
    )
    patcher_orig.enable()
    try:
        orig_results, _ = benchmark_predict_with_fixed_batches(
            model_path=args.model_orig,
            warmup_batches=warmup_batches,
            timed_batches=timed_batches,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            task=args.task,
            batch=args.batch,
            benchmark_mode=args.benchmark_mode,
            source_mode=args.source_mode,
        )
        orig_profile_paths = patcher_orig.end_profiling()
    finally:
        patcher_orig.disable()

    print_single_results("original", orig_results)
    print_timing_breakdown("original", orig_results)

    # ----------------------------
    # Benchmark accelerated
    # ----------------------------
    patcher_accel = OrtSessionPatcher(
        custom_op_so=args.custom_op,
        enable_profiling=args.enable_ort_profiling,
        profile_dir=args.ort_profile_dir,
        profile_prefix="accel_ultra_ort_profile",
    )
    patcher_accel.enable()
    try:
        accel_results, _ = benchmark_predict_with_fixed_batches(
            model_path=args.model_accel,
            warmup_batches=warmup_batches,
            timed_batches=timed_batches,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            task=args.task,
            batch=args.batch,
            benchmark_mode=args.benchmark_mode,
            source_mode=args.source_mode,
        )
        accel_profile_paths = patcher_accel.end_profiling()
    finally:
        patcher_accel.disable()

    print_single_results("accelerated", accel_results)
    print_timing_breakdown("accelerated", accel_results)

    # ----------------------------
    # Comparison + save file
    # ----------------------------
    comparison = build_comparison(orig_results, accel_results)
    report_text = format_comparison_text(orig_results, accel_results, comparison)

    print("\n" + report_text + "\n")
    write_results_file(args.results_file, orig_results, accel_results, comparison)
    print(f"Performance report written to: {os.path.abspath(args.results_file)}")

    if args.enable_ort_profiling:
        print("\nProfiling files:")
        for p in orig_profile_paths:
            print(f"  original:    {p}")
        for p in accel_profile_paths:
            print(f"  accelerated: {p}")

    # ----------------------------
    # Accuracy afterward
    # ----------------------------
    if args.val_model != "none":
        print("\n=== Ultralytics COCO Eval (model.val) ===")

    if args.val_model in ("orig", "both"):
        print("\n[INFO] Running model.val() on ORIGINAL model ...")
        try:
            val_ret, val_stdout = run_ultralytics_val(
                model_path=args.model_orig,
                data=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                conf=args.conf,
                task=args.task,
            )
            if val_stdout.strip():
                print(val_stdout.strip())
            else:
                print("[WARN] val() produced no stdout in this environment.")
            print("\n[INFO] ORIGINAL val() returned:")
            print(repr(val_ret))
        except Exception as e:
            print(f"[ERROR] ORIGINAL model.val() failed: {e}")

    if args.val_model in ("accel", "both"):
        print("\n[INFO] Running model.val() on ACCELERATED model ...")
        # Re-enable patcher only here if the accelerated model requires the custom op for val()
        patcher_val = OrtSessionPatcher(
            custom_op_so=args.custom_op,
            enable_profiling=False,  # do not profile val()
            profile_dir=args.ort_profile_dir,
            profile_prefix="accel_val_unused",
        )
        patcher_val.enable()
        try:
            val_ret, val_stdout = run_ultralytics_val(
                model_path=args.model_accel,
                data=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                conf=args.conf,
                task=args.task,
            )
            if val_stdout.strip():
                print(val_stdout.strip())
            else:
                print("[WARN] val() produced no stdout in this environment.")
            print("\n[INFO] ACCELERATED val() returned:")
            print(repr(val_ret))
        except Exception as e:
            print(f"[ERROR] ACCELERATED model.val() failed: {e}")
            print("If `--data` is a COCO instances JSON and your Ultralytics version expects a data.yaml, "
                  "provide a COCO data.yaml instead (e.g., coco.yaml).")
        finally:
            patcher_val.disable()


if __name__ == "__main__":
    main()
