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
from typing import List, Optional, Tuple

from ultralytics import YOLO


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


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


# ----------------------------
# Benchmark
# ----------------------------
def run_predict_batch(
    model: YOLO,
    batch_paths: List[str],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    half: bool,
):
    return model.predict(
        source=batch_paths,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        half=half,
        verbose=False,
        stream=False,
    )


def benchmark_predict_with_fixed_batches(
    model_path: str,
    warmup_batches: List[List[str]],
    timed_batches: List[List[str]],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    half: bool,
    task: str,
) -> Tuple[dict, YOLO]:
    """
    Bench Ultralytics pipeline on fixed batches.
    Returns metrics dict and model instance.
    """
    model = YOLO(model_path, task=task)

    # Warmup
    for batch_paths in warmup_batches:
        run_predict_batch(model, batch_paths, imgsz, conf, iou, device, half)

    # Timed
    per_image_ms: List[float] = []
    per_batch_ms: List[float] = []
    total_images = 0

    t0 = time.perf_counter()
    for batch_paths in timed_batches:
        start = time.perf_counter()
        _ = run_predict_batch(model, batch_paths, imgsz, conf, iou, device, half)
        end = time.perf_counter()

        dt_s = end - start
        dt_ms = dt_s * 1000.0
        bsz = len(batch_paths)

        total_images += bsz
        per_batch_ms.append(dt_ms)
        per_image_ms.append(dt_ms / max(1, bsz))

    t1 = time.perf_counter()
    total_time_s = t1 - t0

    batch_sorted = sorted(per_batch_ms)
    image_sorted = sorted(per_image_ms)
    throughput = total_images / total_time_s if total_time_s > 0 else float("inf")

    results = {
        "model_path": model_path,
        "timed_batches": len(timed_batches),
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
    }

    return results, model


def print_single_results(label: str, r: dict) -> None:
    print(f"\n=== Ultralytics ONNX Pipeline Benchmark: {label} ===")
    print(f"Model:         {r['model_path']}")
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

    warmup_batches, timed_batches = sample_batches(
        images=images,
        batch=args.batch,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
    )

    print("Prepared shared benchmark batches:")
    print(f"  warmup batches: {len(warmup_batches)}")
    print(f"  timed batches:  {len(timed_batches)}")
    print(f"  timed images:   {sum(len(b) for b in timed_batches)}")

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
        )
        orig_profile_paths = patcher_orig.end_profiling()
    finally:
        patcher_orig.disable()

    print_single_results("original", orig_results)

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
        )
        accel_profile_paths = patcher_accel.end_profiling()
    finally:
        patcher_accel.disable()

    print_single_results("accelerated", accel_results)

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