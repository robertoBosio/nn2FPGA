import os
import time
import json
import argparse
import sys
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


def outputs_exactly_match(expected: list[np.ndarray], produced: list[np.ndarray], image_idx: int) -> bool:
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

        if not np.array_equal(expected_arr, produced_arr):
            print(f"Image {image_idx}, output {output_idx}: FAIL - values differ")
            report_error_stats(
                f"image_{image_idx}_out_{output_idx}",
                expected_arr.flatten(),
                produced_arr.flatten(),
            )
            image_ok = False

    if image_ok:
        print(f"Image {image_idx}: PASS")

    return image_ok


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
        raise RuntimeError("No batches loaded. Check sample_size / dataloader.")

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
):
    batch_lat_s: list[float] = []
    img_lat_s: list[float] = []
    total_images = 0
    total_time_s = 0.0

    for np_features in batches:
        bs = int(np_features.shape[0])

        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: np_features})
        t1 = time.perf_counter()

        dt = t1 - t0
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


# -----------------------------
# Modes / CLI
# -----------------------------
def run_correctness(args) -> int:
    custom_op_so = os.path.abspath(args.custom_op)
    print("Loading the operator:", custom_op_so)
    print("Starting exact correctness sessions with ORT optimizations disabled...")

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

    checked_images = 0
    failed_images = 0

    for image_idx, features in enumerate(dataloader):
        x = features.numpy().astype(np.float32)
        expected = sess_orig.run(None, {input_name: x})
        produced = sess_opt.run(None, {input_name: x})

        checked_images += 1
        if not outputs_exactly_match(expected, produced, image_idx):
            failed_images += 1

    passed_images = checked_images - failed_images
    print("\n===== Exact correctness summary =====")
    print(f"Checked images: {checked_images}")
    print(f"Passed images:  {passed_images}")
    print(f"Failed images:  {failed_images}")
    print("=====================================")

    return 0 if failed_images == 0 else 1


def run_speed(args) -> int:
    custom_op_so = os.path.abspath(args.custom_op)
    print("Loading the operator:", custom_op_so)

    dataloader = coco_dataloader(
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        num_workers=args.num_workers,
    )
    batches = preload_batches(dataloader, measure_batches=args.measure_batches)

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
    opt_results = benchmark_preloaded(sess_opt, batches, input_name, label="optimized")

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
        help="Number of COCO images to check in correctness mode.",
    )

    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for speed mode.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of COCO images to preload for speed mode. Use -1 for all images.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Number of warmup batches for speed mode.",
    )
    parser.add_argument(
        "--measure-batches",
        type=int,
        default=None,
        help="Number of preloaded batches to benchmark in speed mode.",
    )
    parser.add_argument("--results-file", default="performance_improvement.txt")

    args = parser.parse_args()
    if args.num_images <= 0:
        parser.error("--num-images must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.sample_size == -1:
        args.sample_size = None
    elif args.sample_size is not None and args.sample_size <= 0:
        parser.error("--sample-size must be > 0 or -1")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must be >= 0")
    if args.measure_batches is not None and args.measure_batches <= 0:
        parser.error("--measure-batches must be > 0")

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
