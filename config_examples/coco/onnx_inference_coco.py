import os
import time
import json
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
# Main
# -----------------------------
def main():
    MODEL_PATH = "nn2FPGA_yolov5nu.onnx"
    ORIGINAL_MODEL_PATH = "original_model_qcdq.onnx"
    CUSTOM_OP_SO = os.path.abspath("libnn2fpga_customop.so")
    RESULTS_FILE = "performance_improvement.txt"

    batch_size = 1
    sample_size = 10
    warmup_batches = 5
    measure_batches = None

    # -----------------------------
    # Session for correctness only
    # Profiling disabled here on purpose
    # -----------------------------
    so_no_prof = ort.SessionOptions()
    print("Loading the operator:", CUSTOM_OP_SO)
    so_no_prof.register_custom_ops_library(CUSTOM_OP_SO)
    so_no_prof.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so_no_prof.enable_profiling = False

    print("Starting correctness-check sessions...")
    sess_opt_check = ort.InferenceSession(MODEL_PATH, sess_options=so_no_prof, providers=["CPUExecutionProvider"])
    sess_orig_check = ort.InferenceSession(ORIGINAL_MODEL_PATH, sess_options=so_no_prof, providers=["CPUExecutionProvider"])

    input_name = sess_orig_check.get_inputs()[0].name

    # Correctness check
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    check_correctness_single_output(sess_opt_check, sess_orig_check, input_name, x)

    # Explicitly release correctness-only sessions
    del sess_opt_check
    del sess_orig_check

    # -----------------------------
    # Prepare benchmark data
    # -----------------------------
    dataloader = coco_dataloader(
        batch_size=batch_size,
        sample_size=sample_size,
        num_workers=0,
    )
    batches = preload_batches(dataloader, measure_batches=measure_batches)

    # -----------------------------
    # Sessions for benchmark only
    # Profiling enabled here so traces only include actual benchmark work
    # -----------------------------
    so_prof = ort.SessionOptions()
    so_prof.register_custom_ops_library(CUSTOM_OP_SO)
    so_prof.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so_prof.enable_profiling = True

    print("Starting benchmark sessions...")
    sess_orig = ort.InferenceSession(ORIGINAL_MODEL_PATH, sess_options=so_prof, providers=["CPUExecutionProvider"])
    sess_opt = ort.InferenceSession(MODEL_PATH, sess_options=so_prof, providers=["CPUExecutionProvider"])

    # Warmup both models
    warmup_session(sess_orig, input_name, batches, warmup_batches=warmup_batches)
    warmup_session(sess_opt, input_name, batches, warmup_batches=warmup_batches)

    # Benchmark both
    orig_results = benchmark_preloaded(sess_orig, batches, input_name, label="original")
    opt_results = benchmark_preloaded(sess_opt, batches, input_name, label="optimized")

    # Comparison
    comparison = build_comparison(orig_results, opt_results)
    report_text = format_comparison_text(orig_results, opt_results, comparison)
    print("\n" + report_text + "\n")

    # Write report file
    write_results_file(RESULTS_FILE, orig_results, opt_results, comparison)
    print(f"Performance report written to: {os.path.abspath(RESULTS_FILE)}")

    # End profiling
    prof_file_orig = sess_orig.end_profiling()
    print(f"Original profiling trace written to: {prof_file_orig}")

    prof_file_opt = sess_opt.end_profiling()
    print(f"Optimized profiling trace written to: {prof_file_opt}")


if __name__ == "__main__":
    main()