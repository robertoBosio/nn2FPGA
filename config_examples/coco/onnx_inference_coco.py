import os
import time
import numpy as np
import onnxruntime as ort
import torch
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
        shuffle=True,
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


def check_correctness_three_outputs(
    sess_opt: ort.InferenceSession,
    sess_orig: ort.InferenceSession,
    input_name: str,
    x: np.ndarray,
):
    print("Warming up and checking correctness (3 outputs)...")
    actual = sess_opt.run(None, {input_name: x})
    expected = sess_orig.run(None, {input_name: x})

    if len(expected) < 3 or len(actual) < 3:
        raise RuntimeError(
            f"Expected at least 3 outputs from both models, got "
            f"orig={len(expected)}, opt={len(actual)}"
        )

    report_error_stats("global_out_0", np.asarray(expected[0]).flatten(), np.asarray(actual[0]).flatten())
    report_error_stats("global_out_1", np.asarray(expected[1]).flatten(), np.asarray(actual[1]).flatten())
    report_error_stats("global_out_2", np.asarray(expected[2]).flatten(), np.asarray(actual[2]).flatten())


# -----------------------------
# Benchmarking
# -----------------------------
def percentile(values_s: list[float], p: float) -> float:
    if not values_s:
        return float("nan")
    return float(np.percentile(np.array(values_s, dtype=np.float64) * 1e3, p))  # ms


def benchmark(
    sess: ort.InferenceSession,
    dataloader: DataLoader,
    input_name: str,
    warmup_batches: int = 5,
    measure_batches: int | None = None,
):
    # Warmup
    w = 0
    for features in dataloader:
        np_features = features.numpy().astype(np.float32)
        _ = sess.run(None, {input_name: np_features})
        w += 1
        if w >= warmup_batches:
            break

    # Timed runs
    batch_lat_s: list[float] = []
    img_lat_s: list[float] = []
    total_images = 0
    total_time_s = 0.0

    b = 0
    for features in dataloader:
        if measure_batches is not None and b >= measure_batches:
            break

        np_features = features.numpy().astype(np.float32)
        bs = int(np_features.shape[0])

        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: np_features})
        t1 = time.perf_counter()

        dt = t1 - t0
        batch_lat_s.append(dt)
        img_lat_s.append(dt / bs)

        total_images += bs
        total_time_s += dt
        b += 1

    if total_images == 0:
        raise RuntimeError("No batches measured. Check sample_size / dataloader.")

    throughput = total_images / total_time_s
    avg_batch_ms = (sum(batch_lat_s) / len(batch_lat_s)) * 1e3
    avg_img_ms = (sum(img_lat_s) / len(img_lat_s)) * 1e3

    print("\n===== Benchmark results =====")
    print(f"Measured batches:        {len(batch_lat_s)}")
    print(f"Measured images:         {total_images}")
    print(f"Total measured time (s): {total_time_s:.6f}")
    print(f"Throughput (img/s):      {throughput:.2f}")
    print(f"Avg batch latency (ms):  {avg_batch_ms:.3f}")
    print(f"Avg img latency (ms):    {avg_img_ms:.3f}")

    print("\nLatency percentiles (per-batch, ms):")
    print(f"  p50: {percentile(batch_lat_s, 50):.3f}")
    print(f"  p90: {percentile(batch_lat_s, 90):.3f}")
    print(f"  p95: {percentile(batch_lat_s, 95):.3f}")

    print("\nLatency percentiles (per-image, ms):")
    print(f"  p50: {percentile(img_lat_s, 50):.3f}")
    print(f"  p90: {percentile(img_lat_s, 90):.3f}")
    print(f"  p95: {percentile(img_lat_s, 95):.3f}")
    print("=============================\n")


# -----------------------------
# Main
# -----------------------------
def main():
    MODEL_PATH = "nn2FPGA_yolov5n.onnx"
    ORIGINAL_MODEL_PATH = "original_model_qcdq.onnx"
    CUSTOM_OP_SO = os.path.abspath("libnn2fpga_customop.so")

    # Session options
    so = ort.SessionOptions()
    print("Loading the operator:", CUSTOM_OP_SO)
    so.register_custom_ops_library(CUSTOM_OP_SO)

    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = True

    print("Starting sessions...")
    sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    sess_orig = ort.InferenceSession(ORIGINAL_MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])

    # Input name (use original model as source of truth)
    input_name = sess_orig.get_inputs()[0].name

    # -----------------------------
    # Correctness check (3 outputs)
    # -----------------------------
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    check_correctness_three_outputs(sess, sess_orig, input_name, x)

    # -----------------------------
    # Benchmark on COCO subset
    # -----------------------------
    batch_size = 1
    sample_size = 10          # <-- USED
    warmup_batches = 5
    measure_batches = None    # set to an int to cap measured batches

    dataloader = coco_dataloader(batch_size=batch_size, sample_size=sample_size, num_workers=0)
    benchmark(sess, dataloader, input_name, warmup_batches=warmup_batches, measure_batches=measure_batches)

    # End profiling (optimized model)
    prof_file = sess.end_profiling()
    print(f"Profiling trace written to: {prof_file}")

    # If you also want profiling for original model, enable it and call:
    # prof_file_orig = sess_orig.end_profiling()
    # print(f"Original profiling trace written to: {prof_file_orig}")


if __name__ == "__main__":
    main()
