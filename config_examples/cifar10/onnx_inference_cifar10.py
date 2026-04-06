import time
import os
import torch
import torchvision
import numpy as np
import onnxruntime as ort
import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper

# MODEL_PATH = "/workspace/NN2FPGA/models/onnx/resnet8_a8w8b16.onnx"
MODEL_PATH = "/workspace/NN2FPGA/work/resnet8/original_model_qcdq_for_sim.onnx"

def cifar10_dataloader(batch_size: int, sample_size: int):
    cifar10_directory = "/home/datasets/cifar10"

    if not os.path.exists(cifar10_directory):
        raise FileNotFoundError(f"CIFAR-10 dataset directory not found: {cifar10_directory}")

    test_data = torchvision.datasets.CIFAR10(
        root=cifar10_directory,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            lambda x: x.float(),
        ]),
    )

    if sample_size is not None and sample_size > 0:
        sample_size = min(sample_size, len(test_data))
        test_data = torch.utils.data.Subset(test_data, range(sample_size))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    return test_loader

def postprocess(outputs, labels: torch.Tensor) -> int:
    logits = np.asarray(outputs[0])
    labels_np = labels.cpu().numpy().reshape(-1)

    #print("logits shape:", logits.shape)
    #print("labels shape:", labels_np.shape)

    # Handle common classifier output shapes
    # e.g. (B,10), (B,1,1,10), (B,10,1,1)
    if logits.ndim == 4 and logits.shape[-1] == 10:
        logits = logits.reshape(logits.shape[0], 10)
    elif logits.ndim == 4 and logits.shape[1] == 10:
        logits = logits.reshape(logits.shape[0], 10)
    elif logits.ndim > 2:
        logits = logits.reshape(logits.shape[0], -1)

    predicted = np.argmax(logits, axis=1).reshape(-1)

    #print("predicted shape:", predicted.shape)
    #print("predicted:", predicted)
    #print("labels    :", labels_np)

    correct = int(np.sum(predicted == labels_np))
    return correct


def report_error_stats(
    output_name: str,
    expected_output: np.ndarray,
    produced_output: np.ndarray,
    top_k: int = 10,
):
    """
    Report statistics about the error between expected and produced outputs.
    """
    error = np.abs(expected_output - produced_output)

    max_error = np.max(error)
    mean_error = np.mean(error)
    min_error = np.min(error)
    std_error = np.std(error)

    flat_error = error.flatten()
    flat_idx = np.argsort(-flat_error)
    topk_idx = flat_idx[:top_k]
    unraveled_idx = [np.unravel_index(i, error.shape) for i in topk_idx]

    print("=" * 50)
    print(f"Output: {output_name}")
    print(f"Expected Output (first 10 elements): {expected_output.flatten()[:10]}")
    print(f"Produced Output (first 10 elements): {produced_output.flatten()[:10]}")
    print(f"Max Error: {max_error}")
    print(f"Min Error: {min_error}")
    print(f"Mean Error: {mean_error}")
    print(f"Std Dev of Error: {std_error}")
    if max_error == 0:
        print("No errors detected.")
        print("=" * 50)
        return

    print(f"Top {top_k} errors:")
    for rank, (idx, val) in enumerate(zip(unraveled_idx, flat_error[topk_idx]), 1):
        exp_val = expected_output[idx]
        prod_val = produced_output[idx]
        print(f" {rank:2d}. idx={idx}, error={val}, expected={exp_val}, produced={prod_val}")
    print("=" * 50)


def main():
    # Paths
    #model_path = "nn2FPGA_resnet8.onnx"
    # model_path = "original_model_qcdq_for_sim.onnx"
    custom_op_so = os.path.abspath("libnn2fpga_customop.so")

    # Config
    batch_size = 1
    sample_size = 10000  # set to 10000 for full CIFAR-10 test set

    # Session options
    so = ort.SessionOptions()
    # so.register_custom_ops_library(custom_op_so)
    #so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.enable_profiling = True

    # Create session
    sess = ort.InferenceSession(
        MODEL_PATH,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )

    input_name = sess.get_inputs()[0].name
    output_names = [out.name for out in sess.get_outputs()]

    # print(f"Model input name: {input_name}")
    # print(f"Model output names: {output_names}")
    # model = ModelWrapper(MODEL_PATH)

    dataloader = cifar10_dataloader(batch_size=batch_size, sample_size=sample_size)

    total_correct = 0
    total_samples = 0
    batch_times = []

    for batch_idx, (features, labels) in enumerate(dataloader):
        np_features = features.cpu().numpy().astype(np.float32)
        input_data = {"global_in": np_features}

        t1 = time.time()
        outputs = sess.run(None, input_data)
        # outputs = oxe.execute_onnx(model, input_data)
        # outputs = [outputs["global_out"]]
        t2 = time.time()

        elapsed = t2 - t1
        batch_times.append(elapsed)

        batch_correct = postprocess(outputs, labels)
        batch_total = labels.shape[0]

        total_correct += batch_correct
        total_samples += batch_total

        batch_accuracy = batch_correct / batch_total

        #print(
        #    f"Batch {batch_idx:4d} | "
        #    f"time: {elapsed:.6f} s | "
        #    f"correct: {batch_correct}/{batch_total} | "
        #    f"batch acc: {batch_accuracy:.4f}"
        #)

    total_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    total_time = sum(batch_times)
    avg_batch_time = total_time / len(batch_times) if batch_times else 0.0
    throughput = total_samples / total_time if total_time > 0 else 0.0

    print("\n===== Final Results =====")
    print(f"Total samples:        {total_samples}")
    print(f"Total correct:        {total_correct}")
    print(f"Accuracy:             {total_accuracy:.4f}")
    print(f"Total inference time: {total_time:.6f} s")
    print(f"Average batch time:   {avg_batch_time:.6f} s")
    print(f"Throughput:           {throughput:.2f} images/s")

    # prof_file = sess.end_profiling()
    # print(f"Profiling trace written to: {prof_file}")


if __name__ == "__main__":
    main()

