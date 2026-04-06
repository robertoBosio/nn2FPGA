import time
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np
from PIL import Image

class ImageNet(Dataset):
    def __init__(self, root, train, transform=None, sample_size=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        items = os.listdir(root + "/train/")
        sorted_items = sorted(items)
        for class_id, syn_id in enumerate(sorted_items):
            self.syn_to_class[syn_id] = class_id

        if train:
            image_path = root + "/train/"
        else:
            image_path = root + "/val/"
        items = os.listdir(image_path)
        sorted_items = sorted(items)
        for syn_id in sorted_items:
            syn_folder = os.path.join(image_path, syn_id)
            class_id = self.syn_to_class[syn_id]
            for sample in os.listdir(syn_folder):
                sample_path = os.path.join(syn_folder, sample)
                self.samples.append(sample_path)
                self.targets.append(class_id)
        
        if sample_size is not None:
            # Randomly sample a subset of the dataset and targets
            assert len(self.samples) == len(self.targets)
            indices = np.random.choice(len(self.samples), sample_size, replace=False)
            self.samples = [self.samples[i] for i in indices]
            self.targets = [self.targets[i] for i in indices]

    def __len__(self):
            return len(self.samples)
    
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            x = self.transform(x)
            return x, self.targets[idx]

def imagenet_dataloader(batch_size, sample_size=None):
    IMAGENET_DIRECTORY = '/home/datasets/Imagenet/'

    if not os.path.exists(IMAGENET_DIRECTORY):
        print("IMAGENET Dataset not present")
        exit(0)

    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    args = {
        'train': False,
        'transform': transform,
        'root': IMAGENET_DIRECTORY,
        'sample_size': sample_size
    }

    dataset = ImageNet(**args)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    return test_loader

def postprocess(out_buffer, results, accuracy, batch_size):
    predicted = np.argmax(np.asarray(out_buffer[:]), axis=-1)
    accuracy_batch = np.equal(predicted, results)
    accuracy_batch = accuracy_batch.sum()
    accuracy += accuracy_batch
    return accuracy

def report_error_stats(output_name: str, expected_output: np.ndarray, produced_output: np.ndarray, top_k: int = 10):
    """
    Report statistics about the error between expected and produced outputs.

    Args:
        output_name (str): The name of the output tensor.
        expected_output (np.ndarray): The expected output from the original model.
        produced_output (np.ndarray): The output from the transformed model.
        top_k (int): Number of largest errors to report.
    """
    error = np.abs(expected_output - produced_output)

    max_error = np.max(error)
    mean_error = np.mean(error)
    min_error = np.min(error)
    std_error = np.std(error)

    # flatten for sorting
    flat_error = error.flatten()
    flat_idx = np.argsort(-flat_error)  # descending
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
        return
    print(f"Top {top_k} errors:")
    for rank, (idx, val) in enumerate(zip(unraveled_idx, flat_error[topk_idx]), 1):
        exp_val = expected_output[idx]
        prod_val = produced_output[idx]
        print(f" {rank:2d}. idx={idx}, error={val}, expected={exp_val}, produced={prod_val}")
    print("=" * 50)

# Paths
MODEL_PATH = "nn2FPGA_mobilenet_v2.onnx"
ORIGINAL_MODEL_PATH = "original_model_qcdq.onnx"         
CUSTOM_OP_SO = os.path.abspath("libnn2fpga_customop.so")

# Session options
so = ort.SessionOptions()
so.register_custom_ops_library(CUSTOM_OP_SO)

# Enable optimizations
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Enable profiling
so.enable_profiling = True

# Create session
sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])

# Dummy input data (adapt dtype/shape to your model)
input_name = sess.get_inputs()[0].name
input_shape = [d if isinstance(d, int) else 1 for d in sess.get_inputs()[0].shape]
# x = np.random.rand(10, 3, 224, 224).astype(np.float32)

# # Warmup
# print("Warming up and checking correctness")
# actual_result = sess.run(None, {input_name: x})
# expected_result = sess_orig.run(None, {input_name: x})
# # Check correctness
# report_error_stats("output", np.asarray(expected_result).flatten(), np.asarray(actual_result).flatten())
# del sess, sess_orig


# sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])

# Profile multiple runs
dataloader = imagenet_dataloader(batch_size=1, sample_size=100)  # Only 100 samples
accuracy = 0
for batch, (features, expected_results) in enumerate(dataloader):
    np_features = (np.asarray(features).astype(np.float32))
    input_data = {input_name: np_features}
    
    # Run inference
    t1 = time.time()
    actual_result = sess.run(None, input_data)
    t2 = time.time()
    accuracy = postprocess(actual_result, expected_results, accuracy, 1)
    print(f"Batch {batch} processed in {t2 - t1:.3f} seconds")


print(f"Total accuracy: {accuracy / (len(dataloader) * 1):.2f}, which means {accuracy} correct predictions out of {len(dataloader) * 1} total predictions.")
# Close the session to flush the profiling file
prof_file = sess.end_profiling()
print(f"Profiling trace written to: {prof_file}")

