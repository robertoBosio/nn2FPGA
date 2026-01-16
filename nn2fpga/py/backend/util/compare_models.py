import numpy as np
import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto
import logging

logger = logging.getLogger(__name__)

def generate_random_input(model: ModelWrapper) -> dict:
    """
    Generate random input data for the ONNX model based on its input shapes and data types.
    Args:
        model (ModelWrapper): The ONNX model wrapped in QONNX ModelWrapper.
    Returns:
        dict: A dictionary where keys are input names and values are numpy arrays of random data.
    """
    np.random.seed(0)  # For reproducibility
    input_dict = {}
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
        print(f"Generating random input for {inp.name} with shape {shape}")
        dtype = inp.type.tensor_type.elem_type
        np_dtype = {
            TensorProto.FLOAT: np.float32,
            TensorProto.UINT8: np.uint8,
            TensorProto.INT8: np.int8,
            TensorProto.INT32: np.int32,
        }.get(dtype, np.float32)
        input_dict[inp.name] = np.random.randn(*shape).astype(np_dtype)
    return input_dict

def get_output_names(model: ModelWrapper) -> list[str]:
    """
    Get the names of the outputs of the ONNX model.
    Args:
        model (ModelWrapper): The ONNX model wrapped in QONNX ModelWrapper.
    Returns:
        list[str]: A list of output names.
    """
    return [out.name for out in model.graph.output]

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

    logger.info("=" * 50)
    logger.info(f"Output: {output_name}")
    logger.info(f"Expected Output (first 10 elements): {expected_output.flatten()[:10]}")
    logger.info(f"Produced Output (first 10 elements): {produced_output.flatten()[:10]}")
    logger.info(f"Max Error: {max_error}")
    logger.info(f"Min Error: {min_error}")
    logger.info(f"Mean Error: {mean_error}")
    logger.info(f"Std Dev of Error: {std_error}")
    if max_error == 0:
        logger.info("No errors detected.")
        return
    logger.info(f"Top {top_k} errors:")
    for rank, (idx, val) in enumerate(zip(unraveled_idx, flat_error[topk_idx]), 1):
        exp_val = expected_output[idx]
        prod_val = produced_output[idx]
        logger.info(f" {rank:2d}. idx={idx}, error={val}, expected={exp_val}, produced={prod_val}")
    logger.info("=" * 50)

def test_transformation_equivalence(model_pre: ModelWrapper, model_post: ModelWrapper):
    """
    Test if the outputs of two ONNX models are equivalent given the same random input.
    Args:
        model_pre (ModelWrapper): The original ONNX model before transformation.
        model_post (ModelWrapper): The transformed ONNX model after transformation.
    """
    input_dict = generate_random_input(model_pre)
    output_names = get_output_names(model_pre)

    out_expected = oxe.execute_onnx(model_pre, input_dict, return_full_exec_context=True)
    out_produced = oxe.execute_onnx(model_post, input_dict, return_full_exec_context=True)

    for name in output_names:
        flattened_expected = out_expected[name].flatten()
        flattened_produced = out_produced[name].flatten()
        assert name in out_expected and name in out_produced, f"Missing output: {name}"
        assert flattened_expected.shape == flattened_produced.shape, f"Shape mismatch for: {name}"
        report_error_stats(name, flattened_expected, flattened_produced)
