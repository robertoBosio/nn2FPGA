from qonnx.util.basic import get_by_name
from qonnx.core.modelwrapper import ModelWrapper
from onnx import numpy_helper

def check_act_quant(model: ModelWrapper): 
    """ Check quantization of the model activations. Right now, it is only supported per tensor quantization, 
    with full range of values (narrow=0).
    Args:
        model (ModelWrapper): The ONNX model to check, wrapped in QONNX ModelWrapper.
    Returns:
        bool: True if the model has a supported quantization scheme, False otherwise.
    """

    graph = model.graph
    init_dict = {init.name: init for init in graph.initializer}
    supported = True

    for node in graph.node:
        if node.op_type != "IntQuant" and node.op_type != "Quant":
            continue

        # Check if node has initializers. If so, it isn't an activation Quant node.
        input_name = node.input[0]
        # print(f"Checking node {node.name} for activation quantization.")
        if input_name in init_dict:
            # print(f"Skipping node {node.name} as it is not an activation quantization node (input is an initializer).")
            continue

        # Get scale and zero_point initializers
        scale_name = node.input[1]
        zeropt_name = node.input[2]
        scale = numpy_helper.to_array(get_by_name(graph.initializer, scale_name))
        zeropt = numpy_helper.to_array(get_by_name(graph.initializer, zeropt_name))

        # Skip if we can't find the initializers
        if scale is None or zeropt is None:
            raise ValueError(f"Could not find scale or zero_point for node {node.name}. Ensure they are defined in the model.")

        
        # Check if per-channel (length > 1)
        if scale.ndim > 1 or zeropt.ndim > 1:
            print(f"Error: per channel quantization is not supported for activations. Node {node.name} has scale with shape {scale.shape}") 
            supported = False

    return supported

def check_input_same_quant(model: ModelWrapper):
    """ Check if the input quantizations of element-wise operations and Concat operations are the same.
    Args:
        model (ModelWrapper): The ONNX model to check, wrapped in QONNX ModelWrapper.
    Returns:
        bool: True if the input quantizations of element-wise operations are the same, False otherwise.
    """

    elementwise_ops = ["Add", "Sub", "Mul", "Div", "Max", "Min"] 
    concat_like_ops = ["Concat"]

    graph = model.graph
    init_dict = {init.name: init for init in graph.initializer}
    supported = True
    for node in graph.node:
        if node.op_type not in elementwise_ops + concat_like_ops:
            continue

        # Check if node has initializers. If so, it isn't an activation Quant node.
        input_names = node.input

        # Get the quantization parameters for each input
        scales = []
        zeropts = []
        for input_name in input_names:
            # Retrieve Quant node
            quant_node = model.find_producer(input_name)
            if quant_node is None or quant_node.op_type not in ["IntQuant", "Quant"]:
                raise ValueError(f"Input {input_name} of node {node.name} is not quantized.")

            # Get scale and zero_point initializers
            scale_name = quant_node.input[1]
            zeropt_name = quant_node.input[2]
            scale = numpy_helper.to_array(get_by_name(graph.initializer, scale_name))
            zeropt = numpy_helper.to_array(get_by_name(graph.initializer, zeropt_name))
            scales.append(scale)
            zeropts.append(zeropt)

        # Check if all scales and zero points are the same
        if not all((s == scales[0]).all() for s in scales) or not all((z == zeropts[0]).all() for z in zeropts):
            print(f"Error: Input quantizations of node {node.name} are not the same.")
            supported = False
    
    return supported

def check_quantization(model: ModelWrapper):
    """ Check that the quantization of the model is compatible with the backend.
    Args:
        model (ModelWrapper): The ONNX model to check, wrapped in QONNX ModelWrapper.
    Returns:
        bool: True if the model has a supported quantization scheme, False otherwise.
    """

    compatible_quantization = True
    compatible_quantization = compatible_quantization and check_act_quant(model)
    compatible_quantization = compatible_quantization and check_input_same_quant(model)
    return compatible_quantization
