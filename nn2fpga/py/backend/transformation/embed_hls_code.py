from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from backend.util.codegen_utils import cpp_function, cpp_variable, NewCodeWriter
from backend.core.acceleratorpackage import AcceleratorPackage
from backend.core.tensor_fifo import TensorFifo, get_custom_tensor_fifo_metadata
from onnx import NodeProto
import base64
import os
import re
import numpy as np
import logging
logger = logging.getLogger(__name__)

def generate_hls_code(model: ModelWrapper) -> str:

    """Generate HLS code for the given model.
    Args:
        model (ModelWrapper): The model to generate HLS code for.
    Returns:
        str: The generated HLS code as a string.
    """
    cwr = NewCodeWriter()
    cwr.add_autogen_comment()

    # Include sections for HLS
    cwr.include("ap_int.h")
    cwr.include("hls_stream.h")
    cwr.include("hls_vector.h")
    cwr.include("ap_axi_sdata.h")

    # Include files from the nn2FPGA library
    nn2fpga_include_dir = "/workspace/NN2FPGA/nn2fpga/library/include"
    if os.path.isdir(nn2fpga_include_dir):
        for fname in os.listdir(nn2fpga_include_dir):
            if fname.endswith(".hpp"):
                cwr.include(fname)
    
    # Top function definition
    function = cpp_function(model.get_metadata_prop("top_name"), "void")
    function.add_code("#pragma HLS TOP")
    function.add_code("#pragma HLS DATAFLOW disable_start_propagation")
    function.add_code("#pragma HLS INTERFACE ap_ctrl_none port=return")

    for input in model.graph.input:
        tensor_fifo = get_custom_tensor_fifo_metadata(model, input.name)
        var = cpp_variable(
            input.name,
            f"hls::stream<{tensor_fifo.hls_type}>&",
            pragma=[f"#pragma HLS INTERFACE axis port={input.name}"],
        )
        function.add_argument(var)
        for pragma in var.pragma:
            function.add_code(pragma)

    for const_input_name in {init.name for init in model.graph.initializer if "const_" in init.name}:
        var = cpp_variable(f"{const_input_name}_stream", "hls::stream<ap_uint<8>>&")
        function.add_argument(var)
        function.add_code(f"#pragma HLS INTERFACE axis port={const_input_name}_stream")

    for output in model.graph.output:
        tensor_fifo = get_custom_tensor_fifo_metadata(model, output.name)
        var = cpp_variable(
            output.name,
            f"hls::stream<{tensor_fifo.hls_type}>&",
            pragma=[f"#pragma HLS INTERFACE axis port={output.name}"],
        )
        function.add_argument(var)
        for pragma in var.pragma:
            function.add_code(pragma)
    
    stream_vars = {}
    for fifo in model.graph.value_info:

        # Declare the array of streams only for the first stream of the array.
        if fifo.name.endswith("_0_"):
            trimmed_name = fifo.name[:-3]  # Remove the _0_ suffix 
            tensor_fifo = get_custom_tensor_fifo_metadata(model, fifo.name)
            var = cpp_variable(
                trimmed_name,
                f"hls::stream<{tensor_fifo.hls_type}>",
                array=tensor_fifo.n_array,
            )
            stream_vars[trimmed_name] = (var, [1] * tensor_fifo.n_array)
    
    for fifo in model.graph.value_info:

        # Read the index in the array from the name of the fifo.
        m = re.match(r"^(.*)_(\d+)_$", fifo.name)
        if not m:
            raise ValueError(f"Invalid fifo name: {fifo.name}")
        base_name = m.group(1)
        index = int(m.group(2))
        if base_name not in stream_vars:
            raise ValueError(f"Stream {base_name} not found in stream_vars.")
        tensor_fifo = get_custom_tensor_fifo_metadata(model, fifo.name)
        var, arr = stream_vars[base_name]
        arr[index] = tensor_fifo.depth
        stream_vars[base_name] = (var, arr)

    # Declare all the streams.
    for base_name, (var, arr) in stream_vars.items():
        function.add_code(f"{var.generate_declaration()};")
        for i, depth in enumerate(arr):
            function.add_code(f"#pragma HLS STREAM variable={base_name}[{i}] depth={depth}")

    for node in model.graph.node:
        custom_op = getCustomOp(node)

        # Declare the variables used in the custom operation.
        function.add_code(custom_op.get_nodeattr("hls_variable_declarations"))

        # Declare the object.
        function.add_code(custom_op.get_nodeattr("hls_object_declaration"))

        # Generate the run call for the custom operation
        function.add_code(f"{custom_op.get_nodeattr('hls_run_call')};")

    cwr.add_function_definition(function)
    return cwr.code

def encode_array(arr: np.ndarray):
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii")
    }

def generate_constant_input_values(model: ModelWrapper, partition_node: NodeProto) -> dict:
    """
    Generate a dictionary of input values for the costant inputs of the model.
    Args:
        model (ModelWrapper): The model to generate input values for.
    Returns:
        dict: A dictionary mapping input names to their constant values.
    """
    constant_inputs = {}
    init_dict = {init.name: init for init in model.graph.initializer}

    for tensor_name in init_dict:
        if "const_" in tensor_name:
            tensor = model.get_initializer(tensor_name)
            if tensor is not None:
                constant_inputs[tensor_name] = encode_array(tensor)
            else:
                raise ValueError(f"Initializer '{tensor_name}' not found in model.")

    return constant_inputs   

class EmbedHLSCode(Transformation):
    """
    Class to handle the conversion of ONNX models to HLS (High-Level Synthesis) format.
    """

    def __init__(self, nn2fpga_model: ModelWrapper, work_root: str = "/tmp", erase: bool = True):
        """
        Initializes the OnnxToHLS transformation.
        Args:
            work_root (str): The root directory of the project.
            nn2fpga_model (ModelWrapper): The model ready to be converted to HLS.
            erase (bool): If True, the starting onnx models will be erased after the transformation.
        """
        super().__init__()
        self.work_root = work_root
        self.nn2fpga_model = nn2fpga_model
        self.erase = erase

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        partition_nodes = model.get_nodes_by_op_type("nn2fpgaPartition")
        if not partition_nodes:
            raise ValueError(f"Partition nodes not found in model.")
        
        # We are sure that there is only one nn2FPGA partition node in the model.
        # as this is checked in the supported partition transformation.
        partition_node = partition_nodes[0]

        ap = AcceleratorPackage.from_json(
            getCustomOp(partition_node).get_nodeattr("accelerator_package")
        )

        with open("hls_code.cpp", "w") as f:
            f.write(generate_hls_code(self.nn2fpga_model))

        # Update the accelerator package with the HLS code and driver
        ap.work_dir = self.work_root
        ap.hls_code_b64 = base64.b64encode(generate_hls_code(self.nn2fpga_model).encode()).decode("ascii")
        ap.constant_inputs = generate_constant_input_values(self.nn2fpga_model, partition_node)

        getCustomOp(partition_node).set_nodeattr(
            "accelerator_package", ap.to_json()
        )

        if self.erase:
            # Erase the original model file if it exists
            if os.path.exists("partition_FPGA.onnx"):
                os.remove("partition_FPGA.onnx")
            
            if os.path.exists("wrapper_model.onnx"):
                os.remove("wrapper_model.onnx")

        return (model, False)
