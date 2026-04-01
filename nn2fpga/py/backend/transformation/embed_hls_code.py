from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from backend.util.codegen_utils import cpp_function, cpp_variable, NewCodeWriter
from backend.core.acceleratorpackage import AcceleratorPackage
from backend.core.tensor_fifo import TensorFifo, get_custom_tensor_fifo_metadata
from backend.transformation.convert_to_QCDQ import ConvertToQCDQ
from onnx import NodeProto
import base64
import os
import re
import numpy as np
import logging
logger = logging.getLogger(__name__)

def generate_hls_code(model: ModelWrapper, ap: AcceleratorPackage) -> str:

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

    base_include_dir = "/workspace/NN2FPGA/nn2fpga/library/include"

    if os.path.isdir(base_include_dir):
        for root, _, files in os.walk(base_include_dir):
            for fname in files:
                if fname.endswith(".hpp"):
                    rel_path = os.path.relpath(
                        os.path.join(root, fname),
                        base_include_dir
                    )
                    rel_path = rel_path.replace(os.sep, "/")
                    if "testbench" not in rel_path:
                        cwr.include(rel_path)
    
    # Top function definition
    function = cpp_function(model.get_metadata_prop("top_name"), "void")
    function.add_code("#pragma HLS TOP")
    function.add_code("#pragma HLS DATAFLOW disable_start_propagation")
    function.add_code("#pragma HLS INTERFACE ap_ctrl_none port=return")

    for input_name in [input['new_name'] for input in ap.input_map.values()]:
        tensor_fifo = get_custom_tensor_fifo_metadata(model, input_name)
        var = cpp_variable(
            input_name,
            f"hls::stream<{tensor_fifo.hls_type}>&",
            pragma=[f"#pragma HLS INTERFACE axis port={input_name}"],
        )
        function.add_argument(var)
        for pragma in var.pragma:
            function.add_code(pragma)

    for output_name in [output['new_name'] for output in ap.output_map.values()]:
        tensor_fifo = get_custom_tensor_fifo_metadata(model, output_name)
        var = cpp_variable(
            output_name,
            f"hls::stream<{tensor_fifo.hls_type}>&",
            pragma=[f"#pragma HLS INTERFACE axis port={output_name}"],
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

    model_outputs = [output.name for output in model.graph.output]
    for node in model.graph.node:
        custom_op = getCustomOp(node)

        # Declare the variables used in the custom operation.
        function.add_code(custom_op.get_nodeattr("hls_variable_declarations"))

        # Declare the object.
        function.add_code(custom_op.get_nodeattr("hls_object_declaration"))

        # Generate the run call for the custom operation
        function.add_code(f"{custom_op.get_nodeattr('hls_run_call')};")

        # Print the max size of the FIFOs in output
        for output in [output for output in node.output if output not in model_outputs]:
            tensor_fifo = get_custom_tensor_fifo_metadata(model, output)
            m = re.match(r"^(.*)_(\d+)_$", output)
            if not m:
                raise ValueError(f"Invalid fifo name: {output}")
            base_name = m.group(1)
            index = int(m.group(2))
            trimmed_name = output[:-1]  # Remove the _ suffix
            function.add_code(f'#ifndef __SYNTHESIS__')
            if "linebuffer" in base_name:
                function.add_code(f'std::cout << "{trimmed_name},{tensor_fifo.depth}" << std::endl;')
            else:
                function.add_code(f'std::cout << "{trimmed_name}," << {base_name}[{index}].size() << std::endl;')
            function.add_code(f'#endif') 


    cwr.add_function_definition(function)
    return cwr.code

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
            self.nn2fpga_model.get_metadata_prop("accelerator_package")
        )

        # Update the accelerator package with the HLS code and driver
        ap.work_dir = self.work_root
        ap.hls_code_b64 = base64.b64encode(generate_hls_code(self.nn2fpga_model, ap).encode()).decode("ascii")

        getCustomOp(partition_node).set_nodeattr(
            "accelerator_package", ap.to_json()
        )
        model = model.transform(ConvertToQCDQ())

        if self.erase:
            # Erase the original model file if it exists
            if os.path.exists("partition_FPGA.onnx"):
                os.remove("partition_FPGA.onnx")

            if os.path.exists("wrapper_model.onnx"):
                os.remove("wrapper_model.onnx")

        return (model, False)
