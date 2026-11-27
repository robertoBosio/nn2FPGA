from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import SortGraph
from nn2fpga.compiler.utils.codegen_utils import cpp_function, cpp_variable, NewCodeWriter
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.core.tensor_fifo import TensorFifo, get_custom_tensor_fifo_metadata
from nn2fpga.compiler.core.tensor_type import TensorType
from nn2fpga.compiler.transforms.convert_to_QCDQ import ConvertToQCDQ
from nn2fpga.compiler.utils.axi import get_axi_word_count
from nn2fpga.compiler.utils.board_util import read_board_info
from onnx import NodeProto
import base64
import os
import re
import numpy as np
import logging
logger = logging.getLogger(__name__)

def generate_vitis_hls_code(model: ModelWrapper, ap: AcceleratorPackage) -> str:

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
    cwr.include("ap_float.h")
    cwr.include("hls_stream.h")
    cwr.include("hls_vector.h")
    cwr.include("ap_axi_sdata.h")

    base_include_dir = "/workspace/nn2FPGA/nn2fpga/hw/library/include/vitis"

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
                        cwr.include(f"vitis/{rel_path}")

    # Top function definition
    function = cpp_function(model.get_metadata_prop("top_name"), "void")
    function.add_code("#pragma HLS TOP")
    function.add_code("#pragma HLS DATAFLOW disable_start_propagation")

    use_dma = str(model.get_metadata_prop("use_dma")).lower() == "true"
    if not use_dma:
        function.add_code("#pragma HLS INTERFACE s_axilite port=return bundle=control")
    else:
        function.add_code("#pragma HLS INTERFACE ap_ctrl_none port=return")
        
    axi_bitwidth = None
    if any(port["mode"] == "m_axi" for port in ap.input_map.values()) or any(
        port["mode"] == "m_axi" for port in ap.output_map.values()
    ):
        axi_bitwidth = read_board_info(
            model.get_metadata_prop("board_name")
        )["axi_bitwidth"]

    for input in ap.input_map.values():
        input_name = input['new_name']
        mode = input['mode']
        tensor_fifo = get_custom_tensor_fifo_metadata(model, input_name)
        if mode == "axis":
            var = cpp_variable(
                input_name,
                f"hls::stream<{tensor_fifo.hls_type}>&",
                pragma=[f"#pragma HLS INTERFACE axis port={input_name}"],
            )
        elif mode == "m_axi":
            tensor_type = TensorType.from_canonical_name(input["quant"])
            depth = get_axi_word_count(
                input["shape"], tensor_type.bitwidth, axi_bitwidth
            )
            var = cpp_variable(
                input_name,
                f"{tensor_fifo.hls_type}*",
                pragma=[
                    f"#pragma HLS INTERFACE m_axi port={input_name} bundle={input_name}_bundle depth={depth}",
                ],
            )
        else:
            raise ValueError(f"Unsupported mode {mode} for input {input_name}")
        function.add_argument(var)
        for pragma in var.pragma:
            function.add_code(pragma)

    for output in ap.output_map.values():
        output_name = output['new_name']
        mode = output['mode']
        tensor_fifo = get_custom_tensor_fifo_metadata(model, output_name)
        if mode == "axis":
            var = cpp_variable(
                output_name,
                f"hls::stream<{tensor_fifo.hls_type}>&",
                pragma=[f"#pragma HLS INTERFACE axis port={output_name}"],
        )
        elif mode == "m_axi":
            tensor_type = TensorType.from_canonical_name(output["quant"])
            depth = get_axi_word_count(
                output["shape"], tensor_type.bitwidth, axi_bitwidth
            )
            var = cpp_variable(
                output_name,
                f"{tensor_fifo.hls_type}*",
                pragma=[
                    f"#pragma HLS INTERFACE m_axi port={output_name} bundle={output_name}_bundle depth={depth}",
                ],
            )
        else:
            raise ValueError(f"Unsupported mode {mode} for output {output_name}")
        function.add_argument(var)
        for pragma in var.pragma:
            function.add_code(pragma)
    
    for buffer_name, buffer in ap.buffer_map.items():
        read_buffer = f"{buffer_name}_read"
        var = cpp_variable(
            read_buffer,
            f"{buffer['hls_type']}*",
            pragma=[
                f"#pragma HLS INTERFACE m_axi port={read_buffer} bundle={read_buffer}_bundle depth={buffer['depth']}",
                f"#pragma HLS STABLE variable={read_buffer}",
            ],
        )
        function.add_argument(var)
        for pragma in var.pragma:
            function.add_code(pragma)

        write_buffer = f"{buffer_name}_write"
        var = cpp_variable(
            write_buffer,
            f"{buffer['hls_type']}*",
            pragma=[
                f"#pragma HLS INTERFACE m_axi port={write_buffer} bundle={write_buffer}_bundle depth={buffer['depth']}",
                f"#pragma HLS STABLE variable={write_buffer}",
            ],
        )
        function.add_argument(var)
        for pragma in var.pragma:
            function.add_code(pragma)
        function.add_code(f"#pragma HLS ALIAS ports = {read_buffer}, {write_buffer} distance = 0")

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

        # Checks that the fifo name does not start with a digit, as Vitis HLS
        # does not allow it in the variable name.
        if re.match(r"^\d", base_name):
            raise ValueError(f"Stream name {base_name} cannot start with a digit.")

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


def generate_catapult_hls_code(model: ModelWrapper, ap: AcceleratorPackage) -> str:
    """Generate a streaming-only Catapult HLS top-level source file."""
    if str(model.get_metadata_prop("use_dma")).lower() != "true":
        raise ValueError("Catapult requires use_dma=true for streaming boundary ports.")
    if ap.buffer_map:
        raise ValueError("Catapult does not yet support external memory buffers.")

    cwr = NewCodeWriter()
    cwr.add_autogen_comment()
    cwr.include("array")
    cwr.include("ac_int.h")
    cwr.include("ac_channel.h")

    base_include_dir = "/workspace/nn2FPGA/nn2fpga/hw/library/include/catapult"
    if os.path.isdir(base_include_dir):
        for root, _, files in os.walk(base_include_dir):
            for fname in files:
                if fname.endswith(".hpp"):
                    rel_path = os.path.relpath(
                        os.path.join(root, fname), base_include_dir
                    ).replace(os.sep, "/")
                    if "testbench" not in rel_path:
                        cwr.include(f"catapult/{rel_path}")

    cwr.add_line("#pragma hls_design top")
    function = cpp_function(model.get_metadata_prop("top_name"), "void")

    ports = list(ap.input_map.values()) + list(ap.output_map.values())
    for port in ports:
        port_name = port["new_name"]
        if port["mode"] != "axis":
            raise ValueError(
                f"Catapult supports only axis ports; {port_name} uses {port['mode']}."
            )
        tensor_fifo = get_custom_tensor_fifo_metadata(model, port_name)
        if "ap_" in tensor_fifo.hls_type or "hls::" in tensor_fifo.hls_type:
            raise ValueError(
                f"Catapult requires a Catapult FIFO type for {port_name}, got "
                f"{tensor_fifo.hls_type}."
            )
        function.add_argument(
            cpp_variable(port_name, f"ac_channel<{tensor_fifo.hls_type}>&")
        )

    stream_vars = {}
    for fifo in model.graph.value_info:
        if fifo.name.endswith("_0_"):
            trimmed_name = fifo.name[:-3]
            tensor_fifo = get_custom_tensor_fifo_metadata(model, fifo.name)
            if "ap_" in tensor_fifo.hls_type or "hls::" in tensor_fifo.hls_type:
                raise ValueError(
                    f"Catapult requires a Catapult FIFO type for {fifo.name}, got "
                    f"{tensor_fifo.hls_type}."
                )
            stream_vars[trimmed_name] = cpp_variable(
                trimmed_name,
                f"ac_channel<{tensor_fifo.hls_type}>",
                array=tensor_fifo.n_array,
            )

    for fifo in model.graph.value_info:
        match = re.match(r"^(.*)_(\d+)_$", fifo.name)
        if not match:
            raise ValueError(f"Invalid fifo name: {fifo.name}")
        base_name = match.group(1)
        if base_name not in stream_vars:
            raise ValueError(f"Stream {base_name} not found in stream_vars.")
        if re.match(r"^\d", base_name):
            raise ValueError(f"Stream name {base_name} cannot start with a digit.")

    for var in stream_vars.values():
        # Catapult requires channels internal to the synthesized hierarchy to
        # have static storage; top-level channel references remain dynamic.
        function.add_code(f"static {var.generate_declaration()};")

    for node in model.graph.node:
        custom_op = getCustomOp(node)
        for attribute in (
            "hls_variable_declarations",
            "hls_object_declaration",
            "hls_run_call",
        ):
            source = custom_op.get_nodeattr(attribute)
            if "hls::" in source or "#pragma HLS" in source:
                raise ValueError(
                    f"Catapult source for {node.name or node.op_type} has not been lowered."
                )
            suffix = ";" if attribute == "hls_run_call" else ""
            function.add_code(f"{source}{suffix}")

    cwr.add_function_definition(function)
    return cwr.code


def generate_hls_code(model: ModelWrapper, ap: AcceleratorPackage) -> str:
    """Generate HLS source for the backend selected in model metadata."""
    backend = model.get_metadata_prop("hls_backend") or "vitis"
    if backend == "vitis":
        return generate_vitis_hls_code(model, ap)
    if backend == "catapult":
        return generate_catapult_hls_code(model, ap)
    raise ValueError(f"Unsupported HLS backend: {backend}")

class EmbedHLSCode(Transformation):
    """
    Class to handle the conversion of ONNX models to HLS (High-Level Synthesis) format.
    """

    def __init__(self, hls_model: ModelWrapper, work_root: str = "/tmp", erase: bool = True):
        """
        Initializes the OnnxToHLS transformation.
        Args:
            work_root (str): The root directory of the project.
            hls_model (ModelWrapper): The model ready to be converted to HLS.
            erase (bool): If True, the starting onnx models will be erased after the transformation.
        """
        super().__init__()
        self.work_root = work_root
        self.hls_model = hls_model
        self.erase = erase

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        model = model.transform(SortGraph())
        partition_nodes = model.get_nodes_by_op_type("nn2fpgaPartition")
        if not partition_nodes:
            raise ValueError(f"Partition nodes not found in model.")

        # We are sure that there is only one nn2FPGA partition node in the model.
        # as this is checked in the supported partition transformation.
        partition_node = partition_nodes[0]

        ap = AcceleratorPackage.from_json(
            self.hls_model.get_metadata_prop("accelerator_package")
        )

        # Update the accelerator package with the HLS code and driver
        ap.work_dir = self.work_root
        ap.hls_code_b64 = base64.b64encode(generate_hls_code(self.hls_model, ap).encode()).decode("ascii")

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
