from copy import deepcopy
import re
from onnx import TensorProto, helper, StringStringEntryProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from nn2fpga.compiler.core.tensor_fifo import (
    TensorFifo,
    get_custom_tensor_fifo_metadata,
    set_custom_tensor_fifo_metadata,
)
from nn2fpga.compiler.core.hls_schedule_parser import VitisHlsReportParser
from nn2fpga.compiler.core.tensor_type import TensorType, get_custom_tensor_datatype
from nn2fpga.compiler.core.tensor_fifo import get_custom_tensor_fifo_metadata
from nn2fpga.compiler.core.tensor_layout import get_custom_tensor_layout
from qonnx.util.basic import qonnx_make_model
from nn2fpga.compiler.utils.codegen_utils import cpp_function, cpp_object, cpp_variable, NewCodeWriter
from nn2fpga.compiler.utils.board_util import read_board_info
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.transforms.embed_hls_code import EmbedHLSCode
from nn2fpga.compiler.transforms.generate_bitstream import GenerateBitstream
import os
import json
import subprocess
import logging
logger = logging.getLogger(__name__)

def analyze_memory_occupation(model: ModelWrapper) -> list[str]:
    """Analyze the memory occupation of each stream in the model."""
    def from_hls_type_to_dtype_size(hls_type: str) -> int:
        """
        Parse an HLS type string and return its total size in bits.

        Supported examples:
            ap_int<16>
            ap_uint<32>
            float
            double
            std::array<ap_uint<8>, 4>
            std::array<std::array<ap_uint<8>, 4>, 2>
        """

        hls_type = hls_type.strip()

        # Handle std::array<T, N>
        array_match = re.match(r"std::array\s*<(.+),\s*(\d+)\s*>", hls_type)
        if array_match:
            element_type = array_match.group(1).strip()
            array_size = int(array_match.group(2))
            element_bits = from_hls_type_to_dtype_size(element_type)
            return element_bits * array_size

        # Handle ap_int<N> or ap_uint<N>
        ap_match = re.match(r"(ap_int|ap_uint)\s*<\s*(\d+)\s*>", hls_type)
        if ap_match:
            return int(ap_match.group(2))

        # Handle float and double
        if hls_type == "float":
            return 32
        if hls_type == "double":
            return 64

        raise ValueError(f"Unsupported HLS type: {hls_type}")

    stream_occupations = {}
    tot_bits = 0
    for fifo in model.graph.value_info:
        tensor_fifo = get_custom_tensor_fifo_metadata(model, fifo.name)
        if tensor_fifo is not None:
            stream_occupations[fifo.name] = tensor_fifo.depth * (from_hls_type_to_dtype_size(tensor_fifo.hls_type))
            tot_bits += stream_occupations[fifo.name]

    stream_occupations = dict(sorted(stream_occupations.items(), key=lambda item: item[1], reverse=True))
    logger.info(f"Total memory occupation: {tot_bits} bits, {tot_bits / (1024 * 1024 * 8):.2f} MBs")
    logger.info("Memory occupation of the 10 largest streams:")
    model_II = int(model.get_metadata_prop("model_II"))
    streams_to_optimize = []
    for stream_name, occupation in list(stream_occupations.items())[:10]:
        logger.info(f"{stream_name}: {occupation} bits, {occupation * 100 / tot_bits:.2f}% of total")

        if occupation > 600:
            streams_to_optimize.append(stream_name)

    return streams_to_optimize

def from_fifo_to_tensor_name(fifo_name: str) -> str:
    """ Remove the _stream_[+d]_ suffix from the fifo name to get the original tensor name."""
    m = re.match(r"^(.*)_stream_(\d+)_$", fifo_name)
    if m:
        return m.group(1)
    else:
        raise ValueError(f"Invalid fifo name: {fifo_name}")

class OptimizeFifo(Transformation):
    """Moves largest streams to DDR."""

    def __init__(self, nn2fpga_model: ModelWrapper):
        """
        Initializes the OptimizeFifo transformation.
        Args:
            nn2fpga_model (ModelWrapper): The NN2FPGA model to optimize.
        """
        super().__init__()
        self.nn2fpga_model = nn2fpga_model

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        streams_to_optimize = analyze_memory_occupation(model)
        for fifo_name in streams_to_optimize:
            tensor_name = from_fifo_to_tensor_name(fifo_name)
            producer = self.nn2fpga_model.find_producer(tensor_name)
            if producer is None:
                logger.warning(f"Could not find producer for tensor {tensor_name}, skipping optimization for fifo {fifo_name}.")
                continue
            shape = self.nn2fpga_model.get_tensor_shape(tensor_name)
            if shape is None:
                logger.warning(f"Could not infer shape for tensor {tensor_name}, skipping optimization for fifo {fifo_name}.")
                continue
            depth = get_custom_tensor_fifo_metadata(model, fifo_name).depth
            node = getCustomOp(producer)

            logger.info(f"Optimizing stream {fifo_name} (tensor {tensor_name}) produced by node {producer.name}. The stream has depth {depth} and tensor shape {shape}. The interface of the producer node is {node.get_port_interface()}.")
        return model, False
