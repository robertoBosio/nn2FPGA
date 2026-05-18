from copy import deepcopy
import re
import numpy as np
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
from nn2fpga.compiler.core.tensor_type import TensorType, get_custom_tensor_datatype, set_custom_tensor_datatype
from nn2fpga.compiler.core.tensor_fifo import get_custom_tensor_fifo_metadata
from nn2fpga.compiler.core.tensor_layout import get_custom_tensor_layout, set_custom_tensor_layout
from qonnx.util.basic import qonnx_make_model
from nn2fpga.compiler.utils.codegen_utils import cpp_function, cpp_object, cpp_variable, NewCodeWriter
from nn2fpga.compiler.utils.board_util import read_board_info
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.transforms.embed_hls_code import EmbedHLSCode
from nn2fpga.compiler.transforms.generate_bitstream import GenerateBitstream
from nn2fpga.compiler.custom_op.DDRStream import DDRStream
import os
import json
import subprocess
import logging
logger = logging.getLogger(__name__)

def compute_baseline_bandwidth_requirements(model: ModelWrapper) -> float:
    """Compute the baseline bandwidth requirements for inputs/outputs of the model"""
    bandwidth = 0
    ap = AcceleratorPackage.from_json(
        model.get_metadata_prop("accelerator_package")
    )
    model_II = int(model.get_metadata_prop("model_II"))
    frequency = int(model.get_metadata_prop("frequency")) * 1e6
    for value in ap.input_map.values():
        if value["value"] is not None:
            # This is a constant tensor, it is streamed only once at the beginning, 
            # so we can ignore it for bandwidth requirements.
            continue
        shape = value["shape"]
        dtype = TensorType.from_canonical_name(value["quant"])
        bandwidth += (np.prod(shape) * dtype.bitwidth / 8)
    
    for value in ap.output_map.values():
        shape = value["shape"]
        dtype = TensorType.from_canonical_name(value["quant"])
        bandwidth += (np.prod(shape) * dtype.bitwidth / 8)

    return (bandwidth / (model_II / frequency)) # bytes per second

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
        model_II = int(model.get_metadata_prop("model_II"))
        frequency = int(model.get_metadata_prop("frequency")) * 1e6
        ap = AcceleratorPackage.from_json(
            model.get_metadata_prop("accelerator_package")
        )
        bandwidth = compute_baseline_bandwidth_requirements(model)
        candidate_fifos = []
        logger.info(f"Baseline bandwidth requirements: {bandwidth / (1024 * 1024 * 1024):.4f} GB/s")

        for fifo in model.graph.value_info:
            fifo_name = fifo.name
            tensor_name = from_fifo_to_tensor_name(fifo_name)
            producer = self.nn2fpga_model.find_producer(tensor_name)
            if producer is None:
                # logger.warning(f"Could not find producer for tensor {tensor_name}, skipping optimization for fifo {fifo_name}.")
                continue
            shape = self.nn2fpga_model.get_tensor_shape(tensor_name)
            if shape is None:
                # logger.warning(f"Could not infer shape for tensor {tensor_name}, skipping optimization for fifo {fifo_name}.")
                continue
            dtype = get_custom_tensor_datatype(self.nn2fpga_model, tensor_name)
            if dtype is None:
                # logger.warning(f"Could not infer datatype for tensor {tensor_name}, skipping optimization for fifo {fifo_name}.")
                continue
            depth = get_custom_tensor_fifo_metadata(model, fifo_name).depth
            node = getCustomOp(producer)
            interface = node.get_port_interface()
            dim1_unroll = interface.out_stream_array
            dim2_unroll = interface.out_word_array
            memory_requirements = depth * dtype.bitwidth * dim2_unroll

            if dim1_unroll != 1:
                continue

            if (
                memory_requirements < 1000 * 8
            ):  # Less than 1KB, we can ignore it for optimization.
                continue

            candidate_fifos.append((fifo_name, memory_requirements))

        candidate_fifos.sort(key=lambda x: x[1], reverse=True)
        for fifo_name, memory_requirements in candidate_fifos:
            tensor_name = from_fifo_to_tensor_name(fifo_name)
            shape = self.nn2fpga_model.get_tensor_shape(tensor_name)
            dtype = get_custom_tensor_datatype(self.nn2fpga_model, tensor_name)
            layout = get_custom_tensor_layout(self.nn2fpga_model, tensor_name)
            fifo_metadata = get_custom_tensor_fifo_metadata(model, fifo_name)
            producer = self.nn2fpga_model.find_producer(tensor_name)
            consumer = self.nn2fpga_model.find_consumer(tensor_name)
            interface = getCustomOp(producer).get_port_interface()
            dim1_unroll = interface.out_stream_array
            dim2_unroll = interface.out_word_array
            buffer_name = tensor_name

            input_name = f"{tensor_name}_in"
            output_name = f"{tensor_name}_out"

            ddr_stream_node = helper.make_node(
                op_type="DDRStream",
                domain="nn2fpga.compiler.custom_op",
                inputs=[input_name],
                outputs=[output_name],
                name=f"DDRStream_{tensor_name}",
                dim2_unroll=dim2_unroll,
                axiword=128,
                burst_length=4,
                stream_depth=fifo_metadata.depth,
                buffer_name=buffer_name,
                in_stream_array=1,
                in_word_array=dim2_unroll,
                out_stream_array=1,
                out_word_array=dim2_unroll,
            )
            
            self.nn2fpga_model.set_tensor_shape(input_name, shape)
            self.nn2fpga_model.set_tensor_shape(output_name, shape)
            set_custom_tensor_datatype(self.nn2fpga_model, input_name, dtype)
            set_custom_tensor_datatype(self.nn2fpga_model, output_name, dtype)
            set_custom_tensor_layout(self.nn2fpga_model, input_name, layout)
            set_custom_tensor_layout(self.nn2fpga_model, output_name, layout)

            for i, out in enumerate(producer.output):
                if out == tensor_name:
                    producer.output[i] = input_name

            for i, inp in enumerate(consumer.input):
                if inp == tensor_name:
                    consumer.input[i] = output_name

            bandwidth_increase = (2 * np.prod(shape) * dtype.bitwidth / 8) / (model_II / frequency)
            logger.info(
                f"Optimizing fifo {fifo_name} would increase bandwidth requirements by {bandwidth_increase / (1024 * 1024 * 1024):.4f} GB/s, new total would be {(bandwidth + bandwidth_increase) / (1024 * 1024 * 1024):.4f} GB/s"
            )
            bandwidth += bandwidth_increase

            producer_idx = list(self.nn2fpga_model.graph.node).index(producer)
            self.nn2fpga_model.graph.node.insert(producer_idx + 1, ddr_stream_node)
            
        self.nn2fpga_model.save("optimized_fifos.onnx")
        return model, False
