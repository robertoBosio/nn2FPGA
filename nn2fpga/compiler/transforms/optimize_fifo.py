import re
from dataclasses import dataclass

import numpy as np
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from nn2fpga.compiler.core.tensor_fifo import (
    get_custom_tensor_fifo_metadata,
    set_custom_tensor_fifo_metadata,
)
import nn2fpga.compiler.transforms as transformation
from nn2fpga.compiler.core.tensor_type import (
    TensorType,
    get_custom_tensor_datatype,
    set_custom_tensor_datatype,
)
from nn2fpga.compiler.core.tensor_layout import (
    get_custom_tensor_layout,
    set_custom_tensor_layout,
)
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
import logging

logger = logging.getLogger(__name__)

MIN_FIFO_BITS =  100 * 1024 # 100 Kb is a heuristic threshold 
MAX_DDR_FIFOS = 8
MAX_AXIWORD_BITS = 128
DEFAULT_DDR_BURST_LENGTH = 4


@dataclass
class FifoCandidate:
    fifo_name: str
    tensor_name: str
    shape: list[int]
    dtype: object
    layout: object
    fifo_depth: int
    dim2_unroll: int
    memory_bits: int
    axiword_bits: int
    burst_length: int


def choose_axiword_bits(dtype_bitwidth: int, dim2_unroll: int) -> int:
    lane_bits = dtype_bitwidth * dim2_unroll
    if lane_bits <= 0:
        raise ValueError("lane_bits must be positive")

    axiword_bits = (MAX_AXIWORD_BITS // lane_bits) * lane_bits
    if axiword_bits == 0:
        raise ValueError(
            f"Cannot choose AXI word width <= {MAX_AXIWORD_BITS} for "
            f"dtype_bitwidth={dtype_bitwidth}, dim2_unroll={dim2_unroll}"
        )
    return axiword_bits

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

    def __init__(self, nn2fpga_model: ModelWrapper, prj_root: str):
        """
        Initializes the OptimizeFifo transformation.
        Args:
            nn2fpga_model (ModelWrapper): The NN2FPGA model to optimize.
            prj_root (str): The project root directory.
        """
        super().__init__()
        self.nn2fpga_model = nn2fpga_model
        self.prj_root = prj_root

    def apply(self, hls_model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        # hls_model is used to inspect existing FIFO metadata, but graph rewrites
        # are applied to nn2fpga_model and then lowered again.
        model_II = int(self.nn2fpga_model.get_metadata_prop("model_II"))
        frequency = int(self.nn2fpga_model.get_metadata_prop("frequency")) * 1e6
        ap = AcceleratorPackage.from_json(
            hls_model.get_metadata_prop("accelerator_package")
        )
        bandwidth = compute_baseline_bandwidth_requirements(self.nn2fpga_model)
        logger.info(f"Baseline bandwidth requirements: {bandwidth / (1024 * 1024 * 1024):.4f} GB/s")

        candidates = self._find_candidate_fifos(hls_model)
        selected = self._select_candidate_fifos(candidates)
        logger.info(
            "Selected %d FIFOs for DDR: %s",
            len(selected),
            ", ".join(candidate.fifo_name for candidate in selected),
        )

        for candidate in selected:
            bandwidth_increase = (2 * np.prod(candidate.shape) * candidate.dtype.bitwidth / 8) / (model_II / frequency)
            logger.info(
                f"Optimizing fifo {candidate.fifo_name} would increase bandwidth requirements by {bandwidth_increase / (1024 * 1024 * 1024):.4f} GB/s, new total would be {(bandwidth + bandwidth_increase) / (1024 * 1024 * 1024):.4f} GB/s"
            )
            bandwidth += bandwidth_increase

            self._insert_ddr_stream(candidate)
            self._add_buffer_map_entry(ap, candidate)
        
        self.nn2fpga_model.set_metadata_prop("accelerator_package", ap.to_json())
        new_hls_model = self.nn2fpga_model.transform(
            transformation.LowerToHLS(
                infer_fifo_depth=False,
                ste_already_done=True,
                optimize_fifo_storage=False,
                prj_root=self.prj_root
            )
        )

        self._copy_fifo_metadata(hls_model, new_hls_model)
            
        return new_hls_model, False

    def _find_candidate_fifos(self, hls_model: ModelWrapper) -> list[FifoCandidate]:
        candidates = []
        for fifo in hls_model.graph.value_info:
            fifo_name = fifo.name
            try:
                tensor_name = from_fifo_to_tensor_name(fifo_name)
            except ValueError:
                continue
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
            depth = get_custom_tensor_fifo_metadata(hls_model, fifo_name).depth
            node = getCustomOp(producer)
            interface = node.get_port_interface()
            dim1_unroll = interface.out_stream_array
            dim2_unroll = interface.out_word_array
            memory_bits = depth * dtype.bitwidth * dim2_unroll

            if dim1_unroll != 1:
                continue

            if memory_bits < MIN_FIFO_BITS:
                continue

            candidates.append(
                FifoCandidate(
                    fifo_name=fifo_name,
                    tensor_name=tensor_name,
                    shape=shape,
                    dtype=dtype,
                    layout=get_custom_tensor_layout(self.nn2fpga_model, tensor_name),
                    fifo_depth=depth,
                    dim2_unroll=dim2_unroll,
                    memory_bits=memory_bits,
                    axiword_bits=choose_axiword_bits(dtype.bitwidth, dim2_unroll),
                    burst_length=DEFAULT_DDR_BURST_LENGTH,
                )
            )

        return candidates

    def _select_candidate_fifos(self, candidates: list[FifoCandidate]) -> list[FifoCandidate]:
        candidates.sort(key=lambda candidate: (-candidate.memory_bits, candidate.fifo_name))
        return candidates[:MAX_DDR_FIFOS]

    def _insert_ddr_stream(self, candidate: FifoCandidate) -> None:
        tensor_name = candidate.tensor_name
        producer = self.nn2fpga_model.find_producer(tensor_name)
        # At this stage tensors are point-to-point, so one consumer is expected.
        consumer = self.nn2fpga_model.find_consumer(tensor_name)
        input_name = f"{tensor_name}_in"
        output_name = f"{tensor_name}_out"

        ddr_stream_node = helper.make_node(
            op_type="DDRStream",
            domain="nn2fpga.compiler.custom_op",
            inputs=[input_name],
            outputs=[output_name],
            name=f"DDRStream_{tensor_name}",
            dim2_unroll=candidate.dim2_unroll,
            axiword=candidate.axiword_bits,
            burst_length=candidate.burst_length,
            stream_depth=candidate.fifo_depth,
            buffer_name=tensor_name,
            in_stream_array=1,
            in_word_array=candidate.dim2_unroll,
            out_stream_array=1,
            out_word_array=candidate.dim2_unroll,
        )
        
        self.nn2fpga_model.set_tensor_shape(input_name, candidate.shape)
        self.nn2fpga_model.set_tensor_shape(output_name, candidate.shape)
        set_custom_tensor_datatype(self.nn2fpga_model, input_name, candidate.dtype)
        set_custom_tensor_datatype(self.nn2fpga_model, output_name, candidate.dtype)
        set_custom_tensor_layout(self.nn2fpga_model, input_name, candidate.layout)
        set_custom_tensor_layout(self.nn2fpga_model, output_name, candidate.layout)

        for i, out in enumerate(producer.output):
            if out == tensor_name:
                producer.output[i] = input_name

        for i, inp in enumerate(consumer.input):
            if inp == tensor_name:
                consumer.input[i] = output_name

        producer_idx = list(self.nn2fpga_model.graph.node).index(producer)
        self.nn2fpga_model.graph.node.insert(producer_idx + 1, ddr_stream_node)

    def _add_buffer_map_entry(self, ap: AcceleratorPackage, candidate: FifoCandidate) -> None:
        # Shapes include batch; at this stage batch is fixed to 1.
        total_bits = int(np.prod(candidate.shape) * candidate.dtype.bitwidth)
        if total_bits % candidate.axiword_bits != 0:
            raise ValueError(
                f"Tensor {candidate.tensor_name} has {total_bits} bits, not divisible by "
                f"selected AXI word width {candidate.axiword_bits}"
            )
        if total_bits % 8 != 0:
            raise ValueError(
                f"Tensor {candidate.tensor_name} has {total_bits} bits, not divisible by 8"
            )

        ap.buffer_map[candidate.tensor_name] = {
            "hls_type": f"ap_uint<{candidate.axiword_bits}>",
            "depth": f"{total_bits // candidate.axiword_bits}",
            "size_bytes": total_bits // 8,
        }

    def _copy_fifo_metadata(self, old_hls_model: ModelWrapper, new_hls_model: ModelWrapper) -> None:
        for fifo in old_hls_model.graph.value_info:
            fifo_metadata = get_custom_tensor_fifo_metadata(old_hls_model, fifo.name)
            set_custom_tensor_fifo_metadata(new_hls_model, fifo.name, fifo_metadata)
