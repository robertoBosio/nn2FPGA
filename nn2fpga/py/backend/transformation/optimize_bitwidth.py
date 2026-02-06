from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import TensorQuant, is_constant_input_node
from backend.transformation.add_streaming_params import quant_array
import numpy as np
import logging

logger = logging.getLogger(__name__)

def bits_required_for_range(minv: int, maxv: int, signed: bool, narrow: bool) -> int:
    # ensure python ints
    minv = int(minv)
    maxv = int(maxv)

    if signed:
        # small b such that minv,maxv fit in signed range
        b = 1
        while True:
            min_allowed = -(1 << (b - 1))
            if narrow:
                min_allowed += 1
            max_allowed = (1 << (b - 1)) - 1
            if (minv >= min_allowed) and (maxv <= max_allowed):
                return b
            b += 1
    else:
        # unsigned: require non-negative
        if minv < 0:
            raise ValueError(f"Unsigned quant but found negative value {minv}")
        b = 1
        while True:
            min_allowed = 0 + (1 if narrow else 0)
            max_allowed = (1 << b) - 1
            if (minv >= min_allowed) and (maxv <= max_allowed):
                return b
            b += 1

class OptimizeBitwidth(Transformation):
    """Optimize bitwidth of constant tensors based on their values."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        # Find all Quant nodes in the model
        quants = model.get_nodes_by_op_type("Quant")
        for quant in quants:

            if not is_constant_input_node(model, quant):
                continue  # Skip Quant nodes on the activations

            # Get the value of the constant tensor
            tensor_value = model.get_initializer(quant.input[0])
            if tensor_value is None:
                raise ValueError(
                    f"Quant node {quant.name} has no constant input tensor {quant.input[0]}"
                )

            # Quantize tensor values according to the quantization parameters
            tq = TensorQuant.from_quant_node(quant, model)
            q_arr = quant_array(
                tensor_value,
                scale=tq.scale,
                zeropt=tq.zeropt,
                bitwidth=tq.bitwidth,
                signed=tq.signed,
                narrow=tq.narrow,
                rounding_mode=tq.rounding_mode,
            )

            # Determine the minimum bitwidth required to represent the quantized values
            max_val = q_arr.max()
            min_val = q_arr.min()
            required_bits = bits_required_for_range(
                minv=min_val, maxv=max_val, signed=tq.signed, narrow=tq.narrow
            )

            # Update the bitwidth if it can be reduced
            if required_bits < tq.bitwidth:
                logger.info(
                    f"Optimizing bitwidth of tensor {quant.input[0]} from {tq.bitwidth} to {required_bits}"
                )
                model.set_initializer(
                    quant.input[3], np.array(required_bits, dtype=np.int32)
                )
        return model, False
