from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.transforms.propagate_quant import QUANT_INVARIANT_NODES
from nn2fpga.compiler.core.tensor_type import (
    get_custom_tensor_datatype,
    set_custom_tensor_datatype,
)
import numpy as np
import logging

logger = logging.getLogger(__name__)

class InferQuant(Transformation):
    """Infer quantization parameters for nodes without quantization parameters that are quant invariant."""

    @staticmethod
    def _all_input_quants_match(model: ModelWrapper, node) -> bool:
        input_quants = [
            get_custom_tensor_datatype(model, input_name)
            for input_name in node.input
            if input_name != ""
        ]
        input_quants = [quant for quant in input_quants if quant is not None]
        if len(input_quants) <= 1:
            return True
        reference = input_quants[0]
        return all(quant == reference for quant in input_quants[1:])

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        # Find all nodes in the model
        nodes = model.graph.node
        for node in nodes:
            if node.op_type not in QUANT_INVARIANT_NODES:
                continue

            # Check if the node already has quantization parameters in output
            output_quant = get_custom_tensor_datatype(model, node.output[0])
            if output_quant is not None:
                continue  # Skip nodes that already have quantization parameters

            if node.op_type in {"Concat", "StreamingConcat"} and not self._all_input_quants_match(model, node):
                logger.info(
                    "Skipping quant inference for node %s (%s) because its inputs have differing quantization parameters.",
                    node.name,
                    node.op_type,
                )
                continue

            # Infer quantization parameters based on the producer node
            input_quant = get_custom_tensor_datatype(model, node.input[0])
            if input_quant is not None:
                # If the input has quantization parameters, set them for the output
                set_custom_tensor_datatype(model, node.output[0], input_quant)
                logger.info(
                    f"Inferred quantization parameters for node {node.name} ({node.op_type}) based on input quantization."
                )

        return (model, False)
