from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_type import (
    FloatTensorType,
    QuantizedTensorType,
    set_custom_tensor_datatype,
    get_custom_tensor_datatype,
    is_constant_input_node,
)
import logging

logger = logging.getLogger(__name__)

class FoldQuant(Transformation):
    """Fold Quant node into tensors datatype."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        not_folded = 0
        folded = 0
        folded_float = 0

        # Find all Quant nodes in the model
        quants = model.get_nodes_by_op_type("Quant")
        for quant in quants:

            if is_constant_input_node(model, quant):
                continue  # Skip Quant nodes on the parameters (weights and biases)

            # Get quantization parameters
            expected_tensor_type = QuantizedTensorType.from_quant_node(quant, model)

            current_output_tensor_type = get_custom_tensor_datatype(
                model, quant.output[0]
            )

            # Check if the output of the Quant node has already a quantization
            # If it does, check if they are different
            if (
                current_output_tensor_type is not None
                and current_output_tensor_type != expected_tensor_type
            ):

                raise ValueError(
                    f"Quant node {quant.name} has a different quantization parameter than the output tensor {quant.output[0]}.\n"
                    f"Expected: {expected_tensor_type.get_canonical_name()}, "
                    f"current_output: {current_output_tensor_type.get_canonical_name() if current_output_tensor_type else 'None'}"
                )

            # Set the quantization parameters on the output tensor
            set_custom_tensor_datatype(model, quant.output[0], expected_tensor_type)

            # Check if the input of the Quant node has already a quantization
            current_input_tensor_type = get_custom_tensor_datatype(
                model, quant.input[0]
            )

            if (
                current_input_tensor_type is not None
                and current_input_tensor_type != expected_tensor_type
            ):
                # This case can happen if there are multiple Quant nodes with different parameters.
                # It is not an error, but we cannot fold the Quant node,
                # and thus we have to implement it in the hardware.
                not_folded += 1
                continue

            # Bypass and remove the Quant node
            producer = model.find_producer(quant.input[0])

            # In case the input to the quant is a model input,
            # we need to update the consumers of the quant to use directly the model input
            if producer is None:
                set_custom_tensor_datatype(model, quant.input[0], expected_tensor_type)
                consumers = model.find_consumers(quant.output[0])
                for consumer in consumers:
                    for i, inp in enumerate(consumer.input):
                        if inp == quant.output[0]:
                            consumer.input[i] = quant.input[0]
            else:
                # Update the producer to use the output of the Quant node
                for i, out in enumerate(producer.output):
                    if out == quant.input[0]:
                        producer.output[i] = quant.output[0]

            # Remove the Quant node from the graph
            graph.node.remove(quant)
            folded += 1
        
        for tensor in model.get_all_tensor_names():
            tensor_type = get_custom_tensor_datatype(model, tensor)
            if tensor_type is None:
                set_custom_tensor_datatype(model, tensor, FloatTensorType())
                folded_float += 1
            

        logger.info(
            f"Folded {folded} Quant nodes into tensor datatype. Set {folded_float} tensors without quantization to float."
        )
        if not_folded > 0:
            logger.warning(
                f"Could not fold {not_folded} Quant nodes due to multiple quantization parameters on the same tensor."
            )

        return (model, False)
