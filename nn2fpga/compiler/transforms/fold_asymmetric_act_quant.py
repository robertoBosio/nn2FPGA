from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from onnx import helper
from nn2fpga.compiler.core.tensor_quant import get_custom_tensor_datatype
import numpy as np
import logging
logger = logging.getLogger(__name__)

class FoldAsymmetricActQuant(Transformation):
    """ Fold the zero point of asymmetric activation quantization into the bias of the next Conv layer. """
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph

        # Find all Conv layers in the model
        convs = model.get_nodes_by_op_type("StreamingConv")
        for conv in convs:
            custom_node = getCustomOp(conv)

            act_quant_params = get_custom_tensor_datatype(model, conv.input[0])
            if act_quant_params is None or act_quant_params.zeropt == 0:
                continue  # Nothing to fold if zeropt is None or 0

            zeropt_act = act_quant_params.zeropt

            # Get the weight quantization parameters
            w_scale = model.get_initializer(conv.input[2])
            w_value = model.get_initializer(conv.input[1])

            if w_value is None:
                logger.warning(f"Skipping {conv.name} weight data is not available.")
                continue

            # Sum the weights of each filter to compute the bias adjustment
            weight_sums = w_value.reshape(w_value.shape[0], -1).sum(axis=1)

            if len(conv.input) > 5:
                b_scale = model.get_initializer(conv.input[6])
                b_value = model.get_initializer(conv.input[5])

                # Reshape the weight scales to remove any extra dimensions
                weight_scales = np.squeeze(w_scale)

                # Check that the bias scale is close to the product of the weight scale and activation scale
                if not all(np.isclose(b_scale, weight_scales * act_quant_params.scale)):
                    logger.warning(f"Skipping {conv.name} as bias scale does not match weight and activation scales.")
                    continue

                if b_value is None:
                    logger.warning(f"Skipping {conv.name} bias data is not available.")
                    continue
            else:
                # bias_data = np.zeros(weight_data.shape[0], dtype=np.float32)
                # bias_quant = helper.make_node(
                #         "Quant",
                #         name = f"{conv.name}_bias",
                #         inputs=[out, quant_node.input[1], quant_node.input[2], quant_node.input[3]],  # Use the same quantization parameters
                #         outputs=[new_quant_output],
                #         domain=quant_node.domain,
                #     )
                logger.info(f"Skipping {conv.name} as it has no bias input.")
                continue

            # Adjust the bias by subtracting the zero point scaled by the weight sums
            new_bias_data = b_value - (zeropt_act * act_quant_params.scale * weight_sums)
            model.set_initializer(conv.input[5], new_bias_data)

            # Remove the zero point input from the activation quantization node
            conv.attribute.append(
                helper.make_attribute(key="asym_folded", value=1, attr_type=helper.AttributeProto.INT)
            )

            logger.info(f"Folded zero point of asymmetric activation quantization into bias of {conv.name}.")
        return (model, False)
