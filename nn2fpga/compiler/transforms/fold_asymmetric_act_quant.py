from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from nn2fpga.compiler.core.tensor_type import get_custom_tensor_datatype
from nn2fpga.compiler.transforms.adjust_conv_scale import required_bitwidth_from_int_range
from nn2fpga.compiler.utils.quantization import quant_array_no_clipping, safe_int_quant_call
import numpy as np
import logging
logger = logging.getLogger(__name__)


def _reshape_qparam_for_weights(qparam, weight_shape):
    qparam = np.asarray(qparam, dtype=np.float32)
    if qparam.size == 1:
        return qparam.reshape(())
    if qparam.ndim == 1 and qparam.shape[0] == weight_shape[0]:
        return qparam.reshape((weight_shape[0],) + (1,) * (len(weight_shape) - 1))
    raise ValueError(
        f"Unsupported weight quantization parameter shape {qparam.shape} for weights {weight_shape}."
    )
def _effective_dequantized_weights(w_value, w_scale, w_zeropt, w_bitwidth, w_signed, w_narrow, w_rounding_mode):
    scale = _reshape_qparam_for_weights(w_scale, w_value.shape)
    zeropt = _reshape_qparam_for_weights(w_zeropt, w_value.shape)
    return safe_int_quant_call(
        w_value,
        scale=scale,
        zeropt=zeropt,
        bitwidth=w_bitwidth,
        signed=w_signed,
        narrow=w_narrow,
        rounding_mode=w_rounding_mode,
    )

class FoldAsymmetricActQuant(Transformation):
    """ Fold the zero point of asymmetric activation quantization into the bias of the next Conv layer. """
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        # Find all Conv layers in the model
        convs = model.get_nodes_by_op_type("StreamingConv")
        for conv in convs:
            custom_node = getCustomOp(conv)

            act_quant_params = get_custom_tensor_datatype(model, conv.input[0])
            if act_quant_params is None or act_quant_params.zeropt == 0:
                continue  # Nothing to fold if zeropt is None or 0

            zeropt_act = act_quant_params.zeropt

            # Fold against the effective deployed weights, not the ideal float values.
            # If quantization clips, using the raw initializer sum over-corrects the bias.
            w_scale = np.asarray(model.get_initializer(conv.input[2]), dtype=np.float32)
            w_zeropt = np.asarray(model.get_initializer(conv.input[3]), dtype=np.float32)
            w_bitwidth = int(np.asarray(model.get_initializer(conv.input[4])).reshape(()).item())
            w_value = np.asarray(model.get_initializer(conv.input[1]), dtype=np.float32)

            if w_value is None:
                logger.warning(f"Skipping {conv.name} weight data is not available.")
                continue

            weight_scales = np.squeeze(w_scale)

            effective_weights = _effective_dequantized_weights(
                w_value,
                w_scale,
                w_zeropt,
                w_bitwidth,
                bool(custom_node.get_nodeattr("w_signed")),
                bool(custom_node.get_nodeattr("w_narrow")),
                custom_node.get_nodeattr("w_rounding_mode"),
            )

            # Sum the deployed dequantized weights of each filter to compute the bias adjustment.
            weight_sums = effective_weights.reshape(effective_weights.shape[0], -1).sum(axis=1)

            if len(conv.input) > 5:
                b_scale = model.get_initializer(conv.input[6])
                b_value = model.get_initializer(conv.input[5])

                # Check that the bias scale is close to the product of the weight scale and activation scale
                if not np.all(np.isclose(b_scale, weight_scales * act_quant_params.scale)):
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
            logger.info(f"Folding zero point of asymmetric activation quantization into bias of {conv.name}.")
            logger.info(f"Old bias: {b_value}")
            logger.info(f"New bias: {new_bias_data}")
            model.set_initializer(conv.input[5], new_bias_data)

            if len(conv.input) > 8:
                b_zeropt = np.asarray(model.get_initializer(conv.input[7]))
                b_bitwidth = int(np.asarray(model.get_initializer(conv.input[8])).reshape(()).item())
                bias_ints = quant_array_no_clipping(
                    np.asarray(new_bias_data),
                    scale=np.asarray(b_scale),
                    zeropt=b_zeropt,
                    bitwidth=b_bitwidth,
                    signed=bool(custom_node.get_nodeattr("b_signed")),
                    rounding_mode=custom_node.get_nodeattr("b_rounding_mode"),
                )
                i_min = int(np.min(bias_ints))
                i_max = int(np.max(bias_ints))
                required_bits = required_bitwidth_from_int_range(
                    i_min,
                    i_max,
                    signed=bool(custom_node.get_nodeattr("b_signed")),
                    narrow=bool(custom_node.get_nodeattr("b_narrow")),
                )
                if required_bits > b_bitwidth:
                    logger.info(
                        f"Increasing bias bitwidth for {conv.name} from {b_bitwidth} to {required_bits} after asymmetric folding."
                    )
                    model.set_initializer(conv.input[8], np.array(required_bits, dtype=np.int32))

            # Record that this conv now consumes its activation with a local effective zero point of 0.
            custom_node.set_nodeattr("asym_folding", 1)

            logger.info(f"Folded zero point of asymmetric activation quantization into bias of {conv.name}.")
        return (model, False)
