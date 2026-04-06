from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_quant import TensorQuant
from qonnx.custom_op.registry import getCustomOp
import logging
import numpy as np
import math
from nn2fpga.compiler.transforms.add_streaming_params import quant_array, safe_int_quant_call
from nn2fpga.compiler.core.tensor_quant import TensorQuant
logger = logging.getLogger(__name__)

NODES_WITH_BIAS = [
    "Conv",
]

def required_bitwidth_from_int_range(i_min: int, i_max: int, signed: bool, narrow: bool) -> int:
    # Determine minimal bitwidth that can represent [i_min, i_max]
    if signed:
        # For signed b bits:
        # narrow=False: [-2^(b-1), 2^(b-1)-1]
        # narrow=True : [-2^(b-1)+1, 2^(b-1)-1]
        abs_needed = max(abs(i_min), abs(i_max))
        if abs_needed == 0:
            return 1
        # Need (b-1) magnitude bits
        mag_bits = math.ceil(math.log2(abs_needed + 1))
        b = mag_bits + 1
        # If narrow and we need exactly -2^(b-1), bump
        if narrow:
            min_allowed = -(2 ** (b - 1)) + 1
            if i_min < min_allowed:
                b += 1
        return b
    else:
        # Unsigned b bits: [0, 2^b - 1]
        if i_min < 0:
            raise ValueError("Unsigned quant but negative integers required.")
        if i_max == 0:
            return 1
        return math.ceil(math.log2(i_max + 1))

class AdjustConvScale(Transformation):
    """
    Adjust bias tensor so that bias_scale matches input_scale * weight_scale
    while PRESERVING the original bias quantization granularity:

      - If bias was per-tensor -> keep per-tensor (scale stays scalar/[1])
      - If bias was per-channel -> keep per-channel (scale stays [C_out])

    Handles two modes only (no mixed granularity):
      - both per-tensor
      - both per-channel

    When the current bias_scale is smaller than target (ratio < 1) we will
    rescale weights (re-quantizing them by updating weight scale and
    bitwidth) so that input_scale * new_weight_scale == bias_scale.
    """

    def _as_1d(self, x) -> np.ndarray | None:
        if x is None:
            return None
        a = np.asarray(x)
        if a.ndim == 0:
            return a.reshape(1)
        if a.ndim == 1:
            return a
        return a.flatten()

    def _is_per_tensor(self, s: np.ndarray) -> bool:
        return s.size == 1

    def _broadcast_or_error(self, s: np.ndarray, c: int, name: str) -> np.ndarray:
        if s.size == 1:
            return np.full((c,), float(s[0]), dtype=np.float32)
        if s.size == c:
            return s.astype(np.float32)
        raise ValueError(f"{name} scale has {s.size} elements; expected 1 or {c}.")

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        for node in model.graph.node:
            if node.op_type not in NODES_WITH_BIAS:
                continue

            if len(node.input) <= 2 or not node.input[2]:
                continue

            bias_input = node.input[2]

            bias_q = model.find_producer(bias_input)
            w_q    = model.find_producer(node.input[1])
            x_q    = model.find_producer(node.input[0])

            if bias_q is None or w_q is None or x_q is None:
                logger.warning(f"Missing quant producer(s) for node {node.name}. Skipping.")
                continue

            bias_scale  = self._as_1d(model.get_initializer(bias_q.input[1]))
            weight_scale = self._as_1d(model.get_initializer(w_q.input[1]))
            input_scale  = self._as_1d(model.get_initializer(x_q.input[1]))

            if bias_scale is None or weight_scale is None or input_scale is None:
                logger.warning(f"Missing scale initializer(s) for node {node.name}. Skipping.")
                continue

            # Determine C_out for per-channel logic (weight/bias sizes must match in our simplified mode).
            if weight_scale.size > 1:
                c_out = weight_scale.size
            elif bias_scale.size > 1:
                c_out = bias_scale.size
            else:
                c_out = 1

            # Bias float initializer feeding QuantizeLinear (common pattern).
            bias_float_name = bias_q.input[0]
            bias_tensor = model.get_initializer(bias_float_name)
            if bias_tensor is None:
                logger.warning(f"Bias float initializer {bias_float_name} not found for {node.name}. Skipping.")
                continue

            bias_tensor = np.asarray(bias_tensor).astype(np.float32).reshape(-1)

            bias_was_per_tensor = self._is_per_tensor(bias_scale)
            weight_was_per_tensor = self._is_per_tensor(weight_scale)

            # Only support the two symmetric modes: both per-tensor or both per-channel
            if bias_was_per_tensor != weight_was_per_tensor:
                raise ValueError(
                    f"Mixed quant granularity detected in node {node.name}: "
                    f"bias per-tensor={bias_was_per_tensor}, weight per-tensor={weight_was_per_tensor}. "
                    "AdjustBiasScale only supports both per-tensor or both per-channel."
                )

            # Prepare TensorQuant objects to inspect quant params
            bias_tensor_quant = TensorQuant.from_quant_node(bias_q, model)
            weight_tensor_quant = TensorQuant.from_quant_node(w_q, model)

            # If per-tensor mode
            if bias_was_per_tensor:
                b_scalar = float(bias_scale[0])
                w_scalar = float(weight_scale[0])
                x_scalar = float(input_scale[0]) if input_scale.size == 1 else float(np.max(input_scale))

                if w_scalar * x_scalar > b_scalar:
                    # In case bias_scale < input_scale * weight_scale we need to rescale the weights
                    # as rescaling bias would require dividing the bias values, which could lead to
                    # small quantization errors.
                    quant_node = w_q
                    target_tensor_quant = weight_tensor_quant
                    target_tensor_scalar = b_scalar / x_scalar
                    ratio = (w_scalar * x_scalar) / b_scalar 
                else:
                    quant_node = bias_q
                    target_tensor_quant = bias_tensor_quant
                    target_tensor_scalar = w_scalar * x_scalar
                    ratio = b_scalar / (w_scalar * x_scalar)

                # Quantize existing floats according to target tensor quant params to get integer representation
                tensor_quantized = quant_array(
                    model.get_initializer(quant_node.input[0]),
                    target_tensor_quant.scale,
                    target_tensor_quant.zeropt,
                    target_tensor_quant.bitwidth,
                    signed=target_tensor_quant.signed,
                    narrow=target_tensor_quant.narrow,
                    rounding_mode=target_tensor_quant.rounding_mode,
                )

                # In case the original floating values were not quantized, we need to substitute them
                # to being able to change the scaling factor. Otherwise changing only the scaling factor
                # could result in rounding errors if the original floats were not exactly representable in the original quantization scheme.
                tensor_quantized_float = safe_int_quant_call(
                    model.get_initializer(quant_node.input[0]),
                    target_tensor_quant.scale,
                    target_tensor_quant.zeropt,
                    target_tensor_quant.bitwidth,
                    signed=target_tensor_quant.signed,
                    narrow=target_tensor_quant.narrow,
                    rounding_mode=target_tensor_quant.rounding_mode,
                )
                new_ints = tensor_quantized * ratio

                i_min = int(np.min(new_ints))
                i_max = int(np.max(new_ints))
                needed_bw = required_bitwidth_from_int_range(i_min, i_max, target_tensor_quant.signed, target_tensor_quant.narrow)
                if needed_bw > target_tensor_quant.bitwidth:
                    logger.info(f"Updating bitwidth of quant node {quant_node.name} from {target_tensor_quant.bitwidth} to {needed_bw}.")
                    model.set_initializer(quant_node.input[3], np.array(needed_bw, dtype=np.float32))

                # update bias quantizer scale to the target (per-tensor)
                model.set_initializer(quant_node.input[1], np.array([target_tensor_scalar], dtype=np.float32))
                logger.info(f"Adjusted per-tensor scale for node {node.name}")

                # Update the quantized floating tensor.
                model.set_initializer(quant_node.input[0], tensor_quantized_float.astype(np.float32))

            else:
                # per-channel mode: both weight and bias are per-channel and size should be c_out
                w_s = self._broadcast_or_error(weight_scale, c_out, "weight")
                x_s = self._broadcast_or_error(input_scale,  c_out, "input")
                b_s = self._broadcast_or_error(bias_scale,   c_out, "bias")

                # Determine channel-wise decision: if w_s * x_s > b_s then that channel requires
                # weight rescale (because bias scale < input*weight and dividing bias ints would lose bits).
                prod_s = (w_s * x_s).astype(np.float32)
                # small epsilon to avoid division by zero
                eps = 1e-12
                need_rescale_weights = prod_s > (b_s + eps)   # boolean array length c_out

                logger.debug(
                    f"Node {node.name}: per-channel mode: channels_need_weight_rescale={int(np.sum(need_rescale_weights))}/{c_out}"
                )

                # Load float initializers
                weight_float_name = w_q.input[0]
                weight_tensor = model.get_initializer(weight_float_name)
                if weight_tensor is None:
                    logger.warning(f"Weight float initializer {weight_float_name} not found for {node.name}. Cannot rescale weights. Skipping.")
                    continue
                weight_tensor = np.asarray(weight_tensor).astype(np.float32)

                bias_float_name = bias_q.input[0]
                bias_tensor = model.get_initializer(bias_float_name)
                if bias_tensor is None:
                    logger.warning(f"Bias float initializer {bias_float_name} not found for {node.name}. Skipping.")
                    continue
                bias_tensor = np.asarray(bias_tensor).astype(np.float32).reshape(-1)

                # We'll compute new scales per-channel:
                # - for channels where need_rescale_weights == True:
                #       new_w_scale[channel] = b_s[channel] / x_s[channel]
                #   and bias scale remains unchanged for that channel.
                # - for channels where need_rescale_weights == False:
                #       new_b_scale[channel] = w_s[channel] * x_s[channel]
                #   and weight scale remains unchanged for that channel.
                new_w_scale = w_s.copy()
                new_b_scale = b_s.copy()

                # Prepare ratios for integer-scaling computation.
                # For channels where weights are rescaled: we will compute ratio_w = (w_s * x_s) / b_s  (>1)
                # and multiply the existing quantized weight ints by ratio_w to get new ints.
                # For channels where biases are rescaled: ratio_b = b_s / (w_s * x_s)  (>1)
                ratio_w = np.ones((c_out,), dtype=np.float32)
                ratio_b = np.ones((c_out,), dtype=np.float32)

                # Channels that need weight rescale:
                mask_w = need_rescale_weights
                if np.any(mask_w):
                    # new weight scale for those channels
                    new_w_scale[mask_w] = (b_s[mask_w] / np.maximum(x_s[mask_w], eps)).astype(np.float32)
                    # ratio to scale old integer representation -> new integer representation
                    ratio_w[mask_w] = (w_s[mask_w] * x_s[mask_w]) / np.maximum(b_s[mask_w], eps)

                # Channels that will keep weight but rescale bias:
                mask_b = ~mask_w
                if np.any(mask_b):
                    new_b_scale[mask_b] = (w_s[mask_b] * x_s[mask_b]).astype(np.float32)
                    ratio_b[mask_b] = (b_s[mask_b] / np.maximum((w_s[mask_b] * x_s[mask_b]), eps))

                # --- Simulate weight integer changes to compute required bitwidth (if any) ---
                # Quantize weight floats according to current weight quant params to obtain integers
                # The quant_array call should accept a per-channel scale array (weight_tensor_quant.scale)
                # and will broadcast it across the weight tensor shape (C_out, ...).
                current_weight_ints = quant_array(
                    model.get_initializer(w_q.input[0]),
                    weight_tensor_quant.scale,
                    weight_tensor_quant.zeropt,
                    weight_tensor_quant.bitwidth,
                    signed=weight_tensor_quant.signed,
                    narrow=weight_tensor_quant.narrow,
                    rounding_mode=weight_tensor_quant.rounding_mode,
                ).astype(np.int64)

                # Broadcast ratio_w across the weight tensor's output-channel axis.
                # We assume the output channel is axis 0 (Conv weights shape: (C_out, C_in, kH, kW)).
                # Build a shape to broadcast ratio_w: (C_out, 1, 1, 1, ...)
                # Create shape of ones for the remaining axes:
                wt_shape = current_weight_ints.shape
                if wt_shape[0] != c_out:
                    # Unexpected shape; best-effort: try to broadcast ratio along first axis anyway.
                    logger.warning(f"Unexpected weight tensor shape {wt_shape} for node {node.name}; expected C_out=={c_out}. Proceeding with axis-0 broadcast.")
                # Build ratio_w broadcast shape
                broadcast_shape = [c_out] + [1] * (len(wt_shape) - 1)
                ratio_w_broadcast = ratio_w.reshape(broadcast_shape)

                # Compute new simulated weight ints after applying ratio (channels not affected will have ratio 1)
                simulated_new_weight_ints = (current_weight_ints.astype(np.float32) * ratio_w_broadcast).astype(np.int64)

                # Determine required bitwidth for weights from the simulated ints
                w_i_min = int(np.min(simulated_new_weight_ints))
                w_i_max = int(np.max(simulated_new_weight_ints))
                needed_w_bw = required_bitwidth_from_int_range(w_i_min, w_i_max, weight_tensor_quant.signed, weight_tensor_quant.narrow)

                if needed_w_bw > weight_tensor_quant.bitwidth:
                    logger.info(f"Updating bitwidth of weight quant node {w_q.name} from {weight_tensor_quant.bitwidth} to {needed_w_bw}")
                    model.set_initializer(w_q.input[3], np.array(needed_w_bw, dtype=np.float32))

                # Update weight scale initializer per-channel (only channels that needed weight rescale changed)
                model.set_initializer(w_q.input[1], new_w_scale.astype(np.float32))
                logger.info(f"Updated weight scale initializer for node {node.name} (per-channel).")

                # --- Simulate bias integer changes to compute required bitwidth (if any) ---
                # Quantize bias floats according to current bias quant params (per-channel vector)
                current_bias_ints = quant_array(
                    model.get_initializer(bias_q.input[0]),
                    bias_tensor_quant.scale,
                    bias_tensor_quant.zeropt,
                    bias_tensor_quant.bitwidth,
                    signed=bias_tensor_quant.signed,
                    narrow=bias_tensor_quant.narrow,
                    rounding_mode=bias_tensor_quant.rounding_mode,
                ).astype(np.int64)

                # Apply per-channel ratio_b: channels that were designated for bias-rescale will have ratio_b>1, else 1
                simulated_new_bias_ints = (current_bias_ints.astype(np.float32) * ratio_b).astype(np.int64)

                b_i_min = int(np.min(simulated_new_bias_ints))
                b_i_max = int(np.max(simulated_new_bias_ints))
                needed_b_bw = required_bitwidth_from_int_range(b_i_min, b_i_max, bias_tensor_quant.signed, bias_tensor_quant.narrow)

                if needed_b_bw > bias_tensor_quant.bitwidth:
                    logger.info(f"Updating bitwidth of bias quant node {bias_q.name} from {bias_tensor_quant.bitwidth} to {needed_b_bw}")
                    model.set_initializer(bias_q.input[3], np.array(needed_b_bw, dtype=np.float32))

                # Update bias scale initializer per-channel (only channels that needed bias rescale changed)
                model.set_initializer(bias_q.input[1], new_b_scale.astype(np.float32))
                logger.info(f"Updated bias scale initializer for node {node.name} (per-channel).")

                logger.info(
                    f"Per-channel adjustment for node {node.name}: "
                    f"{int(np.sum(mask_w))} channels had weights rescaled, {int(np.sum(mask_b))} channels had bias scales updated."
                )

        return model, False
