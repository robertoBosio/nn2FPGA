from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import TensorQuant
import logging
import numpy as np
from backend.transformation.add_streaming_params import safe_int_quant_call
from backend.core.tensor_quant import TensorQuant
logger = logging.getLogger(__name__)

NODES_WITH_BIAS = [
    "Conv",
]

class AdjustBiasScale(Transformation):
    """
    Adjust bias tensor so that bias_scale matches input_scale * weight_scale
    while PRESERVING the original bias quantization granularity:

      - If bias was per-tensor -> keep per-tensor (scale stays scalar/[1])
      - If bias was per-channel -> keep per-channel (scale stays [C_out])

    Handles mixed cases:
      - input per-tensor, weight per-channel
      - input per-channel, weight per-tensor (rare but supported)
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

            bias_was_per_tensor = self._is_per_tensor(bias_scale)

            # Determine C_out for per-channel logic.
            # Prefer weight scale length if per-channel; else bias scale length if per-channel; else 1.
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

            # If bias is per-channel, enforce it matches c_out.
            # If bias is per-tensor, allow scalar bias or vector bias — we will apply uniform rescale.
            if (not bias_was_per_tensor) and bias_tensor.size != c_out:
                raise ValueError(
                    f"Bias tensor for node {node.name} has {bias_tensor.size} elements "
                    f"but expected {c_out} for per-channel bias."
                )

            bias_tensor_quant = TensorQuant.from_quant_node(bias_q, model)
            bias_tensor_quantized = safe_int_quant_call(
                model.get_initializer(bias_q.input[0]),
                bias_tensor_quant.scale,
                bias_tensor_quant.zeropt,
                bias_tensor_quant.bitwidth,
                signed=bias_tensor_quant.signed,
                narrow=bias_tensor_quant.narrow,
                rounding_mode=bias_tensor_quant.rounding_mode,
            )

            # Build target bias scale, respecting original bias granularity.
            if bias_was_per_tensor:
                # Keep per-tensor bias quantization => choose a SINGLE scalar target.
                # We still want "uniform" bias scale, but if weight/input are per-channel,
                # we must reduce them to a scalar. Use max for safety (smaller int magnitudes).
                x_scalar = float(input_scale[0]) if input_scale.size == 1 else float(np.max(input_scale))
                w_scalar = float(weight_scale[0]) if weight_scale.size == 1 else float(np.max(weight_scale))
                target_bias_scale = np.array([x_scalar * w_scalar], dtype=np.float32)
                current_bias_scale = np.array([float(bias_scale[0])], dtype=np.float32)

                if np.allclose(current_bias_scale, target_bias_scale, rtol=1e-6, atol=0.0):
                    continue

                ratio = current_bias_scale[0] / target_bias_scale[0]
                new_bias_tensor = bias_tensor_quantized * ratio

                model.set_initializer(bias_float_name, new_bias_tensor.astype(np.float32))
                # model.set_initializer(bias_q.input[1], target_bias_scale)  # stays per-tensor
                logger.info(f"Adjusted per-tensor bias scale for node {node.name}")

            else:
                # Keep per-channel bias quantization => target is per-channel product.
                w_s = self._broadcast_or_error(weight_scale, c_out, "weight")
                x_s = self._broadcast_or_error(input_scale,  c_out, "input")
                b_s = self._broadcast_or_error(bias_scale,   c_out, "bias")

                target_bias_scale = (w_s * x_s).astype(np.float32)

                if np.allclose(b_s, target_bias_scale, rtol=1e-6, atol=0.0):
                    continue

                ratio = b_s / target_bias_scale
                # bias_tensor is [C_out] for per-channel case
                new_bias_tensor = bias_tensor_quantized * ratio

                model.set_initializer(bias_float_name, new_bias_tensor.astype(np.float32))
                # model.set_initializer(bias_q.input[1], target_bias_scale.astype(np.float32))  # stays per-channel
                logger.info(f"Adjusted per-channel bias scale for node {node.name}")

        return model, False
