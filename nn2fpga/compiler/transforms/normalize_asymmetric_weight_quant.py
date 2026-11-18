import numpy as np
from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from nn2fpga.compiler.core.tensor_type import QuantizedTensorType

from nn2fpga.compiler.utils.quantization import quant_array


def _scalar_initializer(model: ModelWrapper, name: str, description: str):
    value = model.get_initializer(name)
    if value is None:
        raise ValueError(f"{description} initializer '{name}' is missing.")
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(
            f"{description} must be per-tensor (one value); got shape {value.shape}."
        )
    return value.reshape(()).item()


def _required_signed_bitwidth(i_min: int, i_max: int) -> int:
    for bitwidth in range(1, 33):
        if -(1 << (bitwidth - 1)) <= i_min and i_max <= (1 << (bitwidth - 1)) - 1:
            return bitwidth
    raise ValueError(
        f"Centered weight range [{i_min}, {i_max}] requires more than 32 bits."
    )


def _required_unsigned_bitwidth(i_max: int) -> int:
    for bitwidth in range(1, 33):
        if i_max <= (1 << bitwidth) - 1:
            return bitwidth
    raise ValueError(f"Centered weight maximum {i_max} requires more than 32 bits.")


class NormalizeAsymmetricWeightQuant(Transformation):
    """Center constant per-tensor Conv/Gemm weight quantizers in the FPGA partition."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        for quant in list(model.graph.node):
            if quant.op_type not in {"Quant", "IntQuant"}:
                continue

            consumers = model.find_direct_successors(quant) or []
            if not consumers or any(
                consumer.op_type not in {"Conv", "Gemm"}
                or len(consumer.input) < 2
                or consumer.input[1] != quant.output[0]
                for consumer in consumers
            ):
                continue

            raw_weights = model.get_initializer(quant.input[0])
            if raw_weights is None:
                raise ValueError(
                    f"Weight quantizer '{quant.name}' must have a constant weight initializer."
                )

            quant_type = QuantizedTensorType.from_quant_node(quant, model)

            deployed_codes = quant_array(
                raw_weights,
                scale=quant_type.scale,
                zeropt=quant_type.zeropt,
                bitwidth=quant_type.bitwidth,
                signed=quant_type.signed,
                narrow=quant_type.narrow,
                rounding_mode=quant_type.rounding_mode,
            ).astype(np.int64)
            centered_codes = deployed_codes - quant_type.zeropt
            centered_min = int(np.min(centered_codes))
            centered_max = int(np.max(centered_codes))
            centered_signed = centered_min < 0
            required_bitwidth = (
                _required_signed_bitwidth(centered_min, centered_max)
                if centered_signed
                else _required_unsigned_bitwidth(centered_max)
            )
            centered_bitwidth = max(quant_type.bitwidth, required_bitwidth)

            # Replacing the raw values preserves the quantizer's dequantized output
            # while making the values stored by the accelerator centered integers.
            centered_weights = (centered_codes * quant_type.scale).astype(np.float32)
            reproduced_codes = quant_array(
                centered_weights,
                scale=quant_type.scale,
                zeropt=0,
                bitwidth=centered_bitwidth,
                signed=centered_signed,
                narrow=False,
                rounding_mode=quant_type.rounding_mode,
            ).astype(np.int64)

            if not np.array_equal(reproduced_codes, centered_codes):
                mismatch_indices = np.argwhere(reproduced_codes != centered_codes)
                examples = []
                for index in mismatch_indices[:16]:
                    index = tuple(index)
                    examples.append(
                        "index={index}: raw={raw}, deployed_code={deployed}, "
                        "centered_code={centered}, replacement={replacement}, "
                        "reproduced_code={reproduced}".format(
                            index=index,
                            raw=np.asarray(raw_weights)[index],
                            deployed=deployed_codes[index],
                            centered=centered_codes[index],
                            replacement=centered_weights[index],
                            reproduced=reproduced_codes[index],
                        )
                    )
                raise ValueError(
                    f"Could not exactly center deployed weights for quantizer '{quant.name}': "
                    f"{len(mismatch_indices)} mismatches. "
                    f"scale={quant_type.scale}, zeropt={quant_type.zeropt}, original_bitwidth={quant_type.bitwidth}, "
                    f"centered_bitwidth={centered_bitwidth}, original_signed={quant_type.signed}, "
                    f"centered_signed={centered_signed}, original_narrow={quant_type.narrow}, "
                    f"rounding_mode={quant_type.rounding_mode}.\n"
                    + "\n".join(examples)
                )

            model.set_initializer(quant.input[0], centered_weights)
            model.set_initializer(quant.input[2], np.array(0, dtype=np.int32))
            model.set_initializer(quant.input[3], np.array(centered_bitwidth, dtype=np.int32))
            signed_attr = get_by_name(quant.attribute, "signed")
            narrow_attr = get_by_name(quant.attribute, "narrow")
            signed_attr.i = int(centered_signed)
            if narrow_attr is not None:
                narrow_attr.i = 0

        return model, False
