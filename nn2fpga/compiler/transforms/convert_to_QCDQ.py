from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.qonnx_to_qcdq import QuantToQCDQ
from qonnx.custom_op.registry import getCustomOp 
from nn2fpga.compiler.transforms.add_streaming_params import quant_array
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.core.tensor_quant import TensorQuant 
from onnx import TensorProto, helper, numpy_helper
import onnx.shape_inference as si
from onnxscript.rewriter import pattern, rewrite
from onnxscript import ir
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_tensorproto_dtype(bitwidth, signed):
    """Get the TensorProto data type based on bitwidth and signedness."""
    bitwidth = int(bitwidth)
    signed = bool(signed)

    if bitwidth <= 8:
        return TensorProto.INT8 if signed else TensorProto.UINT8
    elif bitwidth <= 32:
        return TensorProto.INT32 if signed else TensorProto.UINT32
    else:
        raise ValueError(f"Unsupported bitwidth for quantization: {bitwidth}")


def get_numpy_dtype(bitwidth, signed):
    """Get the numpy dtype matching the ONNX quantized type."""
    bitwidth = int(bitwidth)
    signed = bool(signed)

    if bitwidth <= 8:
        return np.int8 if signed else np.uint8
    elif bitwidth <= 32:
        return np.int32 if signed else np.uint32
    else:
        raise ValueError(f"Unsupported bitwidth for quantization: {bitwidth}")

def toNHWC(tensor_shape):
    """Convert a tensor shape from NCHW to NHWC format."""
    NHWC_shape = [tensor_shape[0]]  # Batch size
    NHWC_shape.extend(tensor_shape[2:])  # Height and Width
    NHWC_shape.append(tensor_shape[1])  # Channels
    return NHWC_shape

def toNCHW(tensor_shape):
    """Convert a tensor shape from NHWC to NCHW format."""
    NCHW_shape = [tensor_shape[0]]  # Batch size
    NCHW_shape.append(tensor_shape[-1])  # Channels
    NCHW_shape.extend(tensor_shape[1:-1])  # Height and Width
    return NCHW_shape

def constant_quant_pattern(
    qonnx_op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode
):
    return qonnx_op.Quant(
        x,
        scale,
        zero_point,
        bitwidth,
        signed=signed,
        narrow=narrow,
        _allow_other_attributes=True,
        _domain="qonnx.custom_op.general",
    )


def dynamic_quant_pattern(
    qonnx_op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode
):
    return qonnx_op.Quant(
        x,
        scale,
        zero_point,
        bitwidth,
        signed=signed,
        narrow=narrow,
        _allow_other_attributes=True,
        _domain="qonnx.custom_op.general",
    )


def _extract_scalar_const(v):
    if v.const_value is None:
        return None
    arr = v.const_value.numpy()
    if arr.shape != ():
        return None
    return arr.item()


def _extract_optional_attr_value(v, default=None):
    if v is None:
        return default
    return getattr(v, "value", default)


def _is_supported_quant_config(bitwidth, signed, narrow, rounding_mode):
    if bitwidth is None:
        return False

    bitwidth = int(bitwidth)
    signed = bool(signed)
    narrow = bool(narrow)

    # Match the dtypes you currently support in get_tensorproto_dtype()
    if bitwidth not in (8, 32):
        logger.warning("Skipping Quant lowering: unsupported bitwidth=%s", bitwidth)
        return False

    # ONNX QuantizeLinear cannot represent narrow range directly
    if narrow:
        logger.warning("Skipping Quant lowering: narrow=True is not representable by plain Q/DQ")
        return False

    # QuantizeLinear uses ONNX rounding/saturation semantics; do not silently
    # lower if the model explicitly asked for a different mode.
    if rounding_mode not in (None, "ROUND"):
        logger.warning(
            "Skipping Quant lowering: unsupported rounding_mode=%s for plain Q/DQ",
            rounding_mode,
        )
        return False

    return True


def is_quant_with_constant_input(
    context, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode, **_
):
    # Require constant x/scale/zero_point/bitwidth for full constant folding
    if not all(i.const_value is not None for i in [x, scale, zero_point, bitwidth]):
        return False

    bitwidth_scalar = _extract_scalar_const(bitwidth)
    if bitwidth_scalar is None:
        return False

    signed_val = _extract_optional_attr_value(signed, False)
    narrow_val = _extract_optional_attr_value(narrow, False)
    rounding_mode_val = _extract_optional_attr_value(rounding_mode, "ROUND")

    if not _is_supported_quant_config(
        bitwidth_scalar, signed_val, narrow_val, rounding_mode_val
    ):
        return False

    scale_val = scale.const_value.numpy()
    zero_point_val = zero_point.const_value.numpy()

    # Keep the same restrictions as your original pass
    if scale_val.ndim > 1:
        return False
    if zero_point_val.ndim > 1:
        return False

    return True


def _make_constant_tensor_value(op, name, np_value, onnx_dtype):
    return op.Constant(
        value=helper.make_tensor(
            name=name,
            data_type=onnx_dtype,
            dims=list(np_value.shape),
            vals=np_value.flatten().tolist(),
        )
    )


def _make_zero_point_input(op, zero_point, bitwidth, signed):
    """
    Ensure the zero-point input is an integer tensor of the same type as the
    quantized output, as required by ONNX QuantizeLinear/DequantizeLinear.
    """
    target_onnx_dtype = get_tensorproto_dtype(bitwidth, signed)
    target_np_dtype = get_numpy_dtype(bitwidth, signed)

    # Best case: constant zero-point, possibly float or wrong integer width
    if zero_point.const_value is not None:
        zp_np = zero_point.const_value.numpy()
        zp_np = np.rint(zp_np).astype(target_np_dtype, copy=False)
        return _make_constant_tensor_value(
            op,
            name=f"{zero_point.name}_qcdq_cast",
            np_value=zp_np,
            onnx_dtype=target_onnx_dtype,
        )

    # If zero_point is not constant, we cannot safely retag its type here
    # with the simple rewriter API unless we introduce a full Cast node with
    # the exact target type. In most QONNX Quant nodes zero_point is constant.
    logger.warning(
        "Skipping Quant lowering: zero_point for non-constant input is not constant (%s)",
        zero_point.name,
    )
    return None


def quant_constant_to_dequant(
    op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode
):
    x_np = x.const_value.numpy()
    scale_np = scale.const_value.numpy().squeeze()
    zero_point_np = zero_point.const_value.numpy().squeeze()
    bitwidth_np = int(bitwidth.const_value.numpy().squeeze())
    signed_val = bool(_extract_optional_attr_value(signed, False))
    narrow_val = bool(_extract_optional_attr_value(narrow, False))
    rounding_mode_val = _extract_optional_attr_value(rounding_mode, "ROUND")

    c_x = quant_array(
        x_np,
        scale_np,
        zero_point_np,
        bitwidth_np,
        signed=signed_val,
        narrow=narrow_val,
        rounding_mode=rounding_mode_val,
    )

    data_type = get_tensorproto_dtype(bitwidth_np, signed_val)
    quantized_const = _make_constant_tensor_value(
        op,
        name=f"quantized_{x.name}",
        np_value=np.asarray(c_x),
        onnx_dtype=data_type,
    )

    zp_input = _make_zero_point_input(op, zero_point, bitwidth_np, signed_val)
    if zp_input is None:
        # Should not happen because the checker requires constant zero_point.
        return op.Identity(x)

    return op.DequantizeLinear(quantized_const, scale, zp_input)


def is_dynamic_quant_rewritable(
    context, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode, **_
):
    # Exclude the constant-input case, which is handled above.
    if x.const_value is not None:
        return False

    # We still need static quantization parameters
    if scale.const_value is None or zero_point.const_value is None or bitwidth.const_value is None:
        return False

    bitwidth_scalar = _extract_scalar_const(bitwidth)
    if bitwidth_scalar is None:
        return False

    signed_val = _extract_optional_attr_value(signed, False)
    narrow_val = _extract_optional_attr_value(narrow, False)
    rounding_mode_val = _extract_optional_attr_value(rounding_mode, "ROUND")

    if not _is_supported_quant_config(
        bitwidth_scalar, signed_val, narrow_val, rounding_mode_val
    ):
        return False

    scale_val = scale.const_value.numpy()
    zero_point_val = zero_point.const_value.numpy()

    # Safe/simple lowering path: per-tensor quantization.
    # If you need per-channel too, you must infer and set the axis attribute.
    if scale_val.ndim != 0 or zero_point_val.ndim != 0:
        logger.warning(
            "Skipping Quant lowering: only scalar scale/zero_point are handled in dynamic Q/DQ rewrite"
        )
        return False

    return True

def create_const_initializer(model, value, dtype):
    init_name = model.make_new_valueinfo_name()
    model.set_initializer(
        init_name,
        np.array(value, dtype=dtype),
    )
    return init_name

def quant_to_qcdq(
    op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode
):
    bitwidth_val = int(bitwidth.const_value.numpy().squeeze())
    signed_val = bool(signed.value)

    zp_input = _make_zero_point_input(op, zero_point, bitwidth_val, signed_val)
    if zp_input is None:
        return op.Identity(x)

    q = op.QuantizeLinear(x, scale, zp_input)
    dq = op.DequantizeLinear(q, scale, zp_input)
    return dq


class ConvertToQCDQ(Transformation):
    """Convert QONNX Quant nodes to ONNX Q/DQ."""

    def __init__(self):
        self._rewrite_rule_set = pattern.RewriteRuleSet(
            [
                # Fold Quant(constant) -> DequantizeLinear(Constant(...), scale, zp)
                pattern.RewriteRule(
                    constant_quant_pattern,
                    quant_constant_to_dequant,
                    is_quant_with_constant_input,
                ),
                # Lower remaining Quant(x, scale, zp, bw) -> DQ(Q(x))
                pattern.RewriteRule(
                    dynamic_quant_pattern,
                    quant_to_qcdq,
                    is_dynamic_quant_rewritable,
                ),
            ],
            commute=True,
        )

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        model_proto = model.model

        # First let QONNX handle the easy native cases it supports.
        model = ModelWrapper(model_proto)
        model = model.transform(QuantToQCDQ())

        # Then rewrite the leftover Quant nodes (e.g. nonzero zero-point).
        ir_model = ir.from_proto(model.model)
        ir_model = rewrite(ir_model, pattern_rewrite_rules=self._rewrite_rule_set)
        model_proto = ir.to_proto(ir_model)

        # Re-run shape inference
        model_proto = si.infer_shapes(model_proto)
        model = ModelWrapper(model_proto)

        # Add transpose and QuantizeLinear/DequantizeLinear nodes around the nn2fpgaPartition node
        partition_nodes = model.get_nodes_by_op_type("nn2fpgaPartition")
        partition_node = partition_nodes[0] if partition_nodes else None

        if partition_node:
            ap = AcceleratorPackage.from_json(
                getCustomOp(partition_node).get_nodeattr("accelerator_package")
            )

            new_inputs_map = {}
            for i, inp in enumerate(partition_node.input):

                if (
                    model.find_producer(inp) is not None
                    and model.find_producer(inp).op_type == "QuantizeLinear"
                ):
                    # Skip if the input is already quantized
                    continue

                inp_shape = model.get_tensor_shape(inp)
                if inp_shape is None:
                    continue  # Skip if input shape is not available

                inp_shape_nhwc = toNHWC(inp_shape)

                # If the input is already in NHWC format, skip the transpose
                # This is not a reliable test to understand the layout of the tensor.
                quant_input_name = inp
                if inp_shape != inp_shape_nhwc:
                    quant_input_name = f"{inp}_transposed"

                    perm = list(range(len(inp_shape_nhwc)))
                    perm = toNHWC(perm)  # Convert to NHWC permutation
                    transpose_before = helper.make_node(
                        "Transpose",
                        name=f"{inp}_transpose",
                        inputs=[inp],
                        outputs=[f"{inp}_transposed"],
                        perm=perm,
                    )
                    model.set_tensor_shape(
                        f"{inp}_transposed",
                        inp_shape_nhwc,
                    )
                    model.graph.node.append(transpose_before)

                input_tensor_quant = TensorQuant.from_canonical_name(ap.input_map[inp]["quant"])
                scale_init_name = create_const_initializer(
                    model,
                    input_tensor_quant.scale,
                    np.float32,
                )
                zeropt_init_name = create_const_initializer(
                    model,
                    input_tensor_quant.zeropt,
                    input_tensor_quant.get_numpy_dtype()
                )

                quantize_node = helper.make_node(
                    "QuantizeLinear",
                    inputs=[quant_input_name, scale_init_name, zeropt_init_name],
                    outputs=[f"{inp}_quantized"],
                    name=f"{inp}_quantize",
                    axis=len(inp_shape_nhwc) - 1,  # Channel axis for NHWC
                )
                model.set_tensor_shape(
                    f"{inp}_quantized",
                    inp_shape_nhwc,
                    dtype=input_tensor_quant.get_tensorproto_dtype(),
                )

                model.graph.node.append(quantize_node)
                new_inputs_map[inp] = (i, f"{inp}_quantized")

            if new_inputs_map:
                # Replace the inputs with the transposed versions
                for old_name, (index, new_name) in new_inputs_map.items():
                    partition_node.input[index] = new_name

                # 2) Update the input map in the accelerator package and preserve the order
                rename = {old: new for old, (_, new) in new_inputs_map.items()}
                old_map = ap.input_map
                ap.input_map = {rename.get(k, k): v for k, v in old_map.items()}

                # 3) Update shapes
                for old_name, (_, new_name) in new_inputs_map.items():
                    ap.input_map[new_name]["shape"] = toNHWC(ap.input_map[new_name]["shape"])

            new_outputs_map = {}
            for i, out in enumerate(partition_node.output):
                consumers = model.find_consumers(out)

                if consumers is not None and all(
                    consumer.op_type == "DequantizeLinear" for consumer in consumers
                ):
                    # Skip if the output is already dequantized
                    continue

                out_shape = model.get_tensor_shape(out)
                if out_shape is None:
                    continue

                # Compute the shape in output to the nn2fpgaPartition node which is channel last format
                out_shape_nhwc = toNHWC(out_shape)

                # If the shapes in channel last and channel first formats are the same, skip
                # the transpose node and assign the output directly to the dequantize node
                dequant_output_name = f"{out}_dequantized"
                if out_shape == out_shape_nhwc:
                    dequant_output_name = out

                output_tensor_quant = TensorQuant.from_canonical_name(ap.output_map[out]["quant"])
                scale_init_name = create_const_initializer(
                    model,
                    output_tensor_quant.scale,
                    np.float32,
                )
                zeropt_init_name = create_const_initializer(
                    model,
                    output_tensor_quant.zeropt,
                    output_tensor_quant.get_numpy_dtype()
                )

                dequantize_node = helper.make_node(
                    "DequantizeLinear",
                    inputs=[f"{out}_quantized", scale_init_name, zeropt_init_name],
                    outputs=[dequant_output_name],
                    name=f"{out}_dequantize",
                    axis=len(out_shape_nhwc) - 1,  # Channel axis for NHWC
                )

                model.set_tensor_shape(
                    f"{out}_quantized",
                    out_shape_nhwc,
                    dtype=output_tensor_quant.get_tensorproto_dtype(),
                )

                model.graph.node.append(dequantize_node)
                if out_shape != out_shape_nhwc:
                    # Add a Transpose node after the partition node
                    perm = list(range(len(out_shape_nhwc)))
                    perm = toNCHW(perm)  # Convert to NCHW permutation
                    # Create a Transpose node to convert from NHWC to NCHW
                    # This is needed because the output of the partition node is in NHWC format
                    # but the rest of the model expects NCHW format

                    transpose_after = helper.make_node(
                        "Transpose",
                        name=f"{out}_transpose",
                        inputs=[f"{out}_dequantized"],
                        outputs=[out],
                        perm=perm,  # NHWC to NCHW
                    )
                    model.set_tensor_shape(
                        f"{out}_dequantized",
                        out_shape_nhwc,
                    )

                    model.graph.node.append(transpose_after)
                new_outputs_map[out] = (i, f"{out}_quantized")

            if new_outputs_map:
                # Replace the outputs with the transposed versions
                for old_name, (index, new_name) in new_outputs_map.items():
                    # Update the partition node output
                    partition_node.output[index] = new_name

                # 2) Update the output map in the accelerator package and preserve the order
                rename = {old: new for old, (_, new) in new_outputs_map.items()}
                old_map = ap.output_map
                ap.output_map = {rename.get(k, k): v for k, v in old_map.items()}

                # 3) Update shapes
                for old_name, (_, new_name) in new_outputs_map.items():
                    ap.output_map[new_name]["shape"] = toNHWC(ap.output_map[new_name]["shape"])

            # Set the updated accelerator package back to the partition node
            getCustomOp(partition_node).set_nodeattr(
                "accelerator_package", ap.to_json()
            )

        return model, False
