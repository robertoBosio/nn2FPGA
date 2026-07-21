from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.qonnx_to_qcdq import QuantToQCDQ
from qonnx.custom_op.registry import getCustomOp
from nn2fpga.compiler.transforms.add_streaming_params import quant_array
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.core.tensor_type import TensorType, QuantizedTensorType
from nn2fpga.compiler.core.tensor_layout import TensorLayout
from onnx import TensorProto, helper, numpy_helper
import onnx.shape_inference as si
from onnxscript.rewriter import pattern, rewrite
from onnxscript import ir
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Existing helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def get_tensorproto_dtype(bitwidth, signed):
    bitwidth = int(bitwidth)
    signed   = bool(signed)
    if bitwidth <= 8:
        return TensorProto.INT8  if signed else TensorProto.UINT8
    elif bitwidth <= 16:
        # INT16/UINT16 not supported by Q/DQ before opset 21 — promote to 32-bit
        return TensorProto.INT32 if signed else TensorProto.UINT32
    elif bitwidth <= 32:
        return TensorProto.INT32 if signed else TensorProto.UINT32
    else:
        raise ValueError(f"Unsupported bitwidth for quantization: {bitwidth}")


def get_numpy_dtype(bitwidth, signed):
    bitwidth = int(bitwidth)
    signed   = bool(signed)
    if bitwidth <= 8:
        return np.int8  if signed else np.uint8
    elif bitwidth <= 16:
        return np.int32 if signed else np.uint32
    elif bitwidth <= 32:
        return np.int32 if signed else np.uint32
    else:
        raise ValueError(f"Unsupported bitwidth for quantization: {bitwidth}")


def constant_quant_pattern(
    qonnx_op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode
):
    return qonnx_op.Quant(
        x, scale, zero_point, bitwidth,
        signed=signed, narrow=narrow,
        _allow_other_attributes=True,
        _domain="qonnx.custom_op.general",
    )


def dynamic_quant_pattern(
    qonnx_op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode
):
    return qonnx_op.Quant(
        x, scale, zero_point, bitwidth,
        signed=signed, narrow=narrow,
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
    narrow   = bool(narrow)
    if bitwidth not in (8, 16, 32):
        logger.warning("Skipping Quant lowering: unsupported bitwidth=%s", bitwidth)
        return False
    if narrow:
        logger.warning("Skipping Quant lowering: narrow=True not representable by plain Q/DQ")
        return False
    if rounding_mode not in (None, "ROUND"):
        logger.warning("Skipping Quant lowering: unsupported rounding_mode=%s", rounding_mode)
        return False
    return True


def is_quant_with_constant_input(
    context, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode, **_
):
    if not all(i.const_value is not None for i in [x, scale, zero_point, bitwidth]):
        return False
    bitwidth_scalar   = _extract_scalar_const(bitwidth)
    if bitwidth_scalar is None:
        return False
    signed_val        = _extract_optional_attr_value(signed, False)
    narrow_val        = _extract_optional_attr_value(narrow, False)
    rounding_mode_val = _extract_optional_attr_value(rounding_mode, "ROUND")
    if not _is_supported_quant_config(bitwidth_scalar, signed_val, narrow_val, rounding_mode_val):
        return False
    if scale.const_value.numpy().ndim > 1 or zero_point.const_value.numpy().ndim > 1:
        return False
    return True


def _make_constant_tensor_value(op, name, np_value, onnx_dtype):
    return op.Constant(
        value=helper.make_tensor(
            name=name, data_type=onnx_dtype,
            dims=list(np_value.shape),
            vals=np_value.flatten().tolist(),
        )
    )


def _make_zero_point_input(op, zero_point, bitwidth, signed):
    target_onnx_dtype = get_tensorproto_dtype(bitwidth, signed)
    target_np_dtype   = get_numpy_dtype(bitwidth, signed)
    if zero_point.const_value is not None:
        zp_np = np.rint(zero_point.const_value.numpy()).astype(target_np_dtype, copy=False)
        return _make_constant_tensor_value(
            op, name=f"{zero_point.name}_qcdq_cast",
            np_value=zp_np, onnx_dtype=target_onnx_dtype,
        )
    logger.warning("Skipping Quant lowering: zero_point is not constant (%s)", zero_point.name)
    return None


def quant_constant_to_dequant(
    op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode
):
    x_np              = x.const_value.numpy()
    scale_np          = scale.const_value.numpy().squeeze()
    zero_point_np     = zero_point.const_value.numpy().squeeze()
    bitwidth_np       = int(bitwidth.const_value.numpy().squeeze())
    signed_val        = bool(_extract_optional_attr_value(signed, False))
    narrow_val        = bool(_extract_optional_attr_value(narrow, False))
    rounding_mode_val = _extract_optional_attr_value(rounding_mode, "ROUND")

    c_x = quant_array(
        x_np, scale_np, zero_point_np, bitwidth_np,
        signed=signed_val, narrow=narrow_val, rounding_mode=rounding_mode_val,
    )
    data_type       = get_tensorproto_dtype(bitwidth_np, signed_val)
    quantized_const = _make_constant_tensor_value(
        op, name=f"quantized_{x.name}",
        np_value=np.asarray(c_x), onnx_dtype=data_type,
    )
    zp_input = _make_zero_point_input(op, zero_point, bitwidth_np, signed_val)
    if zp_input is None:
        return op.Identity(x)
    return op.DequantizeLinear(quantized_const, scale, zp_input)


def is_dynamic_quant_rewritable(
    context, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode, **_
):
    if x.const_value is not None:
        return False
    if any(v.const_value is None for v in [scale, zero_point, bitwidth]):
        return False
    bitwidth_scalar   = _extract_scalar_const(bitwidth)
    if bitwidth_scalar is None:
        return False
    signed_val        = _extract_optional_attr_value(signed, False)
    narrow_val        = _extract_optional_attr_value(narrow, False)
    rounding_mode_val = _extract_optional_attr_value(rounding_mode, "ROUND")
    if not _is_supported_quant_config(bitwidth_scalar, signed_val, narrow_val, rounding_mode_val):
        return False
    if scale.const_value.numpy().ndim != 0 or zero_point.const_value.numpy().ndim != 0:
        logger.warning(
            "Skipping Quant lowering: only scalar scale/zero_point handled in dynamic Q/DQ rewrite"
        )
        return False
    return True


def create_const_initializer(model, value, dtype):
    init_name = model.make_new_valueinfo_name()
    model.set_initializer(init_name, np.array(value, dtype=dtype))
    return init_name


def quant_to_qcdq(op, x, scale, zero_point, bitwidth, signed, narrow, rounding_mode):
    bitwidth_val = int(bitwidth.const_value.numpy().squeeze())
    signed_val   = bool(signed.value)
    zp_input = _make_zero_point_input(op, zero_point, bitwidth_val, signed_val)
    if zp_input is None:
        return op.Identity(x)
    q  = op.QuantizeLinear(x, scale, zp_input)
    dq = op.DequantizeLinear(q, scale, zp_input)
    return dq


# ─────────────────────────────────────────────────────────────────────────────
# Main transformation
# ─────────────────────────────────────────────────────────────────────────────

class ConvertToQCDQ(Transformation):
    """Convert QONNX Quant nodes to ONNX Q/DQ.

    Transposes around the nn2fpgaPartition node are inserted in the quantized
    (INT8) domain so they operate on 1-byte elements rather than 4-byte floats:

      Input side:
        float → QuantizeLinear(float→int8) → Transpose(int8) → partition

      Output side:
        partition → Transpose(int8) → DequantizeLinear(int8→float)

    """

    def __init__(self):
        self._rewrite_rule_set = pattern.RewriteRuleSet(
            [
                pattern.RewriteRule(
                    constant_quant_pattern,
                    quant_constant_to_dequant,
                    is_quant_with_constant_input,
                ),
                pattern.RewriteRule(
                    dynamic_quant_pattern,
                    quant_to_qcdq,
                    is_dynamic_quant_rewritable,
                ),
            ],
            commute=True,
        )

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        if model.get_nodes_by_op_type("Quant") != []:
            model = model.transform(QuantToQCDQ())

            ir_model    = ir.from_proto(model.model)
            ir_model    = rewrite(ir_model, pattern_rewrite_rules=self._rewrite_rule_set)
            model_proto = ir.to_proto(ir_model)

            model = ModelWrapper(model_proto)
            assert model.get_nodes_by_op_type("Quant") == [], \
                "Not all Quant nodes were rewritten to QCDQ pattern"

        partition_nodes = model.get_nodes_by_op_type("nn2fpgaPartition")
        partition_node  = partition_nodes[0] if partition_nodes else None

        if partition_node:
            ap = AcceleratorPackage.from_json(
                getCustomOp(partition_node).get_nodeattr("accelerator_package")
            )

            # ── Inputs: QuantizeLinear(float→int8) → Transpose(int8) ─────
            new_inputs_map = {}
            for i, inp in enumerate(partition_node.input):

                if (
                    model.find_producer(inp) is not None
                    and model.find_producer(inp).op_type == "QuantizeLinear"
                ):
                    continue

                inp_shape = model.get_tensor_shape(inp)
                if inp_shape is None:
                    continue

                input_layout = TensorLayout.from_canonical_name(
                    ap.input_map[inp].get("layout")
                )
                assert len(input_layout.perm) == len(
                    inp_shape
                ), f"Input layout perm length {len(input_layout.perm)} does not match input shape length {len(inp_shape)} for input '{inp}'"
                inp_shape_perm = [inp_shape[dim] for dim in input_layout.perm]

                input_tensor_type = TensorType.from_canonical_name(
                    ap.input_map[inp]["quant"]
                )
                
                tensor_name = inp
                if isinstance(input_tensor_type, QuantizedTensorType):
                    # The expected input tensor is quantized, thus we need 
                    # to insert QuantizeLinear before the partition input.
                    scale_init_name  = create_const_initializer(
                        model, input_tensor_type.scale, np.float32
                    )
                    zeropt_init_name = create_const_initializer(
                        model, input_tensor_type.zeropt,
                        input_tensor_type.get_numpy_dtype()
                    )
                    # 1. Quantize the original float tensor (shape unchanged)
                    quantize_node = helper.make_node(
                        "QuantizeLinear",
                        inputs=[inp, scale_init_name, zeropt_init_name],
                        outputs=[f"{inp}_quantized"],
                        name=f"{inp}_quantize",
                        axis=len(inp_shape) - 1,
                    )
                    model.set_tensor_shape(
                        f"{inp}_quantized",
                        inp_shape,
                        dtype=input_tensor_type.get_tensorproto_dtype(),
                    )
                    model.graph.node.append(quantize_node)
                    tensor_name = f"{inp}_quantized"

                if not input_layout.is_identity():
                    # The expected input layout is not the onnx layout, 
                    # thus we need to insert a Transpose.

                    transpose_node = helper.make_node(
                        "Transpose",
                        name=f"{inp}_transpose",
                        inputs=[f"{tensor_name}"],
                        outputs=[f"{tensor_name}_transposed"],
                        perm=input_layout.perm,
                    )
                    model.set_tensor_shape(
                        f"{tensor_name}_transposed",
                        inp_shape_perm,
                        dtype=input_tensor_type.get_tensorproto_dtype(),
                    )
                    model.graph.node.append(transpose_node)
                    tensor_name = f"{tensor_name}_transposed"

                new_inputs_map[inp] = (i, f"{tensor_name}")

            if new_inputs_map:
                for old_name, (index, new_name) in new_inputs_map.items():
                    partition_node.input[index] = new_name

                rename       = {old: new for old, (_, new) in new_inputs_map.items()}
                ap.input_map = {rename.get(k, k): v for k, v in ap.input_map.items()}

                for old_name, (_, new_name) in new_inputs_map.items():
                    input_layout = TensorLayout.from_canonical_name(
                        ap.input_map[new_name].get("layout")
                    )
                    inp_shape    = ap.input_map[new_name]["shape"]
                    inp_perm     = [dim for dim in input_layout.perm if dim < len(inp_shape)]
                    ap.input_map[new_name]["shape"] = [inp_shape[dim] for dim in inp_perm]

            # ── Outputs: Transpose(int8) → DequantizeLinear(int8→float) ──
            new_outputs_map = {}
            for i, out in enumerate(partition_node.output):
                consumers = model.find_consumers(out)

                if len(consumers) > 0 and all(
                    consumer.op_type == "DequantizeLinear" for consumer in consumers
                ):
                    continue

                out_shape = model.get_tensor_shape(out)
                if out_shape is None:
                    continue

                output_layout = TensorLayout.from_canonical_name(
                    ap.output_map[out].get("layout")
                )
                assert len(output_layout.perm) == len(
                    out_shape
                ), f"Output layout perm length {len(output_layout.perm)} does not match output shape length {len(out_shape)} for output '{out}'"
                out_shape_perm = [out_shape[dim] for dim in output_layout.perm]

                output_tensor_type = TensorType.from_canonical_name(
                    ap.output_map[out]["quant"]
                )

                tensor_name = out
                if isinstance(output_tensor_type, QuantizedTensorType):
                    scale_init_name  = create_const_initializer(
                        model, output_tensor_type.scale, np.float32
                    )
                    zeropt_init_name = create_const_initializer(
                        model, output_tensor_type.zeropt,
                        output_tensor_type.get_numpy_dtype()
                    )

                    # Partition output tensor name carries the int8 data
                    model.set_tensor_shape(
                        f"{tensor_name}_quantized",
                        out_shape,
                        dtype=output_tensor_type.get_tensorproto_dtype(),
                    )
                    # 2. Dequantize the transposed int8 tensor back to float
                    dequantize_node = helper.make_node(
                        "DequantizeLinear",
                        inputs=[f"{tensor_name}_quantized",
                                scale_init_name, zeropt_init_name],
                        outputs=[out],
                        name=f"{out}_dequantize",
                        axis=len(out_shape) - 1,
                    )
                    model.graph.node.append(dequantize_node)
                    tensor_name = f"{tensor_name}_quantized"

                if not output_layout.is_identity():
                    transpose_node = helper.make_node(
                        "Transpose",
                        name=f"{out}_transpose",
                        inputs=[f"{tensor_name}_pre_transpose"],
                        outputs=[f"{tensor_name}"],
                        perm=output_layout.inverse().perm,  # Inverse perm to get back to original layout
                    )
                    model.set_tensor_shape(
                        f"{tensor_name}_pre_transpose",
                        out_shape_perm,
                        dtype=output_tensor_type.get_tensorproto_dtype(),
                    )
                    model.graph.node.append(transpose_node)
                    tensor_name = f"{tensor_name}_pre_transpose"

                new_outputs_map[out] = (i, f"{tensor_name}")

            if new_outputs_map:
                for old_name, (index, new_name) in new_outputs_map.items():
                    partition_node.output[index] = new_name

                rename        = {old: new for old, (_, new) in new_outputs_map.items()}
                ap.output_map = {rename.get(k, k): v for k, v in ap.output_map.items()}

                for old_name, (_, new_name) in new_outputs_map.items():
                    output_layout = TensorLayout.from_canonical_name(
                        ap.output_map[new_name].get("layout")
                    )
                    out_shape  = ap.output_map[new_name]["shape"]
                    out_perm   = [dim for dim in output_layout.perm if dim < len(out_shape)]
                    ap.output_map[new_name]["shape"] = [out_shape[dim] for dim in out_perm]

            getCustomOp(partition_node).set_nodeattr(
                "accelerator_package", ap.to_json()
            )
        
        return model, False
