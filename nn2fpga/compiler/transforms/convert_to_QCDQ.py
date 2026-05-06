from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.qonnx_to_qcdq import QuantToQCDQ
from qonnx.custom_op.registry import getCustomOp
from nn2fpga.compiler.transforms.add_streaming_params import quant_array
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.core.tensor_quant import TensorQuant
from nn2fpga.compiler.core.tensor_layout import TensorLayout
from onnx import TensorProto, helper, numpy_helper
import onnx.shape_inference as si
from onnxscript.rewriter import pattern, rewrite
from onnxscript import ir
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Transpose split analysis  (analysis only — no graph rewriting)
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_LINE_BYTES    = 64
_BAD_THRESHOLD_BYTES = _CACHE_LINE_BYTES // 2   # 32 B — half a cache line


def _dtype_itemsize(tensor_type: int) -> int:
    return {
        TensorProto.FLOAT:   4,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE:  8,
        TensorProto.INT8:    1,
        TensorProto.UINT8:   1,
        TensorProto.INT16:   2,
        TensorProto.UINT16:  2,
        TensorProto.INT32:   4,
        TensorProto.UINT32:  4,
        TensorProto.INT64:   8,
        TensorProto.UINT64:  8,
        TensorProto.BOOL:    1,
    }.get(tensor_type, 4)


def analyze_transpose_split(
    tensor_name: str,
    shape: list[int],
    perm: list[int],
    tensor_type: int,
) -> None:
    """
    Analyse whether a Reshape -> Transpose -> Reshape split would help.
    Logs DEBUG when the transpose is fine, WARNING when it is cache-hostile
    together with a concrete split suggestion.  Never modifies the graph.

    Cache-hostility metric:
        The innermost output axis corresponds to perm[-1] in input coordinates.
        If shape[perm[-1]] * itemsize < half a cache line, each output row is
        too short for the prefetcher to be effective.

    Split strategy:
        Keep leading identity-mapped (batch) axes intact so ORT can loop over
        them independently — this proved faster on the Cortex-A53 than
        collapsing them into the matrix rows.  Then walk trailing input axes
        inward, accumulating their product, until the combined size reaches one
        full cache line.  Those axes are collapsed into a single dimension for
        a 3-D transpose that has a cache-line-wide inner dimension.
    """
    itemsize     = _dtype_itemsize(tensor_type)
    ndim         = len(shape)
    total_bytes  = int(np.prod(shape)) * itemsize

    inner_dim_size  = shape[perm[-1]]
    inner_dim_bytes = inner_dim_size * itemsize
    util            = min(inner_dim_bytes, _CACHE_LINE_BYTES) / _CACHE_LINE_BYTES
    wasted          = 1.0 / util if util > 0 else float("inf")
    is_bad          = inner_dim_bytes < _BAD_THRESHOLD_BYTES

    if not is_bad:
        logger.debug(
            "[transpose-analysis] OK | tensor=%s shape=%s perm=%s "
            "inner=%d elems/%dB util=%.0f%% total=%.2fMB",
            tensor_name, shape, perm,
            inner_dim_size, inner_dim_bytes, util * 100, total_bytes / 1e6,
        )
        return

    # ── Identify batch axes (leading identity-mapped dims) ────────────────
    n_batch = 0
    for i, p in enumerate(perm):
        if p == i:
            n_batch += 1
        else:
            break

    # ── Find a collapse group among trailing input axes ───────────────────
    accumulated   = 1
    collapse_axes = []
    for ax in range(ndim - 1, n_batch - 1, -1):
        accumulated *= shape[ax]
        collapse_axes.insert(0, ax)
        if accumulated * itemsize >= _CACHE_LINE_BYTES:
            break

    if len(collapse_axes) > 1 and accumulated * itemsize >= _CACHE_LINE_BYTES:
        batch_shape  = list(shape[:n_batch])
        middle_axes  = [ax for ax in range(n_batch, ndim) if ax not in collapse_axes]
        middle_shape = [shape[ax] for ax in middle_axes]
        inter_shape  = batch_shape + middle_shape + [accumulated]

        # Generalised 3-D perm: keep batch, then swap the last two dims.
        split_perm = list(range(n_batch)) + [n_batch + 1, n_batch]

        out_shape  = [shape[p] for p in perm]
        split_hint = (
            f"Reshape {shape} -> {inter_shape}, "
            f"Transpose {split_perm}, "
            f"Reshape -> {out_shape}  "
            f"[collapsed {len(collapse_axes)} axes into {accumulated} elems "
            f"= {accumulated * itemsize}B >= {_CACHE_LINE_BYTES}B cache line]"
        )
    else:
        split_hint = (
            "No clean collapse group found — "
            "consider redesigning the kernel output layout."
        )

    logger.warning(
        "[transpose-analysis] SLOW | tensor=%s shape=%s perm=%s "
        "inner=%d elems/%dB util=%.0f%% wasted=%.1fx total=%.2fMB | %s",
        tensor_name, shape, perm,
        inner_dim_size, inner_dim_bytes, util * 100, wasted,
        total_bytes / 1e6, split_hint,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Existing helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def get_tensorproto_dtype(bitwidth, signed):
    bitwidth = int(bitwidth)
    signed   = bool(signed)
    if bitwidth <= 8:
        return TensorProto.INT8 if signed else TensorProto.UINT8
    elif bitwidth <= 32:
        return TensorProto.INT32 if signed else TensorProto.UINT32
    else:
        raise ValueError(f"Unsupported bitwidth for quantization: {bitwidth}")


def get_numpy_dtype(bitwidth, signed):
    bitwidth = int(bitwidth)
    signed   = bool(signed)
    if bitwidth <= 8:
        return np.int8 if signed else np.uint8
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
    if bitwidth not in (8, 32):
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

    For every inserted transpose, analyze_transpose_split() is called.
    It logs a WARNING with a concrete Reshape→Transpose→Reshape suggestion
    whenever the inner output dimension is narrower than half a cache line.
    No split rewriting is performed yet.
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
                inp_perm       = [dim for dim in input_layout.perm if dim < len(inp_shape)]
                inp_shape_perm = [inp_shape[dim] for dim in inp_perm]

                input_tensor_quant = TensorQuant.from_canonical_name(
                    ap.input_map[inp]["quant"]
                )
                scale_init_name  = create_const_initializer(
                    model, input_tensor_quant.scale, np.float32
                )
                zeropt_init_name = create_const_initializer(
                    model, input_tensor_quant.zeropt,
                    input_tensor_quant.get_numpy_dtype()
                )

                if not input_layout.is_identity():
                    # 1. Quantize the original float tensor (shape unchanged)
                    quantize_node = helper.make_node(
                        "QuantizeLinear",
                        inputs=[inp, scale_init_name, zeropt_init_name],
                        outputs=[f"{inp}_quantized_pretranspose"],
                        name=f"{inp}_quantize",
                        axis=len(inp_shape) - 1,
                    )
                    model.set_tensor_shape(
                        f"{inp}_quantized_pretranspose",
                        inp_shape,
                        dtype=input_tensor_quant.get_tensorproto_dtype(),
                    )
                    model.graph.node.append(quantize_node)

                    # 2. Transpose the int8 tensor into the layout the partition expects
                    analyze_transpose_split(
                        tensor_name=inp,
                        shape=inp_shape,
                        perm=inp_perm,
                        tensor_type=input_tensor_quant.get_tensorproto_dtype(),
                    )
                    transpose_node = helper.make_node(
                        "Transpose",
                        name=f"{inp}_transpose",
                        inputs=[f"{inp}_quantized_pretranspose"],
                        outputs=[f"{inp}_quantized"],
                        perm=inp_perm,
                    )
                    model.set_tensor_shape(
                        f"{inp}_quantized",
                        inp_shape_perm,
                        dtype=input_tensor_quant.get_tensorproto_dtype(),
                    )
                    model.graph.node.append(transpose_node)

                else:
                    # Identity layout: quantize directly, no transpose needed
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
                        dtype=input_tensor_quant.get_tensorproto_dtype(),
                    )
                    model.graph.node.append(quantize_node)

                new_inputs_map[inp] = (i, f"{inp}_quantized")

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

                if consumers is not None and all(
                    consumer.op_type == "DequantizeLinear" for consumer in consumers
                ):
                    continue

                out_shape = model.get_tensor_shape(out)
                if out_shape is None:
                    continue

                output_layout = TensorLayout.from_canonical_name(
                    ap.output_map[out].get("layout")
                )
                out_perm       = [dim for dim in output_layout.perm if dim < len(out_shape)]
                out_shape_perm = [out_shape[dim] for dim in out_perm]

                output_tensor_quant = TensorQuant.from_canonical_name(
                    ap.output_map[out]["quant"]
                )
                scale_init_name  = create_const_initializer(
                    model, output_tensor_quant.scale, np.float32
                )
                zeropt_init_name = create_const_initializer(
                    model, output_tensor_quant.zeropt,
                    output_tensor_quant.get_numpy_dtype()
                )

                # Partition output tensor name carries the int8 data
                model.set_tensor_shape(
                    f"{out}_quantized",
                    out_shape,
                    dtype=output_tensor_quant.get_tensorproto_dtype(),
                )

                if not output_layout.is_identity():
                    # 1. Transpose the int8 partition output into the expected layout
                    analyze_transpose_split(
                        tensor_name=out,
                        shape=out_shape,
                        perm=out_perm,
                        tensor_type=output_tensor_quant.get_tensorproto_dtype(),
                    )
                    transpose_node = helper.make_node(
                        "Transpose",
                        name=f"{out}_transpose",
                        inputs=[f"{out}_quantized"],
                        outputs=[f"{out}_transposed_quantized"],
                        perm=out_perm,
                    )
                    model.set_tensor_shape(
                        f"{out}_transposed_quantized",
                        out_shape_perm,
                        dtype=output_tensor_quant.get_tensorproto_dtype(),
                    )
                    model.graph.node.append(transpose_node)

                    # 2. Dequantize the transposed int8 tensor back to float
                    dequantize_node = helper.make_node(
                        "DequantizeLinear",
                        inputs=[f"{out}_transposed_quantized",
                                scale_init_name, zeropt_init_name],
                        outputs=[out],
                        name=f"{out}_dequantize",
                        axis=len(out_shape_perm) - 1,
                    )
                    model.graph.node.append(dequantize_node)

                else:
                    # Identity layout: dequantize directly, no transpose needed
                    dequantize_node = helper.make_node(
                        "DequantizeLinear",
                        inputs=[f"{out}_quantized", scale_init_name, zeropt_init_name],
                        outputs=[out],
                        name=f"{out}_dequantize",
                        axis=len(out_shape) - 1,
                    )
                    model.graph.node.append(dequantize_node)

                new_outputs_map[out] = (i, f"{out}_quantized")

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