from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from onnx import helper, TensorProto
from onnxscript.rewriter import pattern, rewrite
from onnxscript import ir
import logging
logger = logging.getLogger(__name__)

def reshape_quant_gemm_pattern(
    op, x, W, B
):
    """
    Pattern for Reshape -> Quant -> Gemm.
    """

    x2d = op.Reshape(x, _allow_other_inputs=True)
    x2d_quant = op.Quant(
        x2d,
        _domain="qonnx.custom_op.general",
        _allow_other_attributes=True,
        _allow_other_inputs=True,
    )
    y = op.Gemm(x2d_quant, W, B, alpha=1.0, beta=1.0, transB=1, _allow_other_attributes=True)
    return y

def flatten_quant_gemm_pattern(
    op, x, W, B
):
    """
    Pattern for Flatten -> Quant -> Gemm.
    """
    x2d = op.Flatten(x, axis=1, _allow_other_inputs=True)
    x2d_quant = op.Quant(
        x2d,
        _domain="qonnx.custom_op.general",
        _allow_other_attributes=True,
        _allow_other_inputs=True,
    )
    y = op.Gemm(x2d_quant, W, B, alpha=1.0, beta=1.0, transB=1, _allow_other_attributes=True)
    return y

def quant_gemm_pattern(
    op, x, W, B
):
    """
    Pattern for Quant -> Gemm.
    """
    x_quant = op.Quant(
        x,
        _domain="qonnx.custom_op.general",
        _allow_other_attributes=True,
        _allow_other_inputs=True,
    )
    y = op.Gemm(x_quant, W, B, alpha=1.0, beta=1.0, transB=1, _allow_other_attributes=True)
    return y

def rewrite_conv1x1(op, x, W, B, **kwargs):

    """
    Rewrite X -> Conv1x1 -> Reshape.
    """

    reshape_weights_shape_tensor = helper.make_tensor(
        name=f"reshape_weights_shape_{x.name}",
        data_type=TensorProto.INT64,
        dims=[4],
        vals=[0, 0, 1, 1],
    )
    reshape_input = op.Constant(value=reshape_weights_shape_tensor)

    W_reshaped = op.Reshape(W, reshape_input)

    y = op.Conv(
        x,
        W_reshaped,
        B,
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    reshape_output_shape_tensor = helper.make_tensor(
        name=f"reshape_output_shape_{x.name}",
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[1, -1],
    )
    reshape_output = op.Constant(value=reshape_output_shape_tensor)

    y = op.Reshape(y, reshape_output)

    return y

def rewrite_conv1x1_with_unsqueeze_input(op, x, W, B, **kwargs):
    """
    Rewrite X -> Reshape -> Conv1x1 -> Reshape.
    """

    reshape_input_shape_tensor = helper.make_tensor(
        name=f"reshape_input_shape_{x.name}",
        data_type=TensorProto.INT64,
        dims=[4],
        vals=[0, 0, 1, 1],
    )
    reshape_input = op.Constant(value=reshape_input_shape_tensor)

    x_reshaped = op.Reshape(x, reshape_input)

    reshape_weights_shape_tensor = helper.make_tensor(
        name=f"reshape_weights_shape_{x.name}",
        data_type=TensorProto.INT64,
        dims=[4],
        vals=[0, 0, 1, 1],
    )
    reshape_input = op.Constant(value=reshape_weights_shape_tensor)

    W_reshaped = op.Reshape(W, reshape_input)

    y = op.Conv(
        x_reshaped,
        W_reshaped,
        B,
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    reshape_output_shape_tensor = helper.make_tensor(
        name=f"reshape_output_shape_{x.name}",
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[1, -1],
    )
    reshape_output = op.Constant(value=reshape_output_shape_tensor)

    y = op.Reshape(y, reshape_output)

    return y


class FullyConnectedToPointwise(Transformation):
    """
    Replace:
      (Flatten|Reshape) -> Quant(act) -> Gemm( Quant(W), [Quant(B)] )
    with:
      Conv1x1 -> Reshape
    """

    def __init__(self):
        self._rewrite_rule_set = pattern.RewriteRuleSet(
            [
                pattern.RewriteRule(reshape_quant_gemm_pattern, rewrite_conv1x1),
                pattern.RewriteRule(flatten_quant_gemm_pattern, rewrite_conv1x1),
                pattern.RewriteRule(quant_gemm_pattern, rewrite_conv1x1_with_unsqueeze_input),
            ],
            commute=True,
        )

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        model = ir.from_proto(model.model)
        model = rewrite(model, pattern_rewrite_rules=self._rewrite_rule_set)
        model = ir.to_proto(model)
        model = ModelWrapper(model)
        model = model.transform(InferShapes())
        return model, False
