from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from onnxscript.rewriter import rewrite, pattern
from onnxscript import ir
from onnx_ir import convenience as ir_convenience
import logging
logger = logging.getLogger(__name__)

def pattern_constant_multiply_by_one(op, a, b):
    return op.Mul(a, b)

def condition_constant_multiply_by_one(op, a, b):
    if ir_convenience.get_const_tensor(b) is not None:
        const_value = ir_convenience.get_const_tensor(b)
        arr = const_value.numpy().reshape(-1)
        return arr.size == 1 and arr[0] == 1

def rewrite_constant_multiply_by_one(op, a, b):
    return a

class RemoveNoopNodes(Transformation):
    """Remove no-op nodes from the model."""

    def __init__(self):
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """
        Apply the transformation to remove no-op nodes from the model.

        Args:
            model (ModelWrapper): The model to transform.

        Returns:
            tuple: A tuple containing the transformed model and a boolean indicating if the transformation needs to be reapplied.
        """
        model = ir.from_proto(model.model)
        model = rewrite(
            model,
            pattern_rewrite_rules=[
                pattern.RewriteRule(
                    pattern_constant_multiply_by_one,
                    rewrite_constant_multiply_by_one,
                    condition_constant_multiply_by_one,
                )
            ],
        )
        model = ir.to_proto(model)
        model = ModelWrapper(model)

        return (model, False)
