from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from onnxscript.rewriter import rewrite
from backend.custom_op.register_rewrite_rule import collect_rule_buckets
from onnxscript import ir
import logging
logger = logging.getLogger(__name__)


class LowerToNN2FPGALayers(Transformation):
    """Lower the model to NN2FPGA layers."""

    def __init__(self):
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """
        Apply the transformation to lower the model to NN2FPGA layers.

        Args:
            model (ModelWrapper): The model to transform.

        Returns:
            tuple: A tuple containing the transformed model and a boolean indicating if the transformation needs to be reapplied.
        """
        model = ir.from_proto(model.model)

        for ruleset in collect_rule_buckets():
            model = rewrite(model, pattern_rewrite_rules=ruleset)
        model = ir.to_proto(model)
        model = ModelWrapper(model)

        return (model, False)
