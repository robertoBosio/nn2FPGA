from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from onnxscript.rewriter import pattern, rewrite
from onnxscript import ir
import logging

logger = logging.getLogger(__name__)


def concat3_pattern(op, x0, x1, x2, axis: int = 0):
    """
    Match: Concat(x0, x1, x2, axis=axis)
    """
    return op.Concat(x0, x1, x2, axis=axis)


def concat3_rewrite(op, x0: ir.Value, x1: ir.Value, x2: ir.Value, axis=None):
    """
    Rewrite:
        y = Concat(x0, x1, x2, axis)
    into:
        t0 = Concat(x0, x1, axis)
        y  = Concat(t0, x2, axis)
    """
    t0 = op.Concat(x0, x1, axis=axis)
    return op.Concat(t0, x2, axis=axis)


def concat4_pattern(op, x0, x1, x2, x3, axis: int = 0):
    """
    Match: Concat(x0, x1, x2, x3, axis=axis)
    """
    return op.Concat(x0, x1, x2, x3, axis=axis)


def concat4_rewrite(
    op,
    x0: ir.Value,
    x1: ir.Value,
    x2: ir.Value,
    x3: ir.Value,
    axis=None,
):
    """
    Rewrite:
        y = Concat(x0, x1, x2, x3, axis)
    into a small binary tree:
        t01 = Concat(x0, x1, axis)
        t23 = Concat(x2, x3, axis)
        y   = Concat(t01, t23, axis)
    """
    t01 = op.Concat(x0, x1, axis=axis)
    t23 = op.Concat(x2, x3, axis=axis)
    return op.Concat(t01, t23, axis=axis)


def concat5_pattern(op, x0, x1, x2, x3, x4, axis: int = 0):
    """
    Match: Concat(x0, x1, x2, x3, x4, axis=axis)
    """
    return op.Concat(x0, x1, x2, x3, x4, axis=axis)


def concat5_rewrite(
    op,
    x0: ir.Value,
    x1: ir.Value,
    x2: ir.Value,
    x3: ir.Value,
    x4: ir.Value,
    axis=None,
):
    """
    Rewrite:
        y = Concat(x0, x1, x2, x3, x4, axis)
    into a chain of binary concats:
        t01   = Concat(x0, x1, axis)
        t012  = Concat(t01, x2, axis)
        t0123 = Concat(t012, x3, axis)
        y     = Concat(t0123, x4, axis)
    """
    t01 = op.Concat(x0, x1, axis=axis)
    t012 = op.Concat(t01, x2, axis=axis)
    t0123 = op.Concat(t012, x3, axis=axis)
    return op.Concat(t0123, x4, axis=axis)


def concat6_pattern(op, x0, x1, x2, x3, x4, x5, axis: int = 0):
    """
    Match: Concat(x0, x1, x2, x3, x4, x5, axis=axis)
    """
    return op.Concat(x0, x1, x2, x3, x4, x5, axis=axis)


def concat6_rewrite(
    op,
    x0: ir.Value,
    x1: ir.Value,
    x2: ir.Value,
    x3: ir.Value,
    x4: ir.Value,
    x5: ir.Value,
    axis=None,
):
    """
    Rewrite:
        y = Concat(x0, x1, x2, x3, x4, x5, axis)
    into a balanced-ish binary tree:
        t01  = Concat(x0, x1, axis)
        t23  = Concat(x2, x3, axis)
        t45  = Concat(x4, x5, axis)
        t0123 = Concat(t01, t23, axis)
        y     = Concat(t0123, t45, axis)
    """
    t01 = op.Concat(x0, x1, axis=axis)
    t23 = op.Concat(x2, x3, axis=axis)
    t45 = op.Concat(x4, x5, axis=axis)
    t0123 = op.Concat(t01, t23, axis=axis)
    return op.Concat(t0123, t45, axis=axis)


class SplitConcat(Transformation):
    """
    Replace:
        Concat(x0, x1, x2, ..., xN, axis)
    with:
        multiple binary Concat operations arranged in a balanced tree.
    """

    def __init__(self):
        self._rewrite_rule_set = pattern.RewriteRuleSet(
            [
                pattern.RewriteRule(
                    concat3_pattern,
                    concat3_rewrite,
                ),
                pattern.RewriteRule(
                    concat4_pattern,
                    concat4_rewrite,
                ),
                pattern.RewriteRule(
                    concat5_pattern,
                    concat5_rewrite,
                ),
                pattern.RewriteRule(
                    concat6_pattern,
                    concat6_rewrite,
                ),
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
