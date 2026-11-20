"""Rewrite the ship-detection SPP MaxPool fan-out as a 5x5 cascade."""

from collections import defaultdict

from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from nn2fpga.compiler.core.tensor_type import QuantizedTensorType


class FuseSPPMaxPool(Transformation):
    """Fuse the fixed 5x5/9x9/13x13 SPP MaxPool pattern into a 5x5 cascade.

    For stride-one max pooling with minimum-value padding, applying 5x5 pooling
    two and three times has the same receptive fields as 9x9 and 13x13 pooling.
    This deliberately recognizes only the quantization-preserving topology used
    by the ship-detection model.
    """

    _KERNELS = ((5, 2), (9, 4), (13, 6))

    @staticmethod
    def _attribute(node, name, default):
        for attribute in node.attribute:
            if attribute.name == name:
                return helper.get_attribute_value(attribute)
        return default

    @classmethod
    def _is_expected_pool(cls, node, kernel, padding) -> bool:
        if node.op_type != "MaxPool" or len(node.input) != 1 or len(node.output) != 1:
            return False
        if list(cls._attribute(node, "kernel_shape", [])) != [kernel, kernel]:
            return False
        if list(cls._attribute(node, "pads", [0, 0, 0, 0])) != [padding] * 4:
            return False
        if list(cls._attribute(node, "strides", [1, 1])) != [1, 1]:
            return False
        if list(cls._attribute(node, "dilations", [1, 1])) != [1, 1]:
            return False
        if cls._attribute(node, "ceil_mode", 0) != 0:
            return False
        return cls._attribute(node, "auto_pad", "NOTSET") in ("", "NOTSET")

    @staticmethod
    def _quantized_type_from_producer(model: ModelWrapper, tensor_name: str):
        producer = model.find_producer(tensor_name)
        if producer is None or producer.op_type not in ("Quant", "IntQuant"):
            return None
        return QuantizedTensorType.from_quant_node(producer, model)

    @staticmethod
    def _quantized_type_from_consumer(model: ModelWrapper, tensor_name: str):
        consumers = model.find_consumers(tensor_name)
        if len(consumers) != 1 or consumers[0].op_type not in ("Quant", "IntQuant"):
            return None
        return QuantizedTensorType.from_quant_node(consumers[0], model)

    @staticmethod
    def _set_pool_5x5(node) -> None:
        kept_attributes = [attribute for attribute in node.attribute if attribute.name not in ("kernel_shape", "pads")]
        del node.attribute[:]
        node.attribute.extend(kept_attributes)
        node.attribute.extend(
            [
                helper.make_attribute("kernel_shape", [5, 5]),
                helper.make_attribute("pads", [2, 2, 2, 2]),
            ]
        )

    def _matches(self, model: ModelWrapper, pools) -> bool:
        shared_input = pools[0].input[0]
        if any(pool.input[0] != shared_input for pool in pools):
            return False
        if not all(
            self._is_expected_pool(pool, kernel, padding)
            for pool, (kernel, padding) in zip(pools, self._KERNELS)
        ):
            return False

        input_shape = model.get_tensor_shape(shared_input)
        if input_shape is None or len(input_shape) != 4:
            return False
        if any(model.get_tensor_shape(pool.output[0]) != input_shape for pool in pools):
            return False

        input_type = self._quantized_type_from_producer(model, shared_input)
        output_types = [self._quantized_type_from_consumer(model, pool.output[0]) for pool in pools]
        return input_type is not None and all(output_type == input_type for output_type in output_types)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        pools_by_input = defaultdict(list)
        for node in model.graph.node:
            if node.op_type == "MaxPool" and len(node.input) == 1:
                pools_by_input[node.input[0]].append(node)

        modified = False
        for pools in pools_by_input.values():
            if len(pools) != 3:
                continue

            pools = sorted(
                pools,
                key=lambda node: list(self._attribute(node, "kernel_shape", [])),
            )
            if not self._matches(model, pools):
                continue

            self._set_pool_5x5(pools[1])
            self._set_pool_5x5(pools[2])
            pools[1].input[0] = pools[0].output[0]
            pools[2].input[0] = pools[1].output[0]
            modified = True

        return model, modified
