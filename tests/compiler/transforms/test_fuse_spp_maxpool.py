import numpy as np
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.transforms.fuse_spp_maxpool import FuseSPPMaxPool


def _quant_node(name, input_name, output_name, scale_name):
    return helper.make_node(
        "Quant",
        inputs=[input_name, scale_name, "zero_point", "bitwidth"],
        outputs=[output_name],
        name=name,
        domain="qonnx.custom_op.general",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )


def _model(output_scales=(0.25, 0.25, 0.25)):
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 24, 24])
    output_info = [
        helper.make_tensor_value_info(f"output_{index}", TensorProto.FLOAT, [1, 3, 24, 24])
        for index in range(3)
    ]
    value_info = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 3, 24, 24])
        for name in ("quantized_input", "pool_5", "pool_9", "pool_13")
    ]
    initializers = [
        helper.make_tensor("input_scale", TensorProto.FLOAT, [1], [0.25]),
        helper.make_tensor("zero_point", TensorProto.INT8, [1], [0]),
        helper.make_tensor("bitwidth", TensorProto.INT32, [1], [8]),
    ]
    nodes = [_quant_node("input_quant", "input", "quantized_input", "input_scale")]
    for index, (kernel, padding) in enumerate(((5, 2), (9, 4), (13, 6))):
        pool_output = f"pool_{kernel}"
        scale_name = f"output_scale_{kernel}"
        initializers.append(helper.make_tensor(scale_name, TensorProto.FLOAT, [1], [output_scales[index]]))
        nodes.extend(
            [
                helper.make_node(
                    "MaxPool",
                    inputs=["quantized_input"],
                    outputs=[pool_output],
                    name=f"pool_{kernel}",
                    kernel_shape=[kernel, kernel],
                    pads=[padding] * 4,
                    strides=[1, 1],
                ),
                _quant_node(f"output_quant_{kernel}", pool_output, f"output_{index}", scale_name),
            ]
        )
    return ModelWrapper(qonnx_make_model(helper.make_graph(nodes, "spp", [input_info], output_info, initializers, value_info=value_info)))


def test_fuses_quantization_preserving_spp_maxpool():
    model = _model().transform(FuseSPPMaxPool())
    pools = {node.name: node for node in model.graph.node if node.op_type == "MaxPool"}

    assert list(helper.get_attribute_value(next(a for a in pools["pool_9"].attribute if a.name == "kernel_shape"))) == [5, 5]
    assert list(helper.get_attribute_value(next(a for a in pools["pool_13"].attribute if a.name == "kernel_shape"))) == [5, 5]
    assert pools["pool_9"].input[0] == "pool_5"
    assert pools["pool_13"].input[0] == "pool_9"


def test_rejects_spp_when_quantization_changes():
    model = _model((0.25, 0.5, 0.25)).transform(FuseSPPMaxPool())
    pools = {node.name: node for node in model.graph.node if node.op_type == "MaxPool"}

    assert list(helper.get_attribute_value(next(a for a in pools["pool_9"].attribute if a.name == "kernel_shape"))) == [9, 9]
    assert pools["pool_9"].input[0] == "quantized_input"


def _max_pool_2d(values, kernel):
    padding = kernel // 2
    padded = np.pad(values, ((0, 0), (0, 0), (padding, padding), (padding, padding)), constant_values=-np.inf)
    return np.maximum.reduce(
        [padded[:, :, row:row + values.shape[2], column:column + values.shape[3]] for row in range(kernel) for column in range(kernel)]
    )


def test_three_5x5_pools_equal_the_original_receptive_fields():
    values = np.random.default_rng(1).integers(-128, 128, size=(1, 3, 24, 24)).astype(np.float32)
    pool_5 = _max_pool_2d(values, 5)
    pool_9 = _max_pool_2d(values, 9)
    pool_13 = _max_pool_2d(values, 13)

    assert np.array_equal(pool_5, _max_pool_2d(values, 5))
    assert np.array_equal(pool_9, _max_pool_2d(pool_5, 5))
    assert np.array_equal(pool_13, _max_pool_2d(_max_pool_2d(pool_5, 5), 5))
