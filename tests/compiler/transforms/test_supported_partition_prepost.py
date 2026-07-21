from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.transforms.supported_partition import (
    PostProcessPartitionModel,
    PreProcessPartitionModel,
)


def _build_resize_model(nodes, outputs, initializers=None):
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    graph = helper.make_graph(
        nodes=nodes,
        name="resize_graph",
        inputs=[input_tensor],
        outputs=outputs,
        initializer=initializers or [],
    )
    return ModelWrapper(qonnx_make_model(graph, producer_name="test_producer"))


def _get_attr(node, name):
    return next((attr for attr in node.attribute if attr.name == name), None)


def _make_passthrough_add(input_name: str, bias_name: str, output_name: str, node_name: str):
    return helper.make_node(
        "Add",
        inputs=[input_name, bias_name],
        outputs=[output_name],
        name=node_name,
    )


def test_resize_with_empty_optional_inputs_round_trips():
    resize_node = helper.make_node(
        "Resize",
        inputs=["input", "", "scales"],
        outputs=["resize_output"],
        name="resize_node",
        mode="nearest",
    )
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 16, 16])
    add_node = _make_passthrough_add("resize_output", "kept", "output", "keep_bias")
    scales_init = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
    preserved_init = helper.make_tensor("kept", TensorProto.FLOAT, [1], [0.0])
    model = _build_resize_model([resize_node, add_node], [output_tensor], [scales_init, preserved_init])

    original_inputs = list(model.graph.node[0].input)
    processed_model = model.transform(PreProcessPartitionModel())
    processed_resize = processed_model.graph.node[0]

    assert list(processed_resize.input) == ["input", "resize_node_dummy_input_1", "scales"]
    input_mask = _get_attr(processed_resize, "__resize_input_mask")
    assert input_mask is not None
    assert list(input_mask.ints) == [0, 1, 0]
    assert {init.name for init in processed_model.graph.initializer} == {
        "scales",
        "kept",
        "resize_node_dummy_input_1",
    }

    roundtrip_model = processed_model.transform(PostProcessPartitionModel())
    roundtrip_resize = roundtrip_model.graph.node[0]

    assert list(roundtrip_resize.input) == original_inputs
    assert _get_attr(roundtrip_resize, "__resize_input_mask") is None
    assert {init.name for init in roundtrip_model.graph.initializer} == {"scales", "kept"}


def test_resize_without_empty_inputs_round_trips_without_dummy_initializers():
    resize_node = helper.make_node(
        "Resize",
        inputs=["input", "roi", "scales"],
        outputs=["resize_output"],
        name="resize_node",
        mode="nearest",
    )
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 16, 16])
    add_node = _make_passthrough_add("resize_output", "bias", "output", "keep_bias")
    roi_init = helper.make_tensor("roi", TensorProto.FLOAT, [0], [])
    scales_init = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
    bias_init = helper.make_tensor("bias", TensorProto.FLOAT, [1], [0.0])
    model = _build_resize_model([resize_node, add_node], [output_tensor], [roi_init, scales_init, bias_init])

    processed_model = model.transform(PreProcessPartitionModel())
    processed_resize = processed_model.graph.node[0]

    assert list(processed_resize.input) == ["input", "roi", "scales"]
    input_mask = _get_attr(processed_resize, "__resize_input_mask")
    assert input_mask is not None
    assert list(input_mask.ints) == [0, 0, 0]
    assert {init.name for init in processed_model.graph.initializer} == {"roi", "scales", "bias"}

    roundtrip_model = processed_model.transform(PostProcessPartitionModel())
    roundtrip_resize = roundtrip_model.graph.node[0]

    assert list(roundtrip_resize.input) == ["input", "roi", "scales"]
    assert _get_attr(roundtrip_resize, "__resize_input_mask") is None
    assert {init.name for init in roundtrip_model.graph.initializer} == {"roi", "scales", "bias"}


def test_multiple_resize_nodes_are_handled_independently():
    resize_a = helper.make_node(
        "Resize",
        inputs=["input", "", "scales_a"],
        outputs=["output_a"],
        name="resize_a",
        mode="nearest",
    )
    resize_b = helper.make_node(
        "Resize",
        inputs=["output_a", "roi_b", "", "sizes_b"],
        outputs=["resize_b_output"],
        name="resize_b",
        mode="nearest",
    )
    output_tensor = helper.make_tensor_value_info("output_b", TensorProto.FLOAT, [1, 3, 32, 32])
    add_node = _make_passthrough_add("resize_b_output", "kept", "output_b", "keep_bias")
    initializers = [
        helper.make_tensor("scales_a", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]),
        helper.make_tensor("roi_b", TensorProto.FLOAT, [0], []),
        helper.make_tensor("sizes_b", TensorProto.INT64, [4], [1, 3, 32, 32]),
        helper.make_tensor("kept", TensorProto.FLOAT, [1], [0.0]),
    ]
    model = _build_resize_model([resize_a, resize_b, add_node], [output_tensor], initializers)

    processed_model = model.transform(PreProcessPartitionModel())
    processed_a = processed_model.graph.node[0]
    processed_b = processed_model.graph.node[1]

    assert list(processed_a.input) == ["input", "resize_a_dummy_input_1", "scales_a"]
    assert list(processed_b.input) == ["output_a", "roi_b", "resize_b_dummy_input_2", "sizes_b"]
    assert list(_get_attr(processed_a, "__resize_input_mask").ints) == [0, 1, 0]
    assert list(_get_attr(processed_b, "__resize_input_mask").ints) == [0, 0, 1, 0]
    assert {init.name for init in processed_model.graph.initializer} == {
        "scales_a",
        "roi_b",
        "sizes_b",
        "kept",
        "resize_a_dummy_input_1",
        "resize_b_dummy_input_2",
    }

    roundtrip_model = processed_model.transform(PostProcessPartitionModel())
    roundtrip_a = roundtrip_model.graph.node[0]
    roundtrip_b = roundtrip_model.graph.node[1]

    assert list(roundtrip_a.input) == ["input", "", "scales_a"]
    assert list(roundtrip_b.input) == ["output_a", "roi_b", "", "sizes_b"]
    assert _get_attr(roundtrip_a, "__resize_input_mask") is None
    assert _get_attr(roundtrip_b, "__resize_input_mask") is None
    assert {init.name for init in roundtrip_model.graph.initializer} == {
        "scales_a",
        "roi_b",
        "sizes_b",
        "kept",
    }


def test_non_resize_graph_is_unchanged_by_pre_and_post_process():
    identity_node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["identity_output"],
        name="identity_node",
    )
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
    add_node = _make_passthrough_add("identity_output", "kept", "output", "keep_bias")
    preserved_init = helper.make_tensor("kept", TensorProto.FLOAT, [1], [0.0])
    model = _build_resize_model([identity_node, add_node], [output_tensor], [preserved_init])

    processed_model = model.transform(PreProcessPartitionModel())
    roundtrip_model = processed_model.transform(PostProcessPartitionModel())

    assert list(roundtrip_model.graph.node[0].input) == ["input"]
    assert [attr.name for attr in roundtrip_model.graph.node[0].attribute] == []
    assert {init.name for init in roundtrip_model.graph.initializer} == {"kept"}
