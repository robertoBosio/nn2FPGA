from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.transforms.supported_partition import match_supported_patterns


def _build_quant_node(
    input_name: str,
    output_name: str,
    scale_value: float,
    zero_point_value: int,
    bitwidth_value: int,
    node_name: str,
):
    scale = helper.make_tensor(
        name=f"{node_name}_scale",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[scale_value],
    )
    zero_point = helper.make_tensor(
        name=f"{node_name}_zero_point",
        data_type=TensorProto.INT8,
        dims=[1],
        vals=[zero_point_value],
    )
    bitwidth = helper.make_tensor(
        name=f"{node_name}_bitwidth",
        data_type=TensorProto.INT32,
        dims=[1],
        vals=[bitwidth_value],
    )
    quant_node = helper.make_node(
        "Quant",
        inputs=[
            input_name,
            scale.name,
            zero_point.name,
            bitwidth.name,
        ],
        outputs=[output_name],
        domain="qonnx.custom_op.general",
        name=node_name,
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )
    return [scale, zero_point, bitwidth], quant_node


def _build_slice_nodes(
    input_name: str,
    output_name: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
    steps: list[int],
    node_name: str,
):
    start_tensor = helper.make_tensor(
        name=f"{node_name}_starts",
        data_type=TensorProto.INT64,
        dims=[len(starts)],
        vals=starts,
    )
    end_tensor = helper.make_tensor(
        name=f"{node_name}_ends",
        data_type=TensorProto.INT64,
        dims=[len(ends)],
        vals=ends,
    )
    axes_tensor = helper.make_tensor(
        name=f"{node_name}_axes",
        data_type=TensorProto.INT64,
        dims=[len(axes)],
        vals=axes,
    )
    steps_tensor = helper.make_tensor(
        name=f"{node_name}_steps",
        data_type=TensorProto.INT64,
        dims=[len(steps)],
        vals=steps,
    )

    start_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{node_name}_starts_out"],
        value=start_tensor,
        name=f"{node_name}_start_const",
    )
    end_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{node_name}_ends_out"],
        value=end_tensor,
        name=f"{node_name}_end_const",
    )
    axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{node_name}_axes_out"],
        value=axes_tensor,
        name=f"{node_name}_axes_const",
    )
    steps_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{node_name}_steps_out"],
        value=steps_tensor,
        name=f"{node_name}_steps_const",
    )
    slice_node = helper.make_node(
        "Slice",
        inputs=[
            input_name,
            start_node.output[0],
            end_node.output[0],
            axes_node.output[0],
            steps_node.output[0],
        ],
        outputs=[output_name],
        name=node_name,
    )
    return [start_node, end_node, axes_node, steps_node, slice_node]


def _build_slice_pattern_model(
    *,
    slice_specs: list[dict],
    output_quant_params: list[tuple[float, int, int]] | None = None,
    non_quant_consumer_index: int | None = None,
):
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 1, 1])
    output_tensors = []

    quant_inits, input_quant = _build_quant_node(
        "input", "quantized_input", 0.1, 0, 8, "input_quant"
    )
    initializers = list(quant_inits)
    nodes = [input_quant]
    value_infos = [
        helper.make_tensor_value_info("quantized_input", TensorProto.FLOAT, [1, 8, 1, 1])
    ]

    if output_quant_params is None:
        output_quant_params = [(0.1, 0, 8)] * len(slice_specs)

    for index, spec in enumerate(slice_specs):
        slice_nodes = _build_slice_nodes(
            "quantized_input",
            f"slice_out_{index}",
            spec["starts"],
            spec["ends"],
            spec["axes"],
            spec["steps"],
            spec["name"],
        )
        nodes.extend(slice_nodes)
        value_infos.append(
            helper.make_tensor_value_info(
                f"slice_out_{index}",
                TensorProto.FLOAT,
                spec.get("shape", [1, spec["ends"][0] - spec["starts"][0], 1, 1]),
            )
        )

        if non_quant_consumer_index == index:
            identity_node = helper.make_node(
                "Identity",
                inputs=[f"slice_out_{index}"],
                outputs=[f"output_{index}"],
                name=f"identity_{index}",
            )
            nodes.append(identity_node)
        else:
            quant_params = output_quant_params[index]
            quant_inits, quant_node = _build_quant_node(
                f"slice_out_{index}",
                f"output_{index}",
                quant_params[0],
                quant_params[1],
                quant_params[2],
                f"output_quant_{index}",
            )
            initializers.extend(quant_inits)
            nodes.append(quant_node)

        output_tensors.append(
            helper.make_tensor_value_info(f"output_{index}", TensorProto.FLOAT, [1, 4, 1, 1])
        )

    graph = helper.make_graph(
        nodes,
        name="slice_pattern_graph",
        inputs=[input_tensor],
        outputs=output_tensors,
        initializer=initializers,
    )
    graph.value_info.extend(value_infos)
    return ModelWrapper(qonnx_make_model(graph, producer_name="test_producer"))


def _build_concat_pattern_model(*, axis: int, quant_params: list[tuple[float, int, int]]):
    input_tensors = [
        helper.make_tensor_value_info(f"input_{index}", TensorProto.FLOAT, [1, 4, 1, 1])
        for index in range(len(quant_params))
    ]
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 1, 1])
    nodes = []
    initializers = []

    for index, params in enumerate(quant_params):
        quant_inits, quant_node = _build_quant_node(
            f"input_{index}",
            f"quant_out_{index}",
            params[0],
            params[1],
            params[2],
            f"input_quant_{index}",
        )
        initializers.extend(quant_inits)
        nodes.append(quant_node)

    concat_node = helper.make_node(
        "Concat",
        inputs=[f"quant_out_{index}" for index in range(len(quant_params))],
        outputs=["output"],
        axis=axis,
        name="concat_node",
    )
    nodes.append(concat_node)
    return ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                nodes,
                name="concat_pattern_graph",
                inputs=input_tensors,
                outputs=[output_tensor],
                initializer=initializers,
            ),
            producer_name="test_producer",
        )
    )


def _build_resize_pattern_model(
    *,
    roi_name: str,
    mode: str,
    coordinate_transformation_mode: str,
    scales_name: str = "scales",
    scales_values: tuple[float, ...] = (1.0, 1.0, 2.0, 2.0),
    scales_as_constant: bool = False,
):
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 16, 16])
    quant_inits, input_quant = _build_quant_node(
        "input", "quantized_input", 0.1, 0, 8, "input_quant"
    )
    nodes = [input_quant]
    initializers = list(quant_inits)

    resize_inputs = ["quantized_input", roi_name, scales_name]
    if scales_as_constant:
        scales_tensor = helper.make_tensor(
            name="scales_value",
            data_type=TensorProto.FLOAT,
            dims=[len(scales_values)],
            vals=list(scales_values),
        )
        scales_const = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[scales_name],
            value=scales_tensor,
            name="scales_const",
        )
        nodes.append(scales_const)
    else:
        initializers.append(
            helper.make_tensor(
                scales_name,
                TensorProto.FLOAT,
                [len(scales_values)],
                list(scales_values),
            )
        )

    resize_node = helper.make_node(
        "Resize",
        inputs=resize_inputs,
        outputs=["output"],
        name="resize_node",
        mode=mode,
        coordinate_transformation_mode=coordinate_transformation_mode,
    )
    nodes.append(resize_node)

    return ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                nodes,
                name="resize_pattern_graph",
                inputs=[input_tensor],
                outputs=[output_tensor],
                initializer=initializers,
            ),
            producer_name="test_producer",
        )
    )


def _build_binary_quant_pattern_model(
    *,
    op_type: str,
    quant_params: list[tuple[float, int, int]],
    second_input_from_quant: bool = True,
):
    input_tensors = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [1, 4, 1, 1]),
        helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [1, 4, 1, 1]),
    ]
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])
    nodes = []
    initializers = []

    quant0_inits, quant0 = _build_quant_node(
        "input_0",
        "quant_out_0",
        quant_params[0][0],
        quant_params[0][1],
        quant_params[0][2],
        "input_quant_0",
    )
    initializers.extend(quant0_inits)
    nodes.append(quant0)

    op_inputs = ["quant_out_0"]
    if second_input_from_quant:
        quant1_inits, quant1 = _build_quant_node(
            "input_1",
            "quant_out_1",
            quant_params[1][0],
            quant_params[1][1],
            quant_params[1][2],
            "input_quant_1",
        )
        initializers.extend(quant1_inits)
        nodes.append(quant1)
        op_inputs.append("quant_out_1")
    else:
        op_inputs.append("input_1")

    op_node = helper.make_node(
        op_type,
        inputs=op_inputs,
        outputs=["output"],
        name=f"{op_type.lower()}_node",
    )
    nodes.append(op_node)

    return ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                nodes,
                name=f"{op_type.lower()}_pattern_graph",
                inputs=input_tensors,
                outputs=[output_tensor],
                initializer=initializers,
            ),
            producer_name="test_producer",
        )
    )


def _build_leakyrelu_quant_pattern_model():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])

    quant_inits, quant_node = _build_quant_node(
        "input",
        "quantized_input",
        0.1,
        0,
        8,
        "input_quant",
    )
    leakyrelu_node = helper.make_node(
        "LeakyRelu",
        inputs=["quantized_input"],
        outputs=["output"],
        name="leakyrelu_node",
        alpha=0.1015625,
    )

    return ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                [quant_node, leakyrelu_node],
                name="leakyrelu_quant_pattern_graph",
                inputs=[input_tensor],
                outputs=[output_tensor],
                initializer=quant_inits,
            ),
            producer_name="test_producer",
        )
    )


def _build_leakyrelu_fusable_pattern_model(producer_op_type: str):
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])
    producer_inputs = ["input"]
    input_tensors = [input_tensor]

    if producer_op_type in ("Conv", "Gemm", "Add"):
        producer_inputs.append("producer_input_1")
        input_tensors.append(
            helper.make_tensor_value_info("producer_input_1", TensorProto.FLOAT, [1, 4, 1, 1])
        )

    producer_node = helper.make_node(
        producer_op_type,
        inputs=producer_inputs,
        outputs=["producer_output"],
        name=f"{producer_op_type.lower()}_node",
    )
    leakyrelu_node = helper.make_node(
        "LeakyRelu",
        inputs=["producer_output"],
        outputs=["output"],
        name="leakyrelu_node",
        alpha=0.1015625,
    )

    return ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                [producer_node, leakyrelu_node],
                name=f"leakyrelu_{producer_op_type.lower()}_pattern_graph",
                inputs=input_tensors,
                outputs=[output_tensor],
            ),
            producer_name="test_producer",
        )
    )


def test_slice_pattern_matches_full_tiling_with_equal_output_quant_params():
    model = _build_slice_pattern_model(
        slice_specs=[
            {"name": "slice_0", "starts": [0], "ends": [4], "axes": [1], "steps": [1]},
            {"name": "slice_1", "starts": [4], "ends": [8], "axes": [1], "steps": [1]},
        ]
    )
    anchor_node = next(node for node in model.graph.node if node.name == "slice_0")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is True
    assert match.pattern_name == "SliceSplitTreeFeasibleQuantized"
    assert {"slice_0", "input_quant"}.issubset(match.covered)
    assert {
        "slice_0_start_const",
        "slice_0_end_const",
        "slice_0_axes_const",
        "slice_0_steps_const",
        "slice_1_start_const",
        "slice_1_end_const",
        "slice_1_axes_const",
        "slice_1_steps_const",
    }.issubset(match.covered)


def test_slice_pattern_rejects_gap_in_tiling():
    model = _build_slice_pattern_model(
        slice_specs=[
            {"name": "slice_0", "starts": [0], "ends": [3], "axes": [1], "steps": [1]},
            {"name": "slice_1", "starts": [4], "ends": [8], "axes": [1], "steps": [1]},
        ]
    )
    anchor_node = next(node for node in model.graph.node if node.name == "slice_0")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("tile [0..D) contiguously" in reason for reason in match.reasons)


def test_slice_pattern_rejects_overlap_in_tiling():
    model = _build_slice_pattern_model(
        slice_specs=[
            {"name": "slice_0", "starts": [0], "ends": [5], "axes": [1], "steps": [1]},
            {"name": "slice_1", "starts": [4], "ends": [8], "axes": [1], "steps": [1]},
        ]
    )
    anchor_node = next(node for node in model.graph.node if node.name == "slice_0")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("tile [0..D) contiguously" in reason for reason in match.reasons)


def test_slice_pattern_rejects_different_output_quant_params():
    model = _build_slice_pattern_model(
        slice_specs=[
            {"name": "slice_0", "starts": [0], "ends": [4], "axes": [1], "steps": [1]},
            {"name": "slice_1", "starts": [4], "ends": [8], "axes": [1], "steps": [1]},
        ],
        output_quant_params=[(0.1, 0, 8), (0.2, 0, 8)],
    )
    anchor_node = next(node for node in model.graph.node if node.name == "slice_0")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("quant params differ" in reason for reason in match.reasons)


def test_slice_pattern_rejects_non_unit_step():
    model = _build_slice_pattern_model(
        slice_specs=[
            {"name": "slice_0", "starts": [0], "ends": [4], "axes": [1], "steps": [2]},
            {"name": "slice_1", "starts": [4], "ends": [8], "axes": [1], "steps": [1]},
        ]
    )
    anchor_node = next(node for node in model.graph.node if node.name == "slice_0")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("non-unit step" in reason for reason in match.reasons)


def test_slice_pattern_rejects_non_quantized_output_consumer():
    model = _build_slice_pattern_model(
        slice_specs=[
            {"name": "slice_0", "starts": [0], "ends": [4], "axes": [1], "steps": [1]},
            {"name": "slice_1", "starts": [4], "ends": [8], "axes": [1], "steps": [1]},
        ],
        non_quant_consumer_index=1,
    )
    anchor_node = next(node for node in model.graph.node if node.name == "slice_0")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("slice output not consumed" in reason for reason in match.reasons)


def test_slice_pattern_rejects_unsupported_axis():
    model = _build_slice_pattern_model(
        slice_specs=[
            {"name": "slice_0", "starts": [0], "ends": [1], "axes": [0], "steps": [1], "shape": [1, 8, 1, 1]},
        ]
    )
    anchor_node = next(node for node in model.graph.node if node.name == "slice_0")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("Axis 0 not allowed" in reason for reason in match.reasons)


def test_concat_pattern_matches_supported_axis_with_equal_quant_params():
    model = _build_concat_pattern_model(axis=1, quant_params=[(0.1, 0, 8), (0.1, 0, 8)])
    anchor_node = next(node for node in model.graph.node if node.name == "concat_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is True
    assert match.pattern_name == "Concat(Quant(...)) same params"
    assert {"concat_node", "input_quant_0", "input_quant_1"}.issubset(match.covered)


def test_concat_pattern_matches_negative_last_axis():
    model = _build_concat_pattern_model(axis=-1, quant_params=[(0.1, 0, 8), (0.1, 0, 8)])
    anchor_node = next(node for node in model.graph.node if node.name == "concat_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is True
    assert match.pattern_name == "Concat(Quant(...)) same params"
    assert {"concat_node", "input_quant_0", "input_quant_1"}.issubset(match.covered)


def test_concat_pattern_matches_negative_height_axis():
    model = _build_concat_pattern_model(axis=-2, quant_params=[(0.1, 0, 8), (0.1, 0, 8)])
    anchor_node = next(node for node in model.graph.node if node.name == "concat_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is True
    assert match.pattern_name == "Concat(Quant(...)) same params"
    assert {"concat_node", "input_quant_0", "input_quant_1"}.issubset(match.covered)


def test_concat_pattern_rejects_different_quant_params():
    model = _build_concat_pattern_model(axis=1, quant_params=[(0.1, 0, 8), (0.2, 0, 8)])
    anchor_node = next(node for node in model.graph.node if node.name == "concat_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("Concat inputs quant params differ" in reason for reason in match.reasons)


def test_concat_pattern_rejects_unsupported_axis():
    model = _build_concat_pattern_model(axis=0, quant_params=[(0.1, 0, 8), (0.1, 0, 8)])
    anchor_node = next(node for node in model.graph.node if node.name == "concat_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("Concat axis must resolve to 1, 2, or 3" in reason for reason in match.reasons)


def test_concat_pattern_rejects_negative_batch_axis():
    model = _build_concat_pattern_model(axis=-4, quant_params=[(0.1, 0, 8), (0.1, 0, 8)])
    anchor_node = next(node for node in model.graph.node if node.name == "concat_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("Concat axis must resolve to 1, 2, or 3" in reason for reason in match.reasons)


def test_resize_pattern_matches_nearest_asymmetric_with_initializer_scales():
    model = _build_resize_pattern_model(
        roi_name="",
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
    )
    anchor_node = next(node for node in model.graph.node if node.name == "resize_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is True
    assert match.pattern_name == "Resize(Quant(act)) upsample nearest/asymmetric + scales [1,1,s,s]"
    assert {"resize_node", "input_quant"}.issubset(match.covered)


def test_resize_pattern_rejects_non_empty_roi():
    model = _build_resize_pattern_model(
        roi_name="roi",
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
    )
    anchor_node = next(node for node in model.graph.node if node.name == "resize_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("Resize roi input must be empty" in reason for reason in match.reasons)


def test_resize_pattern_rejects_constant_scales_node():
    model = _build_resize_pattern_model(
        roi_name="",
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        scales_as_constant=True,
    )
    anchor_node = next(node for node in model.graph.node if node.name == "resize_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("Resize scales must be provided as an initializer" in reason for reason in match.reasons)


def test_resize_pattern_rejects_mismatched_hw_scales():
    model = _build_resize_pattern_model(
        roi_name="",
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        scales_values=(1.0, 1.0, 2.0, 3.0),
    )
    anchor_node = next(node for node in model.graph.node if node.name == "resize_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("Resize scales H and W factors must be equal" in reason for reason in match.reasons)


def test_add_pattern_matches_two_activation_quant_inputs_even_with_different_params():
    model = _build_binary_quant_pattern_model(
        op_type="Add",
        quant_params=[(0.1, 0, 8), (0.2, 1, 8)],
    )
    anchor_node = next(node for node in model.graph.node if node.name == "add_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is True
    assert match.pattern_name == "Add(Quant(a), Quant(b))"
    assert {"add_node", "input_quant_0", "input_quant_1"}.issubset(match.covered)


def test_add_pattern_rejects_non_quantized_input():
    model = _build_binary_quant_pattern_model(
        op_type="Add",
        quant_params=[(0.1, 0, 8), (0.1, 0, 8)],
        second_input_from_quant=False,
    )
    anchor_node = next(node for node in model.graph.node if node.name == "add_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("Both Add inputs must come from activation Quant/IntQuant" in reason for reason in match.reasons)


def test_mul_pattern_matches_two_activation_quant_inputs():
    model = _build_binary_quant_pattern_model(
        op_type="Mul",
        quant_params=[(0.1, 0, 8), (0.2, 0, 8)],
    )
    anchor_node = next(node for node in model.graph.node if node.name == "mul_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is True
    assert match.pattern_name == "Mul(Quant(a), Quant(b))"
    assert {"mul_node", "input_quant_0", "input_quant_1"}.issubset(match.covered)


def test_mul_pattern_rejects_non_quantized_input():
    model = _build_binary_quant_pattern_model(
        op_type="Mul",
        quant_params=[(0.1, 0, 8), (0.1, 0, 8)],
        second_input_from_quant=False,
    )
    anchor_node = next(node for node in model.graph.node if node.name == "mul_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("Both Mul inputs must come from activation Quant/IntQuant" in reason for reason in match.reasons)


def test_leakyrelu_pattern_matches_quantized_input():
    model = _build_leakyrelu_quant_pattern_model()
    anchor_node = next(node for node in model.graph.node if node.name == "leakyrelu_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is True
    assert match.pattern_name == "LeakyRelu(Quant(x)) or fused into Conv/Gemm/Add"
    assert {"leakyrelu_node", "input_quant"}.issubset(match.covered)


def test_leakyrelu_pattern_matches_fusable_conv_input():
    model = _build_leakyrelu_fusable_pattern_model("Conv")
    anchor_node = next(node for node in model.graph.node if node.name == "leakyrelu_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is True
    assert match.pattern_name == "LeakyRelu(Quant(x)) or fused into Conv/Gemm/Add"
    assert match.covered == {"leakyrelu_node"}


def test_leakyrelu_pattern_matches_fusable_gemm_input():
    model = _build_leakyrelu_fusable_pattern_model("Gemm")
    anchor_node = next(node for node in model.graph.node if node.name == "leakyrelu_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is True
    assert match.pattern_name == "LeakyRelu(Quant(x)) or fused into Conv/Gemm/Add"
    assert match.covered == {"leakyrelu_node"}


def test_leakyrelu_pattern_matches_fusable_add_input():
    model = _build_leakyrelu_fusable_pattern_model("Add")
    anchor_node = next(node for node in model.graph.node if node.name == "leakyrelu_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is True
    assert match.pattern_name == "LeakyRelu(Quant(x)) or fused into Conv/Gemm/Add"
    assert match.covered == {"leakyrelu_node"}


def test_leakyrelu_pattern_rejects_non_quantized_non_fusable_input():
    model = _build_leakyrelu_fusable_pattern_model("Identity")
    anchor_node = next(node for node in model.graph.node if node.name == "leakyrelu_node")

    match = match_supported_patterns(model, anchor_node)

    assert match.ok is False
    assert any("LeakyRelu must be quantized or fusable into Conv/Gemm/Add" in reason for reason in match.reasons)
