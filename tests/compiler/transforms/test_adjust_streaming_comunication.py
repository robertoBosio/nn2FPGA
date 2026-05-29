from nn2fpga.compiler.transforms.adjust_streaming_comunication import (
    AdjustStreamingCommunication,
)
from nn2fpga.compiler.core.tensor_layout import (
    TensorLayout,
    get_custom_tensor_layout,
    set_custom_tensor_layout,
)
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto, helper


def annotate_layouts(model, tensor_names, layout):
    for tensor_name in tensor_names:
        set_custom_tensor_layout(model, tensor_name, layout)


def assert_bandwidth_adjust_metadata(model, layout, shape):
    for node in model.graph.node:
        if node.op_type.startswith("BandwidthAdjust"):
            assert get_custom_tensor_layout(model, node.output[0]) == layout
            assert model.get_tensor_shape(node.output[0]) == shape

def test_adjust_streaming_communication_decrease_channels():

    output_tensor = helper.make_tensor_value_info(
        name="output_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 64, 32, 32],
    )

    input_tensor = helper.make_tensor_value_info(
        name="input_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 64, 32, 32],
    )
    
    input_tensor_streamed = helper.make_tensor_value_info(
        name="input_tensor_streamed",
        elem_type=TensorProto.FLOAT,
        shape=[1, 64, 32, 32],
    )

    maxpool_output = helper.make_tensor_value_info(
        name="maxpool_output",
        elem_type=TensorProto.FLOAT,
        shape=[1, 64, 32, 32],
    )

    axitostream_node = helper.make_node(
        "AXIToStream",
        inputs=["input_tensor"],
        outputs=["input_tensor_streamed"],
        name="axitostream_node",
        domain="nn2fpga.compiler.custom_op",
        channel_unroll=16,
        width_unroll=1,
        in_stream_array=1,
        out_stream_array=1,
        in_word_array=16,
        out_word_array=16,
    )

    maxpool_node0 = helper.make_node(
        "StreamingMaxPool",
        inputs=["input_tensor_streamed"],
        outputs=["maxpool_output"],
        name="maxpool_node0",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        ceil_mode=0,
        channel_unroll=16,
        width_unroll=1,
        in_stream_array=1,
        out_stream_array=1,
        in_word_array=16,
        out_word_array=16,
    )
    
    maxpool_node1 = helper.make_node(
        "StreamingMaxPool",
        inputs=["maxpool_output"],
        outputs=["output_tensor"],
        name="maxpool_node1",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        ceil_mode=0,
        channel_unroll=8,
        width_unroll=1,
        in_stream_array=1,
        out_stream_array=1,
        in_word_array=8,
        out_word_array=8,
    )

    graph = helper.make_graph(
        nodes=[axitostream_node, maxpool_node0, maxpool_node1],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        value_info=[input_tensor_streamed, maxpool_output],
    )

    model = qonnx_make_model(graph, producer_name="test_bw")
    model = ModelWrapper(model)
    model.set_metadata_prop("model_II", "8192") 
    layout = TensorLayout((0, 2, 3, 1))
    annotate_layouts(model, ["input_tensor", "input_tensor_streamed", "maxpool_output", "output_tensor"], layout)
    transformed_model = model.transform(AdjustStreamingCommunication())

    # Further assertions can be added here to validate the transformation results
    assert (
        len(transformed_model.get_nodes_by_op_type("BandwidthAdjustDecreaseWord")) == 1
    ), "A BandwidthAdjustDecreaseWord node should have been inserted."
    assert_bandwidth_adjust_metadata(transformed_model, layout, [1, 64, 32, 32])

def test_adjust_streaming_communication_increase_channels():

    output_tensor = helper.make_tensor_value_info(
        name="output_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 64, 32, 32],
    )

    input_tensor = helper.make_tensor_value_info(
        name="input_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 64, 32, 32],
    )
    
    input_tensor_streamed = helper.make_tensor_value_info(
        name="input_tensor_streamed",
        elem_type=TensorProto.FLOAT,
        shape=[1, 64, 32, 32],
    )

    maxpool_output = helper.make_tensor_value_info(
        name="maxpool_output",
        elem_type=TensorProto.FLOAT,
        shape=[1, 64, 32, 32],
    )

    axitostream_node = helper.make_node(
        "AXIToStream",
        inputs=["input_tensor"],
        outputs=["input_tensor_streamed"],
        name="axitostream_node",
        domain="nn2fpga.compiler.custom_op",
        channel_unroll=8,
        width_unroll=1,
        in_stream_array=1,
        out_stream_array=1,
        in_word_array=8,
        out_word_array=8,
    )

    maxpool_node0 = helper.make_node(
        "StreamingMaxPool",
        inputs=["input_tensor_streamed"],
        outputs=["maxpool_output"],
        name="maxpool_node0",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        ceil_mode=0,
        channel_unroll=8,
        width_unroll=1,
        in_stream_array=1,
        out_stream_array=1,
        in_word_array=8,
        out_word_array=8,
    )
    
    maxpool_node1 = helper.make_node(
        "StreamingMaxPool",
        inputs=["maxpool_output"],
        outputs=["output_tensor"],
        name="maxpool_node1",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        ceil_mode=0,
        channel_unroll=16,
        width_unroll=1,
        in_stream_array=1,
        out_stream_array=1,
        in_word_array=16,
        out_word_array=16,
    )

    graph = helper.make_graph(
        nodes=[axitostream_node, maxpool_node0, maxpool_node1],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        value_info=[input_tensor_streamed, maxpool_output],
    )

    model = qonnx_make_model(graph, producer_name="test_bw")
    model = ModelWrapper(model)
    model.set_metadata_prop("model_II", "8192") 
    layout = TensorLayout((0, 2, 3, 1))
    annotate_layouts(model, ["input_tensor", "input_tensor_streamed", "maxpool_output", "output_tensor"], layout)
    transformed_model = model.transform(AdjustStreamingCommunication())

    # Further assertions can be added here to validate the transformation results
    assert (
        len(transformed_model.get_nodes_by_op_type("BandwidthAdjustIncreaseWord")) == 1
    ), "A BandwidthAdjustIncreaseWord node should have been inserted."
    assert_bandwidth_adjust_metadata(transformed_model, layout, [1, 64, 32, 32])


def test_adjust_streaming_communication_increase_streams():

    output_tensor = helper.make_tensor_value_info(
        name="output_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 3, 32, 32],
    )

    input_tensor = helper.make_tensor_value_info(
        name="input_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 3, 32, 32],
    )
    
    input_tensor_streamed = helper.make_tensor_value_info(
        name="input_tensor_streamed",
        elem_type=TensorProto.FLOAT,
        shape=[1, 3, 32, 32],
    )

    maxpool_output = helper.make_tensor_value_info(
        name="maxpool_output",
        elem_type=TensorProto.FLOAT,
        shape=[1, 3, 32, 32],
    )

    axitostream_node = helper.make_node(
        "AXIToStream",
        inputs=["input_tensor"],
        outputs=["input_tensor_streamed"],
        name="axitostream_node",
        domain="nn2fpga.compiler.custom_op",
        channel_unroll=3,
        width_unroll=2,
        in_stream_array=2,
        out_stream_array=2,
        in_word_array=3,
        out_word_array=3,
    )

    maxpool_node0 = helper.make_node(
        "StreamingMaxPool",
        inputs=["input_tensor_streamed"],
        outputs=["maxpool_output"],
        name="maxpool_node0",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        ceil_mode=0,
        channel_unroll=3,
        width_unroll=2,
        in_stream_array=2,
        out_stream_array=2,
        in_word_array=3,
        out_word_array=3,
    )
    
    maxpool_node1 = helper.make_node(
        "StreamingMaxPool",
        inputs=["maxpool_output"],
        outputs=["output_tensor"],
        name="maxpool_node1",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        ceil_mode=0,
        channel_unroll=3,
        width_unroll=4,
        in_stream_array=4,
        out_stream_array=4,
        in_word_array=3,
        out_word_array=3,
    )

    graph = helper.make_graph(
        nodes=[axitostream_node, maxpool_node0, maxpool_node1],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        value_info=[input_tensor_streamed, maxpool_output],
    )

    model = qonnx_make_model(graph, producer_name="test_bw")
    model = ModelWrapper(model)
    model.set_metadata_prop("model_II", "512") 
    layout = TensorLayout((0, 2, 3, 1))
    annotate_layouts(model, ["input_tensor", "input_tensor_streamed", "maxpool_output", "output_tensor"], layout)
    transformed_model = model.transform(AdjustStreamingCommunication())

    # Further assertions can be added here to validate the transformation results
    assert (
        len(transformed_model.get_nodes_by_op_type("BandwidthAdjustIncreaseStreams")) == 1
    ), "A BandwidthAdjustIncreaseStreams node should have been inserted."
    assert_bandwidth_adjust_metadata(transformed_model, layout, [1, 3, 32, 32])

def test_adjust_streaming_communication_decrease_streams():

    output_tensor = helper.make_tensor_value_info(
        name="output_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 3, 32, 32],
    )

    input_tensor = helper.make_tensor_value_info(
        name="input_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 3, 32, 32],
    )
    
    input_tensor_streamed = helper.make_tensor_value_info(
        name="input_tensor_streamed",
        elem_type=TensorProto.FLOAT,
        shape=[1, 3, 32, 32],
    )

    maxpool_output = helper.make_tensor_value_info(
        name="maxpool_output",
        elem_type=TensorProto.FLOAT,
        shape=[1, 3, 32, 32],
    )

    axitostream_node = helper.make_node(
        "AXIToStream",
        inputs=["input_tensor"],
        outputs=["input_tensor_streamed"],
        name="axitostream_node",
        domain="nn2fpga.compiler.custom_op",
        channel_unroll=3,
        width_unroll=2,
        in_stream_array=2,
        out_stream_array=2,
        in_word_array=3,
        out_word_array=3,
    )

    maxpool_node0 = helper.make_node(
        "StreamingMaxPool",
        inputs=["input_tensor_streamed"],
        outputs=["maxpool_output"],
        name="maxpool_node0",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        ceil_mode=0,
        channel_unroll=3,
        width_unroll=2,
        in_stream_array=2,
        out_stream_array=2,
        in_word_array=3,
        out_word_array=3,
    )
    
    maxpool_node1 = helper.make_node(
        "StreamingMaxPool",
        inputs=["maxpool_output"],
        outputs=["output_tensor"],
        name="maxpool_node1",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        ceil_mode=0,
        channel_unroll=3,
        width_unroll=1,
        in_stream_array=1,
        out_stream_array=1,
        in_word_array=3,
        out_word_array=3,
    )

    graph = helper.make_graph(
        nodes=[axitostream_node, maxpool_node0, maxpool_node1],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        value_info=[input_tensor_streamed, maxpool_output],
    )

    model = qonnx_make_model(graph, producer_name="test_bw")
    model = ModelWrapper(model)
    model.set_metadata_prop("model_II", "1024") 
    layout = TensorLayout((0, 2, 3, 1))
    annotate_layouts(model, ["input_tensor", "input_tensor_streamed", "maxpool_output", "output_tensor"], layout)
    transformed_model = model.transform(AdjustStreamingCommunication())

    # Further assertions can be added here to validate the transformation results
    assert (
        len(transformed_model.get_nodes_by_op_type("BandwidthAdjustDecreaseStreams")) == 1
    ), "A BandwidthAdjustDecreaseStreams node should have been inserted."
    assert_bandwidth_adjust_metadata(transformed_model, layout, [1, 3, 32, 32])

def test_adjust_streaming_communication_increase_both():

    output_tensor = helper.make_tensor_value_info(
        name="output_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 15, 14, 14],
    )

    input_tensor = helper.make_tensor_value_info(
        name="input_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 15, 14, 14],
    )
    
    input_tensor_streamed = helper.make_tensor_value_info(
        name="input_tensor_streamed",
        elem_type=TensorProto.FLOAT,
        shape=[1, 15, 14, 14],
    )

    maxpool_output = helper.make_tensor_value_info(
        name="maxpool_output",
        elem_type=TensorProto.FLOAT,
        shape=[1, 15, 14, 14],
    )

    axitostream_node = helper.make_node(
        "AXIToStream",
        inputs=["input_tensor"],
        outputs=["input_tensor_streamed"],
        name="axitostream_node",
        domain="nn2fpga.compiler.custom_op",
        channel_unroll=3,
        width_unroll=2,
        in_stream_array=2,
        out_stream_array=2,
        in_word_array=3,
        out_word_array=3,
    )

    maxpool_node0 = helper.make_node(
        "StreamingMaxPool",
        inputs=["input_tensor_streamed"],
        outputs=["maxpool_output"],
        name="maxpool_node0",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        ceil_mode=0,
        channel_unroll=3,
        width_unroll=2,
        in_stream_array=2,
        out_stream_array=2,
        in_word_array=3,
        out_word_array=3,
    )
    
    maxpool_node1 = helper.make_node(
        "StreamingMaxPool",
        inputs=["maxpool_output"],
        outputs=["output_tensor"],
        name="maxpool_node1",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        ceil_mode=0,
        channel_unroll=5,
        width_unroll=7,
        in_stream_array=7,
        out_stream_array=7,
        in_word_array=5,
        out_word_array=5,
    )

    graph = helper.make_graph(
        nodes=[axitostream_node, maxpool_node0, maxpool_node1],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        value_info=[input_tensor_streamed, maxpool_output],
    )

    model = qonnx_make_model(graph, producer_name="test_bw")
    model = ModelWrapper(model)
    model.set_metadata_prop("model_II", "490") 
    layout = TensorLayout((0, 2, 3, 1))
    annotate_layouts(model, ["input_tensor", "input_tensor_streamed", "maxpool_output", "output_tensor"], layout)
    transformed_model = model.transform(AdjustStreamingCommunication())

    # The expected behavior is:
    # 1. Insert a BandwidthAdjustIncreaseWord node to go from 3 to 15 channel unroll,
    #    because moving to 1 and then to 5 would exceed the model II.
    # 2. Insert a BandwidthAdjustDecreaseWord node to go from 15 to 5 channel unroll.
    # 3. Insert a BandwidthAdjustDecreaseStreams node to go from 2 to 1 width unroll.
    # 4. Insert a BandwidthAdjustIncreaseStreams node to go from 1 to 7 width unroll. 

    # Further assertions can be added here to validate the transformation results
    assert (
        len(transformed_model.get_nodes_by_op_type("BandwidthAdjustDecreaseStreams")) == 1
    ), "A BandwidthAdjustDecreaseStreams node should have been inserted."
    assert (
        len(transformed_model.get_nodes_by_op_type("BandwidthAdjustIncreaseStreams")) == 1
    ), "A BandwidthAdjustIncreaseStreams node should have been inserted."
    assert (
        len(transformed_model.get_nodes_by_op_type("BandwidthAdjustDecreaseWord")) == 1
    ), "A BandwidthAdjustDecreaseWord node should have been inserted."
    assert (
        len(transformed_model.get_nodes_by_op_type("BandwidthAdjustIncreaseWord")) == 1
    ), "A BandwidthAdjustIncreaseWord node should have been inserted."
    assert_bandwidth_adjust_metadata(transformed_model, layout, [1, 15, 14, 14])


def test_adjust_streaming_communication_uses_layout_shape():
    input_tensor = helper.make_tensor_value_info(
        name="input_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 5, 16, 7],
    )

    input_tensor_streamed = helper.make_tensor_value_info(
        name="input_tensor_streamed",
        elem_type=TensorProto.FLOAT,
        shape=[1, 5, 16, 7],
    )

    relu_output = helper.make_tensor_value_info(
        name="relu_output",
        elem_type=TensorProto.FLOAT,
        shape=[1, 5, 16, 7],
    )

    output_tensor = helper.make_tensor_value_info(
        name="output_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1, 5, 16, 7],
    )

    axitostream_node = helper.make_node(
        "AXIToStream",
        inputs=["input_tensor"],
        outputs=["input_tensor_streamed"],
        name="axitostream_node",
        domain="nn2fpga.compiler.custom_op",
        in_stream_array=1,
        out_stream_array=1,
        in_word_array=7,
        out_word_array=7,
    )

    relu_node0 = helper.make_node(
        "StreamingReLU",
        inputs=["input_tensor_streamed"],
        outputs=["relu_output"],
        name="relu_node0",
        domain="nn2fpga.compiler.custom_op",
        in_stream_array=1,
        out_stream_array=1,
        in_word_array=7,
        out_word_array=7,
    )

    relu_node1 = helper.make_node(
        "StreamingReLU",
        inputs=["relu_output"],
        outputs=["output_tensor"],
        name="relu_node1",
        domain="nn2fpga.compiler.custom_op",
        in_stream_array=2,
        out_stream_array=2,
        in_word_array=7,
        out_word_array=7,
    )

    graph = helper.make_graph(
        nodes=[axitostream_node, relu_node0, relu_node1],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        value_info=[input_tensor_streamed, relu_output],
    )

    model = qonnx_make_model(graph, producer_name="test_bw")
    model = ModelWrapper(model)
    model.set_metadata_prop("model_II", "80")
    layout = TensorLayout.identity(4)
    annotate_layouts(model, ["input_tensor", "input_tensor_streamed", "relu_output", "output_tensor"], layout)

    transformed_model = model.transform(AdjustStreamingCommunication())

    assert (
        len(transformed_model.get_nodes_by_op_type("BandwidthAdjustIncreaseStreams")) == 1
    ), "A BandwidthAdjustIncreaseStreams node should have been inserted."
    assert_bandwidth_adjust_metadata(transformed_model, layout, [1, 5, 16, 7])
