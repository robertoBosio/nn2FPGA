from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.util.board_util import read_board_info
from onnx import helper
from qonnx.util.basic import get_by_name
from backend.core.tensor_quant import get_custom_tensor_datatype, TensorQuant, set_custom_tensor_datatype
import logging
logger = logging.getLogger(__name__)

class InsertAXIConverters(Transformation):
    """
    Inserts AXI converters for each input/output tensors in the model.
    This will convert the input/output tensor from/to AXI format.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        board_res = read_board_info(
            board=model.get_metadata_prop("board_name"),
        )

        new_nodes = []
        for i, inp in enumerate(model.graph.input):
            consumers = model.find_consumers(inp.name)
            if (
                consumers is not None
                and len(consumers) == 1
                and consumers[0].op_type == "NHWCToStream"
            ):
                # This input is already transformed, skip it.
                continue

            orig_input_name = inp.name
            produce_stream_output = f"{orig_input_name}_streamed"

            # Create the custom node NHWCToStream
            produce_node = helper.make_node(
                op_type="NHWCToStream",
                domain="backend.custom_op",
                inputs=[orig_input_name],
                outputs=[produce_stream_output],
                normalize=0,
                axi_bitwidth=board_res["axi_bitwidth"],
                name=f"NHWCToStream_{i}",
                in_ch_par=1,
                out_ch_par=1,
                in_w_par=1,
                out_w_par=1,
            )

            model.set_tensor_shape(
                produce_stream_output, model.get_tensor_shape(orig_input_name)
            )
            tq = get_custom_tensor_datatype(model, orig_input_name)
            if tq is not None:
                set_custom_tensor_datatype(model, produce_stream_output, tq)

            # Replace all uses of this input
            for node in model.graph.node:
                node_inputs = list(node.input)
                for j, node_in in enumerate(node_inputs):
                    if node_in == orig_input_name:
                        node.input[j] = produce_stream_output

            new_nodes.append(produce_node)
            logger.info(f"Inserted NHWCToStream node for input {orig_input_name}")

        # Insert all new nodes at the beginning
        for node in reversed(new_nodes):
            model.graph.node.insert(0, node)

        new_nodes = []
        for i, out in enumerate(model.graph.output):
            producer = model.find_producer(out.name)
            if producer is not None and producer.op_type == "StreamToNHWC":
                # This output is already transformed, skip it.
                continue

            orig_output_name = out.name
            consume_stream_output = f"{orig_output_name}_streamed"

            # Create the custom node StreamToNHWC
            consume_node = helper.make_node(
                op_type="StreamToNHWC",
                domain="backend.custom_op",
                outputs=[consume_stream_output],
                inputs=[orig_output_name],
                axi_bitwidth=board_res["axi_bitwidth"],
                name=f"StreamToNHWC_{i}",
                in_ch_par=1,
                out_ch_par=1,
                in_w_par=1,
                out_w_par=1,
            )

            get_by_name(model.graph.output, orig_output_name).name = consume_stream_output 
            model.set_tensor_shape(
                consume_stream_output, model.get_tensor_shape(orig_output_name)
            )
            tq = get_custom_tensor_datatype(model, orig_output_name)
            if tq is not None:
                set_custom_tensor_datatype(model, consume_stream_output, tq)
            new_nodes.append(consume_node)
            logger.info(f"Inserted StreamToNHWC node for output {orig_output_name}")

        # Insert all new nodes at the beginning
        for node in new_nodes:
            model.graph.node.append(node)

        return (model, False)
