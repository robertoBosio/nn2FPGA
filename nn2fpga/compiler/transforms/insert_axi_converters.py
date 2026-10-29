from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from nn2fpga.compiler.utils.board_util import read_board_info
from onnx import helper
from nn2fpga.compiler.core.tensor_type import (
    get_custom_tensor_datatype,
    set_custom_tensor_datatype,
)
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
import logging
logger = logging.getLogger(__name__)

class InsertAXIConverters(Transformation):
    """
    Inserts AXI converters for each input/output tensors in the model.
    This will convert the input/output tensor from/to AXI format.
    """

    def __init__(self, use_dma: bool = True):
        super().__init__()
        self.use_dma = use_dma

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        board_res = read_board_info(
            board=model.get_metadata_prop("board_name"),
        )
        interface_type = "s_axis" if self.use_dma else "m_axi"
        port_mode = "axis" if self.use_dma else "m_axi"
        ap = AcceleratorPackage.from_json(
            model.get_metadata_prop("accelerator_package")
        )

        new_nodes = []
        for i, inp in enumerate(model.graph.input):
            consumers = model.find_consumers(inp.name)
            if (
                consumers is not None
                and len(consumers) == 1
                and consumers[0].op_type == "AXIToStream"
            ):
                getCustomOp(consumers[0]).set_nodeattr(
                    "interface_type", interface_type
                )
                _, input_port = ap.get_input_by_internal_name(inp.name)
                input_port["mode"] = port_mode
                continue

            orig_input_name = inp.name
            produce_stream_output = f"{orig_input_name}_streamed"

            # Create the custom node AXIToStream
            produce_node = helper.make_node(
                op_type="AXIToStream",
                domain="nn2fpga.compiler.custom_op",
                inputs=[orig_input_name],
                outputs=[produce_stream_output],
                normalize=0,
                axi_bitwidth=board_res["axi_bitwidth"],
                interface_type=interface_type,
                name=f"AXIToStream_{i}",
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
            _, input_port = ap.get_input_by_internal_name(orig_input_name)
            input_port["mode"] = port_mode
            logger.info(f"Inserted AXIToStream node for input {orig_input_name}")

        # Insert all new nodes at the beginning
        for node in reversed(new_nodes):
            model.graph.node.insert(0, node)

        new_nodes = []
        for i, out in enumerate(model.graph.output):
            producer = model.find_producer(out.name)
            if producer is None:
                continue
            if producer.op_type == "StreamToAXI":
                getCustomOp(producer).set_nodeattr("interface_type", interface_type)
                _, output_port = ap.get_output_by_internal_name(out.name)
                output_port["mode"] = port_mode
                continue

            orig_output_name = out.name
            consume_stream_output = f"{orig_output_name}_streamed"

            # Create the custom node StreamToAXI
            consume_node = helper.make_node(
                op_type="StreamToAXI",
                domain="nn2fpga.compiler.custom_op",
                inputs=[consume_stream_output],
                outputs=[orig_output_name],
                axi_bitwidth=board_res["axi_bitwidth"],
                interface_type=interface_type,
                name=f"StreamToAXI_{i}",
            )

            # Rewire all the producers of orig_output_name to produce to consume_stream_output
            for j, prod_out in enumerate(producer.output):
                if prod_out == orig_output_name:
                    producer.output[j] = consume_stream_output

            model.set_tensor_shape(
                consume_stream_output, model.get_tensor_shape(orig_output_name)
            )
            tq = get_custom_tensor_datatype(model, orig_output_name)
            if tq is not None:
                set_custom_tensor_datatype(model, consume_stream_output, tq)
            new_nodes.append(consume_node)
            _, output_port = ap.get_output_by_internal_name(orig_output_name)
            output_port["mode"] = port_mode
            logger.info(f"Inserted StreamToAXI node for output {orig_output_name}")

        # Insert all new nodes at the beginning
        for node in new_nodes:
            model.graph.node.append(node)
        
        model.set_metadata_prop("accelerator_package", ap.to_json())
        return (model, False)
