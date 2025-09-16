from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from onnxscript.rewriter import rewrite
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model
from backend.custom_op.register_rewrite_rule import collect_rules
from backend.core.tensor_fifo import TensorFifo, get_custom_tensor_fifo_metadata, set_custom_tensor_fifo_metadata
from backend.core.tensor_quant import TensorQuant, get_custom_tensor_datatype
from backend.core.acceleratorpackage import AcceleratorPackage
from onnxscript import ir
from onnx import TensorProto, helper, StringStringEntryProto
import numpy as np
import logging
logger = logging.getLogger(__name__)

class LowerToHLS(Transformation):
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        nodes = []
        fifo = {}
        inits = []
        tensors = []
        ap = AcceleratorPackage.from_json(model.get_metadata_prop("accelerator_package"))

        # iterate in topological order
        for node in model.graph.node:
            custom_op = getCustomOp(node)

            # ask the op to expand into multiple kernels
            sub_nodes, sub_inits, sub_fifo = custom_op.lower_to_hls(model)

            # stitch in
            nodes.extend(sub_nodes)
            fifo.update(sub_fifo)
            inits.extend(sub_inits)

        for key in fifo.keys():
            tensors.append(
                helper.make_tensor_value_info(
                    key, TensorProto.FLOAT, None
                )  # shape is dynamic
            )

        # Build new graph
        graph = helper.make_graph(
            nodes,
            model.graph.name + "_to_hls",
            list(model.graph.input),
            list(model.graph.output),
            initializer=inits,
            value_info=tensors
        )
        hls_model = qonnx_make_model(graph, producer_name="nn2fpga")
        hls_model = ModelWrapper(hls_model)

        # Copy metadata props
        src = model.model
        dst = hls_model.model

        # Build index of existing keys in dst
        dst_idx = {p.key: i for i, p in enumerate(dst.metadata_props)}

        for p in src.metadata_props:
            if p.key in dst_idx:
                dst.metadata_props[dst_idx[p.key]].value = p.value
            else:
                kv = StringStringEntryProto()
                kv.key = p.key
                kv.value = p.value
                dst.metadata_props.append(kv)

        # Set fifo metadata
        for node in [x.name for x in hls_model.graph.value_info]:
            if node not in fifo:
                raise Exception(f"Node {node} missing fifo metadata")
            set_custom_tensor_fifo_metadata(hls_model, node, fifo[node])

        # Clear types to remove shape info, which are not valid anymore
        # after lowering tensors to hls streams.
        for v in hls_model.model.graph.input:  v.ClearField("type")
        for v in hls_model.model.graph.output: v.ClearField("type")

        # Set fifo metadata for inputs/outputs of the model
        for input in hls_model.graph.input:
            consumer = model.find_consumer(input.name)
            if consumer is None:
                raise ValueError(f"Input {input.name} does not have a consumer.")
            if consumer.op_type != "NHWCToStream":
                raise ValueError(f"Input {input.name} consumer is not NHWCToStream.")
            axi_word = getCustomOp(consumer).get_nodeattr("axi_bitwidth")
            tensor_quant = get_custom_tensor_datatype(model, input.name)
            data_per_word = axi_word // int(tensor_quant.bitwidth)
            input_shape = model.get_tensor_shape(input.name)
            word_per_tensor = int(np.ceil(np.prod(input_shape) / data_per_word))
            set_custom_tensor_fifo_metadata(
                hls_model,
                input.name,
                TensorFifo(
                    depth=0, hls_type=f"ap_axiu<{axi_word}, 0, 0, 0>", n_array=word_per_tensor
                ),
            )

        for output in hls_model.graph.output:
            producer = model.find_producer(output.name)
            if producer is None:
                raise ValueError(f"Output {output.name} does not have a producer.")
            if producer.op_type != "StreamToNHWC":
                raise ValueError(f"Output {output.name} producer is not StreamToNHWC.")
            axi_word = getCustomOp(producer).get_nodeattr("axi_bitwidth")
            tensor_quant = get_custom_tensor_datatype(model, output.name)
            output_shape = model.get_tensor_shape(output.name)
            data_per_word = axi_word // int(tensor_quant.bitwidth)
            word_per_tensor = int(np.ceil(np.prod(output_shape) / data_per_word))
            set_custom_tensor_fifo_metadata(
                hls_model,
                output.name,
                TensorFifo(
                    depth=0, hls_type=f"ap_axiu<{axi_word}, 0, 0, 0>", n_array=word_per_tensor
                ),
            )
        
        return (hls_model, False)
