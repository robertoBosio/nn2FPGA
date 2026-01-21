import base64
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph, GiveUniqueNodeNames, GiveReadableTensorNames
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.custom_op.general.quant as qonnx_quant
from onnx import NodeProto, TensorProto, helper
from backend.core.tensor_quant import (
    TensorQuant,
    set_custom_tensor_datatype,
    get_custom_tensor_datatype,
)
import numpy as np
from qonnx.custom_op.registry import getCustomOp
from backend.core.acceleratorpackage import AcceleratorPackage
from backend.custom_op.op_base import NodeInterface, HasParameters
from backend.util.board_util import read_board_info
import logging
logger = logging.getLogger(__name__)

# Save original references before monkey-patching
_original_min_int = qonnx_quant.min_int
_original_max_int = qonnx_quant.max_int


def _np_min_int(signed: bool, narrow_range: bool, bit_width: int) -> np.generic:
    val = _original_min_int(signed, narrow_range, bit_width)
    return np.array(val)


def _np_max_int(signed: bool, narrow_range: bool, bit_width: int) -> np.generic:
    val = _original_max_int(signed, narrow_range, bit_width)
    return np.array(val)


def safe_int_quant_call(*args, **kwargs):
    # Monkey-patch min/max to return NumPy-safe scalars
    qonnx_quant.min_int = _np_min_int
    qonnx_quant.max_int = _np_max_int

    try:
        return qonnx_quant.quant(*args, **kwargs)
    finally:
        # Restore originals
        qonnx_quant.min_int = _original_min_int
        qonnx_quant.max_int = _original_max_int


def quant_array(inp_tensor, scale, zeropt, bitwidth, signed, narrow, rounding_mode):
    """Quantize an input tensor to a specified bitwidth and return the quantized tensor."""

    # Let QONNX handle the quantization. This function return the quantized floating tensor.
    inp_tensor = safe_int_quant_call(
        inp_tensor,
        scale=scale,
        zeropt=zeropt,
        bitwidth=bitwidth,
        signed=signed,
        narrow=narrow,
        rounding_mode=rounding_mode,
    )

    # Moving from the quantized floating tensor to a int tensor, knowing that clipping
    # and rounding have already been applied.
    inp_tensor = inp_tensor / scale
    inp_tensor = inp_tensor + zeropt
    return inp_tensor.astype(np.int32)  # Convert to uint32 for packing

def hoist_param(model: ModelWrapper, node: NodeProto) -> None:
    op = getCustomOp(node)
    if not isinstance(op, HasParameters):
        return

    mapping = {}  # old_init_name -> new_stream_output
    counter = 0
    for p in op.list_parameters(model):
        raw = model.get_initializer(p.name)
        if raw is None:
            continue

        # Quantize to integer array (same shape as original weights)
        q_arr = quant_array(
            raw,
            scale=p.tensor_quant.scale,
            zeropt=p.tensor_quant.zeropt,
            bitwidth=p.tensor_quant.bitwidth,
            signed=p.tensor_quant.signed,
            narrow=p.tensor_quant.narrow,
            rounding_mode=p.tensor_quant.rounding_mode,
        )

        # Ensure a container dtype consistent with bitwidth/sign
        # (keep shape intact; we don't pack)
        q_arr = q_arr.astype(p.tensor_quant.get_numpy_dtype(),  copy=False)
        # create initializer for the unpacked int array
        init_name = model.make_new_valueinfo_name()
        model.set_initializer(init_name, q_arr)
        model.set_tensor_shape(init_name, list(q_arr.shape))

        # Build StreamingMemory node from spec
        out_name = model.make_new_valueinfo_name()
        model.set_tensor_shape(out_name, list(q_arr.shape))
        set_custom_tensor_datatype(model, out_name, p.tensor_quant)
        sm = helper.make_node(
            "StreamingMemory",
            inputs=[init_name],
            outputs=[out_name],
            name=f"StreamingMemory_{node.name}_{counter}",
            in_channel_unroll=p.in_channel_unroll,
            out_channel_unroll=p.out_channel_unroll,
            width_unroll=p.width_unroll,
            in_word_array=1,
            out_word_array=p.in_channel_unroll * p.out_channel_unroll,
            in_stream_array=1,
            out_stream_array=p.width_unroll,
            data_per_word=p.data_per_word,
            mem_shape=p.shape,
            times=p.times,
            domain="backend.custom_op",
        )
        model.graph.node.extend([sm])

        # Rewire the op’s input at the original index
        node.input[p.input_index] = out_name
        # Remove original initializer
        model.del_initializer(p.name)

        mapping[p.name] = out_name
        counter = counter + 1
    
    op.set_external_storage()

class AddStreamingParams(Transformation):
    """A transformation pass that adds the logic to handle streaming parameters at startup.
    Each node with parameters will have an associated ParamStream node that is in charge of
    streaming the parameters to the node.
    """

    def __init__(self, nn2fpga_root: str = "/tmp"):
        """
        Initializes the AddStreamingParams transformation.
        Args:
            nn2fpga_root (str): The root directory of nn2FPGA.
        """
        super().__init__()
        self.nn2fpga_root = nn2fpga_root

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        board_res = read_board_info(
            board=model.get_metadata_prop("board_name"),
        )
        sequential_streaming = list()
        grouped_initializer = np.array([], dtype=np.uint32)
        params_quant = TensorQuant(
            scale=1.0,
            zeropt=0,
            bitwidth=32,
            signed=False,
            narrow=False,
            rounding_mode="ROUND",
        )

        for node in model.graph.node:
            hoist_param(model, node)

        # Find all nodes with parameters that need streaming
        # and collect them in a list.
        for node in model.get_nodes_by_op_type("StreamingMemory"):
            tensor_quant = get_custom_tensor_datatype(model, node.output[0])
            packed_array = getCustomOp(node).reshape_and_pack_init_to_int32words(
                model.get_initializer(node.input[0]),
                data_bitwidth=tensor_quant.bitwidth,
                word_bitwidth=32,
            )
            grouped_initializer = np.concatenate(
                (grouped_initializer, packed_array), axis=0
            )
            sequential_streaming.append(
                (node, packed_array.size)
            )

        grouped_initializer = grouped_initializer.reshape((1, grouped_initializer.size))

        if len(sequential_streaming) == 0:
            logger.info("No parameters to stream. Skipping AddStreamingParams transformation.")
            return (model, False)
        
        logger.info(f"Packed streaming parameters size: {grouped_initializer.size} words")

        ap = AcceleratorPackage.from_json(
            model.get_metadata_prop("accelerator_package")
        )

        # Add an input to the model for the streaming parameters.
        # Add also an initializer since it's constant.
        # The 'const_' string in the name is mandatory to recognize the initializer
        # as a special in the simulation flow.
        index = len(ap.input_map)
        ap.input_map["const_param_stream"] = {
            "new_name": "const_param_stream",
            "index": index,
            "shape": grouped_initializer.shape,
            "quant": params_quant.get_canonical_name(),
            "value": base64.b64encode(grouped_initializer.tobytes()).decode("ascii"),
        }
        param_stream_input = helper.make_tensor_value_info(
            "const_param_stream", TensorProto.INT32, grouped_initializer.shape
        )
        model.graph.input.extend([param_stream_input])
        input_stream = [f"{param_stream_input.name}_streamed"]
        model.set_tensor_shape(
            input_stream[0], model.get_tensor_shape(param_stream_input.name)
        )
        set_custom_tensor_datatype(model, param_stream_input.name, params_quant)

        # Create the NHWCToStream node
        produce_node = helper.make_node(
            op_type="NHWCToStream",
            domain="backend.custom_op",
            inputs=[param_stream_input.name],
            outputs=input_stream,
            normalize=0,
            channel_unroll=1,
            width_unroll=1,
            in_stream_array=1,
            out_stream_array=1,
            in_word_array=1,
            out_word_array=1,
            axi_bitwidth=board_res["axi_bitwidth"],
            name=f"NHWCToStream_const_param_stream",
        )

        model.graph.node.append(produce_node)

        # Create a ParamStream node for each node with parameters.
        params_to_shift = grouped_initializer.shape[1]
        for node, params_size in sequential_streaming[:-1]:
            custom_op = getCustomOp(node)
            node.output.extend([f"{node.name}_shift_out"])

            params_to_shift -= params_size
            # model.set_tensor_shape(output_stream[0], custom_op.get_nodeattr("mem_shape"))
            custom_op.set_nodeattr("data_to_shift", params_to_shift)
            model.del_initializer(node.input[0])
            node.input[0] = input_stream[0]
            set_custom_tensor_datatype(
                model,
                node.input[0],
                params_quant
            )

            # The next input is the first output of the current ParamStream node
            input_stream = [f"{node.name}_shift_out"]
        else:
            # For the last node we do not need the shift_out output,
            # we just need the stream output.
            node, params_size = sequential_streaming[-1]

            custom_op = getCustomOp(node)
            # node.output.extend([f"{node.name}_shift_out"])
            model.set_tensor_shape(
                input_stream[0], [params_to_shift]
            )

            params_to_shift -= params_size
            # model.set_tensor_shape(output_stream[0], custom_op.get_nodeattr("mem_shape"))
            custom_op.set_nodeattr("data_to_shift", params_to_shift)
            model.del_initializer(node.input[0])
            node.input[0] = input_stream[0]
            set_custom_tensor_datatype(model, node.input[0], params_quant)

        # Sort the graph.
        model = model.transform(SortGraph())
        input_original_names = [i.name for i in model.graph.input]
        output_original_names = [o.name for o in model.graph.output]
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        input_new_names = [i.name for i in model.graph.input]
        output_new_names = [o.name for o in model.graph.output]
        for original, new in zip(input_original_names, input_new_names):
            for value in ap.input_map.values():
                if value['new_name'] == original:
                    value['new_name'] = new
                    break
        for original, new in zip(output_original_names, output_new_names):
            for value in ap.output_map.values():
                if value['new_name'] == original:
                    value['new_name'] = new
                    break
        model.set_metadata_prop("accelerator_package", ap.to_json())


        # os.system(f"mkdir -p {self.nn2fpga_root}/params/")
        # np.save(f"{self.nn2fpga_root}/params/streaming_params.npy", grouped_initializer)

        # # For c++ testbench, we need to save the parameters in a binary file.
        # with open(f"{self.nn2fpga_root}/params/streaming_params.bin", "wb") as file:
        #     file.write(grouped_initializer.tobytes())
        return (model, False)
