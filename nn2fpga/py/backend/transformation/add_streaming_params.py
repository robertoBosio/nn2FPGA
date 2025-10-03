import base64
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph, GiveUniqueNodeNames, GiveReadableTensorNames
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.custom_op.general.quant as qonnx_quant
from onnx import NodeProto, TensorProto, helper
from backend.util.par_utils import get_par_attributes
from backend.core.tensor_quant import (
    TensorQuant,
    set_custom_tensor_datatype,
    get_custom_tensor_datatype,
)
import numpy as np
from qonnx.custom_op.registry import getCustomOp
from backend.core.acceleratorpackage import AcceleratorPackage

NODE_WITH_PARAMS = [
    "StreamingConv",
    "StreamingDepthwiseConv",
    "Gemm",
    "MatMul",
]

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

    # Let QONNX handle the quantization. This function return the dequantized tensor.
    inp_tensor = safe_int_quant_call(
        inp_tensor,
        scale=scale,
        zeropt=zeropt,
        bitwidth=bitwidth,
        signed=signed,
        narrow=narrow,
        rounding_mode=rounding_mode,
    )

    # Moving from a dequantized tensor to a quantized tensor, knowing that clipping
    # and rounding have already been applied.
    inp_tensor = inp_tensor / scale
    inp_tensor = inp_tensor + zeropt
    return inp_tensor.astype(np.int32)  # Convert to uint32 for packing

def _make_streaming_memory_node_unpacked(
    model: ModelWrapper,
    quant_arr: np.ndarray,
    tensor_quant: TensorQuant,
    par: dict,
    out_tensor_shape: list,
    base_name: str,
) -> tuple[str, str]:
    """
    Create:
      - a new initializer with the quantized (unpacked) tensor, same shape
      - a StreamingMemory node that consumes that initializer
      - a fresh output value name whose type/shape mirror the initializer

    Returns (init_name, out_name).
    """
    init_name = model.make_new_valueinfo_name()
    out_name  = model.make_new_valueinfo_name()

    model.set_initializer(init_name, quant_arr)
    model.set_tensor_shape(init_name, list(quant_arr.shape))
    model.set_tensor_shape(out_name, list(quant_arr.shape))
    set_custom_tensor_datatype(model, out_name, tensor_quant)

    data_per_word = 32 // tensor_quant.bitwidth

    # Expand shapes to 4D if needed
    if len(out_tensor_shape) < 4:
        out_tensor_shape = out_tensor_shape + [1] * (4 - len(out_tensor_shape))
    mem_shape = list(quant_arr.shape)
    if len(mem_shape) < 4:
        mem_shape = mem_shape + [1] * (4 - len(mem_shape))
    in_ch_par = min(par.get("in_ch_par", 1), mem_shape[1])
    out_ch_par = min(par.get("out_ch_par", 1), mem_shape[0])
    in_w_par = 1
    out_w_par = np.prod(mem_shape[2:])
    times = out_tensor_shape[2] * out_tensor_shape[3] // par.get("out_w_par", 1)

    sm_node = helper.make_node(
        "StreamingMemory",
        inputs=[init_name],
        outputs=[out_name],
        name=f"StreamingMemory_{base_name}",
        in_ch_par=in_ch_par,
        out_ch_par=out_ch_par,
        in_w_par=in_w_par,
        out_w_par=out_w_par,
        data_per_word=data_per_word,
        mem_shape=quant_arr.shape,
        times=times,
        domain="backend.custom_op",
    )
    model.graph.node.extend([sm_node])
    return init_name, out_name


def hoist_params_to_streaming_memory_unpacked(model: ModelWrapper, node: NodeProto) -> None:
    """
    For a StreamingConv node:
      - Quantize weights (and biases, if present) WITHOUT packing
      - Insert a StreamingMemory node per parameter (weights and optionally biases)
      - Replace the corresponding StreamingConv inputs with the new SM outputs
      - Remove only the original weight/bias initializers (keep scale/zeropt/bitwidth)

    Mutates `model` in-place.
    """
    custom_node = getCustomOp(node)
    par = get_par_attributes(node)
    out_tensor_shape = model.get_tensor_shape(node.output[0])

    def _get_init(name: str) -> np.ndarray:
        arr = model.get_initializer(name)
        if arr is None:
            raise ValueError(f"Expected initializer '{name}' not found.")
        return arr

    # -------- Weights: inputs [1]=W, [2]=scale, [3]=zeropt, [4]=bitwidth --------
    w_name   = node.input[1]
    w_scale  = _get_init(node.input[2])
    w_zeropt = _get_init(node.input[3])
    w_bw     = _get_init(node.input[4])

    # Quantize to integer array (same shape as original weights)
    quant_w = quant_array(
        _get_init(w_name),
        scale=w_scale,
        zeropt=w_zeropt,
        bitwidth=w_bw,
        signed=custom_node.get_nodeattr("w_signed"),
        narrow=custom_node.get_nodeattr("w_narrow"),
        rounding_mode=custom_node.get_nodeattr("w_rounding_mode"),
    )
    tensor_quant = TensorQuant(
        scale=1.0,
        zeropt=0,
        bitwidth=w_bw,
        signed=custom_node.get_nodeattr("w_signed"),
        narrow=custom_node.get_nodeattr("w_narrow"),
        rounding_mode=custom_node.get_nodeattr("w_rounding_mode"),
    )

    # Ensure a container dtype consistent with bitwidth/sign
    # (keep shape intact; we don't pack)
    quant_w = quant_w.astype(tensor_quant.get_numpy_dtype(),  copy=False)

    # Insert StreamingMemory that produces the same shape/type
    _, w_out = _make_streaming_memory_node_unpacked(
        model,
        quant_w,
        tensor_quant,
        par,
        out_tensor_shape,
        base_name=f"{node.name}_weights",
    )
    # Wire into the conv
    node.input[1] = w_out
    # Drop the original weight initializer (keep metadata in [2],[3],[4])
    if model.get_initializer(w_name) is not None:
        model.del_initializer(w_name)

    # -------- Biases (optional): inputs [5]=B, [6]=scale, [7]=zeropt, [8]=bitwidth --------
    if len(node.input) > 5 and node.input[5] != "":
        b_name   = node.input[5]
        b_scale  = _get_init(node.input[6])
        b_zeropt = _get_init(node.input[7])
        b_bw     = _get_init(node.input[8])

        quant_b = quant_array(
            _get_init(b_name),
            scale=b_scale,
            zeropt=b_zeropt,
            bitwidth=b_bw,
            signed=custom_node.get_nodeattr("b_signed"),
            narrow=custom_node.get_nodeattr("b_narrow"),
            rounding_mode=custom_node.get_nodeattr("b_rounding_mode"),
        )
        tensor_quant = TensorQuant(
            scale=1.0,
            zeropt=0,
            bitwidth=b_bw,
            signed=custom_node.get_nodeattr("b_signed"),
            narrow=custom_node.get_nodeattr("b_narrow"),
            rounding_mode=custom_node.get_nodeattr("b_rounding_mode"),
        )

        # Container dtype for biases as above; keep original bias shape
        quant_b = quant_b.astype(tensor_quant.get_numpy_dtype(),  copy=False)

        _, b_out = _make_streaming_memory_node_unpacked(
            model,
            quant_b,
            tensor_quant,
            par,
            out_tensor_shape,
            base_name=f"{node.name}_biases"
        )
        node.input[5] = b_out
        if model.get_initializer(b_name) is not None:
            model.del_initializer(b_name)

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
            if node.op_type in NODE_WITH_PARAMS:
                hoist_params_to_streaming_memory_unpacked(model, node)

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
            return (model, False)

        ap = AcceleratorPackage.from_json(
            model.get_metadata_prop("accelerator_package")
        )
        # Add an input to the model for the streaming parameters.
        # Add also an initializer since it's constant.
        # The 'const_' string in the name is mandatory to recognize the initializer
        # as a special in the simulation flow.
        ap.input_map["const_param_stream"] = {
            "new_name": "const_param_stream",
            "shape": grouped_initializer.shape,
            "quant": params_quant.get_canonical_name(),
            "value": base64.b64encode(grouped_initializer.tobytes()).decode("ascii"),
        }
        param_stream_input = helper.make_tensor_value_info(
            "const_param_stream", TensorProto.INT32, grouped_initializer.shape
        )
        model.graph.input.extend([param_stream_input])

        # Create a ParamStream node for each node with parameters.
        input_stream = [param_stream_input.name]
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
