import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.core.tensor_fifo import TensorFifo
from backend.custom_op.hlskernel import HLSKernel
from backend.custom_op.op_base import NN2FPGAOp, NodeInterface
from backend.util.codegen_utils import (
    cpp_function,
    cpp_variable,
    cpp_object,
    get_struct_type,
    get_stream_type,
    get_hls_quant_type,
)
from backend.core.tensor_quant import TensorQuant
from backend.util.par_utils import get_par_attributes
from backend.custom_op.register_rewrite_rule import register_rules
from onnxscript import ir
from onnx_ir import convenience as ir_convenience
from onnxscript.rewriter import pattern
import logging

logger = logging.getLogger(__name__)

def _get_B_as_python_number(B: ir.Value) -> float:
    t = ir_convenience.get_const_tensor(B)  # handles initializer OR Constant node
    if t is None:
        raise ValueError("B is not a compile-time constant")
    arr = t.numpy().reshape(-1)
    return float(arr[-1])  # keep your original "last element" behavior

class StreamingSigmoid(NN2FPGAOp):

    @staticmethod
    def hardsigmoid_pattern(op, x, B, alpha):
        y = op.HardSigmoid(x, alpha=alpha)
        return op.Mul(y, B)

    @staticmethod
    def _condition(context, x, B, alpha, **_):
        # Only rewrite if B is actually constant
        return ir_convenience.get_const_tensor(B) is not None

    @staticmethod
    def rewrite(op, x, B, alpha, **_):
        b_value = _get_B_as_python_number(B)
        return op.StreamingSigmoid(
            x,
            alpha=alpha,
            B=b_value,
            _domain="backend.custom_op",
        )

    @register_rules
    def register_rules():
        return [
            pattern.RewriteRule(
                StreamingSigmoid.hardsigmoid_pattern,
                StreamingSigmoid.rewrite,
                StreamingSigmoid._condition,
            )
        ]

    def get_nodeattr_types(self):
        return {
            "alpha": ("f", False, 0.01),  # Slope for negative inputs
            "B": ("f", False, 1.0),      # Scaling factor after HardSigmoid

            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),

            "channel_unroll": ("i", False, 1),
            "width_unroll": ("i", False, 1),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        return helper.make_node(
            "Sigmoid",
            inputs=node.input,
            outputs=node.output,
            alpha=self.get_nodeattr("alpha"),
            name=f"{node.name}_shape_compatible",
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        # create a standard relu node to compute the result
        node = self.onnx_node
        node_sigmoid = helper.make_node(
            "Sigmoid",
            inputs=node.input,
            outputs=node.output,
            alpha=self.get_nodeattr("alpha"),
            name=f"{node.name}_shape_compatible",
        )

        # Make single node graph for execution
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)

        graph_sigmoid = helper.make_graph(
            nodes=[node_sigmoid],
            name="single-sigmoid-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_sigmoid = qonnx_make_model(graph_sigmoid, **onnx_kwargs)
        idict = {node.input[0]: inp_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_sigmoid.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_variable_declaration(self, model, input_quant, output_quant) -> str:
        """ Get the internal cpp variables of the StreamingSigmoid node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """

        nbits = input_quant.bitwidth
        LUT_entries = 1 << nbits

        # 1) Choose container dtype for ONNX and NumPy
        if nbits <= 8:
            onnx_int_type = TensorProto.INT8
            np_int_type = np.int8
        elif nbits <= 16:
            onnx_int_type = TensorProto.INT16
            np_int_type = np.int16
        elif nbits <= 32:
            onnx_int_type = TensorProto.INT32
            np_int_type = np.int32
        else:
            raise ValueError(f"Unsupported bitwidth {nbits} (> 32).")

        # 2) Raw code values: 0 .. 2^nbits - 1
        raw_codes = np.arange(LUT_entries, dtype=np.int64)

        # 3) Sign-extend from nbits to container width
        sign_bit = 1 << (nbits - 1)
        full_range = 1 << nbits

        signed_values = raw_codes.copy()
        signed_values[signed_values >= sign_bit] -= full_range
        # At this point, signed_values holds the *mathematical* two's complement values
        # corresponding to each n-bit code.

        # 4) Cast to container dtype expected by ONNX and shape it
        input_tensor = signed_values.astype(np_int_type).reshape((1, LUT_entries, 1, 1))

        # Define the input tensor to accomodate enough values to fill the LUT
        X = helper.make_tensor_value_info(
            "X",
            onnx_int_type,
            [
                1,
                LUT_entries,
                1,
                1,
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_int_type,
            [
                1,
                LUT_entries,
                1,
                1,
            ],
        )

        X_scale = helper.make_tensor(
            "X_scale", TensorProto.FLOAT, [], [input_quant.scale]
        )
        X_zp = helper.make_tensor("X_zp", onnx_int_type, [], [input_quant.zeropt])
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [output_quant.scale]
        )
        Y_zp = helper.make_tensor("Y_zp", onnx_int_type, [], [output_quant.zeropt])
        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        B = helper.make_tensor(
            "B", TensorProto.FLOAT, [], [self.get_nodeattr("B")]
        )

        HardSigmoid = helper.make_node(
            "HardSigmoid",
            alpha=self.get_nodeattr("alpha"),
            inputs=["X_dq"],
            outputs=["A"],
        )

        Mul = helper.make_node(
            "Mul",
            inputs=["A", "B"],
            outputs=["Y_scaled"],
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_scaled", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [dqlinear, HardSigmoid, Mul, qlinear],
            "qhard_sigmoid_test",
            [X],
            [Y],
            initializer=[X_scale, X_zp, Y_scale, Y_zp, B],
        )
        model_qonnx = helper.make_model(graph, producer_name="qonnx")
        sess = rt.InferenceSession(
            model_qonnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"X": input_tensor})[0]
        lut_values = y.flatten().tolist()

        lut_variable = cpp_variable(
            name=f"{self.onnx_node.name}_lut",
            primitive=f"{get_hls_quant_type(output_quant)}",
            value=lut_values,
        )

        return lut_variable.generate_initialization().code

    def __get_object_declaration(self, model) -> cpp_object:

        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if (input_quant is None):
            raise ValueError(f"Input {self.onnx_node.input[0]} has no quantization info")
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if (output_quant is None):
            raise ValueError(f"Output {self.onnx_node.output[0]} has no quantization info")

        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Input {self.onnx_node.input[0]} has no shape info")
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Output {self.onnx_node.output[0]} has no shape info")

        lut_size = 1 << input_quant.bitwidth
        StreamingSigmoid = cpp_object(
            "StreamingLUT",
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_struct_type(input_quant, self.get_nodeattr('in_word_array'))}",
                    f"TInputWord",
                ),
                (
                    f"{get_hls_quant_type(input_quant)}",
                    f"TInput",
                ),
                (
                    f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}",
                    f"TOutputWord",
                ),
                (
                    f"{get_hls_quant_type(output_quant)}",
                    f"TOutput",
                ),
                (f"{lut_size}", "LUT_SIZE"),
                (f"{input_shape[2]}", "IN_HEIGHT"),
                (f"{input_shape[3]}", "IN_WIDTH"),
                (f"{input_shape[1]}", "IN_CH"),
                (f"{self.get_nodeattr('channel_unroll')}", "CH_PAR"),
                (f"{self.get_nodeattr('width_unroll')}", "W_PAR"),
            ]
        )

        return StreamingSigmoid.generate_declaration()

    def __get_run_call(self, hls_tag: int) -> str:

        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TInputWord>", 
                ),
                (
                    f"LUTmem",
                    f"TOutput"
                ),
                (
                    f"o_data",
                    f"hls::stream<TOutputWord>", 
                ),
            )
        )

        return run.generate_call(
            [hls_tag],
            self.__get_stream_name(self.onnx_node.input[0]),
            f"{self.onnx_node.name}_lut",
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def __get_step_call(self) -> str:

        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TInputWord>", 
                ),
                (
                    f"LUTmem",
                    f"TOutput"
                ),
                (
                    f"o_data",
                    f"hls::stream<TOutputWord>", 
                ),
            )
        )

        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            f"{self.onnx_node.name}_lut",
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int) -> None:
        """Lower the node to HLS code."""
        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(
                f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model."
            )

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        ]

        output_names = [
            f"{self.__get_stream_name(self.onnx_node.output[0])}_{i}_"
            for i in range(self.get_nodeattr("out_stream_array"))
        ]

        tensors_fifo_metadata = {}
        for output in output_names:
            tensors_fifo_metadata[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}",
                n_array=self.get_nodeattr("out_stream_array"),
            )

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="backend.custom_op",
            original_op_type="StreamingSigmoid",
            hls_tag=hls_tag,
            hls_object_name=self.onnx_node.name,
            hls_variable_declarations=self.__get_variable_declaration(
                model, input_quant, output_quant
            ),
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag

    def get_latency(self, model: ModelWrapper) -> int:
        """ Estimate the latency of the StreamingAdd operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")

        unroll_factor = self.get_nodeattr("channel_unroll") * self.get_nodeattr("width_unroll")
        return np.prod(input_shape) // unroll_factor

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the StreamingAdd operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """ Estimate the DSP usage of the StreamingAdd operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """
        return 0

    def has_linebuffer(self) -> bool:
        """ Check if the StreamingAdd operation requires a line buffer.
        Returns:
            bool: True if Line Buffering is required, False otherwise.
        """
        return False

    def can_inherit_interface(self):
        return True

    def inherit_interface(self, model: ModelWrapper, upstream: NodeInterface) -> None:
        """ Inherit the interface from the upstream node."""
        self.set_nodeattr("in_stream_array", upstream.out_stream_array)
        self.set_nodeattr("out_stream_array", upstream.out_stream_array)
        self.set_nodeattr("in_word_array", upstream.out_word_array)
        self.set_nodeattr("out_word_array", upstream.out_word_array)

        self.set_nodeattr("channel_unroll", upstream.out_word_array)
        self.set_nodeattr("width_unroll", upstream.out_stream_array)
