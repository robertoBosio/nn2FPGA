import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_quant import get_custom_tensor_datatype
from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp, NodeInterface
from nn2fpga.compiler.utils.codegen_utils import (
    cpp_function,
    cpp_variable,
    cpp_object,
    get_struct_type,
    get_stream_type,
    get_hls_quant_type,
)
from nn2fpga.compiler.core.tensor_quant import TensorQuant
from nn2fpga.compiler.utils.par_utils import get_par_attributes
from nn2fpga.compiler.custom_op.register_rewrite_rule import register_rules, PRule
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

class StreamingSwish(NN2FPGAOp):

    @staticmethod
    def swish_pattern(
        op,
        x,
        B,
        alpha,
        q_scale,
        q_zeropt,
        q_bitwidth,
        q_signed,
        q_narrow,
        q_rounding_mode,
    ):
        x_hardsigmoid = op.HardSigmoid(x, alpha=alpha)
        x_sigmoid = op.Mul(x_hardsigmoid, B)
        x_sigmoid_quant = op.Quant(
            x_sigmoid,
            q_scale,
            q_zeropt,
            q_bitwidth,
            signed=q_signed,
            narrow=q_narrow,
            rounding_mode=q_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )
        return op.Mul(x, x_sigmoid_quant)

    @staticmethod
    def _condition(context, x, B, alpha, **_):
        # Only rewrite if B is actually constant
        return ir_convenience.get_const_tensor(B) is not None

    @staticmethod
    def rewrite(
        op,
        x,
        B,
        alpha,
        q_scale,
        q_zeropt,
        q_bitwidth,
        q_signed,
        q_narrow,
        q_rounding_mode,
        **_,
    ):
        b_value = _get_B_as_python_number(B)
        return op.StreamingSwish(
            x,
            q_scale,
            q_zeropt,
            q_bitwidth,
            alpha=alpha,
            B=b_value,
            signed=q_signed.value,
            narrow=q_narrow.value,
            rounding_mode=q_rounding_mode.value,
            _domain="nn2fpga.compiler.custom_op",
        )

    @register_rules
    def register_rules():

        # We give this a higher priority than the default to ensure it runs before the pattern is decomposed into more basic ops, which would make it impossible to match.
        return [
            PRule(
                pattern.RewriteRule(
                    StreamingSwish.swish_pattern,
                    StreamingSwish.rewrite,
                    StreamingSwish._condition,
                ),
                priority=1,
            )
        ]

    def get_nodeattr_types(self):
        return {
            "alpha": ("f", False, 0.01),  # Slope for negative inputs
            "B": ("f", False, 1.0),      # Scaling factor after HardSigmoid
            "signed": ("i", True, 0),  # 0: unsigned, 1: signed
            "narrow": ("i", True, 0),  # 0: full range, 1: narrow range
            "rounding_mode": ("s", True, "ROUND"),

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
            "Swish",
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
        node_swish = helper.make_node(
            "Swish",
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

        graph_swish = helper.make_graph(
            nodes=[node_swish],
            name="single-swish-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_swish = qonnx_make_model(graph_swish, **onnx_kwargs)
        idict = {node.input[0]: inp_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_swish.SerializeToString())
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
        """ Get the internal cpp variables of the StreamingSwish node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """

        nbits = input_quant.bitwidth
        sigmoid_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[1]),
            zeropt=model.get_initializer(self.onnx_node.input[2]),
            bitwidth=model.get_initializer(self.onnx_node.input[3]),
            signed=bool(self.get_nodeattr("signed")),
            narrow=bool(self.get_nodeattr("narrow")),
            rounding_mode=self.get_nodeattr("rounding_mode"),
        )
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
        X_sigmoid_scale = helper.make_tensor(
            "X_sigmoid_scale", TensorProto.FLOAT, [], [sigmoid_quant.scale]
        )
        X_sigmoid_zp = helper.make_tensor(
            "X_sigmoid_zp", onnx_int_type, [], [sigmoid_quant.zeropt]
        )

        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        B = helper.make_tensor("B", TensorProto.FLOAT, [], [self.get_nodeattr("B")])

        # ---- Nodes to compute the LUT values ----
        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        HardSigmoid = helper.make_node(
            "HardSigmoid",
            alpha=self.get_nodeattr("alpha"),
            inputs=["X_dq"],
            outputs=["A"],
        )

        Mul0 = helper.make_node(
            "Mul",
            inputs=["A", "B"],
            outputs=["X_sigmoid"],
        )

        qlinear_sigmoid = helper.make_node(
            "QuantizeLinear",
            inputs=["X_sigmoid", "X_sigmoid_scale", "X_sigmoid_zp"],
            outputs=["X_sigmoid_q"],
        )

        dqlinear_sigmoid = helper.make_node(
            "DequantizeLinear",
            inputs=["X_sigmoid_q", "X_sigmoid_scale", "X_sigmoid_zp"],
            outputs=["X_sigmoid_dq"],
        )

        Mul1 = helper.make_node(
            "Mul",
            inputs=["X_sigmoid_dq", "X_dq"],
            outputs=["Y_dq"],
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [
                dqlinear,
                HardSigmoid,
                Mul0,
                qlinear_sigmoid,
                dqlinear_sigmoid,
                Mul1,
                qlinear,
            ],
            "qsigmoid_test",
            [X],
            [Y],
            initializer=[
                X_scale,
                X_zp,
                Y_scale,
                Y_zp,
                B,
                X_sigmoid_scale,
                X_sigmoid_zp,
            ],
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
            domain="nn2fpga.compiler.custom_op",
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
