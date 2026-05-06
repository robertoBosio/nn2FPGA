import numpy as np
import onnxruntime as rt
from typing import Optional
from attr import dataclass
from onnx import TensorProto, helper
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_quant import require_tensor_quant
from nn2fpga.compiler.core.tensor_layout import require_tensor_layout
from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import DSECapable, NN2FPGAOp
from nn2fpga.compiler.utils.codegen_utils import (
    cpp_function,
    cpp_variable,
    cpp_object,
    get_struct_type,
    get_hls_quant_type,
)
from nn2fpga.compiler.core.tensor_quant import TensorQuant
from nn2fpga.compiler.custom_op.register_rewrite_rule import register_rules, PRule
from onnx_ir import convenience as ir_convenience
from onnxscript.rewriter import pattern
import logging

logger = logging.getLogger(__name__)

EXP_PRECISION = 12  # Number of bits for LUT output (Q0.12 format for max precision)
DIV_PRECISION = 32 # Number of bits for division result (Q0.36 format for max precision)

def _as_list(x):
    if x is None:
        return None
    arr = np.asarray(x)
    return [int(v) for v in arr.flatten().tolist()]

def _get_const_i64_list(x) -> Optional[list[int]]:
    t = ir_convenience.get_const_tensor(x)
    if t is None:
        return None
    return _as_list(t)

class StreamingYoloHeadSoftmax(NN2FPGAOp, DSECapable):

    @staticmethod
    def softmax_pattern(
        op,
        x,
        shape_in,
        rq_scale,
        rq_zeropt,
        rq_bitwidth,
        rq_signed,
        rq_narrow,
        rq_rounding_mode,
        tq_scale,
        tq_zeropt,
        tq_bitwidth,
        tq_signed,
        tq_narrow,
        tq_rounding_mode,
        axis,
    ):
        x_r = op.Reshape(x, shape_in)
        x_rq = op.Quant(
            x_r,
            rq_scale,
            rq_zeropt,
            rq_bitwidth,
            signed=rq_signed,
            narrow=rq_narrow,
            rounding_mode=rq_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )
        x_t = op.Transpose(x_rq, perm=[0, 2, 1, 3])
        q_tq = op.Quant(
            x_t,
            tq_scale,
            tq_zeropt,
            tq_bitwidth,
            signed=tq_signed,
            narrow=tq_narrow,
            rounding_mode=tq_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )
        return op.Softmax(q_tq, axis=axis)

    @staticmethod
    def rewrite(
        op,
        x,
        shape_in,
        rq_scale,
        rq_zeropt,
        rq_bitwidth,
        rq_signed,
        rq_narrow,
        rq_rounding_mode,
        tq_scale,
        tq_zeropt,
        tq_bitwidth,
        tq_signed,
        tq_narrow,
        tq_rounding_mode,
        axis,
        ):
        return op.StreamingYoloHeadSoftmax(
            x,
            axis=axis,
            shape_in=_get_const_i64_list(shape_in),
            _domain="nn2fpga.compiler.custom_op",
        )

    @register_rules
    def register_rules():
        return [
            PRule(
                pattern.RewriteRule(
                    StreamingYoloHeadSoftmax.softmax_pattern,
                    StreamingYoloHeadSoftmax.rewrite,
                ),
                priority=1,
            )
        ]

    @dataclass(frozen=True)
    class DSEPoint:
        lanes_unroll: int
        reduction_unroll: int

        # optional helpers to interop with old code / ONNX storage
        def to_dict(self) -> dict:
            return {
                "lanes_unroll": self.lanes_unroll,
                "reduction_unroll": self.reduction_unroll,
            }

        @staticmethod
        def from_dict(d: dict) -> "StreamingYoloHeadSoftmax.DSEPoint":
            return StreamingYoloHeadSoftmax.DSEPoint(
                lanes_unroll=d["lanes_unroll"],
                reduction_unroll=d["reduction_unroll"],
            )

    def get_nodeattr_types(self):
        return {
            "axis": ("i", False, -1),  # Axis for Softmax operation
            "shape_in": ("ints", False, []),  # Shape for the initial Reshape

            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),

            "lanes_unroll": ("i", False, 1),
            "reduction_unroll": ("i", False, 1),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        return helper.make_node(
            "Softmax",
            inputs=node.input,
            outputs=node.output,
            axis=self.get_nodeattr("axis"),
            name=f"{node.name}_shape_compatible",
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        # create a standard relu node to compute the result
        node = self.onnx_node
        node_softmax = helper.make_node(
            "Softmax",
            inputs=node.input,
            outputs=node.output,
            axis=self.get_nodeattr("axis"),
            name=f"{node.name}_shape_compatible",
        )

        # Make single node graph for execution
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)

        graph_softmax = helper.make_graph(
            nodes=[node_softmax],
            name="single-softmax-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_softmax = qonnx_make_model(graph_softmax, **onnx_kwargs)
        idict = {node.input[0]: inp_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_softmax.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_accumulator(self, n_sums: int, input_quant: TensorQuant) -> str:
        """
        Get the accumulator type for the given input quantization.
        For softmax, we need to accumulate exponentials, so we need a wider type.
        We can use a simple heuristic of doubling the bitwidth for the accumulator.
        """
        accumulator_bitwidth = EXP_PRECISION + int(np.floor(np.log2(n_sums) + 1))

        signed = False
        acc_quant = TensorQuant(
            bitwidth=accumulator_bitwidth,
            signed=signed,
            scale=input_quant.scale,
            zeropt=input_quant.zeropt,
        )
        return f"{get_hls_quant_type(acc_quant)}"

    def __get_lut_type(self) -> str:
        """
        Get the type for the LUT entries in the softmax computation.
        The LUT stores quantized exponentials.
        """
        lut_quant = TensorQuant(
            bitwidth=EXP_PRECISION,  # Use the output bitwidth for max precision
            signed=False,
            scale=0.0, # Not the actual scale, but it is usless for type generation.
            zeropt=0,
        )
        return f"{get_hls_quant_type(lut_quant)}"

    def __get_division_type(self) -> str:
        """
        Get the type for the division result in the softmax computation.
        """

        div_quant = TensorQuant(
            bitwidth=DIV_PRECISION,
            signed=False,
            scale=0.0, # Not the actual scale, but it is usless for type generation.
            zeropt=0,
        )
        return f"{get_hls_quant_type(div_quant)}"

    def __is_power_of_two(self, value) -> bool:
        """Check if a value is a power of two."""
        return value > 0 and float(np.log2(value)).is_integer()

    def __get_quantizer(self, output_quant) -> str:
        """Returns the quantizer type for the StreamingSoftmax operation."""

        # Check if the scale is a power of two
        if (
            self.__is_power_of_two(output_quant.scale)
        ):
            shift = ((DIV_PRECISION - EXP_PRECISION) + int(np.log2(output_quant.scale)))
            return f"DequantQuantPo2<{shift}, {self.__get_division_type()}, {get_hls_quant_type(output_quant)}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingSoftmax."
            )

    def __get_variable_declaration(self, model, input_quant) -> str:
        """ Get the internal cpp variables of the StreamingSoftmax node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        nbits = input_quant.bitwidth
        x_scale = input_quant.scale
        out_total_bits = EXP_PRECISION  # Number of fractional bits for LUT output (Q0.12 format)
        lut_entries = 1 << nbits

        # ---- build d = 0..2^nbits-1 ----
        d = np.arange(lut_entries, dtype=np.float64)

        # ---- compute real exp(-d * x_scale) ----
        y_real = np.exp(-d * x_scale)

        # ---- quantize to Q0.F with RNE ties-to-even ----
        # y_q = round(y_real * 2^F) with banker's rounding
        scale = float(2 ** out_total_bits)
        y_q = np.rint(y_real * scale).astype(np.int64)  # np.rint = ties-to-even
        qmin, qmax = 0, (1 << out_total_bits) - 1
        y_q = np.clip(y_q, qmin, qmax)

        # ---- emit C++ variable ----
        # Choose a reasonable C++ primitive based on out_total_bits + signedness
        # (you can override outside if you prefer a fixed type)
        primitive = f"ap_uint<{out_total_bits}>"

        lut_values = [int(v) for v in y_q.tolist()]

        lut_variable = cpp_variable(
            name=f"{self.onnx_node.name}_lut",
            primitive=primitive,
            value=lut_values,
        )
        return lut_variable.generate_initialization().code

    def __get_object_declaration(self, model) -> cpp_object:

        input_quant = require_tensor_quant(model, self.onnx_node.input[0])
        output_quant = require_tensor_quant(model, self.onnx_node.output[0])

        # We actually not consider the shape in input but the shape
        # after the initial reshape and the transpose in the pattern, which is stored
        # in the node attribute.
        input_shape = self.get_nodeattr("shape_in")
        input_shape = [1] * (4 - len(input_shape)) + input_shape
        input_shape = np.array(input_shape)[[0, 2, 1, 3]]
        input_shape = input_shape.tolist()
        axis = self.get_nodeattr("axis")
        if axis < 0:
            axis += len(input_shape)
        dim_reduction = input_shape[axis]

        # The input shape is in NCHW format, but we need the NHWC.

        dim_lanes = np.prod(input_shape) // dim_reduction

        lut_size = 1 << input_quant.bitwidth
        StreamingSoftmax = cpp_object(
            "StreamingSoftmax",
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
                (
                    f"{self.__get_lut_type()}",
                    f"TLUT",
                ),
                (f"{self.__get_accumulator(dim_reduction, input_quant)}", "TAcc"),
                (f"{self.__get_division_type()}", "TDiv"),
                (f"{self.__get_quantizer(output_quant)}", "Quantizer"),
                (f"{lut_size}", "LUT_SIZE"),
                (f"{dim_lanes}", "DIM_LANES"),
                (f"{dim_reduction}", "DIM_REDUCTION"),
                (f"{self.get_nodeattr('lanes_unroll')}", "LANE_PAR"),
                (f"{self.get_nodeattr('reduction_unroll')}", "REDUCE_PAR"),
            ]
        )

        return StreamingSoftmax.generate_declaration()

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

    def accepted_input_layout(self) -> tuple | None:
        """ StreamingYoloHeadSoftmax only accepts NHWC layout. """
        return (0, 2, 3, 1)  # NHWC

    def produced_output_layout(self, input_layout: tuple | None) -> tuple:
        """ StreamingYoloHeadSoftmax produces only NWHC layout. """
        return (0, 3, 2, 1)  # NWHC

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int) -> None:
        """Lower the node to HLS code."""
        input_quant = require_tensor_quant(model, self.onnx_node.input[0])
        output_quant = require_tensor_quant(model, self.onnx_node.output[0])

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
            original_op_type="StreamingSoftmax",
            hls_tag=hls_tag,
            hls_object_name=self.onnx_node.name,
            hls_variable_declarations=self.__get_variable_declaration(
                model, input_quant
            ),
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag

    def get_latency(self, model: ModelWrapper) -> int:
        """ Estimate the latency of the StreamingSoftmax operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")

        unroll_factor = self.get_nodeattr("lanes_unroll") * self.get_nodeattr("reduction_unroll")
        return np.prod(input_shape) * 3 // unroll_factor

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the StreamingSoftmax operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """ Estimate the DSP usage of the StreamingSoftmax operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """
        point = self.__current_dse_point()
        return point.lanes_unroll * point.reduction_unroll

    def has_linebuffer(self) -> bool:
        """ Check if the StreamingSoftmax operation requires a line buffer.
        Returns:
            bool: True if Line Buffering is required, False otherwise.
        """
        return False

    def __current_dse_point(self) -> "StreamingYoloHeadSoftmax.DSEPoint":
        """ Returns the current DSE point of the StreamingSoftmax operation. """
        return StreamingYoloHeadSoftmax.DSEPoint(
            lanes_unroll=self.get_nodeattr("lanes_unroll"),
            reduction_unroll=self.get_nodeattr("reduction_unroll"),
        )

    def get_dse_points(self, model: ModelWrapper) -> list["StreamingYoloHeadSoftmax.DSEPoint"]:
        """Generate the list of valid DSE points for the StreamingYoloHeadSoftmax operation."""

        def divisors(n: list[int], clip: int) -> list[int]:
            return [
                i
                for i in range(1, min(n) + 1)
                if (all(x % i == 0 for x in n) and i <= clip)
            ]

        input_quant = require_tensor_quant(model, self.onnx_node.input[0])
        input_bits = input_quant.bitwidth

        output_quant = require_tensor_quant(model, self.onnx_node.output[0])
        output_bits = output_quant.bitwidth

        input_shape = self.get_nodeattr("shape_in")
        input_shape = [1] * (4 - len(input_shape)) + input_shape
        input_shape = np.array(input_shape)[[0, 2, 1, 3]]
        input_shape = input_shape.tolist()
        axis = self.get_nodeattr("axis")
        if axis < 0:
            axis += len(input_shape)
        dim_reduction = input_shape[axis]

        dim_lanes = np.prod(input_shape) // dim_reduction

        # As of now, kernel height and width are completely unrolled.
        DSE_points = []
        for lanes_unroll in divisors([dim_lanes], dim_lanes):
            for reduction_unroll in divisors([dim_reduction], dim_reduction):
                # Check dimension of input streams
                if (input_bits * reduction_unroll) > 4096:
                    continue
                # Check dimension of output streams
                if (output_bits * reduction_unroll) > 4096:
                    continue

                DSE_points.append(
                    self.DSEPoint(
                        lanes_unroll=lanes_unroll,
                        reduction_unroll=reduction_unroll
                    )
                )

        DSE_points = [self.DSEPoint(lanes_unroll=1, reduction_unroll=1)]
        return DSE_points

    def apply_point(self, model: ModelWrapper, point: "StreamingYoloHeadSoftmax.DSEPoint"):
        """ Set the parallelization attributes for the StreamingYoloHeadSoftmax operation.
        Args:
            point (StreamingYoloHeadSoftmax.DSEPoint): The DSE point containing the unrolling parameters.
        """
        self.set_nodeattr("lanes_unroll", point.lanes_unroll)
        self.set_nodeattr("reduction_unroll", point.reduction_unroll)

        self.set_nodeattr("in_stream_array", point.lanes_unroll)
        self.set_nodeattr("out_stream_array", point.lanes_unroll)
        self.set_nodeattr("in_word_array", point.reduction_unroll)
        self.set_nodeattr("out_word_array", point.reduction_unroll)
