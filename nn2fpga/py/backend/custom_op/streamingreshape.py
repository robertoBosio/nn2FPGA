from attr import dataclass
import numpy as np
import onnxruntime as rt
from onnxscript.rewriter import pattern
from onnx import TensorProto, helper
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.core.tensor_fifo import TensorFifo
from backend.custom_op.hlskernel import HLSKernel
from backend.custom_op.op_base import DSECapable, NN2FPGAOp, NodeInterface
from backend.custom_op.register_rewrite_rule import register_rules
from backend.util.codegen_utils import (
    cpp_function,
    cpp_object,
    get_struct_type,
    get_hls_quant_type,
)
from onnx_ir import convenience as ir_convenience

class StreamingReshape(NN2FPGAOp, DSECapable):
    """ Node implementing the Reshape operation. """
    
    @dataclass(frozen=True)
    class DSEPoint:
        channel_unroll: int
        width_unroll: int

        def to_dict(self) -> dict:
            return {
                "channel_unroll": self.channel_unroll,
                "width_unroll": self.width_unroll,
            }

        @staticmethod
        def from_dict(d: dict) -> "StreamingReshape.DSEPoint":
            return StreamingReshape.DSEPoint(
                channel_unroll=d["channel_unroll"],
                width_unroll=d["width_unroll"],
            )

    @staticmethod
    def pattern(op, x, shape):
        return op.Reshape(x, shape, _allow_other_attributes=True)
    @staticmethod
    def rewrite(op, x, shape):
        
        t = ir_convenience.get_const_tensor(shape)  # handles initializer OR Constant node
        if t is None:
            raise ValueError("Shape is not a compile-time constant")
        arr = t.numpy().reshape(-1)
        return op.StreamingReshape(
            x,
            shape=arr.tolist(),
            _domain="backend.custom_op",
        )

    @register_rules
    def register_rules():
        return [
            pattern.RewriteRule(
                StreamingReshape.pattern, StreamingReshape.rewrite
            )
        ]

    def get_nodeattr_types(self):
        return {
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),
            "shape": ("ints", False, []),

            "channel_unroll": ("i", False, 1),
            "width_unroll": ("i", False, 1),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        shape_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"{node.name}_shape_const"],
            name=f"{node.name}_shape_const",
            value=helper.make_tensor(
                name=f"{node.name}_shape_tensor",
                data_type=TensorProto.INT64,
                dims=(len(self.get_nodeattr("shape")),),
                vals=self.get_nodeattr("shape"),
            ),
        )
        return helper.make_node(
            "Reshape",
            inputs=[node.input[0], f"{node.name}_shape_const"],
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        # create a standard reshape node to compute the result
        node = self.onnx_node

        shape_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"{node.name}_shape_const"],
            name=f"{node.name}_shape_const",
            value=helper.make_tensor(
                name=f"{node.name}_shape_tensor",
                data_type=TensorProto.INT64,
                dims=(len(self.get_nodeattr("shape")),),
                vals=self.get_nodeattr("shape"),
            ),
        )
        node_reshape   = helper.make_node(
            "Reshape",
            inputs=[node.input[0], f"{node.name}_shape_const"],
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
        )

        # Make single node graph for execution
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)

        graph_reshape = helper.make_graph(
            nodes=[shape_node, node_reshape],
            name="single-reshape-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_reshape = qonnx_make_model(graph_reshape, **onnx_kwargs)
        idict = {node.input[0]: inp_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_reshape.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass

    def __is_power_of_two(self, value) -> bool:
        """Check if a value is a power of two."""
        return value > 0 and float(np.log2(value)).is_integer()

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_variable_declaration(self, model) -> str:
        """ Get the internal cpp variables of the StreamingReshape node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_quantizer(self, input_quant, output_quant) -> str:
        """ Returns the quantizer type for the Reshape operation. """

        if (
            self.__is_power_of_two(input_quant.scale)
            and self.__is_power_of_two(output_quant.scale)
        ):
            shift = -1 * int(np.log2(output_quant.scale) - np.log2(input_quant.scale))
            return f"DequantQuantPo2<{shift}, {get_hls_quant_type(input_quant)}, {get_hls_quant_type(output_quant)}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingReshape."
            )

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

        StreamingReshape = cpp_object(
            "StreamingReshape",
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
                    f"{self.__get_quantizer(input_quant, output_quant)}",
                    f"Quantizer",
                ),
                (f"{input_shape[2]}", "IN_HEIGHT"),
                (f"{input_shape[3]}", "IN_WIDTH"),
                (f"{input_shape[1]}", "IN_CH"),
                (f"{self.get_nodeattr('width_unroll')}", "W_PAR"),
                (f"{self.get_nodeattr('channel_unroll')}", "CH_PAR"),
            ]
        )

        return StreamingReshape.generate_declaration()

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
                    f"o_data",
                    f"hls::stream<TOutputWord>", 
                ),
            )
        )

        return run.generate_call(
            [hls_tag],
            self.__get_stream_name(self.onnx_node.input[0]),
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
                    f"o_data",
                    f"hls::stream<TOutputWord>", 
                ),
            )
        )

        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def __current_dse_point(self) -> "StreamingReshape.DSEPoint":
        """ Retrieve the current DSE point from the ONNX attributes. """
        return StreamingReshape.DSEPoint(
            channel_unroll=self.get_nodeattr("channel_unroll"),
            width_unroll=self.get_nodeattr("width_unroll"),
        )

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int) -> None:
        """Lower the node to HLS code."""

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
            original_op_type="StreamingReshape",
            hls_tag=hls_tag,
            hls_object_name=self.onnx_node.name,
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag

    def get_latency(self, model: ModelWrapper) -> int:
        """ Estimate the latency of the StreamingReshape operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")

        point = self.__current_dse_point()

        return np.prod(input_shape) // (point.channel_unroll * point.width_unroll)

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the StreamingReshape operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """ Estimate the DSP usage of the StreamingReshape operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """
        return 0
    
    def get_dse_points(self, model: ModelWrapper) -> list["StreamingReshape.DSEPoint"]:
        """ Generate the DSE points for the StreamingReshape operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            list[StreamingReshape.DSEPoint]: List of DSE points.
        """

        def divisors(n: list[int], clip: int) -> list[int]:
            return [
                i
                for i in range(1, min(n) + 1)
                if (all(x % i == 0 for x in n) and i <= clip)
            ]

        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(
                f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model."
            )
        input_bits = input_quant.bitwidth

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )
        output_bits = output_quant.bitwidth

        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(
                f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model."
            )
        input_shape = input_shape + [1] * (4 - len(input_shape))  # Ensure 4D shape.
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(
                f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model."
            )
        output_shape = output_shape + [1] * (4 - len(output_shape))  # Ensure 4D shape.

        # As of now, kernel height and width are completely unrolled.
        DSE_points = []
        for channel_unroll in divisors(
            [input_shape[1], output_shape[1]],
            min(input_shape[1], output_shape[1]),
        ):
            for width_unroll in divisors(
                [input_shape[3], output_shape[3]],
                min(input_shape[3], output_shape[3]),
            ):
                # Check dimension of input streams
                if (input_bits * channel_unroll) > 4096:
                    continue
                # Check dimension of output streams
                if (output_bits * channel_unroll) > 4096:
                    continue

                DSE_points.append(
                    self.DSEPoint(
                        channel_unroll, width_unroll
                    )
                )

        return DSE_points

    def has_linebuffer(self) -> bool:
        """ Check if the StreamingReLU operation requires a line buffer.
        Returns:
            bool: True if Line Buffering is required, False otherwise.
        """
        return False

    def apply_point(self, model: ModelWrapper, point: "StreamingReshape.DSEPoint"):
        """ Set the parallelization attributes for the StreamingReshape operation.
        Args:
            point (StreamingReshape.DSEPoint): The DSE point containing the unrolling parameters.
        """
        self.set_nodeattr("channel_unroll", point.channel_unroll)
        self.set_nodeattr("width_unroll", point.width_unroll)

        self.set_nodeattr("in_stream_array", point.width_unroll)
        self.set_nodeattr("out_stream_array", point.width_unroll)
        self.set_nodeattr("in_word_array", point.channel_unroll)
        self.set_nodeattr("out_word_array", point.channel_unroll)
