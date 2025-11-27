import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper
from qonnx.util.basic import qonnx_make_model
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.core.tensor_fifo import TensorFifo
from backend.custom_op.hlskernel import HLSKernel
from backend.custom_op.op_base import NN2FPGAOp, DSECapable, HasParameters, ParamDesc
from backend.util.codegen_utils import (
    cpp_function,
    cpp_variable,
    cpp_object,
    get_struct_type,
    get_hls_quant_type,
)
from backend.core.tensor_quant import TensorQuant
from qonnx.core.modelwrapper import ModelWrapper
from backend.custom_op.register_rewrite_rule import register_rules
from onnxscript.rewriter import pattern
from backend.util.board_util import bram_usage_evaluator, packing_feature
from dataclasses import dataclass
from typing import Iterable

class StreamingConv(NN2FPGAOp, DSECapable, HasParameters):

    @dataclass(frozen=True)
    class DSEPoint:
        out_channel_unroll: int
        in_channel_unroll: int
        width_unroll: int
        filter_width_unroll: int
        filter_height_unroll: int

        # optional helpers to interop with old code / ONNX storage
        def to_dict(self) -> dict:
            return {
                "out_channel_unroll": self.out_channel_unroll,
                "in_channel_unroll": self.in_channel_unroll,
                "width_unroll": self.width_unroll,
                "filter_width_unroll": self.filter_width_unroll,
                "filter_height_unroll": self.filter_height_unroll,
            }

        @staticmethod
        def from_dict(d: dict) -> "StreamingConv.DSEPoint":
            return StreamingConv.DSEPoint(
                out_channel_unroll=d["out_channel_unroll"],
                in_channel_unroll=d["in_channel_unroll"],
                width_unroll=d["width_unroll"],
                filter_width_unroll=d["filter_width_unroll"],
                filter_height_unroll=d["filter_height_unroll"],
            )

    @staticmethod
    def pattern(
        op,
        x,
        dilations,
        group,
        kernel_shape,
        pads,
        strides,
        w_value,
        w_scale,
        w_zeropt,
        w_bitwidth,
        b_value,
        b_scale,
        b_zeropt,
        b_bitwidth,
        w_signed,
        w_narrow,
        w_rounding_mode,
        b_signed,
        b_narrow,
        b_rounding_mode,
    ):
        w_quant = op.Quant(
            w_value,
            w_scale,
            w_zeropt,
            w_bitwidth,
            signed=w_signed,
            narrow=w_narrow,
            rounding_mode=w_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )
        b_quant = op.Quant(
            b_value,
            b_scale,
            b_zeropt,
            b_bitwidth,
            signed=b_signed,
            narrow=b_narrow,
            rounding_mode=b_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )
        y = op.Conv(
            x,
            w_quant,
            b_quant,
            dilations=dilations,
            group=1,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            _allow_other_attributes=True,
        )
        return y

    @staticmethod
    def rewrite(
        op,
        x,
        dilations,
        group,
        kernel_shape,
        pads,
        strides,
        w_value,
        w_scale,
        w_zeropt,
        w_bitwidth,
        b_value,
        b_scale,
        b_zeropt,
        b_bitwidth,
        w_signed,
        w_narrow,
        w_rounding_mode,
        b_signed,
        b_narrow,
        b_rounding_mode,
    ):

        return op.StreamingConv(
            x,
            w_value,
            w_scale,
            w_zeropt,
            w_bitwidth,
            b_value,
            b_scale,
            b_zeropt,
            b_bitwidth,
            w_signed=w_signed.value,
            w_narrow=w_narrow.value,
            w_rounding_mode=w_rounding_mode.value,
            b_signed=b_signed.value,
            b_narrow=b_narrow.value,
            b_rounding_mode=b_rounding_mode.value,
            dilations=dilations,
            group=1,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            param_storage="INTERNAL",
            _domain="backend.custom_op",
        )

    @register_rules
    def _rewriter_rules():
        return [
            pattern.RewriteRule(
                StreamingConv.pattern,
                StreamingConv.rewrite,
            )
        ]

    def get_nodeattr_types(self):
        return {
            # Standard ONNX attributes for Conv
            "dilations": ("ints", True, [1, 1]),
            "group": ("i", True, 1),
            "kernel_shape": ("ints", True, [1, 1]),
            "pads": ("ints", True, [0, 0]),
            "strides": ("ints", True, [1, 1]),

            # Custom attributes for quantization of weights
            "w_signed": ("i", True, 0),  # 0: unsigned, 1: signed
            "w_narrow": ("i", True, 0),  # 0: full range, 1: narrow range
            "w_rounding_mode": ("s", True, "ROUND"),

            # Custom attributes for quantization of bias
            "b_signed": ("i", False, 0),  # 0: unsigned, 1: signed
            "b_narrow": ("i", False, 0),  # 0: full range, 1: narrow range
            "b_rounding_mode": ("s", False, "ROUND"),

            # Custom attributes for unroll factors of StreamingConv
            "in_channel_unroll": ("i", False, 1),
            "out_channel_unroll": ("i", False, 1),
            "width_unroll": ("i", False, 1),
            "filter_width_unroll": ("i", False, 1),
            "filter_height_unroll": ("i", False, 1),

            # Custom attributes for input/output streams
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),

            # Custom attributes for zero point folding into bias
            "asym_folding": ("i", False, 0),  # 0: no folding, 1: fold zeropt into bias

            # Custom attribute for activation function
            "activation": ("s", False, "NoOp"),  # NoOp, ReLU

            # Custom attribute for internal/external parameters (weights, biases) storage
            "param_storage": ("s", True, "INTERNAL"),  # INTERNAL, EXTERNAL
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        input_list = []
        if len(node.input) == 5:
            # Conv without bias
            input_list = [node.input[0], node.input[1]] + node.input[2:5]
        elif len(node.input) == 9:
            # Conv with bias
            input_list = [node.input[0], node.input[1], node.input[5]] + node.input[2:5] + node.input[6:9]
        else:
            raise ValueError(
                f"Unexpected number of inputs for StreamingConv node {node.name}: {len(node.input)}"
            )

        return helper.make_node(
            "Conv",
            inputs=input_list,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
            dilations=self.get_nodeattr("dilations"),
            group=self.get_nodeattr("group"),
            kernel_shape=self.get_nodeattr("kernel_shape"),
            pads=self.get_nodeattr("pads"),
            strides=self.get_nodeattr("strides"),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        # create a standard conv node to compute the result
        node = self.onnx_node
        node_conv = helper.make_node(
            "Conv",
            inputs=(
                [node.input[0], node.input[1]]
                if len(node.input) == 5
                else [node.input[0], node.input[1], node.input[5]]
            ),
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
            dilations=self.get_nodeattr("dilations"),
            group=self.get_nodeattr("group"),
            kernel_shape=self.get_nodeattr("kernel_shape"),
            pads=self.get_nodeattr("pads"),
            strides=self.get_nodeattr("strides"),
        )

        # Make single node graph for execution
        inp_values = context[node.input[0]]
        weight_values = context[node.input[1]]
        if len(node.input) > 5:
            bias_values = context[node.input[5]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        weight = helper.make_tensor_value_info(
            node.input[1], TensorProto.FLOAT, weight_values.shape
        )
        if len(node.input) > 5:
            bias = helper.make_tensor_value_info(
                node.input[6], TensorProto.FLOAT, bias_values.shape
            )

        graph_conv = helper.make_graph(
            nodes=[node_conv],
            name="single-conv-exec",
            inputs=[inp, weight, bias] if len(node.input) > 5 else [inp, weight],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_conv = qonnx_make_model(graph_conv, **onnx_kwargs)
        if len(node.input) > 5:
            idict = {node.input[0]: inp_values,
                     node.input[1]: weight_values,
                     node.input[6]: bias_values}
        else:
            idict = {node.input[0]: inp_values, 
                    node.input[1]: weight_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_conv.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_accumulator(self, input_quant, weights_quant, bias_quant, weights_shape) -> str:
        """ Returns the accumulator type for the StreamingConv operation. """

        add_ops = np.prod(weights_shape[1:])
        acc_bitwidth = input_quant.bitwidth + weights_quant.bitwidth + int(
            np.ceil(np.log2(add_ops))
        )
        acc_bitwidth = max(acc_bitwidth, bias_quant.bitwidth) + 1
        signed = input_quant.signed or weights_quant.signed or bias_quant.signed
        acc_quant = TensorQuant(
            bitwidth=acc_bitwidth,
            signed=signed,
            scale=input_quant.scale,
            zeropt=input_quant.zeropt,
        )

        return f"{get_hls_quant_type(acc_quant)}"

    def __get_activation(self, input_quant, weights_quant, bias_quant, weights_shape) -> str:
        """ Returns the activation functor for the StreamingConv operation. """

        activation = self.get_nodeattr("activation")
        if activation == "NoOp":
            return f"DequantQuantEqual<{self.__get_accumulator(input_quant, weights_quant, bias_quant, weights_shape)}>"
        elif activation == "ReLU":
            return f"ReLU<{self.__get_accumulator(input_quant, weights_quant, bias_quant, weights_shape)}>"
        else:
            raise ValueError(
                f"Unsupported activation function '{activation}' for StreamingConv."
            )

    def __is_power_of_two(self, value) -> bool:
        """Check if a value is a power of two."""
        return value > 0 and float(np.log2(value)).is_integer()

    def __get_quantizer(self, input_quant, weights_quant, bias_quant, output_quant, weights_shape) -> str:
        """ Returns the quantizer type for the StreamingConv operation. """

        # Check if the scale is a power of two
        if isinstance(weights_quant.scale, (list, np.ndarray)):
            if len(weights_quant.scale) != 1:
                raise ValueError(
                    "Per-channel quantization is currently not supported for StreamingConv.  "
                )
            weights_scale = weights_quant.scale[0]
        else:
            weights_scale = weights_quant.scale

        if (
            self.__is_power_of_two(input_quant.scale)
            and self.__is_power_of_two(output_quant.scale)
            and self.__is_power_of_two(weights_scale)
        ):
            shift = -1 * (
                int(np.log2(input_quant.scale))
                + int(np.log2(weights_scale))
                - int(np.log2(output_quant.scale))
            )
            return f"DequantQuantPo2<{shift}, {self.__get_accumulator(input_quant, weights_quant, bias_quant, weights_shape)}, {get_hls_quant_type(output_quant)}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingConv.  "
            )

    def __get_object_declaration(self, model) -> cpp_object:
        """ Generate the cpp_object for the StreamingConv operation. """

        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model.")

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        weights_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[2]),
            zeropt=model.get_initializer(self.onnx_node.input[3]),
            bitwidth=model.get_initializer(self.onnx_node.input[4]),
            signed=bool(self.get_nodeattr("w_signed")),
            narrow=bool(self.get_nodeattr("w_narrow")),
            rounding_mode=self.get_nodeattr("w_rounding_mode"),
        )

        bias_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[6]),
            zeropt=model.get_initializer(self.onnx_node.input[7]),
            bitwidth=model.get_initializer(self.onnx_node.input[8]),
            signed=bool(self.get_nodeattr("b_signed")),
            narrow=bool(self.get_nodeattr("b_narrow")),
            rounding_mode=self.get_nodeattr("b_rounding_mode"),
        )

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model.")
        output_shape = output_shape + [1] * (4 - len(output_shape))  # Ensure 4D shape.
        weights_shape = model.get_tensor_shape(self.onnx_node.input[1])
        if weights_shape is None:
            raise ValueError(f"Tensor shape for weights '{self.onnx_node.input[1]}' not found in model.")

        point = self.__current_dse_point()

        # Create the StreamingConv object.
        StreamingConv = cpp_object(
            "StreamingConv",
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_struct_type(input_quant, self.get_nodeattr('in_word_array'))}",
                    "TInputWord",
                ),
                (f"{get_hls_quant_type(input_quant)}", "TInput"),
                (
                    f"{get_struct_type(weights_quant, self.get_nodeattr('in_channel_unroll') * self.get_nodeattr('out_channel_unroll'))}",
                    "TWeightWord",
                ),
                (f"{get_hls_quant_type(weights_quant)}", "TWeight"),
                (
                    f"{get_struct_type(bias_quant, self.get_nodeattr('out_channel_unroll'))}",
                    "TBiasWord",
                ),
                (f"{get_hls_quant_type(bias_quant)}", "TBias"),
                (
                    f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}",
                    "TOutputWord",
                ),
                (f"{get_hls_quant_type(output_quant)}", "TOutput"),
                (self.__get_accumulator(input_quant, weights_quant, bias_quant, weights_shape), "TAcc"),
                (self.__get_activation(input_quant, weights_quant, bias_quant, weights_shape), "Activation"),
                (
                    self.__get_quantizer(input_quant, weights_quant, bias_quant, output_quant, weights_shape),
                    "Quantizer",
                ),
                (output_shape[1], "OUT_CH"),
                (input_shape[1], "IN_CH"),
                (output_shape[2], "OUT_HEIGHT"),
                (output_shape[3], "OUT_WIDTH"),
                (self.get_nodeattr("group"), "GROUP"),
                (self.get_nodeattr("kernel_shape")[0], "FH"),
                (self.get_nodeattr("kernel_shape")[1], "FW"),
                (self.get_nodeattr("strides")[0], "STRIDE_H"),
                (self.get_nodeattr("strides")[1], "STRIDE_W"),
                (point.in_channel_unroll, "IN_CH_PAR"),
                (point.out_channel_unroll, "OUT_CH_PAR"),
                (point.width_unroll, "W_PAR"),
            ],
        )

        return StreamingConv.generate_declaration()

    def __get_variable_declaration(self, model) -> str:
        """ Get the internal cpp variables of the StreamingConv node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """

        weights_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[2]),
            zeropt=model.get_initializer(self.onnx_node.input[3]),
            bitwidth=model.get_initializer(self.onnx_node.input[4]),
            signed=bool(self.get_nodeattr("w_signed")),
            narrow=bool(self.get_nodeattr("w_narrow")),
            rounding_mode=self.get_nodeattr("w_rounding_mode"),
        )

        bias_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[6]),
            zeropt=model.get_initializer(self.onnx_node.input[7]),
            bitwidth=model.get_initializer(self.onnx_node.input[8]),
            signed=bool(self.get_nodeattr("b_signed")),
            narrow=bool(self.get_nodeattr("b_narrow")),
            rounding_mode=self.get_nodeattr("b_rounding_mode"),
        )

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model.")
        output_shape = output_shape + [1] * (4 - len(output_shape))  # Ensure 4D shape.
        weights_shape = model.get_tensor_shape(self.onnx_node.input[1])
        if weights_shape is None:
            raise ValueError(f"Tensor shape for weights '{self.onnx_node.input[1]}' not found in model.")

        # Create weights and biases variable declarations if parameters are stored internally.
        param_storage = self.get_nodeattr("param_storage")
        if param_storage == "INTERNAL":
            values = np.random.randint(
                low=0,
                high=2 ** (weights_quant.bitwidth - 1),
                size=np.prod(weights_shape),
                dtype=np.int8,
            )
            weights_var = cpp_variable(
                name=f"{self.onnx_node.name}_weights",
                primitive=get_hls_quant_type(weights_quant),
                pragma=[
                    f"HLS ARRAY_RESHAPE variable={self.onnx_node.name}_weights dim=3 complete",
                    f"HLS ARRAY_RESHAPE variable={self.onnx_node.name}_weights dim=2 complete",
                ],
                value=values.reshape(
                    output_shape[1]
                    * input_shape[1]
                    // (
                        self.get_nodeattr("in_channel_unroll")
                        * self.get_nodeattr("out_channel_unroll")
                    ),
                    self.get_nodeattr("in_channel_unroll")
                    * self.get_nodeattr("out_channel_unroll"),
                    self.get_nodeattr("kernel_shape")[0]
                    * self.get_nodeattr("kernel_shape")[1],
                ),
            )

            values = np.random.randint(
                low=0,
                high=2 ** (bias_quant.bitwidth - 1),
                size=output_shape[1],
                dtype=np.int32,
            )

            bias_var = cpp_variable(
                name=f"{self.onnx_node.name}_biases",
                primitive=get_hls_quant_type(bias_quant),
                pragma=[
                    f"HLS ARRAY_RESHAPE variable={self.onnx_node.name}_biases dim=3 complete",
                    f"HLS ARRAY_RESHAPE variable={self.onnx_node.name}_biases dim=2 complete",
                ],
                value=values.reshape(
                    output_shape[1] // self.get_nodeattr("out_channel_unroll"),
                    self.get_nodeattr("out_channel_unroll"),
                    1,
                ),
            )

            return (
                weights_var.generate_declaration() 
                + ";\n"
                + weights_var.generate_pragma()
                + "\n"
                + bias_var.generate_declaration()
                + ";\n"
                + bias_var.generate_pragma()
            )
        else:
            return ""

    def __get_run_call(self, hls_tag: int) -> str:
        """ Generates the C++ code necessary to run the StreamingConv node. """

        name = f"{self.onnx_node.name}.run_allpartitioned" if hls_tag == 51 else f"{self.onnx_node.name}.run"
        # Generate the call to the StreamingConv run method.
        run = cpp_function(
            name=name,
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TInputStruct>",
                ),
                (
                    f"i_weights",
                    f"hls::stream<TWeightStruct>",
                ),
                (
                    f"i_biases",
                    f"hls::stream<TBiasStruct>",
                ),
                (
                    f"o_data",
                    f"hls::stream<TOutputStruct>",
                ),
            ),
        )

        return run.generate_call(
            [hls_tag],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.input[1]) if self.get_nodeattr("param_storage") == "EXTERNAL" else f"{self.onnx_node.name}_weights",
            self.__get_stream_name(self.onnx_node.input[5]) if self.get_nodeattr("param_storage") == "EXTERNAL" else f"{self.onnx_node.name}_biases",
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def __get_step_call(self) -> str:
        """ Generates the C++ code necessary to step the StreamingConv node. """

        # Generate the call to the StreamingConv step method.
        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TInputStruct>",
                ),
                (
                    f"i_weights",
                    f"hls::stream<TWeightStruct>",
                ),
                (
                    f"i_biases",
                    f"hls::stream<TBiasStruct>",
                ),
                (
                    f"o_data",
                    f"hls::stream<TOutputStruct>",
                ),
            ),
        )

        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.input[1]) if self.get_nodeattr("param_storage") == "EXTERNAL" else f"{self.onnx_node.name}_weights",
            self.__get_stream_name(self.onnx_node.input[5]) if self.get_nodeattr("param_storage") == "EXTERNAL" else f"{self.onnx_node.name}_biases",
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def __current_dse_point(self) -> "StreamingConv.DSEPoint":
        """ Returns the current DSE point of the StreamingConv operation. """
        return StreamingConv.DSEPoint(
            out_channel_unroll=self.get_nodeattr("out_channel_unroll"),
            in_channel_unroll=self.get_nodeattr("in_channel_unroll"),
            width_unroll=self.get_nodeattr("width_unroll"),
            filter_width_unroll=self.get_nodeattr("filter_width_unroll"),
            filter_height_unroll=self.get_nodeattr("filter_height_unroll"),
        )

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        """
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        point = self.__current_dse_point()
        FH = self.get_nodeattr("kernel_shape")[0]
        FW = self.get_nodeattr("kernel_shape")[1]
        STRIDE_W = self.get_nodeattr("strides")[1]
        FW_EXTENDED = FW + (point.width_unroll - 1) * STRIDE_W
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(FH * FW_EXTENDED)
        ]
        input_names.extend(
            [
                f"{self.__get_stream_name(self.onnx_node.input[1])}_{i}_"
                for i in range(np.prod(self.get_nodeattr("kernel_shape")))
            ]
        )
        if len(self.onnx_node.input) > 5:
            input_names.extend(
                [f"{self.__get_stream_name(self.onnx_node.input[5])}_0_"]
            )

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
            original_op_type="StreamingConv",
            hls_tag=hls_tag,
            hls_object_name=self.onnx_node.name,
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag

    def has_linebuffer(self) -> bool:
        """ Check if the StreamingConv operation requires a linebuffer.
        Returns:
            bool: True if a linebuffer is required, False otherwise.
        """

        kernel_shape = self.get_nodeattr("kernel_shape")
        stride = self.get_nodeattr("strides")
        pads = self.get_nodeattr("pads")

        point = self.__current_dse_point()

        # The only case in which a StreamingConv does not need a line buffer is when the kernel is 1x1,
        # there is no stride, and the output width parallelization is 1 with no padding.
        if (
            all(k == 1 for k in kernel_shape)
            and all(s == 1 for s in stride)
            and point.width_unroll == 1
            and all(p == 0 for p in pads)
        ):
            return False
        return True

    def get_latency(self, model: ModelWrapper) -> int:
        """ Estimate the latency of the StreamingConv operation given a set of unrolling parameters.
        Args:
            point (StreamingConv.DSEPoint): The DSE point containing the unrolling parameters.
        Returns:
            int: Estimated latency in clock cycles.
        """
        kernel_shape = self.get_nodeattr("kernel_shape")
        group = self.get_nodeattr("group")
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model.")
        output_shape = output_shape + [1] * (4 - len(output_shape))  # Ensure 4D shape.

        # Retrieve current unroll attributes if not provided.
        point = self.__current_dse_point()

        # Compute the number of MAC operations.
        MACs_conv = (
            input_shape[1] * np.prod(output_shape) * np.prod(kernel_shape)
        ) // group

        # Compute the latency based on unrolling factors.
        latency_conv = MACs_conv // (
            point.in_channel_unroll
            * point.out_channel_unroll
            * point.width_unroll
            * point.filter_width_unroll
            * point.filter_height_unroll
        )

        # Compute the latency of the line buffer in input.
        latency_lb = np.prod(input_shape) // (
            point.in_channel_unroll * point.width_unroll
        )
        return max(latency_conv, latency_lb)

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the StreamingConv operation given a set of parallelization parameters.
        Args:
            point (StreamingConv.DSEPoint): The DSE point containing the parallelization parameters.
        Returns:
            int: Estimated BRAM usage.
        """

        if self.get_nodeattr("param_storage") == "INTERNAL":
            weights_shape = model.get_tensor_shape(self.onnx_node.input[1])
            if weights_shape is None:
                raise ValueError(f"Tensor shape for weights '{self.onnx_node.input[1]}' not found in model.")

            weights_quant = TensorQuant(
                scale=model.get_initializer(self.onnx_node.input[2]),
                zeropt=model.get_initializer(self.onnx_node.input[3]),
                bitwidth=model.get_initializer(self.onnx_node.input[4]),
                signed=bool(self.get_nodeattr("w_signed")),
                narrow=bool(self.get_nodeattr("w_narrow")),
                rounding_mode=self.get_nodeattr("w_rounding_mode"),
            )

            weight_bits = weights_quant.bitwidth
            n_weights = np.prod(weights_shape)

            # Retrieve current parallelization attributes if not provided.
            point = self.__current_dse_point()

            # The unroll factor considers both channel unrolling and kernel size.
            # Width unrolling is not considered as weights are reused across width iterations.
            unroll_factor = (
                point.in_channel_unroll
                * point.out_channel_unroll
                * point.filter_width_unroll
                * point.filter_height_unroll
            )

            return bram_usage_evaluator(weight_bits, n_weights, unroll_factor)
        else:
            return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """Estimate the DSP usage of the StreamingConv operation given a set of parallelization parameters.
        Args:
            point (StreamingConv.DSEPoint): The DSE point containing the parallelization parameters.
        Returns:
            int: Estimated DSP usage.
        """
        silvia_packing = model.get_metadata_prop("silvia_packing") == "true"

        weights_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[2]),
            zeropt=model.get_initializer(self.onnx_node.input[3]),
            bitwidth=model.get_initializer(self.onnx_node.input[4]),
            signed=bool(self.get_nodeattr("w_signed")),
            narrow=bool(self.get_nodeattr("w_narrow")),
            rounding_mode=self.get_nodeattr("w_rounding_mode"),
        )

        act_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if act_quant is None:
            raise ValueError(
                f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model."
            )

        act_bits = act_quant.bitwidth
        weight_bits = weights_quant.bitwidth

        # Retrieve current parallelization attributes if not provided.
        point = self.__current_dse_point()

        mac_per_dsp, _ = packing_feature(
            (act_bits, weight_bits),
            [point.width_unroll, point.out_channel_unroll],
            silvia_packing,
        )
        MACs = (
            point.filter_height_unroll
            * point.filter_width_unroll
            * point.in_channel_unroll
            * point.out_channel_unroll
            * point.width_unroll
        )

        return MACs // mac_per_dsp

    def get_dse_points(self, model: ModelWrapper) -> list["StreamingConv.DSEPoint"]:
        """Generate the list of valid DSE points for the StreamingConv operation."""

        def divisors(n, clip):
            return [i for i in range(1, n + 1) if (n % i == 0 and i <= clip)]

        kernel_height, kernel_width = self.get_nodeattr("kernel_shape")
        weight_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[2]),
            zeropt=model.get_initializer(self.onnx_node.input[3]),
            bitwidth=model.get_initializer(self.onnx_node.input[4]),
            signed=bool(self.get_nodeattr("w_signed")),
            narrow=bool(self.get_nodeattr("w_narrow")),
            rounding_mode=self.get_nodeattr("w_rounding_mode"),
        )
        weight_bits = weight_quant.bitwidth

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
        for in_channel_unroll in divisors(input_shape[1], input_shape[1]):
            for out_channel_unroll in divisors(output_shape[1], output_shape[1]):
                for width_unroll in divisors(output_shape[3], output_shape[3]):
                    # Check dimension of weight streams
                    if (weight_bits * in_channel_unroll * out_channel_unroll) > 4096:
                        continue
                    # Check dimension of input streams
                    if (input_bits * in_channel_unroll) > 4096:
                        continue
                    # Check dimension of output streams
                    if (output_bits * out_channel_unroll) > 4096:
                        continue

                    DSE_points.append(
                        self.DSEPoint(
                            out_channel_unroll, in_channel_unroll, width_unroll, kernel_height, kernel_width
                        )
                    )

        return DSE_points

    def apply_point(self, model: ModelWrapper, point: "StreamingConv.DSEPoint"):
        """ Set the parallelization attributes for the StreamingConv operation.
        Args:
            point (StreamingConv.DSEPoint): The DSE point containing the unrolling parameters.
        """
        self.set_nodeattr("out_channel_unroll", point.out_channel_unroll)
        self.set_nodeattr("in_channel_unroll", point.in_channel_unroll)
        self.set_nodeattr("width_unroll", point.width_unroll)
        self.set_nodeattr("filter_width_unroll", point.filter_width_unroll)
        self.set_nodeattr("filter_height_unroll", point.filter_height_unroll)

        self.set_nodeattr("in_stream_array", point.width_unroll)
        self.set_nodeattr("out_stream_array", point.width_unroll)
        self.set_nodeattr("in_word_array", point.in_channel_unroll)
        self.set_nodeattr("out_word_array", point.out_channel_unroll)

    def list_parameters(self, model: ModelWrapper) -> Iterable[ParamDesc]:

        weights_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[2]),
            zeropt=model.get_initializer(self.onnx_node.input[3]),
            bitwidth=model.get_initializer(self.onnx_node.input[4]),
            signed=bool(self.get_nodeattr("w_signed")),
            narrow=bool(self.get_nodeattr("w_narrow")),
            rounding_mode=self.get_nodeattr("w_rounding_mode"),
        )
        data_per_word = 32 // weights_quant.bitwidth

        # Expand shapes to 4D if needed
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model.")
        output_shape = output_shape + [1] * (4 - len(output_shape))  # Ensure 4D shape.

        mem_shape = model.get_tensor_shape(self.onnx_node.input[1])
        if len(mem_shape) < 4:
            mem_shape = mem_shape + [1] * (4 - len(mem_shape))

        in_channel_unroll = self.get_nodeattr("in_channel_unroll")
        out_channel_unroll = self.get_nodeattr("out_channel_unroll")
        width_unroll = np.prod(mem_shape[2:])
        times = output_shape[2] * output_shape[3] // self.get_nodeattr("width_unroll")

        yield ParamDesc(
            input_index=1,
            name=self.onnx_node.input[1],
            shape=model.get_tensor_shape(self.onnx_node.input[1]),
            tensor_quant=weights_quant,
            in_channel_unroll=in_channel_unroll,
            out_channel_unroll=out_channel_unroll,
            width_unroll=width_unroll,
            data_per_word=data_per_word,
            times=times,
        )

        if len(self.onnx_node.input) > 5:
            bias_quant = TensorQuant(
                scale=model.get_initializer(self.onnx_node.input[6]),
                zeropt=model.get_initializer(self.onnx_node.input[7]),
                bitwidth=model.get_initializer(self.onnx_node.input[8]),
                signed=bool(self.get_nodeattr("b_signed")),
                narrow=bool(self.get_nodeattr("b_narrow")),
                rounding_mode=self.get_nodeattr("b_rounding_mode"),
            )

            data_per_word = 32 // bias_quant.bitwidth
            in_channel_unroll = 1
            out_channel_unroll = self.get_nodeattr("out_channel_unroll")
            width_unroll = 1

            yield ParamDesc(
                input_index=5,
                name=self.onnx_node.input[5],
                shape=model.get_tensor_shape(self.onnx_node.input[5]),
                tensor_quant=bias_quant,
                in_channel_unroll=in_channel_unroll,
                out_channel_unroll=out_channel_unroll,
                width_unroll=width_unroll,
                data_per_word=data_per_word,
                times=times,
            )
