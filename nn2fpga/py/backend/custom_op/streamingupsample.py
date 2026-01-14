import numpy as np
import onnxruntime as rt
from onnxscript.rewriter import pattern
from onnx import TensorProto, helper
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.core.tensor_fifo import TensorFifo
from backend.custom_op.hlskernel import HLSKernel
from backend.custom_op.op_base import NN2FPGAOp, NodeInterface
from backend.custom_op.register_rewrite_rule import register_rules
from backend.util.codegen_utils import (
    cpp_function,
    cpp_object,
    get_struct_type,
    get_hls_quant_type,
)

class StreamingUpsample(NN2FPGAOp):
    """ Node implementing the StreamingUpsample operation. """

    @staticmethod
    def pattern_resize(op, x, scales):
        return op.Resize(x, None, scales, _allow_other_inputs=True, _allow_other_attributes=True)

    @staticmethod
    def pattern_upsample(op, x, scales):
        return op.Upsample(x, scales, _allow_other_attributes=True)
    
    @staticmethod
    def rewrite(op, x, scales, **kwargs):

        # Get only the scale factor for height/width
        scale = scales.const_value.numpy()[-1]
        return op.StreamingUpsample(
            x,
            scale_factor=int(scale),
            mode="nearest",
            _domain="backend.custom_op",
        )

    @register_rules
    def register_rules():
        return [
            pattern.RewriteRule(
                StreamingUpsample.pattern_resize, StreamingUpsample.rewrite
            ),
            pattern.RewriteRule(
                StreamingUpsample.pattern_upsample, StreamingUpsample.rewrite
            ),
        ]

    def get_nodeattr_types(self):
        return {
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),

            "channel_unroll": ("i", False, 1),
            "in_width_unroll": ("i", False, 1),
            "out_width_unroll": ("i", False, 1),

            "mode": ("s", False, "nearest"),
            "scale_factor": ("i", False, None),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        
        scales = helper.make_tensor(
            name=f"{node.name}_scales",
            data_type=TensorProto.FLOAT,
            dims=[4],
            vals=[1.0, 1.0, float(self.get_nodeattr("scale_factor")), float(self.get_nodeattr("scale_factor"))],
        )

        return helper.make_node(
            "Upsample",
            inputs=[node.input[0], scales.name],
            outputs=node.output,
            mode=self.get_nodeattr("mode"),
            name=f"{node.name}_shape_compatible",
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        # create a standard Upsample node to compute the result
        node = self.onnx_node

        scales = helper.make_tensor(
            name=f"{node.name}_scales",
            data_type=TensorProto.FLOAT,
            dims=[4],
            vals=[1.0, 1.0, float(self.get_nodeattr("scale_factor")), float(self.get_nodeattr("scale_factor"))],
        )

        node_upsample = helper.make_node(
            "Upsample",
            inputs=[node.input[0], scales.name],
            outputs=node.output,
            mode=self.get_nodeattr("mode"),
            name=f"{node.name}_shape_compatible",
        )

        # Make single node graph for execution
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)

        graph_upsample = helper.make_graph(
            nodes=[node_upsample],
            name="single-upsample-exec",
            inputs=[inp],
            outputs=[outp],
            initializer=[scales],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_upsample = qonnx_make_model(graph_upsample, **onnx_kwargs)
        idict = {node.input[0]: inp_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_upsample.SerializeToString())
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
        """ Get the internal cpp variables of the StreamingUpsample node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""
    
    def __get_quantizer(self, input_quant, output_quant) -> str:
        """ Returns the quantizer type for the StreamingUpsample operation. """

        if (
            self.__is_power_of_two(input_quant.scale)
            and self.__is_power_of_two(output_quant.scale)
        ):
            return f"DequantQuantPo2<0, {get_hls_quant_type(input_quant)}, {get_hls_quant_type(output_quant)}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingUpsample."
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

        StreamingUpsample = cpp_object(
            "StreamingUpsample",
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_struct_type(input_quant, self.get_nodeattr('in_word_array'))}",
                    f"TInputWord",
                ),
                (
                    f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}",
                    f"TOutputWord",
                ),
                (
                    f"{self.__get_quantizer(input_quant, output_quant)}",
                    f"Quantizer",
                ),
                (f"{input_shape[2]}", "IN_HEIGHT"),
                (f"{input_shape[3]}", "IN_WIDTH"),
                (f"{input_shape[1]}", "IN_CH"),
                (f"{output_shape[2]}", "OUT_HEIGHT"),
                (f"{output_shape[3]}", "OUT_WIDTH"),
                (f"{self.get_nodeattr('scale_factor')}", "SCALE_FACTOR"),
                (f"{self.get_nodeattr('channel_unroll')}", "CH_PAR"),
                (f"{self.get_nodeattr('in_width_unroll')}", "IN_W_PAR"),
                (f"{self.get_nodeattr('out_width_unroll')}", "OUT_W_PAR"),
            ]
        )

        return StreamingUpsample.generate_declaration()

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
            original_op_type="StreamingUpsample",
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
        """ Estimate the latency of the StreamingUpsample operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model.")

        unroll_factor = self.get_nodeattr("channel_unroll") * self.get_nodeattr("out_width_unroll")
        return np.prod(output_shape) // unroll_factor

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the StreamingUpsample operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """ Estimate the DSP usage of the StreamingUpsample operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """
        return 0
    
    def has_linebuffer(self) -> bool:
        """ Check if the StreamingUpsample operation requires a line buffer.
        Returns:
            bool: True if Line Buffering is required, False otherwise.
        """
        return False
    
    def can_inherit_interface(self):
        return True
    
    def inherit_interface(self, model: ModelWrapper, upstream: NodeInterface) -> None:
        """ Inherit the interface from the upstream node."""
        self.set_nodeattr("in_stream_array", upstream.out_stream_array)
        self.set_nodeattr("out_stream_array", upstream.out_stream_array * self.get_nodeattr("scale_factor"))
        self.set_nodeattr("in_word_array", upstream.out_word_array)
        self.set_nodeattr("out_word_array", upstream.out_word_array)

        self.set_nodeattr("channel_unroll", upstream.out_word_array)
        self.set_nodeattr("in_width_unroll", upstream.out_stream_array)
        self.set_nodeattr("out_width_unroll", upstream.out_stream_array * self.get_nodeattr("scale_factor"))