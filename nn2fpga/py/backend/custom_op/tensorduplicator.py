from onnx import helper
from qonnx.custom_op.base import CustomOp
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.core.tensor_fifo import TensorFifo
from backend.custom_op.hlskernel import HLSKernel
from backend.util.codegen_utils import (
    cpp_function,
    cpp_variable,
    cpp_object,
    get_struct_type,
    get_stream_type,
    get_hls_quant_type,
)
from backend.util.par_utils import get_par_attributes

class TensorDuplicator(CustomOp):
    """ Node duplicating a tensor to ensure that each consumer gets a separate copy. """

    def get_nodeattr_types(self):
        return {
            "copies": ("i", True, 2),
            "in_ch_par": ("i", True, 1),
            "out_ch_par": ("i", True, 1),
            "in_w_par": ("i", True, 1),
            "out_w_par": ("i", True, 1),
        }

    def infer_node_datatype(self, model: ModelWrapper):
        node = self.onnx_node
        in_dtype = model.get_tensor_datatype(node.input[0])
        for out in node.output:
            model.set_tensor_datatype(out, in_dtype)

    def make_shape_compatible_op(self, model: ModelWrapper):
        node = self.onnx_node
        shape_compatible_nodes = [] 
        for i in range(self.get_nodeattr("copies")):
            identity_node = helper.make_node(
                "Identity",
                inputs=[node.input[0]],
                outputs=[node.output[i]],
                name=f"{node.name}_shape_compatible_{i}"
            )
            shape_compatible_nodes.append(identity_node)
        return shape_compatible_nodes

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_name = node.input[0]
        input_val = context[input_name]
        num_copies = self.get_nodeattr("copies")

        for i in range(num_copies):
            out_name = node.output[i]
            context[out_name] = input_val.copy()

    def verify_node(self):
        pass

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_variable_declaration(self, model) -> str:
        """ Get the internal cpp variables of the TensorDuplicator node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_object_declaration(self, model) -> str:
        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model.")

        # Retrieve parallelization attributes.
        par_attribute = get_par_attributes(self.onnx_node)

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")
        input_shape = input_shape + [1] * (4 - len(input_shape))  # Pad to 4D if needed.

        # Create the TensorDuplicator object.
        TensorDuplicator = cpp_object(
            "TensorDuplicator",
            f"{self.onnx_node.name}",
            template_args=[
                (f"{get_struct_type(input_quant, par_attribute['in_ch_par'])}", "TWord"),
                (input_shape[2], "IN_HEIGHT"),
                (input_shape[3], "IN_WIDTH"),
                (input_shape[1], "IN_CH"),
                (par_attribute["in_ch_par"], "CH_PAR"),
                (par_attribute["in_w_par"], "W_PAR"),
            ],
        )
        return TensorDuplicator.generate_declaration()

    def __get_run_call(self, hls_tag: int) -> str:
        """ Generates the C++ code necessary to run the TensorDuplicator node. """

        # Generate the call to the TensorDuplicator run method.
        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TWord>",
                ),
                (
                    f"o_data0",
                    f"hls::stream<TWord>",
                ),
                (
                    f"o_data1",
                    f"hls::stream<TWord>",
                ),
            ),
        )

        return run.generate_call(
            [hls_tag],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
            self.__get_stream_name(self.onnx_node.output[1]),
        )

    def __get_step_call(self) -> str:
        """ Generates the C++ code necessary to run the TensorDuplicator node in step mode. """

        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TWord>",
                ),
                (
                    f"o_data0",
                    f"hls::stream<TWord>",
                ),
                (
                    f"o_data1",
                    f"hls::stream<TWord>",
                ),
            ),
        )

        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
            self.__get_stream_name(self.onnx_node.output[1]),
        )

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int) -> tuple[list, list, dict]:
        """
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        par = get_par_attributes(self.onnx_node)
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(par["in_w_par"])
        ]

        output_names = [
            f"{self.__get_stream_name(self.onnx_node.output[0])}_{i}_"
            for i in range(par["out_w_par"])
        ]
        output_names.extend([
            f"{self.__get_stream_name(self.onnx_node.output[1])}_{i}_"
            for i in range(par["out_w_par"])]
        )

        tensors_fifo_metadata = {}
        for output in output_names:
            tensors_fifo_metadata[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_quant, par['out_ch_par'])}",
                n_array=par["out_w_par"],
            )

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="backend.custom_op",
            original_op_type=self.onnx_node.op_type,
            hls_object_name=self.onnx_node.name,
            hls_tag=hls_tag,
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(hls_tag=hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag
