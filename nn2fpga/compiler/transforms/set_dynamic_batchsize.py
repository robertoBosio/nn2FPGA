from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from nn2fpga.compiler.core.tensor_quant import TensorQuant
from qonnx.core.modelwrapper import ModelWrapper
import numpy as np
from onnx import helper, OperatorSetIdProto
import onnx.shape_inference as si

class SetDynamicBatchSize(Transformation):

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to set dynamic batch size."""
        for input_tensor in model.graph.input:
            old_shape = model.get_tensor_shape(input_tensor.name)
            new_shape = [None] + list(old_shape[1:])
            model.set_tensor_shape(input_tensor.name, new_shape)

        while len(model.graph.value_info) > 0:
            model.graph.value_info.remove(model.graph.value_info[0])

        # Remove output shapes
        new_outputs = []
        for out in model.graph.output:
            if not out.type.HasField("tensor_type"):
                new_outputs.append(out)
                continue

            tensor_type = out.type.tensor_type
            elem_type = tensor_type.elem_type

            # Create output without shape
            new_vi = helper.make_tensor_value_info(
                name=out.name,
                elem_type=elem_type,
                shape=None  # No shape -> makes it completely unspecified
            )
            new_outputs.append(new_vi)

        # Replace graph outputs with shape-less versions
        del model.graph.output[:]
        model.graph.output.extend(new_outputs)

        reshape_nodes = model.get_nodes_by_op_type("Reshape")
        for node in reshape_nodes:
            shape_input = node.input[1]  # The second input is the shape
            # Check if shape is stored as an initializer (static values)
            shape_array = model.get_initializer(shape_input)
            if shape_array.size > 0 and shape_array[0] > 0:
                # Set the first dimension to None for dynamic batch
                new_shape_tensor = np.array([0] + list(shape_array[1:]))
                model.set_initializer(shape_input, new_shape_tensor)

        partition_nodes = model.get_nodes_by_op_type("nn2fpgaPartition")
        if partition_nodes is not None and len(partition_nodes) > 0:
            partition_node = partition_nodes[0]
            ap = AcceleratorPackage.from_json(
                getCustomOp(partition_node).get_nodeattr("accelerator_package")
            )
            for output in partition_node.output:
                output_tensor_shape = ap.output_map[output]["shape"]
                dynamic_output_shape = [None] + output_tensor_shape[1:]
                output_quant = TensorQuant.from_canonical_name(
                    ap.output_map[output]["quant"]
                )
                model.set_tensor_shape(
                    output, dynamic_output_shape, dtype=output_quant.get_tensorproto_dtype()
                )

        # Change the domain of nn2fpgaPartition nodes to support ONNX inference.
        model = model.transform(ChangeDomainOnnxInference())

        # We can't do anymore qonnx InferShapes because the domain required for
        # having the implementation of nn2fpgaPartition node in Python is "ai.onnx.contrib".
        # So we need to run the shape inference from ONNX.
        model = ModelWrapper(si.infer_shapes(model.model))

        return model, False

class ChangeDomainOnnxInference(Transformation):
    """A transformation to change the domain of nn2fpgaPartition nodes to support ONNX inference."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to change the domain of nn2fpgaPartition nodes."""
        partition_nodes = model.get_nodes_by_op_type("nn2fpgaPartition")
        for node in partition_nodes:
            node.domain = "ai.nn2FPGA"

        model.model.opset_import.append(
            OperatorSetIdProto(domain="ai.nn2FPGA", version=1)
        )
        return model, False
