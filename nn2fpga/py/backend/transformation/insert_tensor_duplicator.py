from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from onnx import helper
from backend.core.acceleratorpackage import AcceleratorPackage

class InsertTensorDuplicator(Transformation):
    """
    Inserts a TensorDuplicator node in each fork node of the model.
    This node will duplicate the tensor to ensure that each consumer gets a separate copy.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        fork_nodes = [node for node in model.graph.node if model.is_fork_node(node)]
        for node in fork_nodes:

            fork_out = node.output[0]  
            consumers = model.find_consumers(fork_out)
            # Include graph outputs as consumers
            graph_output_consumers = [x for x in model.graph.output if x.name == fork_out]
            num_copies = len(consumers) + len(graph_output_consumers)
            dup_outputs = [f"{fork_out}_copy_{i}" for i in range(num_copies)]

            # Create the Duplicate node
            dup_node = helper.make_node(
                op_type="TensorDuplicator",
                domain="backend.custom_op",
                inputs=[fork_out],
                outputs=dup_outputs,
                name=f"Duplicate_{fork_out}",
                copies=num_copies
            )
            model.graph.node.insert(0, dup_node)  # Insert early

            # Rewire each consumer to use its own copy
            for consumer in consumers:
                for j, inp_name in enumerate(consumer.input):
                    if inp_name == fork_out:
                        consumer.input[j] = dup_outputs.pop(0)  # Assign a copy and remove it from the list

            # Update output names if fork_out is a graph output
            ap = AcceleratorPackage.from_json(
                model.get_metadata_prop("accelerator_package")
            )
            for graph_out in model.graph.output:
                if graph_out.name == fork_out:
                    graph_out.name = dup_outputs.pop(0)  # Assign a copy and remove it from the list

                    # Search the key in the ap.output_map of the deprecated name
                    # and update the value to the new name
                    for value in ap.output_map.values():
                        if value['new_name'] == fork_out:
                            value['new_name'] = graph_out.name
                            break
            
            model.set_metadata_prop("accelerator_package", ap.to_json())

            # Copy shape and datatype info
            shape = model.get_tensor_shape(fork_out)
            dtype = model.get_tensor_datatype(fork_out)
            for out in dup_outputs:
                model.set_tensor_shape(out, shape)
                model.set_tensor_datatype(out, dtype)

        return (model, False)
