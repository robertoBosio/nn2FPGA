from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from onnx import helper
from backend.core.acceleratorpackage import AcceleratorPackage
    
def _fresh_node_name(model: ModelWrapper, base: str) -> str:
    """Generate a node name not used in the graph."""
    used = {n.name for n in model.graph.node}
    name = base
    k = 0
    while name in used:
        k += 1
        name = f"{base}_{k}"
    return name

class InsertTensorDuplicator(Transformation):
    """
    Inserts a TensorDuplicator node in each fork node of the model.
    This node will duplicate the tensor to ensure that each consumer gets a separate copy.
    It runs multiple times, duplicating tensors until there are no more fork nodes left.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        ap = AcceleratorPackage.from_json(
            model.get_metadata_prop("accelerator_package")
        )
        fork_nodes = [node for node in model.graph.node if model.is_fork_node(node) and node.op_type != "TensorDuplicator"]

        modified = False
        for node in fork_nodes:

            fork_out = node.output[0]
            consumers = model.find_consumers(fork_out)
            
            # Extract two consumers from the dictionary (sorting to have a deterministic order)
            node_pos = {id(n): i for i, n in enumerate(model.graph.node)}
            consumers = sorted(consumers, key=lambda n: node_pos.get(id(n), 10**9))

            consumer_list = [(c, "node") for c in consumers]
            graph_output_consumers = [
                x for x in model.graph.output if x.name == fork_out
            ]
            for graph_out in graph_output_consumers:
                consumer_list.append((graph_out, "graph_output"))
            
            consumers = consumer_list[:2]
            dup_outputs = [model.make_new_valueinfo_name() for _ in range(2)]
            dup_outputs_all = dup_outputs.copy()

            # Create the Duplicate node
            dup_node = helper.make_node(
                op_type="TensorDuplicator",
                domain="backend.custom_op",
                inputs=[fork_out],
                outputs=dup_outputs,
                name=_fresh_node_name(model, f"Duplicate_{fork_out}"),
            )
            producer = model.find_producer(fork_out)
            if producer is None:
                model.graph.node.insert(0, dup_node)  # or append, or a safer location
            else:
                prod_idx = list(model.graph.node).index(producer)
                model.graph.node.insert(prod_idx + 1, dup_node)

            # Rewire each consumer to use its own copy
            for consumer, consumer_type in consumers:
                if consumer_type == "node":
                    for j, inp_name in enumerate(consumer.input):
                        if inp_name == fork_out:
                            consumer.input[j] = dup_outputs.pop(
                                0
                            )  # Assign a copy and remove it from the list

                elif consumer_type == "graph_output":
                    if consumer.name == fork_out:
                        consumer.name = dup_outputs.pop(
                            0
                        )  # Assign a copy and remove it from the list

                    # Search the key in the ap.output_map of the deprecated name
                    # and update the value to the new name
                    for value in ap.output_map.values():
                        if value["new_name"] == fork_out:
                            value["new_name"] = consumer.name
                            break

            # Copy shape and datatype info
            shape = model.get_tensor_shape(fork_out)
            dtype = model.get_tensor_datatype(fork_out)
            for out in dup_outputs_all:
                model.set_tensor_shape(out, shape)
                model.set_tensor_datatype(out, dtype)
            modified = True

        model.set_metadata_prop("accelerator_package", ap.to_json())

        return (model, modified)
