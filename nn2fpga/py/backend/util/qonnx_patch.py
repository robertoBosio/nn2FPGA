from qonnx.util import basic as qonnx_basic

# Store original in case needed
original_is_finn_op = qonnx_basic.is_finn_op

# List of extra custom op domains to treat as QONNX-compatible
EXTRA_CUSTOM_DOMAINS = ["backend"]

def patched_is_finn_op(op_type):
    if original_is_finn_op(op_type):
        return True
    return any(op_type.startswith(dom) for dom in EXTRA_CUSTOM_DOMAINS)

def patch_qonnx_ops():
    """
    Monkey patch QONNX to:
    - Treat some custom ops as FINN-compatible
    - Override PartitionFromLambda.apply with a custom version
    """

    # Patch is_finn_op first
    qonnx_basic.is_finn_op = patched_is_finn_op

    # Patch PartitionFromLambda apply
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.create_generic_partitions import PartitionFromLambda
    from backend.core.acceleratorpackage import AcceleratorPackage
    from backend.core.tensor_quant import TensorQuant
    from onnx import helper
    import copy
    import pathlib
    import tempfile

    original_partition_from_lambda_apply = PartitionFromLambda.apply

    def custom_partition_from_lambda_apply(self, model: ModelWrapper):
        original_nodes = list(model.graph.node)
        partition_ids = set(list(map(self.partitioning, original_nodes)))
        partition_ids.discard(-1)

        # prepare dir for generated .onnx models
        if self.partition_dir is None:
            self.partition_dir = tempfile.mkdtemp(prefix="partitioning_")
        else:
            pathlib.Path(self.partition_dir).mkdir(parents=True, exist_ok=True)

        for partition_id in partition_ids:
            all_nodes = list(model.graph.node)
            partition_nodes = list(filter(lambda x: self.partitioning(x) == partition_id, all_nodes))
            non_partition_nodes = list(filter(lambda x: x not in partition_nodes, all_nodes))

            # partition the model into two models
            p_model = copy.deepcopy(model)
            non_p_model = model
            # remove all non-partition nodes from the partition model
            for node_to_remove in non_partition_nodes:
                p_model.graph.node.remove(node_to_remove)

            # identify the entry and exit points for the partition part
            p_in = []
            p_out = []
            p_start_ind = 0
            for node in p_model.graph.node:
                for in_tensor in node.input:
                    # check if producer has been removed = lies outside the partition
                    has_initializer = in_tensor in [x.name for x in p_model.graph.initializer]
                    has_producer = p_model.find_producer(in_tensor) is not None
                    if not has_initializer and not has_producer:
                        # the same tensor could feed multiple nodes within the partition
                        # (e.g. for residual connections), so we avoid duplicates
                        if in_tensor not in p_in:
                            p_in.append(in_tensor)
                        # keep track of where this partition starts topologically
                        if p_start_ind == 0:
                            p_start_ind = all_nodes.index(node)
                for out_tensor in node.output:
                    # check if tensor is top-level output
                    # or has a consumer outside the partition
                    if out_tensor in [x.name for x in model.graph.output]:
                        if out_tensor not in p_out:
                            p_out.append(out_tensor)
                    else:
                        for consumer in model.find_consumers(out_tensor):
                            if self.partitioning(consumer) != partition_id:
                                if out_tensor not in p_out:
                                    p_out.append(out_tensor)

            p_in_vi = list(map(lambda x: p_model.get_tensor_valueinfo(x), p_in))
            p_out_vi = list(map(lambda x: p_model.get_tensor_valueinfo(x), p_out))

            # check if partitioning is legal (i.e. creates no cycles)
            to_check = [model.find_producer(x) for x in p_in]
            already_checked = set()
            while len(to_check) > 0:
                next_to_check = []
                for node in to_check:
                    if node is not None:
                        assert (
                            self.partitioning(node) != partition_id
                        ), """cycle-free graph violated: partition depends on itself"""
                        # print(node)
                        already_checked.add(node.name)
                        predecessors = model.find_direct_predecessors(node)
                        if predecessors is not None:
                            for predecessor in predecessors:
                                if predecessor.name not in already_checked:
                                    next_to_check.append(predecessor)
                            # next_to_check.extend(predecessors)
                to_check = next_to_check

            # set p graph in/out to be p_in/p_out
            while len(p_model.graph.input) > 0:
                p_model.graph.input.pop()
            for i in p_in_vi:
                p_model.graph.input.append(i)

            while len(p_model.graph.output) > 0:
                p_model.graph.output.pop()
            for o in p_out_vi:
                p_model.graph.output.append(o)

            # remove redundant input and output value_info entries
            for i in p_in_vi:
                if i in p_model.graph.value_info:
                    p_model.graph.value_info.remove(i)

            for o in p_out_vi:
                if o in p_model.graph.value_info:
                    p_model.graph.value_info.remove(o)
            
            # Build the accelerator package input/output maps
            # It is fundamental to maintain consistency between the maps and the inputs/outputs names
            # in the partition model. We use the same name convention as qonnx's GiveReadableTensorNames
            # transformation, i.e. global_in and global_out.
            index = 0
            ap_input_map = {}
            for i, input in enumerate(p_model.graph.input):
                first_node = p_model.find_consumer(input.name)
                if first_node is None:
                    raise ValueError(
                        f"Partition input {input.name} does not have a consumer."
                    )
                new_name = f"global_in" if i == 0 else f"global_in_{i}"
                ap_input_map[input.name] = {
                    "new_name": new_name,
                    "index": index,
                    "shape": p_model.get_tensor_shape(input.name),
                    "quant": None,
                    "value": None,
                }
                if first_node.op_type == "Quant":
                    tensor_quant = TensorQuant.from_quant_node(first_node, p_model)
                    ap_input_map[input.name]["quant"] = tensor_quant.get_canonical_name()
                else:
                    # Currently, we do not support non quantized inputs in nn2fpgaPartition,
                    # so there is something wrong if we reach this point.
                    raise ValueError(
                        f"Partition input {input.name} is not quantized."
                    )
                index += 1

            index = 0 
            ap_output_map = {}
            for i, output in enumerate(p_model.graph.output):
                last_node = p_model.find_producer(output.name)
                if last_node is None:
                    raise ValueError(
                        f"Partition output {output.name} does not have a producer."
                    )
                new_name = f"global_out" if i == 0 else f"global_out_{i}"
                ap_output_map[output.name] = {
                    "new_name": new_name,
                    "index": index,
                    "shape": p_model.get_tensor_shape(output.name),
                    "quant": None,
                    "value": None,
                }
                if last_node.op_type == "Quant":
                    tensor_quant = TensorQuant.from_quant_node(last_node, p_model)
                    ap_output_map[output.name]["quant"] = tensor_quant.get_canonical_name()
                else:
                    # Currently, we do not support non quantized outputs in nn2fpgaPartition,
                    # so there is something wrong if we reach this point.
                    raise ValueError(
                        f"Partition output {output.name} is not quantized."
                    )
                index += 1

            # Create the accelerator package
            ap = AcceleratorPackage(
                input_map=ap_input_map,
                output_map=ap_output_map,
                board_name=p_model.get_metadata_prop("board_name"),
                top_name=p_model.get_metadata_prop("top_name"),
                frequency=p_model.get_metadata_prop("frequency"),
                hls_version=p_model.get_metadata_prop("hls_version"),
            )
            p_model.set_metadata_prop("accelerator_package", ap.to_json())

            # save partition model
            p_model_filename = self.partition_dir + "/partition_" + str(partition_id) + ".onnx"
            p_model.cleanup()
            p_model.save(p_model_filename)

            # insert nn2fpgaPartition node
            p_node = helper.make_node(
                "nn2fpgaPartition",
                p_in,
                p_out,
                name="nn2fpgaPartition_" + str(partition_id),
                domain="backend.custom_op",
            )
            non_p_model.graph.node.insert(p_start_ind, p_node)

            # remove all partition nodes from the parent model
            # do this after inserting the p_node for easier p_start_ind handling
            for node_to_remove in partition_nodes:
                non_p_model.graph.node.remove(node_to_remove)

            model = non_p_model

        return (model, False)

    PartitionFromLambda.apply = custom_partition_from_lambda_apply
