from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.core.modelwrapper import ModelWrapper
from onnx import helper
from collections import deque
from backend.util.par_utils import get_par_attributes, check_par_attributes
import backend.transformation as transformation
import math
import logging
import numpy as np

logger = logging.getLogger(__name__)


def insert_comm_node(
    model,
    name,
    op_type,
    input_name,
    output_name,
    ch_par_in,
    ch_par_out,
    w_par_in,
    w_par_out,
    tensor_shape,
):
    node = helper.make_node(
        op_type,
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        domain="backend.custom_op",
        in_ch_par=ch_par_in,
        out_ch_par=ch_par_out,
        in_w_par=w_par_in,
        out_w_par=w_par_out,
    )

    model.set_tensor_shape(output_name, tensor_shape)
    model.graph.node.append(node)
    return output_name, ch_par_out, w_par_out


def adjust_bandwidth(
    model,
    output,
    last_output,
    last_ch_par,
    last_w_par,
    target_ch_par,
    target_w_par,
    producer,
    consumer,
    tensor_shape,
    model_II,
):
    def adjust(par_in, par_out, axis, op_decr, op_incr, name_suffix):
        nonlocal last_output, last_ch_par, last_w_par
        if par_in == par_out:
            return par_out
        op_base = f"{output}_{name_suffix}"

        # Determine if a middle node is needed
        middle = (
            math.gcd(par_in, par_out)
            if par_in % par_out != 0 and par_out % par_in != 0
            else None
        )

        if middle is not None:
            node_II = np.prod(tensor_shape) // (middle * last_w_par)
            if node_II > model_II:
                logger.info(
                    f"Bandwidth adjustment computed based on GCD may not meet the throughput requirement of the model. Switching to LCM."
                )
                middle = math.lcm(par_in, par_out)

        if middle:
            # Decrease to middle
            middle_op = op_decr if par_in > middle else op_incr
            last_output, last_ch_par, last_w_par = insert_comm_node(
                model,
                f"{middle_op}_{producer.name}_middle_{consumer.name}",
                middle_op,
                last_output,
                f"{op_base}_gcd",
                par_in if axis == "ch" else last_ch_par,
                middle if axis == "ch" else last_ch_par,
                par_in if axis == "w" else last_w_par,
                middle if axis == "w" else last_w_par,
                tensor_shape,
            )
            par_in = middle

        final_op = op_incr if par_out > par_in else op_decr
        last_output, last_ch_par, last_w_par = insert_comm_node(
            model,
            f"{final_op}_{producer.name}_{consumer.name}",
            final_op,
            last_output,
            f"{op_base}",
            par_in if axis == "ch" else last_ch_par,
            par_out if axis == "ch" else last_ch_par,
            par_in if axis == "w" else last_w_par,
            par_out if axis == "w" else last_w_par,
            tensor_shape,
        )
        return par_out

    II_chfirst_graph = max(
        np.prod(tensor_shape) // (last_w_par * min(target_ch_par, last_ch_par)),
        np.prod(tensor_shape) // (target_ch_par * min(last_w_par, target_w_par)),
    )
    II_wfirst_graph = max(
        np.prod(tensor_shape) // (last_ch_par * min(target_w_par, last_w_par)),
        np.prod(tensor_shape) // (target_w_par * min(last_ch_par, target_ch_par)),
    )
    if min(II_chfirst_graph, II_wfirst_graph) > model_II:
        raise ValueError(
            f"Cannot adjust bandwidth between {producer.name} and {consumer.name} to meet the model II requirement."
        )
    
    if II_chfirst_graph <= II_wfirst_graph:

        # Adjust channels
        last_ch_par = adjust(
            last_ch_par,
            target_ch_par,
            "ch",
            "BandwidthAdjustDecreaseChannels",
            "BandwidthAdjustIncreaseChannels",
            "bwch",
        )

        # Adjust streams (width)
        last_w_par = adjust(
            last_w_par,
            target_w_par,
            "w",
            "BandwidthAdjustDecreaseStreams",
            "BandwidthAdjustIncreaseStreams",
            "bww",
        )
    
    else:
        # Adjust streams (width)
        last_w_par = adjust(
            last_w_par,
            target_w_par,
            "w",
            "BandwidthAdjustDecreaseStreams",
            "BandwidthAdjustIncreaseStreams",
            "bww",
        )

        # Adjust channels
        last_ch_par = adjust(
            last_ch_par,
            target_ch_par,
            "ch",
            "BandwidthAdjustDecreaseChannels",
            "BandwidthAdjustIncreaseChannels",
            "bwch",
        )

    return last_output, last_ch_par, last_w_par


class AdjustStreamingCommunication(Transformation):
    """
    A transformation pass that adjusts the streaming communication
    between nodes by inserting necessary communication nodes adjusting
    the number of streaming channels or the channel packed in a single packet.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        # Traversing the graph to find nodes that require adjustment
        communication_nodes = []
        queue = deque(model.get_nodes_by_op_type("NHWCToStream"))
        mark_visited = set()
        model_II = int(model.get_metadata_prop("model_II"))

        while queue:
            node = queue.popleft()
            mark_visited.add(node.name)
            par = get_par_attributes(node)

            consumers = model.find_direct_successors(node)
            if consumers is not None:
                for consumer in consumers:
                    if check_par_attributes(consumer):
                        consumer_par = get_par_attributes(consumer)

                        # Need to balance node.och_par -> consumer.ich_par if ich_par is present,
                        # This represents the number of channels packed in a single packet.
                        if (
                            par["out_ch_par"] != consumer_par["in_ch_par"]
                            or par["out_w_par"] != consumer_par["in_w_par"]
                        ):
                            # Insert a communication node.
                            communication_nodes.append(
                                (
                                    node,
                                    consumer,
                                    par["out_ch_par"],
                                    consumer_par["in_ch_par"],
                                    par["out_w_par"],
                                    consumer_par["in_w_par"],
                                )
                            )

                    if consumer.name not in mark_visited:
                        # If the consumer is not already visited, add it to the queue.
                        queue.append(consumer)

        for (
            producer,
            consumer,
            in_ch_par,
            out_ch_par,
            in_w_par,
            out_w_par,
        ) in communication_nodes:
            for output in producer.output:
                probable_consumer = model.find_consumer(output)
                if probable_consumer is None or probable_consumer.name != consumer.name:
                    continue

                last_output = output
                last_ch_par = in_ch_par
                last_w_par = in_w_par
                tensor_shape = model.get_tensor_shape(output)

                last_output, last_ch_par, last_w_par = adjust_bandwidth(
                    model,
                    output,
                    last_output,
                    last_ch_par,
                    last_w_par,
                    out_ch_par,
                    out_w_par,
                    producer,
                    consumer,
                    tensor_shape,
                    model_II,
                )

                for i, input_name in enumerate(consumer.input):
                    if input_name == output:
                        consumer.input[i] = last_output


        if communication_nodes:
            # If any communication nodes were added, we need to sort the graph
            # and infer shapes again to ensure the model is valid.
            model = model.transform(SortGraph())
            # model = model.transform(transformation.CustomInferShapes())
            logger.info(
                f"Inserted {len(communication_nodes)} communication nodes to adjust streaming communication."
            )
        return (model, False)
