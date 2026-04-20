from __future__ import annotations

import logging

import numpy as np
from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    SortGraph,
)

import backend.transformation as transformation
from backend.core.tensor_quant import TensorQuant

logger = logging.getLogger(__name__)

QUANT_NODE_TYPES = {"IntQuant", "Quant"}

QUANT_INVARIANT_NODES = [
    "BandwidthAdjustDecreaseChannels",  # nn2FPGA
    "BandwidthAdjustDecreaseStreams",  # nn2FPGA
    "BandwidthAdjustIncreaseChannels",  # nn2FPGA
    "BandwidthAdjustIncreaseStreams",  # nn2FPGA
    "Concat",
    "Flatten",
    "GlobalMaxPool",
    "Identity",
    "MaxPool",
    "NHWCToStream",  # nn2FPGA
    "Pad",
    "Reshape",
    "Slice",
    "Split",
    "StreamingConcat",  # nn2FPGA
    "StreamingLineBuffer",  # nn2FPGA
    "StreamingMaxPool",  # nn2FPGA
    "StreamingSplit",  # nn2FPGA
    "StreamToNHWC",  # nn2FPGA
    "TensorDuplicator",  # nn2FPGA
    "Transpose",
]


def _initializer_names(model: ModelWrapper) -> set[str]:
    """Return the set of initializer tensor names in the model."""
    return {init.name for init in model.model.graph.initializer}


def _graph_output_names(model: ModelWrapper) -> set[str]:
    """Return the set of graph output tensor names."""
    return {out.name for out in model.graph.output}


def get_non_constant_inputs(node: NodeProto, model: ModelWrapper) -> list[str]:
    """Return the non-constant inputs of a node.

    Initializers and outputs of Constant nodes are excluded.
    """
    init_names = _initializer_names(model)
    non_constant_inputs: list[str] = []

    for inp in node.input:
        if inp in init_names:
            continue

        producer = model.find_producer(inp)
        if producer is not None and producer.op_type == "Constant":
            continue

        non_constant_inputs.append(inp)

    return non_constant_inputs


def get_non_constant_outputs(node: NodeProto, model: ModelWrapper) -> list[str]:
    """Return the non-initializer outputs of a node.

    In practice, normal computational node outputs are not initializers, but this
    helper keeps the filtering symmetric with get_non_constant_inputs.
    """
    init_names = _initializer_names(model)
    return [out for out in node.output if out not in init_names]


def _all_nodes_have_same_quant(
    quant_nodes: list[NodeProto], model: ModelWrapper
) -> TensorQuant | None:
    """Return the shared TensorQuant if all nodes have identical quant params."""
    if not quant_nodes:
        return None

    if not all(node.op_type in QUANT_NODE_TYPES for node in quant_nodes):
        return None

    reference_quant = TensorQuant.from_quant_node(quant_nodes[0], model)
    if all(TensorQuant.from_quant_node(node, model) == reference_quant for node in quant_nodes):
        return reference_quant

    return None


def _record_quant_proposal(
    tensor_name: str,
    reference_quant: TensorQuant,
    proposed_quant_nodes: dict[str, str],
    conflicted_tensors: set[str],
) -> None:
    """Record a proposed quantization for a tensor, tracking conflicts explicitly."""
    if tensor_name in conflicted_tensors:
        return

    new_quant_canonical = reference_quant.get_canonical_name()
    existing_quant_canonical = proposed_quant_nodes.get(tensor_name)

    if existing_quant_canonical is None:
        proposed_quant_nodes[tensor_name] = new_quant_canonical
        return

    if existing_quant_canonical != new_quant_canonical:
        logger.warning(
            "Conflict in proposed quantization for tensor %s: existing quantization %s "
            "vs new quantization %s. Skipping propagation for this tensor.",
            tensor_name,
            existing_quant_canonical,
            new_quant_canonical,
        )
        proposed_quant_nodes.pop(tensor_name, None)
        conflicted_tensors.add(tensor_name)


def forward_propagate_quantization(
    producers: list[NodeProto | None],
    consumers: list[NodeProto | None],
    node: NodeProto,
    model: ModelWrapper,
    proposed_quant_nodes: dict[str, str],
    conflicted_tensors: set[str],
) -> None:
    """Propose quantization on node outputs from quantized node inputs.

    Propagation is allowed when:
    - all non-constant input producers exist and are quant nodes
    - all such quant nodes have identical quantization parameters
    - none of the output consumers are quant nodes
    """
    if not producers or any(producer is None for producer in producers):
        return

    quantized_producers = [producer for producer in producers if producer is not None]
    reference_quant = _all_nodes_have_same_quant(quantized_producers, model)
    if reference_quant is None:
        return

    if any(
        consumer is not None and consumer.op_type in QUANT_NODE_TYPES
        for consumer in consumers
    ):
        return

    for output_tensor in get_non_constant_outputs(node, model):
        _record_quant_proposal(
            output_tensor,
            reference_quant,
            proposed_quant_nodes,
            conflicted_tensors,
        )


def backward_propagate_quantization(
    consumers: list[NodeProto | None],
    producers: list[NodeProto | None],
    node: NodeProto,
    model: ModelWrapper,
    proposed_quant_nodes: dict[str, str],
    conflicted_tensors: set[str],
) -> None:
    """Propose quantization on node inputs from quantized node outputs.

    Propagation is allowed when:
    - all non-constant output consumers exist and are quant nodes
    - all such quant nodes have identical quantization parameters
    - none of the input producers are quant nodes
    """
    if not consumers or any(consumer is None for consumer in consumers):
        return

    quantized_consumers = [consumer for consumer in consumers if consumer is not None]
    reference_quant = _all_nodes_have_same_quant(quantized_consumers, model)
    if reference_quant is None:
        return

    if any(
        producer is not None and producer.op_type in QUANT_NODE_TYPES
        for producer in producers
    ):
        return

    for input_tensor in get_non_constant_inputs(node, model):
        _record_quant_proposal(
            input_tensor,
            reference_quant,
            proposed_quant_nodes,
            conflicted_tensors,
        )


def _unique_consumers_for_outputs(
    output_tensors: list[str], model: ModelWrapper
) -> list[NodeProto]:
    """Return a deduplicated list of consumers for the provided output tensors."""
    consumers: list[NodeProto] = []
    seen: set[str] = set()

    for tensor_name in output_tensors:
        for consumer in model.find_consumers(tensor_name):
            consumer_name = consumer.name if consumer.name != "" else str(id(consumer))
            if consumer_name in seen:
                continue
            seen.add(consumer_name)
            consumers.append(consumer)

    return consumers


def _materialize_quant_node_for_tensor(
    model: ModelWrapper,
    tensor_name: str,
    tensor_quant: TensorQuant,
) -> NodeProto:
    """Insert a Quant node for tensor_name and rewire graph edges accordingly."""
    graph_output_names = _graph_output_names(model)

    scale_name = f"{tensor_name}_quant_scale"
    zeropt_name = f"{tensor_name}_quant_zero_point"
    bitwidth_name = f"{tensor_name}_quant_bitwidth"
    quantized_tensor_name = f"{tensor_name}_quant"

    model.set_initializer(
        scale_name,
        np.array([tensor_quant.scale], dtype=np.float32),
    )
    model.set_initializer(
        zeropt_name,
        np.array([tensor_quant.zeropt], dtype=tensor_quant.get_numpy_dtype()),
    )
    model.set_initializer(
        bitwidth_name,
        np.array([tensor_quant.bitwidth], dtype=np.int32),
    )

    if tensor_name not in graph_output_names:
        new_quant_node = helper.make_node(
            "Quant",
            inputs=[
                tensor_name,
                scale_name,
                zeropt_name,
                bitwidth_name,
            ],
            outputs=[quantized_tensor_name],
            name=f"Quant_propagated_{tensor_name}",
            signed=tensor_quant.signed,
            narrow=tensor_quant.narrow,
            rounding_mode=tensor_quant.rounding_mode,
            domain="qonnx.custom_op.general",
        )

        for consumer in model.find_consumers(tensor_name):
            for idx, inp in enumerate(consumer.input):
                if inp == tensor_name:
                    consumer.input[idx] = quantized_tensor_name

        return new_quant_node

    # Special case: preserve the public graph output tensor name.
    producer = model.find_producer(tensor_name)
    if producer is not None:
        for idx, out in enumerate(producer.output):
            if out == tensor_name:
                producer.output[idx] = quantized_tensor_name

    new_quant_node = helper.make_node(
        "Quant",
        inputs=[
            quantized_tensor_name,
            scale_name,
            zeropt_name,
            bitwidth_name,
        ],
        outputs=[tensor_name],
        name=f"Quant_propagated_{tensor_name}",
        signed=tensor_quant.signed,
        narrow=tensor_quant.narrow,
        rounding_mode=tensor_quant.rounding_mode,
        domain="qonnx.custom_op.general",
    )
    return new_quant_node


class PropagateQuant(Transformation):
    """Propagate quantization parameters through quantization-invariant nodes.

    This transformation identifies quantization-invariant operators and propagates
    quantization metadata through them in two directions:

    - Forward propagation:
      if all non-constant inputs are produced by quant nodes with identical
      parameters, and none of the outputs are already consumed by quant nodes,
      propose quantization on the outputs.

    - Backward propagation:
      if all non-constant outputs are consumed by quant nodes with identical
      parameters, and none of the inputs are already produced by quant nodes,
      propose quantization on the inputs.

    Proposals are collected first, then materialized as Quant nodes in a second pass.
    Conflicting proposals for the same tensor are dropped.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        is_changed = False

        proposed_quant_nodes: dict[str, str] = {}
        conflicted_tensors: set[str] = set()

        invariant_nodes = [
            node for node in graph.node if node.op_type in QUANT_INVARIANT_NODES
        ]

        for node in invariant_nodes:
            non_constant_inputs = get_non_constant_inputs(node, model)
            non_constant_outputs = get_non_constant_outputs(node, model)

            producers = [model.find_producer(inp) for inp in non_constant_inputs]
            consumers = _unique_consumers_for_outputs(non_constant_outputs, model)

            forward_propagate_quantization(
                producers=producers,
                consumers=consumers,
                node=node,
                model=model,
                proposed_quant_nodes=proposed_quant_nodes,
                conflicted_tensors=conflicted_tensors,
            )
            backward_propagate_quantization(
                consumers=consumers,
                producers=producers,
                node=node,
                model=model,
                proposed_quant_nodes=proposed_quant_nodes,
                conflicted_tensors=conflicted_tensors,
            )

        for tensor_name, quant_canonical in proposed_quant_nodes.items():
            tensor_quant = TensorQuant.from_canonical_name(quant_canonical)
            new_quant_node = _materialize_quant_node_for_tensor(
                model=model,
                tensor_name=tensor_name,
                tensor_quant=tensor_quant,
            )
            graph.node.append(new_quant_node)
            is_changed = True

        # Always normalize the graph after this pass. This is especially important
        # when the graph was modified, but it is also harmless and useful otherwise.
        model = model.transform(SortGraph())
        model = model.transform(transformation.CustomInferShapes())
        model = model.transform(GiveUniqueParameterTensors())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

        return model, is_changed