from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import TensorQuant
import logging

logger = logging.getLogger(__name__)

class RemoveRedundantQuant(Transformation):
    """Remove IntQuant/Quant nodes that are immediately fed by another IntQuant/Quant
    with identical quantization parameters.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.model.graph

        new_nodes: list = []
        # alias_map[a] = b means "tensor a should be replaced by tensor b"
        alias_map: dict[str, str] = {}
        # producer_map[tensor_name] = node_that_produces_it (for current rewritten graph state)
        producer_map: dict[str, object] = {}

        def is_quant(n) -> bool:
            return n.op_type in ("IntQuant", "Quant")

        def resolve(t: str) -> str:
            # resolve aliases transitively
            while t in alias_map:
                t = alias_map[t]
            return t

        run_again = False

        for node in graph.node:
            # Always resolve inputs to account for already-removed nodes
            for k, inp in enumerate(node.input):
                if inp:
                    node.input[k] = resolve(inp)

            if not is_quant(node):
                new_nodes.append(node)
                # Update producer map for all outputs
                for out in node.output:
                    if out:
                        producer_map[out] = node
                continue

            # Find the *actual* producer of node.input[0]
            if len(node.input) == 0 or not node.input[0]:
                # Degenerate quant node; keep it
                new_nodes.append(node)
                for out in node.output:
                    if out:
                        producer_map[out] = node
                continue

            inp0 = node.input[0]
            prev_node = producer_map.get(inp0)

            # Must be directly fed by a quant node (producer of the tensor), not "previous kept node"
            if prev_node is None or not is_quant(prev_node):
                new_nodes.append(node)
                for out in node.output:
                    if out:
                        producer_map[out] = node
                continue

            # Confirm it's truly the direct edge: producer output[0] == this input[0]
            # (producer_map already suggests that, but be safe about multi-output nodes)
            if len(prev_node.output) == 0 or prev_node.output[0] != inp0:
                new_nodes.append(node)
                for out in node.output:
                    if out:
                        producer_map[out] = node
                continue

            # Compare quantization parameters
            q1 = TensorQuant.from_quant_node(prev_node, model)
            q2 = TensorQuant.from_quant_node(node, model)

            if q1 == q2:
                logger.info(
                    f"Removing redundant quant node {node.name} with identical quant params to {prev_node.name}"
                )
                # Redirect consumers of node.output[0] to prev_node.output[0]
                if len(node.output) > 0 and node.output[0]:
                    alias_map[node.output[0]] = prev_node.output[0]
                    # Important: treat node.output[0] as produced by prev_node now,
                    # so later nodes can still detect quant->quant patterns correctly.
                    producer_map[node.output[0]] = prev_node
                run_again = True
                continue

            logger.info(
                f"Keeping quant node {node.name} as quant params differ from {prev_node.name}"
            )
            new_nodes.append(node)
            for out in node.output:
                if out:
                    producer_map[out] = node

        # Final patching pass (important for nodes already appended before later aliases)
        if alias_map:
            for node in new_nodes:
                for k, inp in enumerate(node.input):
                    if inp:
                        node.input[k] = resolve(inp)

            for output in graph.output:
                if output.name:
                    output.name = resolve(output.name)

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        return (model, run_again)


