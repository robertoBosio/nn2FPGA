import logging

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from nn2fpga.compiler.core.tensor_type import require_tensor_type
from nn2fpga.compiler.utils.board_util import read_board_info


logger = logging.getLogger(__name__)

_URAM_DEPTH = 4096
_URAM_WIDTH_BITS = 72
_MIN_URAM_UTILIZATION = 0.5


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


class AllocateStreamingMemoryURAM(Transformation):
    """Report a capacity-aware URAM allocation for StreamingMemory nodes.

    Memories must use at least half of their allocated URAM capacity before
    they are considered for selection.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        streaming_memories = model.get_nodes_by_op_type("StreamingMemory")
        candidates = []
        for node in streaming_memories:
            op = getCustomOp(node)
            brams = op.get_brams(model)
            if brams <= 0:
                continue

            output_shape = op.require_4d_output_shape(model, 0)
            output_type = require_tensor_type(model, node.output[0])
            array_parallelism = op.get_nodeattr("width_unroll")
            word_parallelism = (
                op.get_nodeattr("in_channel_unroll")
                * op.get_nodeattr("out_channel_unroll")
            )
            words_per_address = array_parallelism * word_parallelism
            depth = _ceil_div(int(np.prod(output_shape)), words_per_address)
            width_bits = output_type.bitwidth * words_per_address
            urams = _ceil_div(depth, _URAM_DEPTH) * _ceil_div(
                width_bits, _URAM_WIDTH_BITS
            )
            payload_bits = int(np.prod(output_shape)) * output_type.bitwidth
            allocated_bits = urams * _URAM_DEPTH * _URAM_WIDTH_BITS
            if payload_bits / allocated_bits < _MIN_URAM_UTILIZATION:
                continue
            candidates.append((node, brams, urams))

        selected = (0, ())
        if candidates:
            uram_capacity = read_board_info(
                model.get_metadata_prop("board_name")
            )["uram"]
            if uram_capacity > 0:
                # Each state stores (released_bram, selected candidate indexes).
                # Iterating backwards prevents selecting a memory more than once.
                states = [None] * (uram_capacity + 1)
                states[0] = (0, ())
                for candidate_index, (_, brams, urams) in enumerate(candidates):
                    if urams > uram_capacity:
                        continue
                    for used_urams in range(uram_capacity, urams - 1, -1):
                        previous = states[used_urams - urams]
                        if previous is None:
                            continue
                        proposal = (
                            previous[0] + brams,
                            previous[1] + (candidate_index,),
                        )
                        current = states[used_urams]
                        if current is None or proposal[0] > current[0] or (
                            proposal[0] == current[0] and proposal[1] < current[1]
                        ):
                            states[used_urams] = proposal

                selected_used_urams, selected = 0, states[0]
                for used_urams, state in enumerate(states[1:], start=1):
                    if state is None:
                        continue
                    if state[0] > selected[0] or (
                        state[0] == selected[0]
                        and (
                            used_urams < selected_used_urams
                            or (
                                used_urams == selected_used_urams
                                and state[1] < selected[1]
                            )
                        )
                    ):
                        selected_used_urams, selected = used_urams, state

        selected_nodes = {id(candidates[index][0]) for index in selected[1]}
        modified = False
        for node in streaming_memories:
            op = getCustomOp(node)
            storage_impl = "uram" if id(node) in selected_nodes else "auto"
            if op.get_nodeattr("storage_impl") != storage_impl:
                op.set_nodeattr("storage_impl", storage_impl)
                modified = True

        for candidate_index in selected[1]:
            node, brams, urams = candidates[candidate_index]
            logger.info(
                "Selected %s: %d URAM, releases estimated %d BRAM",
                node.name,
                urams,
                brams,
            )

        return model, modified
