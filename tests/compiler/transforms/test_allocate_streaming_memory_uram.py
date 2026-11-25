from types import SimpleNamespace

from nn2fpga.compiler.transforms.allocate_streaming_memory_uram import (
    AllocateStreamingMemoryURAM,
)
import nn2fpga.compiler.transforms.allocate_streaming_memory_uram as allocate_uram


class _StreamingMemoryOp:
    def __init__(self):
        self.attrs = {
            "storage_impl": "auto",
            "width_unroll": 8,
            "in_channel_unroll": 1,
            "out_channel_unroll": 1,
        }

    def get_nodeattr(self, name):
        return self.attrs[name]

    def set_nodeattr(self, name, value):
        self.attrs[name] = value

    def get_brams(self, model):
        return 1

    def require_4d_output_shape(self, model, output_index):
        return [1, 1, 4096, 8]


class _Model:
    def __init__(self):
        self.node = SimpleNamespace(name="weights", output=["weights_stream"])
        self.node.op = _StreamingMemoryOp()

    def get_nodes_by_op_type(self, op_type):
        assert op_type == "StreamingMemory"
        return [self.node]

    def get_metadata_prop(self, name):
        assert name == "board_name"
        return "test"


def test_allocation_reaches_a_fixed_point(monkeypatch):
    model = _Model()
    monkeypatch.setattr(allocate_uram, "getCustomOp", lambda node: node.op)
    monkeypatch.setattr(
        allocate_uram,
        "require_tensor_type",
        lambda model, tensor_name: SimpleNamespace(bitwidth=8),
    )
    monkeypatch.setattr(
        allocate_uram,
        "read_board_info",
        lambda board: {"uram": 1},
    )

    _, modified = AllocateStreamingMemoryURAM().apply(model)

    assert modified
    assert model.node.op.attrs["storage_impl"] == "uram"
    _, modified = AllocateStreamingMemoryURAM().apply(model)
    assert not modified
