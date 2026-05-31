#!/usr/bin/env python3
"""Generate a CPU -> custom op -> CPU ONNX graph for allocator probing."""

import argparse

import onnx
from onnx import TensorProto, helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Path to write allocator_probe.onnx")
    args = parser.parse_args()

    shape = [1, 3, 16, 16]
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

    numel = 1
    for dim in shape:
        numel *= dim
    add_const = helper.make_tensor("add_const", TensorProto.FLOAT, shape, [2.0] * numel)
    mul_const = helper.make_tensor("mul_const", TensorProto.FLOAT, shape, [3.0] * numel)

    nodes = [
        helper.make_node("Add", ["input", "add_const"], ["added"], name="add"),
        helper.make_node(
            "AllocatorProbe",
            ["added"],
            ["probed"],
            name="allocator_probe",
            domain="ai.nn2FPGA.test",
        ),
        helper.make_node("Mul", ["probed", "mul_const"], ["output"], name="mul"),
    ]

    graph = helper.make_graph(
        nodes,
        "allocator_probe_graph",
        [inp],
        [out],
        initializer=[add_const, mul_const],
    )
    model = helper.make_model(
        graph,
        producer_name="nn2fpga_allocator_probe_test",
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("ai.nn2FPGA.test", 1),
        ],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
