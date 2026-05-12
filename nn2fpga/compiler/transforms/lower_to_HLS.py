import copy
import logging
import os

import numpy as np
from onnx import TensorProto, helper, StringStringEntryProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import qonnx_make_model
from tabulate import tabulate

from nn2fpga.compiler.core.tensor_fifo import (
    TensorFifo,
    get_custom_tensor_fifo_metadata,
    set_custom_tensor_fifo_metadata,
)
from nn2fpga.compiler.core.tensor_type import get_custom_tensor_datatype, set_custom_tensor_datatype, TensorType
from nn2fpga.compiler.core.tensor_layout import get_custom_tensor_layout, set_custom_tensor_layout
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.transforms.compute_fifo_depth import ComputeFifoDepth
from nn2fpga.compiler.transforms.embed_hls_code import EmbedHLSCode
from nn2fpga.compiler.transforms.generate_bitstream import GenerateBitstream
from nn2fpga.compiler.core.hls_schedule_parser import VitisHlsReportParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model_II(model: ModelWrapper) -> int:
    model_II = model.get_metadata_prop("model_II")
    if model_II is None:
        raise ValueError("model_II metadata property not set on model.")
    return int(model_II)


def _compute_word_count(model: ModelWrapper, tensor_name: str, axi_word: int) -> int:
    tensor_type = get_custom_tensor_datatype(model, tensor_name)
    if tensor_type is None:
        raise ValueError(f"Tensor type for '{tensor_name}' not found in model.")
    data_per_word = axi_word // int(tensor_type.bitwidth)
    shape = model.get_tensor_shape(tensor_name)
    return int(np.ceil(np.prod(shape) / data_per_word))

def _insert_input_throttlers(model: ModelWrapper, ap: AcceleratorPackage) -> ModelWrapper:
    """Replace each graph input with a FixedThroughputDMA source node."""
    model_II = _get_model_II(model)
    for node in model.get_nodes_by_op_type("AXIToStream"):
        input_name = node.input[0]
        axi_word = getCustomOp(node).get_nodeattr("axi_bitwidth")
        words_per_tensor = _compute_word_count(model, input_name, axi_word)
        input_shape = model.get_tensor_shape(input_name)
        input_layout = get_custom_tensor_layout(model, input_name)
        input_type = get_custom_tensor_datatype(model, input_name)

        # Stop the FixedThroughputDMA from producing data if the input is a constant
        # since during STE constant inputs are not read to optimize simulation time.
        is_constant_input = False
        for v in ap.input_map.values():
            if v['new_name'] == input_name and v['value'] is not None:
                is_constant_input = True
                break
        if is_constant_input:
            words_per_tensor = 0

        to_remove = [i for i in model.graph.input if i.name == input_name]
        for i in to_remove:
            model.graph.input.remove(i)

        new_tensor_name = f"{input_name}_0_" # Suffix to respect stream array naming convention
        model.graph.node.insert(0, helper.make_node(
            "FixedThroughputDMA",
            inputs=[],
            outputs=[new_tensor_name],
            name=f"FixedThroughputDMA_{input_name}",
            domain="nn2fpga.compiler.custom_op",
            words_per_tensor=words_per_tensor,
            axi_bitwidth=axi_word,
            model_II=model_II,
            in_stream_array=1, out_stream_array=1,
            in_word_array=1, out_word_array=1,
        ))
        node.input[0] = new_tensor_name

        model.set_tensor_shape(new_tensor_name, input_shape)
        set_custom_tensor_layout(model, new_tensor_name, input_layout)
        set_custom_tensor_datatype(model, new_tensor_name, input_type)
    return model


def _insert_output_sinks(model: ModelWrapper) -> ModelWrapper:
    """Replace each graph output with an InfiniteThroughputDMA sink node."""
    for node in model.get_nodes_by_op_type("StreamToAXI"):
        output_name = node.output[0]
        axi_word = getCustomOp(node).get_nodeattr("axi_bitwidth")
        words_per_tensor = _compute_word_count(model, output_name, axi_word)
        output_shape = model.get_tensor_shape(output_name)
        output_layout = get_custom_tensor_layout(model, output_name)
        output_type = get_custom_tensor_datatype(model, output_name)
        to_remove = [o for o in model.graph.output if o.name == output_name]
        for o in to_remove:
            model.graph.output.remove(o)

        new_tensor_name = f"{output_name}_0_" # Suffix to respect stream array naming convention 
        model.graph.node.append(helper.make_node(
            "InfiniteThroughputDMA",
            inputs=[new_tensor_name],
            outputs=[],
            name=f"InfiniteThroughputDMA_{output_name}",
            domain="nn2fpga.compiler.custom_op",
            words_per_tensor=words_per_tensor,
            axi_bitwidth=axi_word,
            in_stream_array=1, out_stream_array=1,
            in_word_array=1, out_word_array=1,
        ))
        node.output[0] = new_tensor_name

        model.set_tensor_shape(new_tensor_name, output_shape)
        set_custom_tensor_layout(model, new_tensor_name, output_layout)
        set_custom_tensor_datatype(model, new_tensor_name, output_type)
    return model


def _lower_nodes_to_hls(
    model: ModelWrapper,
) -> tuple[list, list, dict, list, list]:
    """Iterate nodes in topological order and lower each to HLS sub-nodes.

    Returns:
        nodes:        flattened HLS sub-nodes
        inits:        initializer tensors
        fifo:         mapping tensor_name -> TensorFifo
        tensors:      value_info entries for all fifo streams
        report_lines: one row per original node for the resource report
    """
    nodes, inits = [], []
    fifo: dict = {}
    report_lines = []
    hls_tag = 0

    for node in model.graph.node:
        custom_op = getCustomOp(node)
        interface = custom_op.get_port_interface()
        report_lines.append([
            node.name,
            hls_tag,
            custom_op.get_latency(model),
            custom_op.get_brams(model),
            custom_op.get_dsps(model),
            interface.in_word_array,
            interface.in_stream_array,
            interface.out_word_array,
            interface.out_stream_array,
        ])
        sub_nodes, sub_inits, sub_fifo, hls_tag = custom_op.lower_to_hls(model, hls_tag)
        nodes.extend(sub_nodes)
        inits.extend(sub_inits)
        fifo.update(sub_fifo)

    tensors = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        for name in fifo
    ]
    return nodes, inits, fifo, tensors, report_lines

def _annotate_pipeline_depths(hls_model: ModelWrapper, ste_model: ModelWrapper, work_root: str, ste_already_done: bool = False) -> None:

    if not ste_already_done:
        # Create a dummy model with only the nn2FPGAPartition node, to generate the schedule.
        ap = AcceleratorPackage.from_json(hls_model.get_metadata_prop("accelerator_package"))
        inputs = []
        outputs = []
        inputs_names = []
        outputs_names = []
        for k, v in [(k, v) for k, v in ap.input_map.items() if v["value"] is None]:
            inputs.append(helper.make_tensor_value_info(k, TensorProto.FLOAT, v["shape"]))
            inputs_names.append(k)
            logger.info(f"Creating value info for input {k} with shape {v['shape']}")
        for k, v in [(k, v) for k, v in ap.output_map.items() if v["value"] is None]:
            outputs.append(helper.make_tensor_value_info(k, TensorProto.FLOAT, v["shape"]))
            outputs_names.append(k)
            logger.info(f"Creating value info for output {k} with shape {v['shape']}")

        nn2FPGA_node_copy = helper.make_node(
            "nn2fpgaPartition",
            inputs=inputs_names,
            outputs=outputs_names,
            name="nn2fpgaPartition_0",
            domain="nn2fpga.compiler.custom_op",
        )

        graph = helper.make_graph(
            nodes=[nn2FPGA_node_copy],
            name="schedule_graph",
            inputs=inputs,
            outputs=outputs,
        )

        schedule_model = qonnx_make_model(graph)
        schedule_model = ModelWrapper(schedule_model)

        # Build index of existing keys in dst
        dst_idx = {}
        for p in hls_model.model.metadata_props:
            if p.key in dst_idx:
                schedule_model.model.metadata_props[dst_idx[p.key]].value = p.value
            else:
                kv = StringStringEntryProto()
                kv.key = p.key
                kv.value = p.value
                schedule_model.model.metadata_props.append(kv)

        for input in schedule_model.graph.input:
            tensor_type = TensorType.from_canonical_name(ap.input_map[input.name]["quant"])
            set_custom_tensor_datatype(schedule_model, input.name, tensor_type)
        for output in schedule_model.graph.output:
            tensor_type = TensorType.from_canonical_name(ap.output_map[output.name]["quant"])
            set_custom_tensor_datatype(schedule_model, output.name, tensor_type)
        
        schedule_model = schedule_model.transform(EmbedHLSCode(hls_model=hls_model, erase=False, work_root=work_root))
        schedule_model = schedule_model.transform(GenerateBitstream(work_dir=work_root, erase=False, only_synthesize=True))
    
    # Back-annotate the STE model with pipeline depths.
    for node in ste_model.graph.node:
        custom_op = getCustomOp(node)
        hls_tag = custom_op.get_nodeattr("hls_tag")
        if float(ste_model.get_metadata_prop("hls_version")) > 2025:
            scheduling_report_file = os.path.join(work_root, f"vivado/hlsproj/hls/.autopilot/db/run_{hls_tag}ul_s.verbose.sched.rpt")
        else:
            scheduling_report_file = os.path.join(work_root, f"vivado/hlsproj/solution0/.autopilot/db/run_{hls_tag}ul_s.verbose.sched.rpt")
        if not os.path.exists(scheduling_report_file):
            logger.warning(f"Scheduling report file not found for node {node.name}. Skipping depth adjustment.")
            read_skew = 0
            write_skew = 0
            pipeline_stages = 1

        else:
            scheduling_parser = VitisHlsReportParser(scheduling_report_file)
            if not scheduling_parser.single_loop_function:
                logger.info(f"Node {node.name} is not single loop pipelined.")
                read_skew = 0
                write_skew = 0
                pipeline_stages = 1
                custom_op.set_nodeattr("read_skew", read_skew)
                custom_op.set_nodeattr("write_skew", write_skew)
                custom_op.set_nodeattr("pipeline_stages", pipeline_stages)
                continue

            read_skew = 0
            write_skew = 0
            max_read_state = 0
            min_read_state = scheduling_parser.pipeline_depth + 1
            max_write_state = 0
            min_write_state = scheduling_parser.pipeline_depth + 1
            write_op = False
            read_op = False
            for op in scheduling_parser.fifo_ops:
                sequential_state = scheduling_parser.pipeline_states.index(op["state"])
                if op["op_type"] == "read":
                    max_read_state = max(max_read_state, sequential_state)
                    min_read_state = min(min_read_state, sequential_state)
                    read_op = True
                elif op["op_type"] == "write":
                    max_write_state = max(max_write_state, sequential_state)
                    min_write_state = min(min_write_state, sequential_state)
                    write_op = True
            read_skew = max_read_state - min_read_state if read_op else 0
            write_skew = max_write_state - min_write_state if write_op else 0

            # Vitis HLS is able to optimize concurrent processes inside a single function.
            # Therefore, a state of the FSM is not monolithic, but can contain multiple unrelated processes.
            # This means that there could be an actual skew between read and write operations that are scheduled in the same state.
            # This could happen only to processes that can be logically divided into independent parts, such as
            # StreamingPad where each stream is indipendent from the others.
            if custom_op.get_nodeattr("original_op_type") in [
                "StreamingPad",
                "StreamingAdd",
                "BandwidthAdjustIncreaseStreams",
                "BandwidthAdjustDecreaseStreams",
                "TensorDuplicator",
            ]:
                if read_op:
                    read_skew += 1
                if write_op:
                    write_skew += 1
                logger.info(f"Node {node.name} is a {custom_op.get_nodeattr('original_op_type')} with possible independent processes. Incrementing read_skew to {read_skew} and write_skew to {write_skew}.")
            if write_op and read_op:
                pipeline_stages = max_write_state - min_read_state + 1
            else:
                pipeline_stages = 1
            if pipeline_stages < 1:
                logger.error(f"Node {node.name} has invalid pipeline stages: {pipeline_stages} because {max_write_state} - {min_read_state} + 1 < 1. Setting to 1.")

        custom_op.set_nodeattr("read_skew", read_skew)
        custom_op.set_nodeattr("write_skew", write_skew)
        custom_op.set_nodeattr("pipeline_stages", pipeline_stages)

def _build_hls_model(
    model: ModelWrapper,
    nodes: list,
    inits: list,
    fifo: dict,
    tensors: list,
) -> ModelWrapper:
    """Assemble a new ModelWrapper from lowered HLS nodes, copy metadata, and
    set FIFO metadata for all stream tensors."""
    graph = helper.make_graph(
        nodes,
        model.graph.name + "_to_hls",
        list(model.graph.input),
        list(model.graph.output),
        initializer=inits,
        value_info=tensors,
    )
    hls_model = ModelWrapper(qonnx_make_model(graph, producer_name="nn2fpga"))

    # Copy metadata props from source model
    dst = hls_model.model
    dst_idx = {p.key: i for i, p in enumerate(dst.metadata_props)}
    for p in model.model.metadata_props:
        if p.key in dst_idx:
            dst.metadata_props[dst_idx[p.key]].value = p.value
        else:
            kv = StringStringEntryProto()
            kv.key, kv.value = p.key, p.value
            dst.metadata_props.append(kv)

    # Set FIFO metadata for all interior stream tensors
    for stream in hls_model.graph.value_info:
        if stream.name not in fifo:
            raise ValueError(f"Stream '{stream.name}' missing FIFO metadata.")
        set_custom_tensor_fifo_metadata(hls_model, stream.name, fifo[stream.name])

    # Shape info is no longer valid after lowering to streams
    for v in hls_model.model.graph.input:  v.ClearField("type")
    for v in hls_model.model.graph.output: v.ClearField("type")

    return hls_model


def _set_boundary_fifo_metadata(
    model: ModelWrapper,
    hls_model: ModelWrapper,
) -> None:
    """Attach FIFO metadata to graph input/output boundary tensors.

    Uses the original pre-lowering model to resolve AXIToStream / StreamToAXI
    nodes and read their axi_bitwidth attributes.
    """
    for inp in hls_model.graph.input:
        consumer = model.find_consumer(inp.name)
        if consumer is None:
            raise ValueError(f"Input '{inp.name}' has no consumer.")
        if consumer.op_type != "AXIToStream":
            raise ValueError(f"Input '{inp.name}' consumer is not AXIToStream.")
        axi_word = getCustomOp(consumer).get_nodeattr("axi_bitwidth")
        set_custom_tensor_fifo_metadata(
            hls_model, inp.name,
            TensorFifo(depth=0, hls_type=f"ap_axiu<{axi_word}, 0, 0, 0>", n_array=1),
        )

    for out in hls_model.graph.output:
        producer = model.find_producer(out.name)
        if producer is None:
            raise ValueError(f"Output '{out.name}' has no producer.")
        if producer.op_type != "StreamToAXI":
            raise ValueError(f"Output '{out.name}' producer is not StreamToAXI.")
        axi_word = getCustomOp(producer).get_nodeattr("axi_bitwidth")
        set_custom_tensor_fifo_metadata(
            hls_model, out.name,
            TensorFifo(depth=0, hls_type=f"ap_axiu<{axi_word}, 0, 0, 0>", n_array=1),
        )


def _print_report(report_file: str, lines: list) -> None:
    """Format and write the per-layer resource report to disk."""
    header = [
        "Layer name", "HLS Tag", "Latency (cc)",
        "DSPs", "BRAMs",
        "In word", "In stream", "Out word", "Out stream",
    ]
    total_dsps = total_brams = 0
    rows = []
    for layer in lines:
        # indices: 0=name,1=tag,2=latency,3=brams,4=dsps,5=in_word,6=in_stream,7=out_word,8=out_stream
        brams, dsps = layer[3], layer[4]
        total_brams += brams
        total_dsps += dsps
        rows.append([layer[0], layer[1], layer[2], dsps, brams,
                     layer[5], layer[6], layer[7], layer[8]])

    footer = ["Totals", "", "", total_dsps, total_brams, "", "", "", ""]
    table = tabulate([header] + rows + [footer], headers="firstrow", tablefmt="grid")

    with open(report_file, "w+") as f:
        f.write(table)
        print("", file=f)


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------

class LowerToHLS(Transformation):

    def __init__(
        self,
        infer_fifo_depth: bool = False,
        ste_already_done: bool = False,
        optimize_fifo_storage: bool = False,
        prj_root: str = "/tmp",
    ):
        """Lower the model to HLS by expanding custom ops into HLS kernels and
        replacing tensors with HLS streams.

        Args:
            infer_fifo_depth:      Whether to infer FIFO depth via Self-Timed
                                   Execution simulation. If False, all FIFOs
                                   get depth 0.
            ste_already_done:      Whether the STE pass has already been run, 
                                   so that it can be skipped in the main apply() 
                                   flow and only used for depth annotation.
            optimize_fifo_storage: Whether to move the largest FIFOs to
                                   external memory to save on-chip resources.
            prj_root:              Path to the project root directory.
        """
        self.infer_fifo_depth = infer_fifo_depth
        self.optimize_fifo_storage = optimize_fifo_storage
        self.report_file = os.path.join(prj_root, "hls_report.txt")
        self.prj_root = prj_root

    def _lower_for_ste(self, model: ModelWrapper) -> ModelWrapper:
        """Lower a deep-copied model with DMA throttlers/sinks for Self-Timed
        Execution, then run FIFO depth inference."""
        ste_model = copy.deepcopy(model)

        # Insert throttler/sinks in place of graph inputs/outputs.
        ap = AcceleratorPackage.from_json(
            model.get_metadata_prop("accelerator_package")
        )
        ste_model = _insert_input_throttlers(ste_model, ap)
        ste_model = _insert_output_sinks(ste_model)
        ap.input_map = {}
        ap.output_map = {}
        ste_model.set_metadata_prop("accelerator_package", ap.to_json())
        nodes, inits, fifo, tensors, _ = _lower_nodes_to_hls(ste_model)
        ste_model = _build_hls_model(ste_model, nodes, inits, fifo, tensors)
        return ste_model

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        # --- Lower the real model -------------------------------------------
        nodes, inits, fifo, tensors, report_lines = _lower_nodes_to_hls(model)
        _print_report(self.report_file, report_lines)
        hls_model = _build_hls_model(model, nodes, inits, fifo, tensors)
        _set_boundary_fifo_metadata(model, hls_model)

        # --- Optional STE pass to infer FIFO depths -------------------------
        ste_model = None
        if self.infer_fifo_depth:
            ste_model = self._lower_for_ste(model)
            _annotate_pipeline_depths(
                hls_model,
                ste_model,
                os.path.join(self.prj_root, "depth-sim"),
                ste_already_done=self.ste_already_done,
            )
            ste_model = ste_model.transform(
                ComputeFifoDepth(
                    work_root=self.prj_root,
                    erase=False,
                    ste_already_done=self.ste_already_done,
                )
            )

        # --- Copy inferred FIFO depths back, skipping STE-only streams ------
        if ste_model is not None:
            hls_stream_names = {v.name for v in hls_model.graph.value_info}
            for stream_vi in ste_model.graph.value_info:
                if stream_vi.name not in hls_stream_names:
                    continue
                set_custom_tensor_fifo_metadata(
                    hls_model,
                    stream_vi.name,
                    get_custom_tensor_fifo_metadata(ste_model, stream_vi.name),
                )

        return hls_model, False
