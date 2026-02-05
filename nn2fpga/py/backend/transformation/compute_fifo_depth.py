from copy import deepcopy
import re
from onnx import TensorProto, helper, StringStringEntryProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from backend.core.tensor_fifo import (
    TensorFifo,
    get_custom_tensor_fifo_metadata,
    set_custom_tensor_fifo_metadata,
)
from backend.core.hls_schedule_parser import VitisHlsReportParser
from backend.core.tensor_quant import TensorQuant, set_custom_tensor_datatype
from qonnx.util.basic import qonnx_make_model
from backend.util.codegen_utils import cpp_function, cpp_object, cpp_variable, NewCodeWriter
from backend.util.board_util import read_board_info
from backend.core.acceleratorpackage import AcceleratorPackage
from backend.transformation.embed_hls_code import EmbedHLSCode
from backend.transformation.generate_bitstream import GenerateBitstream
import os
import json
import subprocess
import logging
logger = logging.getLogger(__name__)

def generate_hls_code(model: ModelWrapper, work_root: str) -> str:
    """ Generate the HLS code to execute the model in fifo-depth mode. """

    # Retrieve model II
    model_II = int(model.get_metadata_prop("model_II"))
    ap = AcceleratorPackage.from_json(model.get_metadata_prop("accelerator_package"))
    constant_inputs = [value['new_name'] for value in ap.input_map.values() if value['value'] is not None]

    cwr = NewCodeWriter()
    cwr.add_autogen_comment()

    # Include sections for HLS
    cwr.include("ap_int.h")
    cwr.include("hls_stream.h")
    cwr.include("hls_vector.h")
    cwr.include("ap_axi_sdata.h")
    cwr.include("<chrono>")

    # Include files from the nn2FPGA library
    nn2fpga_include_dir = "/workspace/NN2FPGA/nn2fpga/library/include"
    if os.path.isdir(nn2fpga_include_dir):
        for fname in os.listdir(nn2fpga_include_dir):
            if fname.endswith(".hpp"):
                cwr.include(fname)

    cwr.include("utils/CSDFG_utils.hpp")
    cwr.include("utils/DMA_utils.hpp")
    cwr.include("<fstream>")
    cwr.include("<iostream>")
    cwr.include("<unordered_map>")

    # Top function definition
    function = cpp_function(model.get_metadata_prop("top_name"), "void")
    
    # Record start time
    function.add_code("auto start_time = std::chrono::high_resolution_clock::now();")

    stream_vars = []
    stream_count = len(model.graph.value_info)
    for fifo in model.graph.value_info:

        # Declare the array of streams only for the first stream of the array.
        if fifo.name.endswith("_0_"):
            trimmed_name = fifo.name[:-3]  # Remove the _0_ suffix 
            tensor_fifo = get_custom_tensor_fifo_metadata(model, fifo.name)
            var = cpp_variable(
                trimmed_name,
                f"hls::stream<{tensor_fifo.hls_type}>",
                array=tensor_fifo.n_array,
            )
            stream_vars.append((var, True))

    for node in model.graph.node:
        custom_op = getCustomOp(node)

        # Declare the variables used in the custom operation.
        function.add_code(custom_op.get_nodeattr("hls_variable_declarations"))

        # Declare the object.
        function.add_code(custom_op.get_nodeattr("hls_object_declaration"))

        # Set the pipeline depth
        if "StreamingWindowSelector" in custom_op.get_nodeattr("original_op_type"):
            tensor_fifo = get_custom_tensor_fifo_metadata(model, node.input[0])
            size = tensor_fifo.depth
            size = max(size, 1)
            function.add_code(f"{custom_op.get_nodeattr('hls_object_name')}.step_init({custom_op.get_nodeattr('pipeline_stages')}, {size});")
        else:
            function.add_code(f"{custom_op.get_nodeattr('hls_object_name')}.step_init({custom_op.get_nodeattr('pipeline_stages')});")

    # Declare the output streams.
    consumers_step_calls = []
    for output in model.graph.output:
        tensor_fifo = get_custom_tensor_fifo_metadata(model, output.name)
        func = cpp_object(
            class_name="InfiniteThroughputDMA",
            obj_name=f"InfiniteThroughputDMA_{output.name}",
            template_args=[tensor_fifo.hls_type],
            constructor_args=[tensor_fifo.n_array],
        )
        function.add_code(func.generate_declaration())
        consumers_step_calls.append(
            f"InfiniteThroughputDMA_{output.name}.step({output.name});"
        )
        var = cpp_variable(
            output.name,
            f"hls::stream<{tensor_fifo.hls_type}>",
        )
        stream_vars.append((var, False))

    producers_step_calls = []
    for input in model.graph.input:
        tensor_fifo = get_custom_tensor_fifo_metadata(model, input.name)
        var = cpp_variable(
            input.name,
            f"hls::stream<{tensor_fifo.hls_type}>",
        )
        stream_vars.append((var, False))
        if input.name in constant_inputs:
            continue

        func = cpp_object(
            class_name="FixedThroughputDMA",
            obj_name=f"FixedThroughputDMA_{input.name}",
            template_args=[tensor_fifo.hls_type],
            constructor_args=[tensor_fifo.n_array, model_II]
        )
        function.add_code(func.generate_declaration())
        producers_step_calls.append(
            f"FixedThroughputDMA_{input.name}.step({input.name});"
        )

    # Declare all the internal streams.
    for stream, _ in stream_vars:
        function.add_code(f"{stream.generate_declaration()};")

    # Declare the array of streams sizes.
    stream_sizes = cpp_variable("stream_max_size", primitive="size_t", value=[1] * stream_count) 
    function.add_code(stream_sizes.generate_initialization())

    # Declare the clock cycle counter.
    clock_cycle = cpp_variable("clock_cycle", primitive="size_t", value=0)
    function.add_code(clock_cycle.generate_initialization())
    actual_II = cpp_variable("actual_II", primitive="size_t", value=0)
    function.add_code(actual_II.generate_initialization())

    # Declare the CSDFGState and CSDFGStateHasher for the visited states.
    function.add_code("std::unordered_map<StateSig, std::vector<StateRef>> visited_states;")
    function.add_code("CSDFGState current_state;")

    # Allocating correct size for the actor statuses.
    num_consumers = len(consumers_step_calls)
    num_producers = len(producers_step_calls)
    num_actors = num_consumers + num_producers
    function.add_code("std::vector<ActorStatus> actor_statuses;")
    function.add_code(f"actor_statuses.reserve({num_actors});")

    # Allocating correct size for the channel quantities.
    num_channels = sum([stream.array if stream.array is not None else 1 for stream, _ in stream_vars])
    function.add_code("std::vector<size_t> channel_quantities;")
    function.add_code(f"channel_quantities.reserve({num_channels});")

    # Start of the simulation loop.
    function.add_code("while (true) {")
    function.codewriter.indent()
    function.add_code("ActorStatus actor_status;")
    function.add_code("actor_statuses.clear();")
    function.add_code("channel_quantities.clear();")

    for step_call in consumers_step_calls:
        function.add_code(f"actor_status = {step_call}")
        function.add_code("actor_statuses.push_back(actor_status);")

    # Execute a step for each node in the model in reverse order.
    # It must be done in reverse order to ensure that nodes cannot immediately consume the data produced by the previous node.
    for node in reversed(model.graph.node):
        function.add_code(f"actor_status = {getCustomOp(node).get_nodeattr('hls_step_call')};")
        function.add_code("actor_statuses.push_back(actor_status);")

    # Execute a step for each input producer.
    for step_call in producers_step_calls:
        function.add_code(f"actor_status = {step_call}")
        function.add_code("actor_statuses.push_back(actor_status);")

    # Update the fifo max size for each stream.
    iter = 0
    for stream, is_internal in stream_vars:
        if is_internal:
            for s in range(stream.array):
                function.add_code(f"stream_max_size[{iter}] = std::max<size_t>({stream.name}[{s}].size(), stream_max_size[{iter}]);")
                function.add_code(f"channel_quantities.push_back({stream.name}[{s}].size());")
                iter += 1
        else:
            function.add_code(f"channel_quantities.push_back({stream.name}.size());")

    function.add_code("current_state = CSDFGState(actor_statuses, channel_quantities);")
    function.add_code("CompactState compact = make_compact_state(current_state);")
    function.add_code("StateSig sig = make_signature(compact);")
    function.add_code("auto &bucket = visited_states[sig];")
    function.add_code("for (const auto &ref : bucket) {")
    function.codewriter.indent()
    function.add_code("if (states_equal_on_disk(ref.offset, compact.data)) {")
    function.codewriter.indent()
    function.add_code("actual_II = clock_cycle - ref.clock;")
    function.add_code("goto done_simulation;")
    function.codewriter.dedent()
    function.add_code("}")
    function.codewriter.dedent()
    function.add_code("}")
    function.add_code("uint64_t offset = append_state_to_file(compact.data);")
    function.add_code("bucket.push_back(StateRef{offset, static_cast<uint32_t>(clock_cycle)});")
    function.add_code("clock_cycle++;")
    function.add_code(f"if (clock_cycle > {10 * model_II}) {{")
    function.codewriter.indent()
    function.add_code('std::cout << "Warning: Exceeded maximum clock cycles. The model might be deadlocked." << std::endl;')
    function.add_code("actual_II = 0;")
    function.add_code("break;")
    function.codewriter.dedent()
    function.add_code("}")
    function.add_code(f"if (clock_cycle % 100000 == 0) {{")
    function.codewriter.indent()
    function.add_code('std::cout << "Current clock cycle: " << clock_cycle << std::endl;')
    function.codewriter.dedent()
    function.add_code("}")
    function.codewriter.dedent()
    function.add_code("};")

    # Add the final code to save the json report.
    function.add_code("done_simulation:")
    function.add_code("auto end_time = std::chrono::high_resolution_clock::now();")
    function.add_code("auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();")
    function.add_code(f"std::ofstream report_file(\"{work_root}/fifo_depth.json\");")
    function.add_code("report_file << \"{\\n\";")
    function.add_code("report_file << \"\t\\\"fifo_depth\\\": {\\n\";")

    # Write the fifo depth for each stream.
    iter = 0
    for stream in [s for s, is_internal in stream_vars if is_internal]:
        for s in range(stream.array):
            if iter < stream_count - 1:
                function.add_code(f'report_file << "\t\t\\\"{stream.name}_{s}_\\\": " << stream_max_size[{iter}] << ",\\n";')
            else:
                function.add_code(f'report_file << "\t\t\\\"{stream.name}_{s}_\\\": " << stream_max_size[{iter}] << "\\n";')
            iter += 1

    function.add_code("report_file << \"\t},\\n\";")
    function.add_code("report_file << \"\t\\\"Simulation cycles\\\": \" << clock_cycle << \",\\n\";")
    function.add_code("report_file << \"\t\\\"II\\\": \" << actual_II << \",\\n\";")
    function.add_code("report_file << \"\t\\\"Simulation time (ms)\\\": \" << duration << \"\\n\";")
    function.add_code("report_file << \"}\\n\";")
    function.add_code("report_file.close();")
    cwr.add_function_definition(function)
    return cwr.code

def generate_hls_driver(model: ModelWrapper) -> str:
    """Generate HLS driver code for the given model.
    Args:
        model (ModelWrapper): The model to generate HLS driver code for.
    Returns:
        str: The generated HLS driver code as a string.
    """
    cwr = NewCodeWriter()
    cwr.add_autogen_comment()

    # Include sections for HLS
    cwr.include("ap_int.h")
    cwr.include("hls_stream.h")
    cwr.include("hls_vector.h")
    cwr.include("ap_axi_sdata.h")

    # Accelerator kernel function definition
    kernel_function = cpp_function(
        name=model.get_metadata_prop("top_name"),
        return_type="void",
        qualifiers=["extern"],
    )

    # Add the function prototype, which will be called from the main function.
    cwr.add_function_prototype(kernel_function)
    
    # Main testbench function definition
    main_function = cpp_function(
        name="main",
        return_type="int",
        arguments=[cpp_variable("argc", "int"), cpp_variable("argv", "char**")],
    )

    # Add the kernel function call
    main_function.add_code(f"{kernel_function.generate_call()};")

    main_function.add_code("return 0;")
    cwr.add_function_definition(main_function)
    return cwr.code

def generate_tcl_script(top_name, part_name, frequency, hls_version):
    """Dump a TCL script to set up the HLS project and run the simulation."""

    t_clk = f"{1e3 / int(frequency):.2f}ns" # Convert frequency in MHz to clock period in ns
    lines = list()
    lines.append("# Auto-generated TCL script for HLS project setup")
    lines.append("# Generated by nn2FPGA simulation flow.")
    lines.append("")

    # Check the HLS version to determine the correct syntax
    if float(hls_version) > 2025:
        lines.append(
            'open_component -reset "proj_{top_name}" -flow_target vivado',
        )
    else:
        lines.extend(
            [
                'open_project -reset "proj_{top_name}"',
                'open_solution -reset solution0',
            ]
        )

    lines.extend(
        [
            'add_files fifo_depth.cpp -cflags "-O2 -I/workspace/NN2FPGA/nn2fpga/library/include"',
            'add_files -tb testbench.cpp -cflags "-I/workspace/NN2FPGA/nn2fpga/library/include"',
            'set_top "{top_name}"',
            'set_part {part_name}',
            'create_clock -period {t_clk}',
            'csim_design',
            'exit',
        ]
    )

    return "\n".join(lines).format(top_name=top_name, part_name=part_name, t_clk=t_clk)

def generate_schedule(model: ModelWrapper, work_root: str):

    # Create a dummy model with only the nn2FPGAPartition node, to generate the schedule.
    ap = AcceleratorPackage.from_json(model.get_metadata_prop("accelerator_package"))
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
        domain="backend.custom_op",
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
    for p in model.model.metadata_props:
        if p.key in dst_idx:
            schedule_model.model.metadata_props[dst_idx[p.key]].value = p.value
        else:
            kv = StringStringEntryProto()
            kv.key = p.key
            kv.value = p.value
            schedule_model.model.metadata_props.append(kv)

    for input in schedule_model.graph.input:
        tensor_quant = TensorQuant.from_canonical_name(ap.input_map[input.name]["quant"])
        set_custom_tensor_datatype(schedule_model, input.name, tensor_quant)
    for output in schedule_model.graph.output:
        tensor_quant = TensorQuant.from_canonical_name(ap.output_map[output.name]["quant"])
        set_custom_tensor_datatype(schedule_model, output.name, tensor_quant)

    schedule_model = schedule_model.transform(EmbedHLSCode(nn2fpga_model=model, erase=False, work_root=work_root))
    schedule_model = schedule_model.transform(GenerateBitstream(work_dir=work_root, erase=False, only_synthesize=True))


def adjust_depth_based_on_scheduling(model: ModelWrapper, fifo_depths: dict, work_root: str) -> dict:
    """Adjust the FIFO depth based on the scheduling information."""
    
    # Extract the scheduling information from the HLS report for each node in the original model.
    for node in model.graph.node:
        custom_op = getCustomOp(node)
        hls_tag = custom_op.get_nodeattr("hls_tag")
        if float(model.get_metadata_prop("hls_version")) > 2025:
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

    for stream_name in fifo_depths.keys():
        producer = model.find_producer(stream_name)
        consumer = model.find_consumer(stream_name)
        if producer is None or consumer is None:
            logger.warning(f"Could not find producer or consumer for stream {stream_name}. Skipping depth adjustment.")
            continue

        read_skew = getCustomOp(consumer).get_nodeattr("read_skew")
        write_skew = getCustomOp(producer).get_nodeattr("write_skew")
        fifo_depths[stream_name] += read_skew + write_skew
    
    return fifo_depths

def make_build_dir(work_dir: str) -> None:
    """Create the working directory for the simulation."""
    os.makedirs(work_dir, exist_ok=True)

class ComputeFifoDepth(Transformation):
    """Compute the FIFO depth for each node in the model."""
    
    def __init__(self, work_root: str = "/tmp", erase: bool = True, ste_already_done: bool = False):
        """
        Initializes the ComputeFifoDepth transformation.
        Args:
            work_root (str): The root directory of the project.
            erase (bool): If True, the HLS project directory will be erased after the simulation.
            ste_already_done (bool): If True, skip the synthesis step and the ste simulation.
        """
        super().__init__()
        self.work_root = f"{work_root}/depth-sim"
        self.erase = erase
        self.ste_already_done = ste_already_done

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Compute the FIFO depth for each node in the model."""

        if not self.ste_already_done:

            # Create the working directory for the simulation.
            make_build_dir(self.work_root)

            # Schedule the model to retrieve information about the pipelines.
            generate_schedule(model, self.work_root)

            with open(os.path.join(self.work_root, "fifo_depth.cpp"), "w") as f:
                f.write(generate_hls_code(model, self.work_root))
            
            # Write the driver code.
            with open(os.path.join(self.work_root, "testbench.cpp"), "w") as f:
                f.write(generate_hls_driver(model))

            # Generate the TCL script for the HLS project.
            part_name = read_board_info(
                board=model.get_metadata_prop("board_name"),
            )['part']
            tcl_script = generate_tcl_script(
                top_name=model.get_metadata_prop("top_name"),
                part_name=part_name,
                frequency=model.get_metadata_prop("frequency"),
                hls_version=model.get_metadata_prop("hls_version"),
            )
            with open(os.path.join(self.work_root, "setup.tcl"), "w") as f:
                f.write(tcl_script)

            # run the simulation
            if float(model.get_metadata_prop("hls_version")) > 2025:
                subprocess.run(
                    ["vitis-run", "--mode", "hls", "--tcl", f"{self.work_root}/setup.tcl"],
                    cwd=self.work_root,
                    check=True
                )
            else:
                subprocess.run(
                    ["vitis_hls", "-f", f"{self.work_root}/setup.tcl"],
                    cwd=self.work_root,
                    check=True
                )
                
        else:
            logger.info("Skipping synthesis and STE simulation as ste_already_done is set to True.")

        # Read the fifo depth from the generated json file.
        fifo_depth_file = os.path.join(self.work_root, "fifo_depth.json")
        if not os.path.exists(fifo_depth_file):
            raise FileNotFoundError(f"FIFO depth file not found: {fifo_depth_file}")
        
        fifo_depth_data = {}
        with open(fifo_depth_file, "r") as f:
            fifo_depth_data = json.load(f)
        
        fifo_depths = fifo_depth_data.get("fifo_depth", {})
        if not fifo_depths:
            raise ValueError("No FIFO depth data found in the generated file.")

        # Adjust the FIFO depths based on scheduling information.
        fifo_depths = adjust_depth_based_on_scheduling(model, fifo_depths, self.work_root)
        
        # Store the FIFO depth in the model metadata.
        for stream_name, depth in fifo_depths.items():
            current_meta = get_custom_tensor_fifo_metadata(model, stream_name)
            if current_meta is None:
                logger.warning(f"No existing FIFO metadata found for stream {stream_name}. Skipping update.")
                continue
            if current_meta.depth == 0:
                current_meta.depth = depth + 1
                set_custom_tensor_fifo_metadata(model, stream_name, current_meta)

        # Dump fifo depths for debugging.
        with open(os.path.join(self.work_root, "fifo_depths.csv"), "w") as f:
            f.write("stream_name,depth\n")
            for stream_name, depth in fifo_depths.items():
                current_meta = get_custom_tensor_fifo_metadata(model, stream_name)
                if current_meta is None:
                    logger.warning(f"No existing FIFO metadata found for stream {stream_name}. Skipping writing to CSV.")
                    continue
                f.write(f"{stream_name},{current_meta.depth}\n")
        
        # Optionally erase the working directory.
        if self.erase:
            subprocess.run(["rm", "-rf", self.work_root], check=True)

        return model, False
