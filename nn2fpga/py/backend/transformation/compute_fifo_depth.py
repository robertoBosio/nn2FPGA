import re
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from backend.core.tensor_fifo import (
    TensorFifo,
    get_custom_tensor_fifo_metadata,
    set_custom_tensor_fifo_metadata,
)
from backend.util.codegen_utils import cpp_function, cpp_object, cpp_variable, NewCodeWriter
from backend.util.board_util import read_board_info
from backend.core.acceleratorpackage import AcceleratorPackage
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

    # Include files from the nn2FPGA library
    nn2fpga_include_dir = "/workspace/NN2FPGA/nn2fpga/library/include"
    if os.path.isdir(nn2fpga_include_dir):
        for fname in os.listdir(nn2fpga_include_dir):
            if fname.endswith(".hpp"):
                cwr.include(fname)

    cwr.include("utils/CSDFG_utils.hpp")
    cwr.include("utils/DMA_utils.hpp")
    cwr.include("<fstream>")
    cwr.include("<unordered_map>")

    # Top function definition
    function = cpp_function(model.get_metadata_prop("top_name"), "void")

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
        
    known_stream_depth = set()
    for node in model.graph.node:
        if getCustomOp(node).get_nodeattr("original_op_type") == "StreamingMemory":
            for stream in list(node.input) + list(node.output):
                m = re.match(r"(.*)_(\d+)_$", stream)
                if m:
                    known_stream_depth.add(m.group(1))
    logger.info(f"Known stream depth for: {known_stream_depth}")

    for node in model.graph.node:
        custom_op = getCustomOp(node)

        # Declare the variables used in the custom operation.
        function.add_code(custom_op.get_nodeattr("hls_variable_declarations"))

        # Declare the object.
        function.add_code(custom_op.get_nodeattr("hls_object_declaration"))

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
    stream_sizes = cpp_variable("stream_max_size", primitive="size_t", value=[2] * stream_count) 
    function.add_code(stream_sizes.generate_initialization())

    # Declare the clock cycle counter.
    clock_cycle = cpp_variable("clock_cycle", primitive="size_t", value=0)
    function.add_code(clock_cycle.generate_initialization())
    actual_II = cpp_variable("actual_II", primitive="size_t", value=0)
    function.add_code(actual_II.generate_initialization())

    # Declare the CSDFGState and CSDFGStateHasher for the visited states.
    function.add_code("std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;")
    function.add_code("CSDFGState current_state;")

    # Write the while loop to process the data until all the processors are waiting.
    function.add_code("while (true) {")
    function.codewriter.indent()
    function.add_code("std::vector<ActorStatus> actor_statuses;")
    function.add_code("std::vector<size_t> channel_quantities;")
    function.add_code("ActorStatus actor_status;")

    for step_call in consumers_step_calls:
        function.add_code(f"actor_status = {step_call}")
        function.add_code("actor_statuses.push_back(actor_status);")

    # Execute a step for each node in the model in reverse order.
    # It must be done in reverse order to ensure that nodes cannot immediately consume the data produced by the previous node.
    for node in reversed(model.graph.node):
        function.add_code(f"actor_status = {getCustomOp(node).get_nodeattr('hls_step_call')};")
        if getCustomOp(node).get_nodeattr('original_op_type') != "StreamingMemory":
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
                if stream.name not in known_stream_depth:
                    function.add_code(f"stream_max_size[{iter}] = std::max<size_t>({stream.name}[{s}].size(), stream_max_size[{iter}]);")
                    function.add_code(f"channel_quantities.push_back({stream.name}[{s}].size());")
                iter += 1
        else:
            function.add_code(f"channel_quantities.push_back({stream.name}.size());")

    function.add_code("current_state = CSDFGState(actor_statuses, channel_quantities);")
    function.add_code("if (visited_states.find(current_state) != visited_states.end()) {")
    function.codewriter.indent()
    function.add_code("actual_II = clock_cycle - visited_states[current_state];")
    function.add_code("break;")
    function.codewriter.dedent()
    function.add_code("}")
    function.add_code("visited_states.emplace(current_state, clock_cycle);")
    function.add_code("clock_cycle++;")
    function.add_code(f"if (clock_cycle > {10 * model_II}) {{")
    function.codewriter.indent()
    function.add_code('std::cout << "Warning: Exceeded maximum clock cycles. The model might be deadlocked." << std::endl;')
    function.add_code("actual_II = 0;")
    function.add_code("break;")
    function.codewriter.dedent()
    function.add_code("}")
    function.codewriter.dedent()
    function.add_code("};")

    # Add the final code to save the json report.
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
    function.add_code("report_file << \"\t\\\"II\\\": \" << actual_II << \"\\n\";")
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
            'add_files fifo_depth.cpp -cflags " -I/workspace/NN2FPGA/nn2fpga/library/include"',
            'add_files -tb testbench.cpp -cflags "-I/workspace/NN2FPGA/nn2fpga/library/include"',
            'set_top "{top_name}"',
            'set_part {part_name}',
            'create_clock -period {t_clk}',
            'csim_design',
            'exit',
        ]
    )

    return "\n".join(lines).format(top_name=top_name, part_name=part_name, t_clk=t_clk)

def make_build_dir(work_dir: str) -> None:
    """Create the working directory for the simulation."""
    os.makedirs(work_dir, exist_ok=True)

class ComputeFifoDepth(Transformation):
    """Compute the FIFO depth for each node in the model."""
    
    def __init__(self, work_root: str = "/tmp", erase: bool = True):
        """
        Initializes the ComputeFifoDepth transformation.
        Args:
            work_root (str): The root directory of the project.
            erase (bool): If True, the HLS project directory will be erased after the simulation.
        """
        super().__init__()
        self.work_root = f"{work_root}/depth-sim"
        self.erase = erase

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Compute the FIFO depth for each node in the model."""

        # Create the working directory for the simulation.
        make_build_dir(self.work_root)

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
        subprocess.run(
            ["vitis_hls", "-f", f"{self.work_root}/setup.tcl"],
            cwd=self.work_root,
            check=True
        )

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
        
        # Store the FIFO depth in the model metadata.
        for stream_name, depth in fifo_depths.items():
            current_meta = get_custom_tensor_fifo_metadata(model, stream_name)
            current_meta.depth = depth
            set_custom_tensor_fifo_metadata(model, stream_name, current_meta)
        
        # Optionally erase the working directory.
        if self.erase:
            subprocess.run(["rm", "-rf", self.work_root], check=True)

        return model, False
