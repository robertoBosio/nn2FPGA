import pulp
import math
import time
import numpy as np
from pulp.apis import PULP_CBC_CMD
from tabulate import tabulate
from collections import deque
from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from backend.core.tensor_quant import TensorQuant
from backend.util.par_utils import get_par_attributes, set_par_attributes, check_par_attributes
from backend.util.board_util import read_board_info
from backend.transformation.insert_streaming_line_buffer import has_streaming_linebuffer
from backend.core.tensor_quant import get_custom_tensor_datatype
from onnx import NodeProto
from qonnx.custom_op.registry import getCustomOp
import logging
logger = logging.getLogger(__name__)

PARALLELIZABLE_LAYERS = [
    "StreamingConv",
    "StreamingDepthwiseConv",
    "StreamingGlobalAveragePool",
    "StreamingGlobalMaxPool",
    "AveragePool",
    "MaxPool",
    "NHWCToStream",
    "StreamToNHWC",
]

def packing_feature(operands_bitwidth, par, silvia_packing):
    """ Returns the number of operation that can be packed in a single DSP. 
    
    Arguments:
        operand_bitwidth: Tuple containing information about the bitwidth of the operands.
        par: Tuple containing the parallelization chosen for the layer in the format (och, ich, ow).

    Returns:
        int: The number of operations that can be packed in a single DSP.
        tuple: The packing for each dimension.
    """

    operand_bits = max(operands_bitwidth)
    if (operand_bits == 8):
        if (par[2] % 2 == 0):
            return 2, (1, 2)
        elif (par[0] % 2 == 0):
            return 2, (2, 1)
    elif (operand_bits == 4):
        if (silvia_packing):
            if (par[0] % 4 == 0):
                return 4, (4, 1)
            if (par[2] % 4 == 0):
                return 4, (1, 4)
        else:
            if (par[0] % 2 == 0 and par[2] % 2 == 0):
                return 4, (2, 2)
        if (par[2] % 2 == 0):
            return 2, (1, 2)
        elif (par[0] % 2 == 0):
            return 2, (2, 1)
    return 1, (1, 1)

def compute_bram_layer(weight_bits, weight_number, parallelism):
    """Compute the number of BRAMs needed to store the weights, given the parallelism """

    bram9 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=9)
    bram18 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=18)
    bram36 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=36)
    bram72 = bram_consumption(weight_bits, weight_number, parallelism, WIDTH=72)
    
    return min(bram9, bram18, bram36, bram72)

def bram_consumption(weight_bits, weight_number, parallelism, WIDTH=36):
    """Compute the number of BRAMs needed to store the weights, given the parallelism """

    # Useful space in BRAM18. Each BRAM18 is 18kb with a maximum word width of
    # 36 bits, in which 4 bits are reserved to ECC code
    SIZE_BRAM18 = (18 * 1024)
    
    # Useful space in BRAM36, composed by two BRAM18.
    SIZE_BRAM36 = SIZE_BRAM18 * 2

    WIDTH_BRAM36 = WIDTH

    # Assuming is implemented using LUTRAM
    if (weight_number * weight_bits) <= SIZE_BRAM36:
        return 0
    
    very_long_word = parallelism * weight_bits
    mem_width = very_long_word // WIDTH_BRAM36
    mem_width_rem = very_long_word % WIDTH_BRAM36
    word_depth = weight_number // parallelism
    mem_depth = int(math.ceil(word_depth / (SIZE_BRAM36 // WIDTH_BRAM36)))
    tot_bram = mem_width * mem_depth

    rem_bram = 0
    if (mem_width_rem > 36):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 72)))
    elif (mem_width_rem > 18 and mem_width_rem <= 36):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 36)))
    elif (mem_width_rem > 8 and mem_width_rem <= 18):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 18)))
    elif (mem_width_rem > 0 and mem_width_rem <= 8):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 9)))
    
    tot_bram += rem_bram
    
    return tot_bram

def dsp_consumption(layer, parallelism, silvia_packing):
    """Compute the number of DSPs needed to implement the layer, given the parallelism and the packing."""

    if (layer["depth"]):
        # If the layer is depthwise-like, we have one less loop.
        parallelism = (1, parallelism[1], parallelism[2])
    
    if (layer["type"] in ["StreamingConv", "StreamingDepthwiseConv"]):
        # For Conv layers, the unrolling is done over the output width, output channels and input channels.
        # The DSPs considered are coming from the MAC operation, considering the packing.
        op_per_dsp, _ = packing_feature((layer["weight_bits"], layer["act_bits"]), parallelism, silvia_packing)
        dsp_used = (np.prod(parallelism) * layer["kernel"]) / op_per_dsp

    elif (layer["type"] in ["StreamingGlobalAveragePool", "AveragePool"]):
        # StreamingGlobalAveragePool and AveragePool are unrolled over the output width and input channels.
        # The DSPs considered are coming from the division operation. Each single integer division requires 1 DSPs.
        dsp_used = (np.prod(parallelism))

    else:
        # All the other layers do not involve DSPs, so they are not considered.
        dsp_used = 0

    return int(dsp_used)

def generate_valid_combinations(och, ich, w, och_clip=2**10, ich_clip=2**10, w_clip=2**10, op_clip=2**20, depth=False):
    """ Generate valid combinations of parallelization over input channels, output channels and width.
    
    Arguments:
        och: Number of output channels.
        ich: Number of input channels.
        w: Width.
        och_clip: Maximum number of output channels to consider for parallelization.
        ich_clip: Maximum number of input channels to consider for parallelization.
        w_clip: Maximum width to consider for parallelization.
        op_clip: Maximum number of operations to consider for parallelization.
        
    Returns:
        list: A list of tuples containing the valid combinations of parallelization.
        """
    
    combinations = []
    
    # If the layer is depthwise-like, we have one less loop.
    if depth:
        och = 1

    def divisors(n, clip):
        return [i for i in range(1, n + 1) if (n % i == 0 and i <= clip)]

    for div_och in divisors(och, och_clip):
        for div_ich in divisors(ich, ich_clip):
            for div_w in divisors(w, w_clip):

                # Condition on DSPs.
                if (div_och * div_ich * div_w > op_clip):
                    continue

                # Condition on maximum stream width.
                if (div_och * div_ich > 512):
                    continue

                # If the layer is depthwise-like, output and input channels parallelization are the same thing.
                if depth:
                    combinations.append((div_ich, div_ich, div_w))
                else:
                    combinations.append((div_och, div_ich, div_w))
    return combinations 

def find_common_mult(a, b):
    """Return the least common multiple (LCM) of a and b."""
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def generate_architectures(
    layers_info: list, NUM_DSP: int, axi_bitwidth: int, silvia_packing: bool
) -> list:
    """Given a list of layers, generate all the valid parallelization for each layer."""

    valid_par_solutions = []
    for layer in layers_info:
        iw_clip = 20  # Heuristic clip for input width
        och_clip = 20  # Heuristic clip for output channels
        max_och_par = layer["och"]
        max_ich_par = layer["ich"]
        max_w_par = layer["ow"]
        valid_par_solutions_layer = []

        if layer["type"] == "StreamingConv":

            # Clipping the maximum parallelization to the available DSPs, since it is not
            # possible that one layer uses all the DSPs. Considering also the packing.
            op_per_dsp, _ = packing_feature(
                (layer["weight_bits"], layer["act_bits"]),
                (max_och_par, max_ich_par, max_w_par),
                silvia_packing,
            )
            op_clip = (NUM_DSP / layer["kernel"]) * op_per_dsp
            valid_par_solutions_layer = generate_valid_combinations(
                och=max_och_par,
                ich=max_ich_par,
                w=max_w_par,
                w_clip=iw_clip,
                och_clip=och_clip,
                op_clip=op_clip,
                depth=layer["depth"],
            )

        elif layer["type"] in ["StreamToNHWC", "NHWCToStream"]:

            # Clipping the maximum parallelization of StreamToNHWC and NHWCToStream to the bandwidth of the AXI bus.
            op_clip = axi_bitwidth // layer["act_bits"]

            valid_par_solutions_layer = generate_valid_combinations(
                och=max_och_par,
                ich=max_ich_par,
                w=max_w_par,
                w_clip=iw_clip,
                och_clip=och_clip,
                op_clip=op_clip,
                depth=layer["depth"],
            )

            # For NHWCToStream and StreamToNHWC, the output width can be bigger than 1 only if the channels are packed in a single word.
            temp_par_solutions = []
            for solution in valid_par_solutions_layer:
                if solution[2] > 1 and solution[0] != layer["och"]:
                    continue
                temp_par_solutions.append(solution)
            valid_par_solutions_layer = temp_par_solutions

        else:
            valid_par_solutions_layer = generate_valid_combinations(
                och=max_och_par,
                ich=max_ich_par,
                w=max_w_par,
                w_clip=iw_clip,
                och_clip=och_clip,
                op_clip=2**20,
                depth=layer["depth"],
            )
        
        valid_par_solutions.append(valid_par_solutions_layer)

    return valid_par_solutions

def layers_extractions(model: ModelWrapper) -> list:
    """ Extracts computation-intensive layers from the model and returns their useful information
    for balancing the computation load across the model.
    """ 

    layers_info = []
    for node in model.graph.node:
        if node.op_type in PARALLELIZABLE_LAYERS:

            input_shape = model.get_tensor_shape(node.input[0])
            output_shape = model.get_tensor_shape(node.output[0])
        

            # Ensure input and output shapes are 4D (NCHW format)
            input_shape = input_shape + [1] * (4 - len(input_shape))
            output_shape = output_shape + [1] * (4 - len(output_shape))

            if node.op_type == "StreamingConv":
                kernel_shape = get_by_name(node.attribute, "kernel_shape").ints
                kernel = int(math.prod(kernel_shape))
                group = getCustomOp(node).get_nodeattr("group")
                ops = (
                    output_shape[1]
                    * output_shape[2]
                    * output_shape[3]
                    * input_shape[1]
                    * kernel
                    // group
                )
                weight_bits = int(model.get_initializer(node.input[4]))
                act_bits = get_custom_tensor_datatype(model, node.input[0]).bitwidth
                depth = False

            elif node.op_type == "StreamingDepthwiseConv":
                kernel_shape = get_by_name(node.attribute, "kernel_shape").ints
                kernel = int(math.prod(kernel_shape))
                ops = (
                    output_shape[2]
                    * output_shape[3]
                    * input_shape[1]
                    * kernel
                )
                weight_bits = int(model.get_initializer(node.input[4]))
                act_bits = get_custom_tensor_datatype(model, node.input[0]).bitwidth
                depth = True

            elif node.op_type in ["StreamingGlobalAveragePool", "StreamingGlobalMaxPool"]:
                kernel_shape = (input_shape[2], input_shape[3])
                # kernel = int(math.prod(kernel_shape))
                kernel = 1
                depth = True
                ops = math.prod(input_shape)
                act_bits = get_custom_tensor_datatype(model, node.input[0]).bitwidth
                weight_bits = 0

            elif node.op_type in ["AveragePool", "MaxPool"]:
                kernel_shape = get_by_name(node.attribute, "kernel_shape").ints
                kernel = int(math.prod(kernel_shape))
                depth = True
                ops = math.prod(output_shape) * kernel
                act_bits = get_custom_tensor_datatype(model, node.input[0]).bitwidth
                weight_bits = 0

            elif node.op_type == "StreamToNHWC":
                kernel = 1
                depth = True
                ops = math.prod(output_shape)
                weight_bits = 0
                act_bits = get_custom_tensor_datatype(model, node.input[0]).bitwidth

            elif node.op_type == "NHWCToStream":
                kernel = 1
                depth = True
                ops = math.prod(input_shape)
                weight_bits = 0
                act_bits = get_custom_tensor_datatype(model, node.output[0]).bitwidth

            layers_info.append(
                {
                    "type": node.op_type,
                    "name": node.name,
                    "total": ops,
                    "kernel": kernel,
                    "weight_bits": weight_bits,
                    "act_bits": act_bits,
                    "ich": input_shape[1],
                    "ih": input_shape[2],
                    "iw": input_shape[3],
                    "och": output_shape[1],
                    "oh": output_shape[2],
                    "ow": output_shape[3],
                    "depth": depth,
                }
            )

    return layers_info

def parallelismILP(layers_info, valid_par_solutions, NUM_DSP, NUM_PORTS, silvia_packing, prj_root="/tmp"):
    """ Find the parallelization for each layer that maximize the throughput of the network."""

    constraints_counter = 0

    # valid_tot_par_solutions stores the total parallelization for each valid
    # solution and it is useful to use lpDot to compute the parallelization
    # chosen for a layer
    valid_tot_par_solutions = []
    for i, par_sol in enumerate(valid_par_solutions):
        valid_tot_par_solutions.append([])
        for single_par in par_sol:
            valid_tot_par_solutions[i].append(np.prod(single_par))

    # valid_iter_linebuffer stores the line buffer number of iteration for each valid
    # solution and it is useful to linearize the constraint of the line buffer
    valid_iter_linebuffer = []
    for par_sol, layer in zip(valid_par_solutions, layers_info):
        valid_iter_linebuffer.append([])
        layer_iter = layer["ich"] * layer["iw"] * layer["ih"]
        for single_par in par_sol:
            if not (has_streaming_linebuffer(layer["type"], layer["kernel"], single_par[2])):
                valid_iter_linebuffer[-1].append(0)
            else:
                valid_iter_linebuffer[-1].append(
                    layer_iter // (single_par[1] * single_par[2])
                )

    # valid_iter_solutions stores the number of iteration for each valid
    # parallelization.
    valid_iter_solutions = []
    for layer, layer_par in zip(layers_info, valid_par_solutions):
        valid_iter_solutions.append([])
        layer_iter = layer["total"]
        for single_par in layer_par:
            if layer["depth"]:
                unroll_factor = layer["kernel"] * np.prod(single_par[1:])
            else:
                unroll_factor = layer["kernel"] * np.prod(single_par)
            valid_iter_solutions[-1].append(layer_iter // unroll_factor)

    # valid_dsp_solutions stores the DSPs used for each valid solution
    # considering the possible packing
    valid_dsp_solutions = []
    for layer, layer_par in zip(layers_info, valid_par_solutions):
        valid_dsp_solutions.append([])
        for single_par in layer_par:
            valid_dsp_solutions[-1].append(dsp_consumption(layer, single_par, silvia_packing))

    # valid_bram_solutions stores the BRAMs used for each valid solution.
    valid_bram_solutions = []
    for layer, layer_par in zip(layers_info, valid_par_solutions):
        valid_bram_solutions.append([])
        n_weights = 0

        if (layer["type"] == "StreamingConv"):
            n_weights = layer["ich"] * layer["och"] * layer["kernel"]

        elif (layer["type"] == "StreamingDepthwiseConv"):
            n_weights = layer["ich"] * layer["kernel"]

        for single_par in layer_par:
            bram_used = compute_bram_layer(layer["weight_bits"], n_weights, np.prod(single_par[:2]) * layer["kernel"])
            valid_bram_solutions[-1].append(bram_used)

    # Minimize latencies
    prob = pulp.LpProblem("Parallel_ops", pulp.LpMinimize)

    # Variables of the problem: for each layer each valid parallelization has a
    # binary variable associated that select the chosen parameters.
    layer_binary_variables = []
    for i, solution_set in enumerate(valid_par_solutions):
        layer_binary_variables.append(pulp.LpVariable.dicts(
            f"Choice_l{i}", range(len(solution_set)), cat="Binary"))

    var = pulp.LpVariable("slack", lowBound=0, cat="Integer")

    # Objective function: minimize the maximum latency of the layers.
    prob += (
        var,
        "Minimization variable"
    )

    # Constraint: Only one binary variable per layer should be equal to 1
    for layer_index, layer in enumerate(layers_info):
        constraints_counter += 1
        ones = [1] * len(layer_binary_variables[layer_index])
        prob += (
            pulp.lpDot(layer_binary_variables[layer_index].values(), ones) == 1,
            f"One_choice_constraint_layer_{layer_index}",
        )

    # Constraint: The total number of DSPs used to achieve the chosen
    # parallelization should be lower than the available ones.
    constraints_counter += 1
    prob += (
        pulp.lpSum(
            [
                pulp.lpDot(layer_binary_variables[i].values(), valid_dsp_solutions[i])
                for i, layer in enumerate(layers_info)
            ]
        )
        <= NUM_DSP,
        f"DSP_constraint",
    )

    # Constraint: The total number of BRAMs used to achieve the chosen
    # parallelization should be lower than the available ones.
    constraints_counter += 1
    prob += (
        pulp.lpSum(
            [
                pulp.lpDot(layer_binary_variables[i].values(), valid_bram_solutions[i])
                for i, layer in enumerate(layers_info)
            ]
        )
        <= NUM_PORTS,
        f"BRAM_constraint",
    )

    # Constraints: The latency of each layer should be equal or lower to the minimization variable.
    for layer_index, layer in enumerate(layers_info):
        constraints_counter += 1
        prob += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_iter_solutions[layer_index]) <= var,
            f"Latency_constraint_layer_{layer_index}"
        )

    # Constraints: The latency of each line buffer should be equal or lower to the minimization variable.
    for layer_index, layer in enumerate(layers_info):
        constraints_counter += 1
        prob += (
            ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                valid_iter_linebuffer[layer_index])) <= var,
            f"Linebuffer_constraint_layer_{layer_index}"
        )

    start_time = time.time()
    prob.solve(PULP_CBC_CMD(timeLimit=10, msg=0))
    end_time = time.time()
    if (prob.status == pulp.LpStatusInfeasible):
        logger.error("Throughput problem unfeasible")
        exit(0)

    # Recovering the values of the paralellism for each layer from the binary variables.
    parallel_op = {}
    for i, layer in enumerate(valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[f"{layers_info[i]['name']}"] = layer[s]

    return parallel_op, int(pulp.value(prob.objective)), sum([len(s) for s in valid_par_solutions]), constraints_counter, (end_time - start_time)

def resourceILP(layers_info, model_II, valid_par_solutions, parallel_op, NUM_DSP, NUM_PORTS, silvia_packing, prj_root="/tmp"):
    """ Given the throughput of the network, find the parallelization for each layer that minimize the resources usage."""

    # Retriving only the parallelism combinations for lower throughput to save
    # resources in fast layers. The parallelization over ow is fixed
    layer_binary_variables = []
    clamped_valid_par_solutions = []
    valid_tot_par_solutions = []
    for i, solution_set in enumerate(valid_par_solutions):
        clamped_valid_par_solutions.append([])
        valid_tot_par_solutions.append([])
        chosen_par = np.prod(parallel_op[layers_info[i]['name']][0:2])
        chosen_ow = parallel_op[layers_info[i]['name']][2]
        for combination in solution_set:
            tot_par = np.prod(combination[0:2])
            ow_par = combination[2]
            if (tot_par <= chosen_par and ow_par == chosen_ow):
                clamped_valid_par_solutions[i].append(combination)
                valid_tot_par_solutions[i].append(np.prod(combination))

        layer_binary_variables.append(pulp.LpVariable.dicts(
            f"Choice_l{i}", range(len(clamped_valid_par_solutions[i])), cat="Binary"))

    # valid_iter_linebuffer stores the line buffer number of iteration for each valid
    # solution and it is useful to linearize the constraint of the line buffer
    valid_iter_solutions = []
    valid_iter_linebuffer = []
    for layer, layer_par in zip(layers_info, clamped_valid_par_solutions):
        valid_iter_solutions.append([])
        valid_iter_linebuffer.append([])
        layer_iter = layer["total"]
        line_iter = layer["ich"] * layer["iw"] * layer["ih"]
        for single_par in layer_par:
            if layer["depth"]:
                unroll_factor = layer["kernel"] * np.prod(single_par[1:])
            else:
                unroll_factor = layer["kernel"] * np.prod(single_par)
            valid_iter_solutions[-1].append(layer_iter // unroll_factor)
            if not (has_streaming_linebuffer(layer["type"], layer["kernel"], single_par[2])):
                valid_iter_linebuffer[-1].append(0)
            else:
                valid_iter_linebuffer[-1].append(
                    line_iter // (single_par[1] * single_par[2])
                )

    # valid_dsp_solutions stores the DSPs used for each valid solution
    # considering the possible packing
    valid_dsp_solutions = []
    for layer, layer_par in zip(layers_info, clamped_valid_par_solutions):
        valid_dsp_solutions.append([])
        for single_par in layer_par:
            valid_dsp_solutions[-1].append(dsp_consumption(layer, single_par, silvia_packing))

    # valid_bram_solutions stores the BRAMs used for each valid solution.
    valid_bram_solutions = []
    for layer, layer_par in zip(layers_info, clamped_valid_par_solutions):
        valid_bram_solutions.append([])
        n_weights = 0

        if (layer["type"] == "StreamingConv"):
            n_weights = layer["ich"] * layer["och"] * layer["kernel"]

            if (layer["depth"]):
                n_weights = layer["ich"] * layer["kernel"]

        for single_par in layer_par:
            bram_used = compute_bram_layer(layer["weight_bits"], n_weights, np.prod(single_par[:2]) * layer["kernel"])
            valid_bram_solutions[-1].append(bram_used)

    # Minimize resource usage
    prob_min = pulp.LpProblem("Resource_usage", pulp.LpMinimize)

    # Objective function: minimize the BRAMs + DSPs required to run the whole network.
    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
                    valid_bram_solutions[i]) for i, _ in enumerate(layers_info)]) +
        pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
                    valid_dsp_solutions[i]) for i, _ in enumerate(layers_info)]),
        f"Resource_objective"
    )

    # Objective function: minimize the DSPs required to run the whole network.
    # prob_min += (
    #     pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
    #                 valid_dsp_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]),
    #     f"DSP_constraint"
    # )

    # Constraint: Only one binary variable per layer should be equal to 1
    for layer_index, _ in enumerate(layers_info):
        ones = [1] * len(layer_binary_variables[layer_index])
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(), ones) == 1,
            f"One_choice_constraint_layer_{layer_index}"
        )

    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
                    valid_dsp_solutions[i]) for i, _ in enumerate(layers_info)]) <= NUM_DSP,
        f"DSP_constraint"
    )

    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
                    valid_bram_solutions[i]) for i, _ in enumerate(layers_info)]) <= NUM_PORTS,
        f"BRAM_constraint"
    )

    for layer_index, _ in enumerate(layers_info):
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_iter_solutions[layer_index]) <= model_II,
            f"Throughtput_constraint_layer_{layer_index}"
        )

    for layer_index, _ in enumerate(layers_info):
        prob_min += (
            ( pulp.lpDot(layer_binary_variables[layer_index].values(),
                valid_iter_linebuffer[layer_index])) <= model_II,
            f"Linebuffer_constraint_layer_{layer_index}"
        )

    prob_min.solve(PULP_CBC_CMD(msg=0))
    if (prob_min.status == pulp.LpStatusInfeasible):
        logger.error("Resource problem unfeasible")
        exit(0)

    parallel_op = {}
    for i, layer in enumerate(clamped_valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[f"{layers_info[i]['name']}"] = layer[s]

    return parallel_op

def update_model(model: ModelWrapper, parallel_op: dict) -> ModelWrapper:
    """Update the model with the parallelization chosen for each layer."""

    for node in model.graph.node:
        if node.name in parallel_op:
            par = parallel_op[node.name]
            par_dict = {
                "out_ch_par": par[0],
                "in_ch_par": par[1],
                "in_w_par": par[2],
                "out_w_par": par[2],
            }
            set_par_attributes(node, par_dict)

    return model

def propagate_parallelism(model: ModelWrapper) -> ModelWrapper:
    """Propagate the parallelism information through the model."""
    
    # Retrieving the NHWCToStream nodes to propagate the parallelism.
    queue = deque(model.get_nodes_by_op_type("NHWCToStream"))
    mark_visited = set()

    while queue:
        node = queue.popleft()
        mark_visited.add(node.name)
        par = get_par_attributes(node)

        # Set the input channels parallelization to the output channels parallelization
        # since we are propagating the parallelism to a consumer.
        par['in_ch_par'] = par['out_ch_par']

        consumers = model.find_direct_successors(node)
        if consumers is not None:
            # If the node has consumers, propagate the parallelization to them.
            for consumer in consumers:
                if not check_par_attributes(consumer):
                    # If the consumer does not have parallelization attributes, set them.
                    logger.info(f"Propagating parallelism from {node.name} to {consumer.name}")
                    set_par_attributes(consumer, par)
                if consumer.name not in mark_visited:
                # If the consumer is not already visited, add it to the queue.
                    queue.append(consumer)
            
    return model

def opt_steps(layers_info, parallel_op, valid_par_solutions, prj_root="/tmp"):
    """ Balancing the mismatches between the parallelization of consecutive layers."""

    # Retriving only the parallelism combinations with same throughput.
    clamped_valid_par_solutions = []
    for i, solution_set in enumerate(valid_par_solutions):
        clamped_valid_par_solutions.append([])
        chosen_par = np.prod(parallel_op[layers_info[i]['name']][0:2])
        chosen_ow = parallel_op[layers_info[i]['name']][2]
        
        # Do not choose combination which remove the packing feature over och.
        packing_over_och = parallel_op[layers_info[i]
                                       ['name']][0] % 2 == 0 and chosen_ow % 2 != 0
        for combination in solution_set:
            tot_par = np.prod(combination[0:2])
            ow_par = combination[2]

            if (tot_par == chosen_par and ow_par == chosen_ow):
                if (packing_over_och and combination[0] % 2 != 0):
                    continue
                clamped_valid_par_solutions[i].append(combination)
    
    # Computing a value representing the mismatch between the parallelization of
    # consecutive layers. The mismatch is computed as the difference between the
    # parallelization of the output channels of the previous layer and the input
    # channels of the next layer.
    par_prev = parallel_op[layers_info[0]["name"]]
    tot_mismatch_before = 0
    for layer in layers_info[1:]:
        par = parallel_op[layer["name"]]
        name = layer["name"]

        # For depth conv the parallelization in output is the one over ich
        if (layer["depth"]):
            par_prev_out = par_prev[1]
        else:
            par_prev_out = par_prev[0]

        par_in = par[1]
        if (par_prev_out % par_in != 0 and par_in % par_prev_out != 0):
            adjust = find_common_mult(par_prev_out, par_in)
            if adjust > max(par_prev_out, par_in):
                tot_mismatch_before += adjust - max(par_prev_out, par_in)
        par_prev = par

    new_parallel_op = parallel_op.copy()
    # Trying to minimize the mismatch between the parallelization of consecutive
    # layers, choosing between the valid parallelization combinations. Low effort,
    # if after the iteration the mismatches are increased, recover previous result.
    par_prev = new_parallel_op[layers_info[0]["name"]]
    tot_mismatch_after = 0
    for layer in layers_info[1:]:
        par = new_parallel_op[layer["name"]]
        name = layer["name"]
        
        # For depth conv the parallelization in output is the one over ich
        if (layer["depth"]):
            par_prev_out = par_prev[1]
        else:
            par_prev_out = par_prev[0]
        
        par_in = par[1]
        if (par_prev_out % par_in != 0 and par_in % par_prev_out != 0):
            logger.error(f"Error: och_ops i -> {par_prev_out} % ich_ops i+1 -> {par_in} != 0, using {find_common_mult(par_prev_out, par_in)}")

            for i, combination in enumerate(clamped_valid_par_solutions[layer['index']]):
                if (par_prev_out % combination[1] == 0):
                    logger.info(f"\t\tAssigning {combination} to {name}")
                    new_parallel_op[name] = combination
                    break
            else:
                tot_mismatch_after += find_common_mult(par_prev_out, par_in) - max(par_prev_out, par_in)
        par_prev = par

    logger.info(f"Before: {tot_mismatch_before}, After: {tot_mismatch_after}") 
    if (tot_mismatch_after > tot_mismatch_before):
        return parallel_op
    else:
        return new_parallel_op

def print_report(layers_info, layer_par, n_variables, n_constraints, model_II, time_spent, silvia_packing, generate_report_file, prj_root="/tmp"):
    with open(generate_report_file, "w+") as f:
        print("=" * 40, file=f)
        print("== Parallelization report", file=f)
        print("=" * 40, file=f)
        print(f"Number of variables: \t\t\t{n_variables}", file=f)
        print(f"Number of constraints:\t\t\t{n_constraints}", file=f)
        print(f"Time to solve: \t\t\t\t\t{time_spent:.2f}s", file=f)
        print(f"Initiation interval: \t\t\t{model_II}cc", file=f)
        print(
            f"Theorical throughput @ 200MHz: \t{1000000000.0 / (model_II * 5):.2f}FPS\n",
            file=f,
        )
        table_data = []

        # header row
        header = [
            "Layer name",
            "ICH",
            "OCH",
            "OW",
            "ich_ops",
            "och_ops",
            "ow_ops",
            "DSPs",
            "BRAMs",
            "Iter",
        ]
        table_data.append(header)

        DSPs = 0
        PORTs = 0
        for layer in layers_info:
            pack = False
            ow_ops = layer_par[layer["name"]][2]
            ich_ops = layer_par[layer["name"]][1]
            och_ops = layer_par[layer["name"]][0]

            port = dsp = 0
            if (layer["type"] == "StreamingConv" or layer["type"] == "StreamingDepthwiseConv"):
                bits = layer["weight_bits"]
                dsp = dsp_consumption(layer, layer_par[layer['name']], silvia_packing)

                op_per_dsp, _ = packing_feature((layer["weight_bits"], layer["act_bits"]), layer_par[layer['name']], silvia_packing)
                pack = str(op_per_dsp)

                weights = layer["ich"] * layer["och"] * layer["kernel"]
                if layer["depth"]:
                    weights = layer["ich"] * layer["kernel"]

                port = compute_bram_layer(bits, weights,  och_ops * ich_ops * layer["kernel"])

            if layer["depth"]:
                iter = int(layer["total"] / (ich_ops * ow_ops * layer["kernel"]))
            else:
                iter = int(layer["total"] / (ich_ops * och_ops * ow_ops * layer["kernel"]))
            PORTs += port
            DSPs += dsp

            string_dsp = f"{dsp}"
            if pack:
                string_dsp += f" ({pack})"

            row_data = [
                layer['name'],
                layer['ich'],
                layer['och'],
                layer['ow'],
                ich_ops,
                och_ops,
                ow_ops,
                string_dsp,
                port,
                iter
            ]

            table_data.append(row_data)

        footer = ["Totals", "", "", "", "", "", "", DSPs, PORTs, ""]
        table_data.append(footer)

        # Print the tabulated data to the file
        f.write(tabulate(table_data, headers="firstrow", tablefmt="grid"))
        print("\n", file=f)

class BalanceComputation(Transformation):
    """
    This transformation balances the computation load across the model by distributing
    resourcses evenly among the model's operations.
    """

    def __init__(self, silvia_packing: bool = False, nn2fpga_root: str = "/tmp"):
        """
        Initializes the BalanceComputation transformation.
        Args:
            silvia_packing (bool): If True, uses Silvia packing for DSPs.
            nn2fpga_root (str): The root directory of nn2FPGA.
        """
        super().__init__()
        self.nn2fpga_root = nn2fpga_root
        self.silvia_packing = silvia_packing

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """ Applies the transformation to the model.
        
        Args:
            model (ModelWrapper): The ONNX model to transform, wrapped in QONNX ModelWrapper.
        
        Returns:
            tuple: A tuple containing the transformed model and a boolean indicating if the transformation was applied.
        """

        board_res = read_board_info(
            board=model.get_metadata_prop("board_name"),
        )

        NUM_PORTS = (board_res["bram"] + board_res["uram"] * 8)
        NUM_PORTS = int(NUM_PORTS * 1.2)  # 85% of the BRAMs are used for parallelization
        NUM_DSP = board_res["dsp"] * 0.4  # 40% of the DSPs are used for parallelization

        # Extract layers information
        layers_info = layers_extractions(model)

        # Generate valid parallelization solutions for each layer
        valid_par_solutions = generate_architectures(
            layers_info, NUM_DSP, board_res["axi_bitwidth"], self.silvia_packing
        )

        # Balance the computation load across the model using ILP
        layer_par, model_II, n_variables, n_constraints, time_spent = parallelismILP(
            layers_info,
            valid_par_solutions,
            NUM_DSP,
            NUM_PORTS,
            self.silvia_packing,
            self.nn2fpga_root,
        )

        layer_par = resourceILP(
            layers_info,
            model_II,
            valid_par_solutions,
            layer_par,
            NUM_DSP,
            NUM_PORTS,
            self.silvia_packing,
            self.nn2fpga_root,
        )

        # Print the report
        generate_report_file = f"{self.nn2fpga_root}/balance_computation.rpt"
        print_report(
            layers_info,
            layer_par,
            n_variables,
            n_constraints,
            model_II,
            time_spent,
            self.silvia_packing,
            generate_report_file,
            prj_root=self.nn2fpga_root,
        )
        exit(0)

        # Update the model with the parallelization chosen for each layer
        model = update_model(model, layer_par)
        model = propagate_parallelism(model)


        # layer_par = opt_steps(
        #     layers_info,
        #     layer_par,
        #     valid_par_solutions,
        #     self.nn2fpga_root,
        # )

        # Add model II to the model metadata
        model.set_metadata_prop("model_II", str(model_II))

        logger.info(f"Balanced model with II {model_II} using {n_variables} variables and {n_constraints} constraints in {time_spent:.2f}s")
        return (model, False)
