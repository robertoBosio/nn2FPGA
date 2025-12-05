import pulp
import math
import time
import numpy as np
from pulp.apis import PULP_CBC_CMD
from tabulate import tabulate
from collections import deque
from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.util.board_util import read_board_info
from qonnx.custom_op.registry import getCustomOp
from backend.custom_op.op_base import DSECapable
import logging
logger = logging.getLogger(__name__)

def find_common_mult(a, b):
    """Return the least common multiple (LCM) of a and b."""
    return abs(a * b) // math.gcd(a, b) if a and b else 0

def parallelismILP(nodes, valid_par_solutions, NUM_DSP, NUM_PORTS, model):
    """ Find the parallelization for each layer that maximize the throughput of the network."""

    constraints_counter = 0
    nn2fpga_nodes = [getCustomOp(node) for node in nodes]

    # Corresponding latencies for each valid parallelization solution.
    points_latency = []
    for node, layer_par in zip(nn2fpga_nodes, valid_par_solutions):
        points_latency.append([])
        for single_par in layer_par:
            node.apply_point(model, single_par)
            points_latency[-1].append(node.get_latency(model))

    # Corresponding DSPs for each valid parallelization solution.
    points_dsp = []
    for node, layer_par in zip(nn2fpga_nodes, valid_par_solutions):
        points_dsp.append([])
        for single_par in layer_par:
            node.apply_point(model, single_par)
            points_dsp[-1].append(node.get_dsps(model))

    # Corresponding BRAMs for each valid parallelization solution.
    points_bram = []
    for node, layer_par in zip(nn2fpga_nodes, valid_par_solutions):
        points_bram.append([])
        for single_par in layer_par:
            node.apply_point(model, single_par)
            points_bram[-1].append(node.get_brams(model))

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
    for layer_index in range(len(nodes)):
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
                pulp.lpDot(layer_binary_variables[i].values(), points_dsp[i])
                for i in range(len(nn2fpga_nodes))
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
                pulp.lpDot(layer_binary_variables[i].values(), points_bram[i])
                for i in range(len(nn2fpga_nodes))
            ]
        )
        <= NUM_PORTS,
        f"BRAM_constraint",
    )

    # Constraints: The latency of each layer should be equal or lower to the minimization variable.
    for layer_index in range(len(nn2fpga_nodes)):
        constraints_counter += 1
        prob += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    points_latency[layer_index]) <= var,
            f"Latency_constraint_layer_{layer_index}"
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
                parallel_op[nodes[i].name] = layer[s]

    return (
        parallel_op,
        int(pulp.value(prob.objective)),
        sum([len(s) for s in valid_par_solutions]),
        constraints_counter,
        (end_time - start_time),
    )

def resourceILP(nodes, model_II, valid_par_solutions, parallel_op, NUM_DSP, NUM_PORTS, model):
    """ Given the throughput of the network, find the parallelization for each layer that minimize the resources usage."""

    nn2fpga_nodes = [getCustomOp(node) for node in nodes]
    clamped_valid_par_solutions = valid_par_solutions

    # Variables of the problem: for each layer each valid parallelization has a
    # binary variable associated that select the chosen parameters.
    layer_binary_variables = []
    for i, solution_set in enumerate(valid_par_solutions):
        layer_binary_variables.append(pulp.LpVariable.dicts(
            f"Choice_l{i}", range(len(solution_set)), cat="Binary"))

    valid_iter_solutions = []
    for node, layer_par in zip(nn2fpga_nodes, clamped_valid_par_solutions):
        valid_iter_solutions.append([])
        for single_par in layer_par:
            node.apply_point(model, single_par)
            valid_iter_solutions[-1].append(node.get_latency(model))

    # valid_dsp_solutions stores the DSPs used for each valid solution
    # considering the possible packing
    valid_dsp_solutions = []
    for node, layer_par in zip(nn2fpga_nodes, clamped_valid_par_solutions):
        valid_dsp_solutions.append([])
        for single_par in layer_par:
            node.apply_point(model, single_par)
            valid_dsp_solutions[-1].append(node.get_dsps(model))

    # valid_bram_solutions stores the BRAMs used for each valid solution.
    valid_bram_solutions = []
    for node, layer_par in zip(nn2fpga_nodes, clamped_valid_par_solutions):
        valid_bram_solutions.append([])
        for single_par in layer_par:
            node.apply_point(model, single_par)
            valid_bram_solutions[-1].append(node.get_brams(model))

    # Minimize resource usage
    prob_min = pulp.LpProblem("Resource_usage", pulp.LpMinimize)

    # Objective function: minimize the BRAMs + DSPs required to run the whole network.
    # prob_min += (
    #     pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
    #                 valid_bram_solutions[i]) for i in range(len(nn2fpga_nodes))]) +
    #     pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
    #                 valid_dsp_solutions[i]) for i in range(len(nn2fpga_nodes))]),
    #     f"Resource_objective"
    # )
    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
                    valid_bram_solutions[i]) for i in range(len(nn2fpga_nodes))]),
        f"Resource_objective"
    )

    # Objective function: minimize the DSPs required to run the whole network.
    # prob_min += (
    #     pulp.lpSum([pulp.lpDot(layer_binary_variables[layer['index']].values(),
    #                 valid_dsp_solutions[i]) for i, layer in enumerate(layers_info_unmerged)]),
    #     f"DSP_constraint"
    # )

    # Constraint: Only one binary variable per layer should be equal to 1
    for layer_index in range(len(nn2fpga_nodes)):
        ones = [1] * len(layer_binary_variables[layer_index])
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(), ones) == 1,
            f"One_choice_constraint_layer_{layer_index}"
        )

    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
                    valid_dsp_solutions[i]) for i in range(len(nn2fpga_nodes))]) <= NUM_DSP,
        f"DSP_constraint"
    )

    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
                    valid_bram_solutions[i]) for i in range(len(nn2fpga_nodes))]) <= NUM_PORTS,
        f"BRAM_constraint"
    )

    for layer_index in range(len(nn2fpga_nodes)):
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_iter_solutions[layer_index]) <= model_II,
            f"Throughtput_constraint_layer_{layer_index}"
        )

    prob_min.solve(PULP_CBC_CMD(msg=0))
    if (prob_min.status == pulp.LpStatusInfeasible):
        logger.error("Resource problem unfeasible")
        exit(0)

    parallel_op = {}
    for i, layer in enumerate(clamped_valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[nodes[i].name] = layer[s]

    return parallel_op

def update_model(model: ModelWrapper, parallel_op: dict) -> ModelWrapper:
    """Update the model with the parallelization chosen for each layer."""

    for node in model.graph.node:
        if node.name in parallel_op:
            selected_point = parallel_op[node.name]
            getCustomOp(node).apply_point(model, selected_point)

    return model

def propagate_parallelism(model: ModelWrapper) -> ModelWrapper:
    """Propagate the parallelism information through the model."""
    
    # Retrieving the NHWCToStream nodes to propagate the parallelism.
    queue = deque(model.get_nodes_by_op_type("NHWCToStream"))
    mark_visited = set()

    while queue:
        node = queue.popleft()
        mark_visited.add(node.name)

        # Creating the propagation interface.
        interface = getCustomOp(node).get_port_interface()
        consumers = model.find_direct_successors(node)
        if consumers is not None:
            # If the node has consumers, propagate the parallelization to them.
            for consumer in consumers:
                if not isinstance(getCustomOp(consumer), DSECapable):
                    # If the consumer does not have parallelization attributes, set them.
                    logger.info(f"Propagating parallelism from {node.name} to {consumer.name}")
                    getCustomOp(consumer).inherit_interface(model, interface)

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

def print_report(nodes, layer_par, n_variables, n_constraints, model_II, time_spent, generate_report_file, model):
    with open(generate_report_file, "w+") as f:
        print("=" * 40, file=f)
        print("== DSE report", file=f)
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
            "DSPs",
            "BRAMs",
            "Latency (cc)",
        ]
        table_data.append(header)

        DSPs = 0
        PORTs = 0
        for layer in nodes:
            dsp = getCustomOp(layer).get_dsps(model)
            DSPs += dsp
            port = getCustomOp(layer).get_brams(model)
            PORTs += port

            row_data = [
                layer.name,
                f"{dsp}",
                f"{port}",
                f"{getCustomOp(layer).get_latency(model)}"
            ]

            table_data.append(row_data)

        footer = ["Totals", DSPs, PORTs, ""]
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
        NUM_PORTS = int(NUM_PORTS * 1)  # 85% of the BRAMs are used for parallelization
        NUM_DSP = board_res["dsp"] * 0.05  # 50% of the DSPs are used for parallelization
        NUM_DSP = 2000

        # Extract layers information
        DSE_nodes = [node for node in model.graph.node if isinstance(getCustomOp(node), DSECapable)]

        # Generate valid parallelization solutions for each layer
        DSE_points = []
        for node in DSE_nodes:
            DSE_points.append(getCustomOp(node).get_dse_points(model))

        # Balance the computation load across the model using ILP
        layer_par, model_II, n_variables, n_constraints, time_spent = parallelismILP(
            DSE_nodes,
            DSE_points,
            NUM_DSP,
            NUM_PORTS,
            model,
        )

        layer_par = resourceILP(
            DSE_nodes,
            model_II,
            DSE_points,
            layer_par,
            NUM_DSP,
            NUM_PORTS,
            model,
        )

        # Update the model with the parallelization chosen for each layer
        model = update_model(model, layer_par)
        model = propagate_parallelism(model)

        # layer_par = opt_steps(
        #     layers_info,
        #     layer_par,
        #     valid_par_solutions,
        #     self.nn2fpga_root,
        # )

        # Print the report
        generate_report_file = f"{self.nn2fpga_root}/balance_computation.rpt"
        print_report(
            DSE_nodes,
            layer_par,
            n_variables,
            n_constraints,
            model_II,
            time_spent,
            generate_report_file,
            model,
        )

        # Add model II to the model metadata
        model.set_metadata_prop("model_II", str(model_II))

        logger.info(f"Balanced model with II {model_II} using {n_variables} variables and {n_constraints} constraints in {time_spent:.2f}s")
        return (model, False)
