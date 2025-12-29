import pulp
import math
import time
from pulp.apis import PULP_CBC_CMD
from tabulate import tabulate
from collections import deque
from typing import Any, Dict, List, Set, Tuple
from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from backend.util.board_util import read_board_info
from qonnx.custom_op.registry import getCustomOp
from backend.custom_op.op_base import DSECapable, NodeInterface
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


def reduce_mismatches(model: ModelWrapper, parallel_op: Dict[str, Any], model_II: int) -> ModelWrapper:
    """
    Evaluate search-space size for reducing mismatches between DSE ops.
    The objective is to minimize the number of mismatches of words and streams between
    consecutive DSE-capable nodes, while keeping the same DSP/BRAM/latency usage and throughput. 
    """

    def is_dse_node(node) -> bool:
        return isinstance(getCustomOp(node), DSECapable)

    def compute_alternatives(node, model_II) -> Tuple[List[NodeInterface], List[Any]]:
        """
        Returns alternative DSE points that do not increase DSP/BRAM/latency
        relative to the current applied point in `parallel_op`.
        """
        node_name = node.name
        op = getCustomOp(node)

        # Current metrics at the point already applied in the model.
        cur_dsps = op.get_dsps(model)
        cur_brams = op.get_brams(model)

        alt: List[NodeInterface] = []
        points: List[Any] = []
        try:
            for point in op.get_dse_points(model):
                op.apply_point(model, point)
                new_dsps = op.get_dsps(model)
                new_brams = op.get_brams(model)
                new_lat = op.get_latency(model)

                if new_dsps <= cur_dsps and new_brams <= cur_brams and new_lat <= model_II:
                    alt.append(op.get_port_interface())
                    points.append(point)
        finally:
            # Always restore original point, even if metrics queries fail.
            op.apply_point(model, parallel_op[node_name])

        return alt, points

    def find_downstream_dse_consumers(start_node) -> List[Any]:
        """
        BFS from `start_node` to find all reachable DSECapable nodes,
        traversing through non-DSE nodes. Does not traverse past a DSE node.
        """
        consumers: List[Any] = []
        visited: Set[str] = set()

        q = deque(model.find_direct_successors(start_node) or [])
        while q:
            n = q.popleft()
            if n.name in visited:
                continue
            visited.add(n.name)

            if is_dse_node(n):
                consumers.append(n)
                continue  # stop traversal past DSE nodes
            q.extend(model.find_direct_successors(n) or [])

        return consumers
    
    # Collect all DSE-capable nodes once.
    dse_nodes = [n for n in model.graph.node if is_dse_node(n)]
    node2i = {n.name: i for i, n in enumerate(dse_nodes)}

    # Cache alternatives per node to avoid recomputation.
    iface_cache: Dict[str, List[NodeInterface]] = {}
    points_cache: Dict[str, List[Any]] = {}
    for n in dse_nodes:
        if n.name not in iface_cache:
            iface_cache[n.name], points_cache[n.name] = compute_alternatives(n, model_II)   # list of NodeInterface

    # Build mismatch costs between DSE nodes.
    edges: List[Tuple[int, int]] = [] # (u_idx, v_idx)
    mismatch_cost: Dict[Tuple[int, int, int, int], int] = {} # (u_idx, v_idx, p, q)
    current_mismatch = 0
    for u_node in dse_nodes:
        u = node2i[u_node.name]
        u_iface = getCustomOp(u_node).get_port_interface()
        consumers = find_downstream_dse_consumers(u_node)

        for v_node in consumers:
            v = node2i[v_node.name]
            v_iface = getCustomOp(v_node).get_port_interface()
            
            # Current mismatch between u -> v
            c_mismatch = 0
            c_mismatch += 1 if u_iface.out_stream_array != v_iface.in_stream_array else 0
            c_mismatch += 1 if u_iface.out_word_array != v_iface.in_word_array else 0
            current_mismatch += c_mismatch

            edges.append((u, v))

            u_alts = iface_cache[u_node.name]
            v_alts = iface_cache[v_node.name]

            for p, p_alt in enumerate(u_alts):
                for q, c_alt in enumerate(v_alts):
                    c = 0
                    c += 1 if p_alt.out_stream_array != c_alt.in_stream_array else 0
                    c += 1 if p_alt.out_word_array != c_alt.in_word_array else 0
                    mismatch_cost[(u, v, p, q)] = c
    
    # One-hot variables for alternative choices.
    x = []
    for i, node in enumerate(dse_nodes):
        n_alts = len(iface_cache[node.name])
        x.append(pulp.LpVariable.dicts(f"Choice_{i}", range(n_alts), cat="Binary"))
    
    # One-hot variables for coupled alternative choices.
    y = {}  # (u,v,p,q) -> pulp var
    for (u, v) in edges:
        n_u = len(iface_cache[dse_nodes[u].name])
        n_v = len(iface_cache[dse_nodes[v].name])

        for p in range(n_u):
            for q in range(n_v):
                var = pulp.LpVariable(f"Y_{u}_{p}_{v}_{q}", cat="Binary")
                y[(u, v, p, q)] = var

    # --- Problem ---
    prob = pulp.LpProblem("Minimize_Mismatches", pulp.LpMinimize)

    # --- Constraints: one-hot per node (exactly one alternative chosen) ---
    for i, node in enumerate(dse_nodes):
        n_alts = len(iface_cache[node.name])
        prob += pulp.lpSum(x[i][p] for p in range(n_alts)) == 1, f"OneHot_{i}"

    # --- Constraints: linearize y = x_u AND x_v for each edge (u,v) and choice pair (p,q) ---
    for (u, v) in edges:
        n_u = len(iface_cache[dse_nodes[u].name])
        n_v = len(iface_cache[dse_nodes[v].name])

        for p in range(n_u):
            for q in range(n_v):
                y_uvpq = y[(u, v, p, q)]

                # Constrain y_uvpq to be 1 iff both x[u][p] and x[v][q] are 1
                prob += y_uvpq <= x[u][p], f"Yub_u{u}p{p}_v{v}q{q}_1"
                prob += y_uvpq <= x[v][q], f"Yub_u{u}p{p}_v{v}q{q}_2"
                prob += y_uvpq >= x[u][p] + x[v][q] - 1, f"Ylb_u{u}p{p}_v{v}q{q}"

    # --- Objective: minimize total mismatch cost ---
    prob += pulp.lpSum(
        mismatch_cost[(u, v, p, q)] * y[(u, v, p, q)]
        for (u, v) in edges
        for p in range(len(iface_cache[dse_nodes[u].name]))
        for q in range(len(iface_cache[dse_nodes[v].name]))
    ), "TotalMismatchCost"

    prob.solve(pulp.PULP_CBC_CMD(msg=True))

    chosen = {}
    for i, node in enumerate(dse_nodes):
        for p in range(len(iface_cache[node.name])):
            if pulp.value(x[i][p]) > 0.5:
                chosen[node.name] = p  # index into iface_cache[node.name]
                getCustomOp(node).apply_point(model, points_cache[node.name][p])
                break

    if (prob.status == pulp.LpStatusInfeasible):
        logger.error("Mismatch minimization problem unfeasible")
    
    logger.info(f"Starting mismatches: {current_mismatch}, minimized to: {int(pulp.value(prob.objective))}")
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
        NUM_DSP = board_res["dsp"] * 0.15  # 50% of the DSPs are used for parallelization
        NUM_DSP = 1700

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
        model = reduce_mismatches(model, layer_par, model_II)
        model = propagate_parallelism(model)

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
