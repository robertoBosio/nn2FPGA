# --- stdlib ---
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# --- third-party ---
import pulp
from pulp.apis import PULP_CBC_CMD
from tabulate import tabulate
from onnx import NodeProto

# --- project ---
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from backend.custom_op.op_base import DSECapable, NodeInterface
from backend.util.board_util import read_board_info
import logging
logger = logging.getLogger(__name__)

@dataclass
class DSEBlock:
    nodes: Set[int]                 # indices into dse_nodes
    internal_edges: List[Tuple[int, int]]  # (u,v) with u,v in nodes

def is_dse_node(node: NodeProto) -> bool:
    """ Return true if the node is DSECapable. """
    return isinstance(getCustomOp(node), DSECapable)

def find_downstream_dse_consumers(model: ModelWrapper, start_node: NodeProto) -> List[Any]:
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

def split_dse_graph(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    K: int = 100,
    alpha: float = 0.3,
) -> List[DSEBlock]:
    """
    Partition a directed graph into blocks with core size <= K using
    seeded neighbor-growth to minimize cut edges.

    Returns a list of blocks, each block dict contains:
    - "core": Set[int]          core nodes whose decisions will be committed
    - "internal_edges": List[(u,v)] edges with u,v in solve
    Notes:
    - A best-effort Kahn topo is attempted; if cycles exist,
        remaining nodes are appended arbitrarily.
    """
    blocks: List[DSEBlock] = []

    # Build adjacency lists.
    out_n: Dict[int, List[int]] = defaultdict(list)
    in_n: Dict[int, List[int]] = defaultdict(list)
    for u, v in edges:
        out_n[u].append(v)
        in_n[v].append(u)

    node_set = set(nodes)

    # Compute indegrees for Kahn topo sort.
    # It will start from nodes with zero indegree.
    indeg = {n: 0 for n in nodes}
    for u, v in edges:
        if u in node_set and v in node_set:
            indeg[v] += 1

    # Compute topological order.
    # Pop a node with indeg 0, decrement its neighbors, and enqueue any that reach indeg 0.
    q = deque([n for n in nodes if indeg[n] == 0])
    topo_calc = []
    while q:
        n = q.popleft()
        topo_calc.append(n)
        for w in out_n.get(n, []):
            if w in indeg:
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)

    # If cycles exist, append remaining nodes arbitrarily.
    if len(topo_calc) < len(nodes):
        remaining = [n for n in nodes if n not in set(topo_calc)]
        topo_calc.extend(remaining)
    topo = topo_calc

    def neighbors(v: int) -> Iterable[int]:
        return list(out_n.get(v, [])) + list(in_n.get(v, []))

    # Tracks nodes already assigned to a core block
    assigned: Set[int] = set()

    # Helps scoring candidates quickly
    def internal_degree_to_block(v: int, B: Set[int]) -> int:
        """# directed edges between v and B, counting both in/out."""
        cnt = 0
        for u in out_n.get(v, []):
            if u in B:
                cnt += 1
        for u in in_n.get(v, []):
            if u in B:
                cnt += 1
        return cnt

    def external_degree_unassigned(v: int, B: Set[int]) -> int:
        """# directed edges between v and unassigned nodes outside B."""
        cnt = 0
        for u in out_n.get(v, []):
            if u not in B and u not in assigned:
                cnt += 1
        for u in in_n.get(v, []):
            if u not in B and u not in assigned:
                cnt += 1
        return cnt

    # ---- main loop: seed + grow ----
    topo_idx = 0
    while len(assigned) < len(nodes):

        # pick next unassigned seed in topo order
        seed = None
        while topo_idx < len(topo):
            cand = topo[topo_idx]
            topo_idx += 1
            if cand in node_set and cand not in assigned:
                seed = cand
                break
        if seed is None:
            # fallback: any remaining node
            remaining = [n for n in nodes if n not in assigned]
            seed = remaining[0]

        core: Set[int] = {seed}
        frontier: Set[int] = set()
        for nb in neighbors(seed):
            if nb in node_set and nb not in assigned and nb not in core:
                frontier.add(nb)

        # grow core up to K
        while len(core) < K:

            # If frontier is empty, add from topo to stall-fill
            if not frontier:
                for cand in topo:
                    if cand not in assigned and cand not in core:
                        core.add(cand)
                        if len(core) >= K:
                            break
                break

            # pick best candidate from frontier
            best_v = None
            best_score = None
            best_internal = -1

            for v in frontier:
                if v in assigned or v in core:
                    continue
                internal = internal_degree_to_block(v, core)
                external = external_degree_unassigned(v, core)
                score = internal - alpha * external

                # tie-break: prefer higher internal, then higher score
                if (
                    best_v is None
                    or internal > best_internal
                    or (internal == best_internal and score > best_score)
                ):
                    best_v = v
                    best_score = score
                    best_internal = internal

            if best_v is None:
                # frontier only had assigned/core; clear and continue to stall-fill
                frontier.clear()
                continue

            # add chosen node
            core.add(best_v)
            frontier.discard(best_v)

            # expand frontier around best_v
            for nb in neighbors(best_v):
                if nb in node_set and nb not in assigned and nb not in core:
                    frontier.add(nb)

        # build edge lists for this block
        internal_edges: List[Tuple[int, int]] = []
        for u, v in edges:
            u_in = u in core
            v_in = v in core
            if u_in and v_in:
                internal_edges.append((u, v))

        blocks.append(
            DSEBlock(
                nodes=core,
                internal_edges=internal_edges,
            )
        )

        # commit assignment only for core nodes
        assigned |= core

    return blocks

def evaluate_points_for_nodes(
    nodes: List[Any],
    valid_par_solutions: List[List[Any]],
    model: ModelWrapper,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    For each node and each candidate point, evaluate (latency, DSPs, BRAMs).

    Returns:
        latencies[i][k], dsps[i][k], brams[i][k] for node i, point k.
    """
    latencies: List[List[int]] = []
    dsps: List[List[int]] = []
    brams: List[List[int]] = []

    nn2fpga_nodes = [getCustomOp(node) for node in nodes]

    for op, layer_par in zip(nn2fpga_nodes, valid_par_solutions):
        layer_lat: List[int] = []
        layer_dsp: List[int] = []
        layer_bram: List[int] = []

        for point in layer_par:
            op.apply_point(model, point)
            layer_lat.append(op.get_latency(model))
            layer_dsp.append(op.get_dsps(model))
            layer_bram.append(op.get_brams(model))

        latencies.append(layer_lat)
        dsps.append(layer_dsp)
        brams.append(layer_bram)

    return latencies, dsps, brams

def solve_ilp(
    prob: pulp.LpProblem,
    name: str,
    time_limit_s: Optional[int] = None,
    gap_rel: Optional[float] = None,
    msg: bool = False,
    raise_on_infeasible: bool = True,
) -> Tuple[int, float]:
    """
    Solve a PuLP ILP with CBC and consistent logging.

    Returns:
        (status, elapsed_seconds)

    Notes:
      - status is the PuLP status code (e.g. pulp.LpStatusOptimal).
      - You can decide whether infeasible should raise.
    """
    solver_kwargs = {"msg": int(msg)}
    if time_limit_s is not None:
        solver_kwargs["timeLimit"] = time_limit_s
    if gap_rel is not None:
        solver_kwargs["gapRel"] = gap_rel

    logger.info(f"Starting ILP solver: {name}")
    start = time.time()
    prob.solve(PULP_CBC_CMD(**solver_kwargs))
    elapsed = time.time() - start

    status = prob.status
    logger.info(f"ILP solver finished: {name} in {elapsed:.2f}s, status={pulp.LpStatus[status]}")

    if raise_on_infeasible and status == pulp.LpStatusInfeasible:
        raise RuntimeError(f"ILP infeasible: {name}")

    return status, elapsed

def solve_mismatch_block_ilp(
    model: ModelWrapper,
    dse_nodes: List[NodeProto],                      # index -> onnx node
    block: DSEBlock,
    iface_cache: Dict[str, List[NodeInterface]],
    points_cache: Dict[str, List[Any]],
    mismatch_cost: Dict[Tuple[int, int, int, int], int],  # (u,v,p,q)->cost
    time_limit_s: Optional[int] = None,
    gap_rel: Optional[float] = None,
    msg: bool = False,
) -> None:
    """
    Solve mismatch minimization for a single block.

    Side effect:
      - applies selected points to `model` for nodes in block.nodes
    """
    solve_nodes: Set[int] = set(block.nodes)
    internal_edges: List[Tuple[int, int]] = list(block.internal_edges)

    # ---- sanity: ensure each node has >=1 alternative ----
    for i in solve_nodes:
        name = dse_nodes[i].name
        if name not in iface_cache or len(iface_cache[name]) == 0:
            raise RuntimeError(f"Node {name} has 0 alternatives; cannot solve mismatch block.")

    # ---- x vars: one-hot per node ----
    x: Dict[int, Dict[int, pulp.LpVariable]] = {}
    for i in solve_nodes:
        name = dse_nodes[i].name
        n_alts = len(iface_cache[name])
        x[i] = pulp.LpVariable.dicts(f"Choice_{i}", range(n_alts), cat="Binary")

    # ---- y vars: only for internal edges ----
    y: Dict[Tuple[int, int, int, int], pulp.LpVariable] = {}
    for (u, v) in internal_edges:
        nu = len(iface_cache[dse_nodes[u].name])
        nv = len(iface_cache[dse_nodes[v].name])
        for p in range(nu):
            for q in range(nv):
                y[(u, v, p, q)] = pulp.LpVariable(f"Y_{u}_{p}_{v}_{q}", cat="Binary")

    prob = pulp.LpProblem("Minimize_Mismatches_Block", pulp.LpMinimize)

    # ---- one-hot constraints ----
    for i in solve_nodes:
        n_alts = len(iface_cache[dse_nodes[i].name])
        prob += pulp.lpSum(x[i][p] for p in range(n_alts)) == 1, f"OneHot_{i}"

    # ---- linearization constraints: y = x_u AND x_v ----
    for (u, v) in internal_edges:
        nu = len(iface_cache[dse_nodes[u].name])
        nv = len(iface_cache[dse_nodes[v].name])
        for p in range(nu):
            for q in range(nv):
                y_uvpq = y[(u, v, p, q)]
                prob += y_uvpq <= x[u][p], f"Yub_u{u}p{p}_v{v}q{q}_1"
                prob += y_uvpq <= x[v][q], f"Yub_u{u}p{p}_v{v}q{q}_2"
                prob += y_uvpq >= x[u][p] + x[v][q] - 1, f"Ylb_u{u}p{p}_v{v}q{q}"

    # ---- objective ----
    prob += pulp.lpSum(
        mismatch_cost[(u, v, p, q)] * y[(u, v, p, q)]
        for (u, v) in internal_edges
        for p in range(len(iface_cache[dse_nodes[u].name]))
        for q in range(len(iface_cache[dse_nodes[v].name]))
    ), "BlockMismatchCost"

    # ---- solve ----
    solve_ilp(
        prob,
        name=f"Reduce mismatches block |V|={len(solve_nodes)} |E|={len(internal_edges)}",
        time_limit_s=time_limit_s,
        gap_rel=gap_rel,
        msg=msg,
    )

    # ---- extract & apply ----
    for i in solve_nodes:
        name = dse_nodes[i].name
        n_alts = len(iface_cache[name])
        chosen_p = None
        for p in range(n_alts):
            val = pulp.value(x[i][p])
            if val is not None and val > 0.5:
                chosen_p = p
                break
        if chosen_p is None:
            chosen_p = 0  # fallback (shouldn't happen)

        getCustomOp(dse_nodes[i]).apply_point(model, points_cache[name][chosen_p])

def parallelismILP(nodes, valid_par_solutions, NUM_DSP, NUM_PORTS, model):
    """Find the parallelization for each layer that maximize the throughput of the network."""

    constraints_counter = 0

    # Evaluate latency / DSP / BRAM for each node and point
    points_latency, points_dsp, points_bram = evaluate_points_for_nodes(
        nodes, valid_par_solutions, model
    )

    # Minimize latencies
    prob = pulp.LpProblem("Parallel_ops", pulp.LpMinimize)

    # Variables of the problem: for each layer each valid parallelization has a
    # binary variable associated that select the chosen parameters.
    layer_binary_variables = []
    for i, solution_set in enumerate(valid_par_solutions):
        layer_binary_variables.append(
            pulp.LpVariable.dicts(
                f"Choice_l{i}", range(len(solution_set)), cat="Binary"
            )
        )

    var = pulp.LpVariable("slack", lowBound=0, cat="Integer")

    # Objective function: minimize the maximum latency of the layers.
    prob += (var, "Minimization variable")

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
                for i in range(len(nodes))
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
                for i in range(len(nodes))
            ]
        )
        <= NUM_PORTS,
        f"BRAM_constraint",
    )

    # Constraints: The latency of each layer should be equal or lower to the minimization variable.
    for layer_index in range(len(nodes)):
        constraints_counter += 1
        prob += (
            pulp.lpDot(
                layer_binary_variables[layer_index].values(),
                points_latency[layer_index],
            )
            <= var,
            f"Latency_constraint_layer_{layer_index}",
        )

    status, elapsed = solve_ilp(prob, name="Throughput optimization", time_limit_s=10, msg=False)

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
        elapsed,
    )

def resourceILP(nodes, model_II, valid_par_solutions, parallel_op, NUM_DSP, NUM_PORTS, model):
    """ Given the throughput of the network, find the parallelization for each layer that minimize the resources usage."""

    # Variables of the problem: for each layer each valid parallelization has a
    # binary variable associated that select the chosen parameters.
    layer_binary_variables = []
    for i, solution_set in enumerate(valid_par_solutions):
        layer_binary_variables.append(pulp.LpVariable.dicts(
            f"Choice_l{i}", range(len(solution_set)), cat="Binary"))

    # Precompute valid solutions for DSP, BRAM, and iterations
    valid_iter_solutions, valid_dsp_solutions, valid_bram_solutions = (
        evaluate_points_for_nodes(nodes, valid_par_solutions, model)
    )

    # Minimize resource usage
    prob_min = pulp.LpProblem("Resource_usage", pulp.LpMinimize)

    # Objective function: minimize the BRAMs + DSPs required to run the whole network.
    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
                    valid_bram_solutions[i]) for i in range(len(nodes))]) +
        pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
                    valid_dsp_solutions[i]) for i in range(len(nodes))]),
        f"Resource_objective"
    )
    # prob_min += (
    #     pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
    #                 valid_bram_solutions[i]) for i in range(len(nodes))]),
    #     f"Resource_objective"
    # )

    # Constraint: Only one binary variable per layer should be equal to 1
    for layer_index in range(len(nodes)):
        ones = [1] * len(layer_binary_variables[layer_index])
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(), ones) == 1,
            f"One_choice_constraint_layer_{layer_index}"
        )

    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
                    valid_dsp_solutions[i]) for i in range(len(nodes))]) <= NUM_DSP,
        f"DSP_constraint"
    )

    prob_min += (
        pulp.lpSum([pulp.lpDot(layer_binary_variables[i].values(),
                    valid_bram_solutions[i]) for i in range(len(nodes))]) <= NUM_PORTS,
        f"BRAM_constraint"
    )

    for layer_index in range(len(nodes)):
        prob_min += (
            pulp.lpDot(layer_binary_variables[layer_index].values(),
                    valid_iter_solutions[layer_index]) <= model_II,
            f"Throughtput_constraint_layer_{layer_index}"
        )

    status, elapsed = solve_ilp(prob_min, name="Resource optimization", msg=False)

    parallel_op = {}
    for i, layer in enumerate(valid_par_solutions):
        for s in range(len(layer)):
            if int(layer_binary_variables[i][s].value()) == 1:
                parallel_op[nodes[i].name] = layer[s]

    return parallel_op

def reduce_mismatches(model: ModelWrapper, parallel_op: Dict[str, Any], model_II: int) -> ModelWrapper:
    """
    Evaluate search-space size for reducing mismatches between DSE ops.
    The objective is to minimize the number of mismatches of words and streams between
    consecutive DSE-capable nodes, while keeping the same DSP/BRAM/latency usage and throughput. 
    """
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

    # Collect all DSE-capable nodes.
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
        consumers = find_downstream_dse_consumers(model, u_node)

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

    nodes = list(range(len(dse_nodes)))
    blocks = split_dse_graph(nodes, edges, K=25)
    logger.info(f"Reduced mismatch ILP: {len(dse_nodes)} DSE nodes, {len(blocks)} blocks.")

    for block in blocks:

        solve_mismatch_block_ilp(
            model=model,
            dse_nodes=dse_nodes,
            block=block,
            iface_cache=iface_cache,
            points_cache=points_cache,
            mismatch_cost=mismatch_cost,
            time_limit_s=100,
            msg=False,
        )
    
    return model

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
    mark_visited = {node.name for node in queue}

    while queue:
        node = queue.popleft()

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
                    mark_visited.add(consumer.name)
            
    return model

def print_report(nodes, n_variables, n_constraints, model_II, frequency, time_spent, generate_report_file, model):
    with open(generate_report_file, "w+") as f:
        print("=" * 40, file=f)
        print("== DSE report", file=f)
        print("=" * 40, file=f)
        print(f"Number of variables: \t\t\t{n_variables}", file=f)
        print(f"Number of constraints:\t\t\t{n_constraints}", file=f)
        print(f"Time to solve: \t\t\t\t\t{time_spent:.2f}s", file=f)
        print(f"Initiation interval: \t\t\t{model_II}cc", file=f)
        print(
            f"Theorical throughput @ {frequency}MHz: \t{1000000000.0 / (model_II * 1000 / float(frequency)):.2f}FPS\n",
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

    def __init__(self, nn2fpga_root: str = "/tmp"):
        """
        Initializes the BalanceComputation transformation.
        Args:
            nn2fpga_root (str): The root directory of nn2FPGA.
        """
        super().__init__()
        self.nn2fpga_root = nn2fpga_root

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
        dsp_limit = model.get_metadata_prop("dsp_limit")
        frequency = model.get_metadata_prop("frequency")
        if dsp_limit is not None:
            dsp_limit = int(dsp_limit)

        NUM_PORTS = (board_res["bram"] + board_res["uram"] * 8)
        NUM_PORTS = int(NUM_PORTS * 1.2)  # 85% of the BRAMs are used for parallelization
        NUM_DSP = board_res["dsp"] * 0.4  # 40% of the DSPs are used for parallelization

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
            n_variables,
            n_constraints,
            model_II,
            frequency,
            time_spent,
            generate_report_file,
            model,
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
