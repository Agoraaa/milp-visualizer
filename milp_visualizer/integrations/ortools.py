"""OR-Tools model integration: build co-occurrence graphs from pywraplp Solver models."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from ..graph import VariableGraph, ConstraintGraph, ExcludeSpec, _compile_exclude
from ..mps_parser import MPS


def _ortools_to_variable_graph(solver, exclude: ExcludeSpec = None) -> tuple[VariableGraph, MPS]:
    variables = solver.variables()
    constraints = solver.constraints()
    pred = _compile_exclude(exclude)

    binary_vars: set[str] = set()
    integer_vars: set[str] = set()
    for v in variables:
        if v.integer():
            integer_vars.add(v.name())
            if v.lb() == 0.0 and v.ub() == 1.0:
                binary_vars.add(v.name())

    adj: dict[str, dict[str, int]] = defaultdict(dict)
    nodes: set[str] = {v.name() for v in variables}
    constraint_count: dict[str, int] = {}

    for constr in constraints:
        if pred is not None and pred(constr.name()):
            continue
        vars_in_constr = [v.name() for v in variables if constr.GetCoefficient(v) != 0.0]
        for name in vars_in_constr:
            constraint_count[name] = constraint_count.get(name, 0) + 1
        for u, v in combinations(vars_in_constr, 2):
            if v in adj[u]:
                adj[u][v] += 1
                adj[v][u] += 1
            else:
                adj[u][v] = 1
                adj[v][u] = 1

    graph = VariableGraph(adj=adj, nodes=nodes, constraint_count=constraint_count)
    pseudo_mps = MPS(binary_vars=binary_vars, integer_vars=integer_vars)
    return graph, pseudo_mps


def _ortools_to_constraint_graph(solver, exclude: ExcludeSpec = None) -> ConstraintGraph:
    variables = solver.variables()
    constraints = solver.constraints()
    pred = _compile_exclude(exclude)

    adj: dict[str, dict[str, int]] = defaultdict(dict)
    nodes: set[str] = {c.name() for c in constraints}
    variable_count: dict[str, int] = {}
    constraint_types: dict[str, str] = {}

    for constr in constraints:
        name = constr.name()
        lo, hi = constr.lb(), constr.ub()
        if lo == hi:
            constraint_types[name] = "E"
        elif lo == float("-inf"):
            constraint_types[name] = "L"
        elif hi == float("inf"):
            constraint_types[name] = "G"
        else:
            constraint_types[name] = "L"

    var_to_constrs: dict[str, list[str]] = defaultdict(list)
    for constr in constraints:
        name = constr.name()
        vars_in_constr = [v.name() for v in variables if constr.GetCoefficient(v) != 0.0]
        variable_count[name] = len(vars_in_constr)
        for var in vars_in_constr:
            if pred is not None and pred(var):
                continue
            var_to_constrs[var].append(name)

    for constrs in var_to_constrs.values():
        for u, v in combinations(constrs, 2):
            if v in adj[u]:
                adj[u][v] += 1
                adj[v][u] += 1
            else:
                adj[u][v] = 1
                adj[v][u] = 1

    return ConstraintGraph(
        adj=adj,
        nodes=nodes,
        variable_count=variable_count,
        constraint_types=constraint_types,
    )
