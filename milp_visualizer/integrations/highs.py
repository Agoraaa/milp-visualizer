"""HiGHS model integration: build co-occurrence graphs from live highspy models."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

from ..graph import VariableGraph, ConstraintGraph, ExcludeSpec, _compile_exclude
from ..mps_parser import MPS


def _highs_to_variable_graph(model, exclude: ExcludeSpec = None) -> tuple[VariableGraph, MPS]:
    lp = model.getLp()
    col_names = list(lp.col_names_)
    row_names = list(lp.row_names_)
    matrix = lp.a_matrix_
    starts = list(matrix.start_)
    indices = list(matrix.index_)

    integrality = list(lp.integrality_) if lp.integrality_ else []
    col_lower = list(lp.col_lower_)
    col_upper = list(lp.col_upper_)

    binary_vars: set[str] = set()
    integer_vars: set[str] = set()
    for i, name in enumerate(col_names):
        if integrality and integrality[i] == 1:
            integer_vars.add(name)
            if col_lower[i] == 0.0 and col_upper[i] == 1.0:
                binary_vars.add(name)

    # Build row -> vars from CSC matrix
    row_to_vars: dict[int, list[str]] = defaultdict(list)
    for col_idx, col_name in enumerate(col_names):
        start = starts[col_idx]
        end = starts[col_idx + 1] if col_idx + 1 < len(starts) else len(indices)
        for row_idx in indices[start:end]:
            row_to_vars[row_idx].append(col_name)

    pred = _compile_exclude(exclude)
    adj: dict[str, dict[str, int]] = defaultdict(dict)
    nodes: set[str] = set(col_names)
    constraint_count: dict[str, int] = {}

    for row_idx, vars_in_row in row_to_vars.items():
        if pred is not None and pred(row_names[row_idx]):
            continue
        for var in vars_in_row:
            constraint_count[var] = constraint_count.get(var, 0) + 1
        for u, v in combinations(vars_in_row, 2):
            if v in adj[u]:
                adj[u][v] += 1
                adj[v][u] += 1
            else:
                adj[u][v] = 1
                adj[v][u] = 1

    graph = VariableGraph(adj=adj, nodes=nodes, constraint_count=constraint_count)
    pseudo_mps = MPS(
        name=getattr(model, "model_name_", "") or "",
        binary_vars=binary_vars,
        integer_vars=integer_vars,
    )
    return graph, pseudo_mps


def _highs_to_constraint_graph(model, exclude: ExcludeSpec = None) -> ConstraintGraph:
    lp = model.getLp()
    col_names = list(lp.col_names_)
    row_names = list(lp.row_names_)
    matrix = lp.a_matrix_
    starts = list(matrix.start_)
    indices = list(matrix.index_)
    row_lower = list(lp.row_lower_)
    row_upper = list(lp.row_upper_)

    def _sense(lo: float, hi: float) -> str:
        if lo == hi:
            return "E"
        if math.isinf(lo):
            return "L"
        if math.isinf(hi):
            return "G"
        return "L"

    adj: dict[str, dict[str, int]] = defaultdict(dict)
    nodes: set[str] = set(row_names)
    variable_count: dict[str, int] = {}
    constraint_types: dict[str, str] = {}

    for i, name in enumerate(row_names):
        constraint_types[name] = _sense(row_lower[i], row_upper[i])

    # Build row -> vars from CSC matrix
    row_to_vars: dict[int, list[str]] = defaultdict(list)
    for col_idx, col_name in enumerate(col_names):
        start = starts[col_idx]
        end = starts[col_idx + 1] if col_idx + 1 < len(starts) else len(indices)
        for row_idx in indices[start:end]:
            row_to_vars[row_idx].append(col_name)

    pred = _compile_exclude(exclude)
    var_to_constrs: dict[str, list[str]] = defaultdict(list)
    for row_idx, vars_in_row in row_to_vars.items():
        name = row_names[row_idx]
        variable_count[name] = len(vars_in_row)
        for var in vars_in_row:
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
        name=getattr(model, "model_name_", "") or "",
    )
