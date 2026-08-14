"""OR-Tools model integration: build co-occurrence graphs from pywraplp Solver models.

pywraplp has no bulk "vars in constraint" API — building B still costs
O(variables x constraints) via GetCoefficient per pair, same as before. What
changes is that this now only builds the incidence matrix; drop/group/matmul
are handled by the shared sparse builders in graph.py.
"""

from __future__ import annotations

import scipy.sparse as sp

from ..graph import (
    VariableGraph,
    ConstraintGraph,
    ExcludeSpec,
    GroupSpec,
    build_variable_graph_from_incidence,
    build_constraint_graph_from_incidence,
)
from ..mps_parser import MPS


def _ortools_incidence(solver) -> tuple[sp.csr_matrix, list, list[str], list[str]]:
    variables = solver.variables()
    constraints = solver.constraints()
    var_names = [v.name() for v in variables]
    constr_names = [c.name() for c in constraints]
    col_idx = {n: j for j, n in enumerate(var_names)}

    rows, cols, data = [], [], []
    for i, constr in enumerate(constraints):
        for v in variables:
            if constr.GetCoefficient(v) != 0.0:
                rows.append(i)
                cols.append(col_idx[v.name()])
                data.append(1.0)
    B = sp.csr_matrix(
        (data, (rows, cols)) if data else ([], ([], [])),
        shape=(len(constraints), len(var_names)),
    )
    return B, constraints, constr_names, var_names


def _ortools_to_variable_graph(
    solver, exclude: ExcludeSpec = None, groups: GroupSpec = None
) -> tuple[VariableGraph, MPS]:
    variables = solver.variables()
    B, _constraints, constr_names, var_names = _ortools_incidence(solver)

    binary_vars: set[str] = set()
    integer_vars: set[str] = set()
    for v in variables:
        if v.integer():
            integer_vars.add(v.name())
            if v.lb() == 0.0 and v.ub() == 1.0:
                binary_vars.add(v.name())

    graph = build_variable_graph_from_incidence(B, constr_names, var_names, exclude=exclude, groups=groups)
    pseudo_mps = MPS(binary_vars=binary_vars, integer_vars=integer_vars)
    return graph, pseudo_mps


def _ortools_to_constraint_graph(
    solver, exclude: ExcludeSpec = None, groups: GroupSpec = None
) -> ConstraintGraph:
    B, constraints, constr_names, var_names = _ortools_incidence(solver)

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

    return build_constraint_graph_from_incidence(
        B, constr_names, var_names, constraint_types, exclude=exclude, groups=groups,
    )
