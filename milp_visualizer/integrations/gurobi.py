"""Gurobi model integration: build co-occurrence graphs from live models.

Builds a constraint x variable incidence matrix directly from the model, then
hands off to the shared sparse-matmul builders in graph.py — no dict/combinations
adjacency building here.
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


def _gurobi_to_variable_graph(
    gurobi_model, exclude: ExcludeSpec = None, groups: GroupSpec = None
) -> tuple[VariableGraph, MPS]:
    """Return (VariableGraph, pseudo-MPS) built from a live Gurobi model."""
    gurobi_model.update()

    gvars = gurobi_model.getVars()
    var_names = [v.VarName for v in gvars]
    col_idx = {v: j for j, v in enumerate(var_names)}
    binary_vars: set[str] = set()
    integer_vars: set[str] = set()
    for v in gvars:
        if v.VType == "B":
            binary_vars.add(v.VarName)
            integer_vars.add(v.VarName)
        elif v.VType in ("I", "N"):
            integer_vars.add(v.VarName)

    constrs = gurobi_model.getConstrs()
    constr_names = [c.ConstrName for c in constrs]
    rows, cols, data = [], [], []
    for i, constr in enumerate(constrs):
        row = gurobi_model.getRow(constr)
        for k in range(row.size()):
            rows.append(i)
            cols.append(col_idx[row.getVar(k).VarName])
            data.append(1.0)
    B = sp.csr_matrix(
        (data, (rows, cols)) if data else ([], ([], [])),
        shape=(len(constrs), len(var_names)),
    )

    # quadratic constraints contribute extra co-occurrence signal as pseudo-rows
    q_rows, q_cols, q_data = [], [], []
    q_count = 0
    try:
        for qconstr in gurobi_model.getQConstrs():
            qrow = gurobi_model.getQCRow(qconstr)
            seen: set[str] = set()
            lin = qrow.getLinExpr()
            for i in range(lin.size()):
                seen.add(lin.getVar(i).VarName)
            for i in range(qrow.size()):
                seen.add(qrow.getVar1(i).VarName)
                seen.add(qrow.getVar2(i).VarName)
            for name in seen:
                if name in col_idx:
                    q_rows.append(q_count)
                    q_cols.append(col_idx[name])
                    q_data.append(1.0)
            q_count += 1
    except Exception:
        pass

    if q_count:
        Bq = sp.csr_matrix((q_data, (q_rows, q_cols)), shape=(q_count, len(var_names)))
        B = sp.vstack([B, Bq]).tocsr()
        constr_names = constr_names + [f"_qconstr{i}" for i in range(q_count)]

    graph = build_variable_graph_from_incidence(B, constr_names, var_names, exclude=exclude, groups=groups)
    pseudo_mps = MPS(
        name=gurobi_model.ModelName,
        binary_vars=binary_vars,
        integer_vars=integer_vars,
    )
    return graph, pseudo_mps


def _gurobi_to_constraint_graph(
    gurobi_model, exclude: ExcludeSpec = None, groups: GroupSpec = None
) -> ConstraintGraph:
    """Return ConstraintGraph built from a live Gurobi model."""
    gurobi_model.update()
    _sense_map = {"<": "L", ">": "G", "=": "E"}

    var_names = [v.VarName for v in gurobi_model.getVars()]
    col_idx = {v: j for j, v in enumerate(var_names)}

    constrs = gurobi_model.getConstrs()
    constr_names = [c.ConstrName for c in constrs]
    constraint_types = {c.ConstrName: _sense_map.get(c.Sense, "L") for c in constrs}

    rows, cols, data = [], [], []
    for i, constr in enumerate(constrs):
        row = gurobi_model.getRow(constr)
        for k in range(row.size()):
            rows.append(i)
            cols.append(col_idx[row.getVar(k).VarName])
            data.append(1.0)
    B = sp.csr_matrix(
        (data, (rows, cols)) if data else ([], ([], [])),
        shape=(len(constrs), len(var_names)),
    )

    return build_constraint_graph_from_incidence(
        B, constr_names, var_names, constraint_types,
        exclude=exclude, groups=groups, name=gurobi_model.ModelName,
    )
