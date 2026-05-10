"""Gurobi model integration: build co-occurrence graphs from live models."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from ..graph import VariableGraph, ConstraintGraph
from ..mps_parser import MPS


def _gurobi_to_variable_graph(gurobi_model) -> tuple[VariableGraph, MPS]:
    """Return (VariableGraph, pseudo-MPS) built from a live Gurobi model."""
    gurobi_model.update()

    gvars = gurobi_model.getVars()
    var_names = [v.VarName for v in gvars]
    binary_vars: set[str] = set()
    integer_vars: set[str] = set()
    for v in gvars:
        if v.VType == "B":
            binary_vars.add(v.VarName)
            integer_vars.add(v.VarName)
        elif v.VType in ("I", "N"):
            integer_vars.add(v.VarName)

    adj: dict[str, dict[str, int]] = defaultdict(dict)
    nodes: set[str] = set(var_names)
    constraint_count: dict[str, int] = {}

    for constr in gurobi_model.getConstrs():
        row = gurobi_model.getRow(constr)
        vars_in_row = [row.getVar(i).VarName for i in range(row.size())]
        for var in vars_in_row:
            constraint_count[var] = constraint_count.get(var, 0) + 1
        for u, v in combinations(vars_in_row, 2):
            if v in adj[u]:
                adj[u][v] += 1
                adj[v][u] += 1
            else:
                adj[u][v] = 1
                adj[v][u] = 1

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
            for u, v in combinations(seen, 2):
                if v in adj[u]:
                    adj[u][v] += 1
                    adj[v][u] += 1
                else:
                    adj[u][v] = 1
                    adj[v][u] = 1
    except Exception:
        pass

    graph = VariableGraph(adj=adj, nodes=nodes, constraint_count=constraint_count)
    pseudo_mps = MPS(
        name=gurobi_model.ModelName,
        binary_vars=binary_vars,
        integer_vars=integer_vars,
    )
    return graph, pseudo_mps


def _gurobi_to_constraint_graph(gurobi_model) -> ConstraintGraph:
    """Return ConstraintGraph built from a live Gurobi model."""
    gurobi_model.update()

    _sense_map = {"<": "L", ">": "G", "=": "E"}
    adj: dict[str, dict[str, int]] = defaultdict(dict)
    nodes: set[str] = set()
    variable_count: dict[str, int] = {}
    constraint_types: dict[str, str] = {}

    var_to_constrs: dict[str, list[str]] = defaultdict(list)
    for constr in gurobi_model.getConstrs():
        name = constr.ConstrName
        nodes.add(name)
        constraint_types[name] = _sense_map.get(constr.Sense, "L")
        row = gurobi_model.getRow(constr)
        var_names = [row.getVar(i).VarName for i in range(row.size())]
        variable_count[name] = len(var_names)
        for var in var_names:
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
        name=gurobi_model.ModelName,
    )
