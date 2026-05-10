"""Graph data structures for MILP co-occurrence graphs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class CooccurrenceGraph:
    adj: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    nodes: set[str] = field(default_factory=set)

    def neighbors(self, node: str) -> dict[str, int]:
        return self.adj.get(node, {})

    def weight(self, u: str, v: str) -> int:
        return self.adj.get(u, {}).get(v, 0)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return sum(len(nbrs) for nbrs in self.adj.values()) // 2


@dataclass
class VariableGraph(CooccurrenceGraph):
    # number of constraints each variable appears in
    constraint_count: dict[str, int] = field(default_factory=dict)


@dataclass
class ConstraintGraph(CooccurrenceGraph):
    # number of variables each constraint contains
    variable_count: dict[str, int] = field(default_factory=dict)
    # row sense: 'L' (<=), 'G' (>=), 'E' (=)
    constraint_types: dict[str, str] = field(default_factory=dict)
    name: str = ""


def build_variable_graph(model) -> VariableGraph:
    """Build variable co-occurrence graph from an MPS model.

    Nodes = variables. Edge weight = number of shared constraints.
    """
    graph = VariableGraph()
    graph.nodes = set(model.variables)

    for row, row_coeffs in model.coefficients.items():
        if row == model.objective:
            continue
        vars_in_row = list(row_coeffs.keys())
        for var in vars_in_row:
            graph.constraint_count[var] = graph.constraint_count.get(var, 0) + 1
        for u, v in combinations(vars_in_row, 2):
            if v in graph.adj[u]:
                graph.adj[u][v] += 1
                graph.adj[v][u] += 1
            else:
                graph.adj[u][v] = 1
                graph.adj[v][u] = 1

    return graph


def build_constraint_graph(model) -> ConstraintGraph:
    """Build constraint co-occurrence graph from an MPS model.

    Nodes = constraints. Edge weight = number of shared variables.
    """
    graph = ConstraintGraph()

    for row, sense in model.row_types.items():
        if row == model.objective:
            continue
        graph.nodes.add(row)
        graph.constraint_types[row] = sense

    var_to_constrs: dict[str, list[str]] = defaultdict(list)
    for row, row_coeffs in model.coefficients.items():
        if row == model.objective or row not in graph.nodes:
            continue
        graph.variable_count[row] = len(row_coeffs)
        for var in row_coeffs:
            var_to_constrs[var].append(row)

    for constrs in var_to_constrs.values():
        for u, v in combinations(constrs, 2):
            if v in graph.adj[u]:
                graph.adj[u][v] += 1
                graph.adj[v][u] += 1
            else:
                graph.adj[u][v] = 1
                graph.adj[v][u] = 1

    return graph
