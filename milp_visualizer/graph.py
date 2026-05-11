"""Graph data structures for MILP co-occurrence graphs."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

ExcludeSpec = str | re.Pattern | list[str | re.Pattern] | set[str] | None
GroupSpec = list[str | re.Pattern] | None


def _compile_exclude(exclude: ExcludeSpec):
    """Return a predicate that returns True if a node name should be excluded.

    Plain str: prefix match (startswith). re.Pattern: fullmatch.
    """
    if exclude is None:
        return None
    items = [exclude] if isinstance(exclude, (str, re.Pattern)) else list(exclude)
    prefixes = [x for x in items if isinstance(x, str)]
    patterns = [x for x in items if isinstance(x, re.Pattern)]

    def should_exclude(name: str) -> bool:
        return (
            any(name.startswith(p) for p in prefixes)
            or any(p.fullmatch(name) for p in patterns)
        )

    return should_exclude


def _drop_nodes(graph: CooccurrenceGraph, predicate) -> None:
    """Remove nodes matching predicate in-place."""
    remove = {n for n in graph.nodes if predicate(n)}
    if not remove:
        return
    graph.nodes -= remove
    for n in remove:
        graph.adj.pop(n, None)
    for nbrs in graph.adj.values():
        for n in remove:
            nbrs.pop(n, None)
    if isinstance(graph, VariableGraph):
        for n in remove:
            graph.constraint_count.pop(n, None)
    elif isinstance(graph, ConstraintGraph):
        for n in remove:
            graph.variable_count.pop(n, None)
            graph.constraint_types.pop(n, None)


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


def _glob_to_pattern(spec: str) -> re.Pattern | None:
    """Convert a glob-style string to a compiled regex.

    '*' → (\\d+) capturing group (partitions by value).
    '?' → (?:\\d+) non-capturing group (matches any integer, ignored in partition key).
    Returns None if spec contains neither '*' nor '?'.
    """
    if "*" not in spec and "?" not in spec:
        return None
    tokens = re.split(r"([*?])", spec)
    pattern = ""
    for tok in tokens:
        if tok == "*":
            pattern += r"(\d+)"
        elif tok == "?":
            pattern += r"(?:\d+)"
        else:
            pattern += re.escape(tok)
    return re.compile("^" + pattern)


def _assign_groups(nodes: set[str], groups: list[str | re.Pattern]) -> dict[str, str]:
    """Map each node to a super-node name. First-match wins; unmatched map to themselves.

    Plain str without '*': prefix match → one group.
    Plain str with '*': each '*' captures an integer; partition by captured tuple.
    Regex without capture groups: all matches → one group.
    Regex with capture groups: partitioned by captured tuple → one group per unique capture.
    Super-node name = alphabetically first member of each group.
    """
    node_to_group: dict[str, str] = {}
    for spec in groups:
        if isinstance(spec, str):
            pattern = _glob_to_pattern(spec)
            if pattern is not None:
                # glob with captures — reuse regex capture-group path
                buckets: dict[tuple, list[str]] = defaultdict(list)
                for n in nodes:
                    if n in node_to_group:
                        continue
                    m = pattern.search(n)
                    if m:
                        buckets[m.groups()].append(n)
                for bucket in buckets.values():
                    bucket.sort()
                    group_name = bucket[0]
                    for n in bucket:
                        node_to_group[n] = group_name
            else:
                matched = sorted(n for n in nodes if n.startswith(spec) and n not in node_to_group)
                if not matched:
                    continue
                group_name = matched[0]
                for n in matched:
                    node_to_group[n] = group_name
        else:
            if spec.groups:
                # partition by captured tuple
                buckets: dict[tuple, list[str]] = defaultdict(list)
                for n in nodes:
                    if n in node_to_group:
                        continue
                    m = spec.search(n)
                    if m:
                        buckets[m.groups()].append(n)
                for bucket in buckets.values():
                    bucket.sort()
                    group_name = bucket[0]
                    for n in bucket:
                        node_to_group[n] = group_name
            else:
                matched = sorted(n for n in nodes if spec.search(n) and n not in node_to_group)
                if not matched:
                    continue
                group_name = matched[0]
                for n in matched:
                    node_to_group[n] = group_name
    for n in nodes:
        if n not in node_to_group:
            node_to_group[n] = n
    return node_to_group


def collapse_groups(graph: CooccurrenceGraph, groups: list[str | re.Pattern]) -> CooccurrenceGraph:
    """Collapse each group of matching nodes into one super-node before embedding.

    Super-node name = alphabetically first matched node. Intra-group edges dropped;
    inter-group edge weights summed over all member pairs.
    """
    node_to_group = _assign_groups(graph.nodes, groups)
    new_nodes = set(node_to_group.values())

    new_adj: dict[str, dict[str, int]] = defaultdict(dict)
    for u, nbrs in graph.adj.items():
        gu = node_to_group[u]
        for v, w in nbrs.items():
            gv = node_to_group[v]
            if gu == gv:
                continue
            new_adj[gu][gv] = new_adj[gu].get(gv, 0) + w

    if isinstance(graph, VariableGraph):
        new_graph = VariableGraph()
        new_graph.nodes = new_nodes
        new_graph.adj = defaultdict(dict, new_adj)
        for n, g in node_to_group.items():
            new_graph.constraint_count[g] = (
                new_graph.constraint_count.get(g, 0) + graph.constraint_count.get(n, 0)
            )
    elif isinstance(graph, ConstraintGraph):
        new_graph = ConstraintGraph()
        new_graph.nodes = new_nodes
        new_graph.adj = defaultdict(dict, new_adj)
        for n, g in node_to_group.items():
            new_graph.variable_count[g] = (
                new_graph.variable_count.get(g, 0) + graph.variable_count.get(n, 0)
            )
        for n in sorted(node_to_group):
            g = node_to_group[n]
            if g not in new_graph.constraint_types and n in graph.constraint_types:
                new_graph.constraint_types[g] = graph.constraint_types[n]
        new_graph.name = graph.name
    else:
        new_graph = CooccurrenceGraph()
        new_graph.nodes = new_nodes
        new_graph.adj = defaultdict(dict, new_adj)

    return new_graph


def build_variable_graph(model, exclude: ExcludeSpec = None) -> VariableGraph:
    """Build variable co-occurrence graph from an MPS model.

    Nodes = variables. Edge weight = number of shared constraints.
    Constraints matching exclude are skipped during edge building.
    """
    pred = _compile_exclude(exclude)
    graph = VariableGraph()
    graph.nodes = set(model.variables)

    for row, row_coeffs in model.coefficients.items():
        if row == model.objective:
            continue
        if pred is not None and pred(row):
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


def build_constraint_graph(model, exclude: ExcludeSpec = None) -> ConstraintGraph:
    """Build constraint co-occurrence graph from an MPS model.

    Nodes = constraints. Edge weight = number of shared variables.
    Variables matching exclude are skipped during edge building.
    """
    pred = _compile_exclude(exclude)
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
            if pred is not None and pred(var):
                continue
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
