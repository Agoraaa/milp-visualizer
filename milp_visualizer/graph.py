"""Graph data structures for MILP co-occurrence graphs — sparse-matrix native.

A CooccurrenceGraph holds its adjacency as a scipy sparse CSR matrix `A` plus an
index-ordered `node_list`. Nothing before rendering ever materializes a dict —
exclude/group filtering happens on the constraint x variable incidence matrix
`B` before the co-occurrence matmul (Bᵀ@B or B@Bᵀ), and pruning (`_top_neighbors`)
runs on `A` directly. A dict-of-dicts view (`to_adj`) is built once, only at
render time, from the smallest the matrix will ever be.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

ExcludeSpec = str | list[str] | set[str] | None
GroupSpec = list[str] | None

_EMPTY = sp.csr_matrix((0, 0))


def _compile_exclude(exclude: ExcludeSpec):
    """Return a predicate that returns True if a node name should be excluded.

    Plain str without '*'/'?': prefix match (startswith).
    Plain str with '*'/'?': glob match (see _glob_to_pattern) — any match excludes.
    """
    if exclude is None:
        return None
    items = [exclude] if isinstance(exclude, str) else list(exclude)
    prefixes = [x for x in items if _glob_to_pattern(x) is None]
    patterns = [_glob_to_pattern(x) for x in items if _glob_to_pattern(x) is not None]

    def should_exclude(name: str) -> bool:
        return (
            any(name.startswith(p) for p in prefixes)
            or any(p.search(name) for p in patterns)
        )

    return should_exclude


@dataclass
class CooccurrenceGraph:
    A: sp.csr_matrix = field(default_factory=lambda: _EMPTY)
    node_list: list[str] = field(default_factory=list)

    @property
    def nodes(self) -> set[str]:
        return set(self.node_list)

    @property
    def num_nodes(self) -> int:
        return len(self.node_list)

    @property
    def num_edges(self) -> int:
        return self.A.nnz // 2

    def neighbors(self, node: str) -> dict[str, int]:
        try:
            i = self.node_list.index(node)
        except ValueError:
            return {}
        row = self.A.getrow(i).tocoo()
        return {self.node_list[j]: int(w) for j, w in zip(row.col, row.data) if w}

    def weight(self, u: str, v: str) -> int:
        return self.neighbors(u).get(v, 0)

    def to_adj(self) -> dict[str, dict[str, int]]:
        """Materialize dict-of-dicts adjacency — render-time convenience only."""
        coo = self.A.tocoo()
        adj: dict[str, dict[str, int]] = defaultdict(dict)
        for i, j, w in zip(coo.row, coo.col, coo.data):
            if w:
                adj[self.node_list[i]][self.node_list[j]] = int(w)
        return adj


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


def _assign_groups(nodes: set[str], groups: list[str]) -> dict[str, str]:
    """Map each node to a super-node name. First-match wins; unmatched map to themselves.

    Plain str without '*'/'?': prefix match → one group.
    Plain str with '*'/'?': each '*' captures an integer; partition by captured tuple
        ('?' matches any integer but doesn't partition on it).
    Super-node name = alphabetically first member of each group.
    """
    node_to_group: dict[str, str] = {}
    for spec in groups:
        pattern = _glob_to_pattern(spec)
        if pattern is not None:
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
    for n in nodes:
        if n not in node_to_group:
            node_to_group[n] = n
    return node_to_group


def _drop_from_incidence(B: sp.csr_matrix, row_names: list[str], col_names: list[str], pred):
    """Drop rows/columns whose name matches pred, before any matmul."""
    if pred is None:
        return B, row_names, col_names
    row_keep = np.array([not pred(n) for n in row_names], dtype=bool)
    col_keep = np.array([not pred(n) for n in col_names], dtype=bool)
    B = B[row_keep][:, col_keep]
    row_names = [n for n, keep in zip(row_names, row_keep) if keep]
    col_names = [n for n, keep in zip(col_names, col_keep) if keep]
    return B, row_names, col_names


def _group_matrix(names: list[str], groups: list[str]):
    """Build a names x supernodes indicator matrix for merging by group."""
    node_to_group = _assign_groups(set(names), groups)
    supernodes = sorted(set(node_to_group.values()))
    sn_idx = {s: i for i, s in enumerate(supernodes)}
    rows = list(range(len(names)))
    cols = [sn_idx[node_to_group[n]] for n in names]
    data = [1.0] * len(names)
    G = sp.csr_matrix((data, (rows, cols)), shape=(len(names), len(supernodes)))
    return G, supernodes, node_to_group


def _top_neighbors_sparse(A: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep only the k heaviest entries per row, sparse throughout."""
    A = A.tocsr()
    indptr, data = A.indptr, A.data.copy()
    for i in range(A.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        row_len = end - start
        if row_len <= k:
            continue
        row = data[start:end]
        drop = np.argpartition(row, row_len - k)[: row_len - k]
        row[drop] = 0
    pruned = sp.csr_matrix((data, A.indices, indptr), shape=A.shape)
    pruned.eliminate_zeros()
    return pruned


def build_variable_graph_from_incidence(
    B: sp.csr_matrix,
    constraints: list[str],
    variables: list[str],
    exclude: ExcludeSpec = None,
    groups: GroupSpec = None,
) -> VariableGraph:
    """Build a VariableGraph from a constraint x variable incidence matrix.

    Shared core for every source (MPS/LP, Gurobi, HiGHS, OR-Tools) — each source
    only needs to build B; drop → group → co-occurrence (Bᵀ@B) happens here, sparse
    throughout.
    """
    pred = _compile_exclude(exclude)
    B, constraints, variables = _drop_from_incidence(B, constraints, variables, pred)

    if groups is not None:
        G, variables, _ = _group_matrix(variables, groups)
        B = B @ G

    A = (B.T @ B).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    counts = np.asarray(B.sum(axis=0)).flatten()

    graph = VariableGraph(A=A, node_list=variables)
    graph.constraint_count = {v: int(c) for v, c in zip(variables, counts) if c > 0}
    return graph


def build_constraint_graph_from_incidence(
    B: sp.csr_matrix,
    constraints: list[str],
    variables: list[str],
    constraint_types: dict[str, str],
    exclude: ExcludeSpec = None,
    groups: GroupSpec = None,
    name: str = "",
) -> ConstraintGraph:
    """Build a ConstraintGraph from a constraint x variable incidence matrix.

    Shared core — see build_variable_graph_from_incidence. Co-occurrence = B@Bᵀ.
    """
    pred = _compile_exclude(exclude)
    B, constraints, variables = _drop_from_incidence(B, constraints, variables, pred)
    types = {r: constraint_types.get(r, "L") for r in constraints}

    if groups is not None:
        G, constraints, node_to_group = _group_matrix(constraints, groups)
        B = (G.T @ B).tocsr()
        grouped_types: dict[str, str] = {}
        for n in sorted(node_to_group):
            g = node_to_group[n]
            if g not in grouped_types and n in types:
                grouped_types[g] = types[n]
        types = grouped_types

    A = (B @ B.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    counts = np.asarray(B.sum(axis=1)).flatten()

    graph = ConstraintGraph(A=A, node_list=constraints)
    graph.variable_count = {c: int(v) for c, v in zip(constraints, counts) if v > 0}
    graph.constraint_types = types
    graph.name = name
    return graph


def _build_incidence(model) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Build sparse constraint x variable incidence matrix from a parsed MPS model."""
    constraints = model.constraints
    variables = model.variables
    row_idx = {r: i for i, r in enumerate(constraints)}
    col_idx = {v: j for j, v in enumerate(variables)}

    rows, cols, data = [], [], []
    for row in constraints:
        for var in model.coefficients.get(row, {}):
            rows.append(row_idx[row])
            cols.append(col_idx[var])
            data.append(1.0)

    B = sp.csr_matrix(
        (data, (rows, cols)) if data else ([], ([], [])),
        shape=(len(constraints), len(variables)),
    )
    return B, constraints, variables


def build_variable_graph(model, exclude: ExcludeSpec = None, groups: GroupSpec = None) -> VariableGraph:
    """Build variable co-occurrence graph from a parsed MPS model."""
    B, constraints, variables = _build_incidence(model)
    return build_variable_graph_from_incidence(B, constraints, variables, exclude=exclude, groups=groups)


def build_constraint_graph(model, exclude: ExcludeSpec = None, groups: GroupSpec = None) -> ConstraintGraph:
    """Build constraint co-occurrence graph from a parsed MPS model."""
    B, constraints, variables = _build_incidence(model)
    types = {r: model.row_types.get(r, "L") for r in constraints}
    return build_constraint_graph_from_incidence(
        B, constraints, variables, types, exclude=exclude, groups=groups, name=model.name
    )
