"""HiGHS model integration: build co-occurrence graphs from live highspy models.

HiGHS already stores its constraint matrix as CSC (columns = variables, row indices
= constraints touched) — exactly scipy's csc_matrix layout, so B is a zero-copy wrap
of matrix.start_/index_, no Python-level loop needed. Shared sparse-matmul builders
in graph.py take it from there.
"""

from __future__ import annotations

import math

import numpy as np
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


def _highs_incidence(model) -> tuple[sp.csr_matrix, list[str], list[str]]:
    lp = model.getLp()
    col_names = list(lp.col_names_)
    row_names = list(lp.row_names_)
    matrix = lp.a_matrix_
    indptr = np.asarray(matrix.start_)
    indices = np.asarray(matrix.index_)
    data = np.ones(len(indices))
    B = sp.csc_matrix((data, indices, indptr), shape=(len(row_names), len(col_names))).tocsr()
    return B, row_names, col_names


def _highs_to_variable_graph(
    model, exclude: ExcludeSpec = None, groups: GroupSpec = None
) -> tuple[VariableGraph, MPS]:
    lp = model.getLp()
    col_names = list(lp.col_names_)
    B, row_names, col_names = _highs_incidence(model)

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

    graph = build_variable_graph_from_incidence(B, row_names, col_names, exclude=exclude, groups=groups)
    pseudo_mps = MPS(
        name=getattr(model, "model_name_", "") or "",
        binary_vars=binary_vars,
        integer_vars=integer_vars,
    )
    return graph, pseudo_mps


def _highs_to_constraint_graph(
    model, exclude: ExcludeSpec = None, groups: GroupSpec = None
) -> ConstraintGraph:
    lp = model.getLp()
    row_lower = list(lp.row_lower_)
    row_upper = list(lp.row_upper_)
    B, row_names, col_names = _highs_incidence(model)

    def _sense(lo: float, hi: float) -> str:
        if lo == hi:
            return "E"
        if math.isinf(lo):
            return "L"
        if math.isinf(hi):
            return "G"
        return "L"

    constraint_types = {name: _sense(row_lower[i], row_upper[i]) for i, name in enumerate(row_names)}

    return build_constraint_graph_from_incidence(
        B, row_names, col_names, constraint_types,
        exclude=exclude, groups=groups, name=getattr(model, "model_name_", "") or "",
    )
