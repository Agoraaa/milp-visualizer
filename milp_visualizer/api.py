"""Unified visualize() entry point — dispatches on source type."""

from __future__ import annotations

from pathlib import Path

from .mps_parser import parse
from .graph import build_variable_graph, build_constraint_graph
from .visualize import _visualize_variables, _visualize_constraints
from .integrations.gurobi import _gurobi_to_variable_graph, _gurobi_to_constraint_graph

_VALID_MODES = ("variables", "constraints")


def visualize(
    source,
    output: str | None = None,
    *,
    mode: str = "variables",
    **kwargs,
) -> None:
    """Visualize a MILP model's co-occurrence structure.

    Args:
        source: one of —
            str | Path       : path to an MPS file
            gurobipy.Model   : live Gurobi model (need not be solved)
        output: output file path — extension determines format: ".png" (matplotlib) or ".html" (Plotly interactive)
        mode: "variables" (default) or "constraints"
        **kwargs: forwarded to the renderer —
            max_neighbors : max edges drawn per node
            label_nodes   : annotate node names (default: auto for <=50 nodes)
            node_colors   : {node_name: hex_color} override
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

    if output is None:
        output = "variable_graph.html" if mode == "variables" else "constraint_graph.html"

    if isinstance(source, (str, Path)):
        mps = parse(source)
        if mode == "variables":
            g = build_variable_graph(mps)
            _visualize_variables(g, model=mps, output=output, **kwargs)
        else:
            g = build_constraint_graph(mps)
            _visualize_constraints(g, output=output, **kwargs)

    elif hasattr(source, "getConstrs"):  # gurobipy.Model
        if mode == "variables":
            g, pseudo_mps = _gurobi_to_variable_graph(source)
            _visualize_variables(g, model=pseudo_mps, output=output, **kwargs)
        else:
            g = _gurobi_to_constraint_graph(source)
            _visualize_constraints(g, output=output, **kwargs)

    else:
        raise TypeError(
            f"unsupported source type {type(source).__name__!r}; "
            "expected a file path (str/Path) or a supported solver model (gurobipy.Model)"
        )
