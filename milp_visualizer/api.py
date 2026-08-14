"""Unified visualize() entry point — dispatches on source type."""

from __future__ import annotations

from pathlib import Path

from .mps_parser import parse as _parse_mps
from .lp_parser import parse_lp as _parse_lp
from .graph import build_variable_graph, build_constraint_graph, ExcludeSpec, GroupSpec
from .visualize import _visualize_variables, _visualize_constraints
from .integrations.gurobi import _gurobi_to_variable_graph, _gurobi_to_constraint_graph
from .integrations.highs import _highs_to_variable_graph, _highs_to_constraint_graph
from .integrations.ortools import _ortools_to_variable_graph, _ortools_to_constraint_graph

_VALID_MODES = ("variables", "constraints")


def visualize(
    source,
    output: str | None = None,
    *,
    mode: str = "variables",
    exclude: ExcludeSpec = None,
    max_neighbors: int | None = None,
    label_nodes: bool | int | None = None,
    node_categories: dict[str, str] | None = None,
    groups: GroupSpec = None,
) -> None:
    """Visualize a MILP model's co-occurrence structure.

    Args:
        source: one of —
            str | Path       : path to an MPS or LP file
            gurobipy.Model   : live Gurobi model (need not be solved)
            highspy.Highs    : live HiGHS model
            pywraplp.Solver  : live OR-Tools model
        output: output file path — extension determines format: ".png" (matplotlib) or ".html" (Plotly interactive)
        mode: "variables" (default) or "constraints"
        exclude: nodes to drop before graph construction —
            str              : prefix match, or glob ('*'/'?') if it contains either
            list/set         : mix of the above
        max_neighbors: max edges drawn per node
        label_nodes: annotate node names (default: auto for <=50 nodes)
        node_categories: {node_name: hex_color} override
        groups: list of str — prefix match, or glob ('*'/'?') to partition by captured
            integer — matched nodes collapsed into one super-node before embedding;
            super-node name = first matched node
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

    if output is None:
        output = "variable_graph.html" if mode == "variables" else "constraint_graph.html"

    if isinstance(source, (str, Path)):
        p = Path(source)
        mps = _parse_lp(p) if p.suffix.lower() == '.lp' else _parse_mps(p)
        if mode == "variables":
            g = build_variable_graph(mps, exclude=exclude, groups=groups)
            _visualize_variables(g, model=mps, output=output,
                                 max_neighbors=max_neighbors, label_nodes=label_nodes,
                                 node_categories=node_categories)
        else:
            g = build_constraint_graph(mps, exclude=exclude, groups=groups)
            _visualize_constraints(g, output=output,
                                   max_neighbors=max_neighbors, label_nodes=label_nodes,
                                   node_categories=node_categories)

    elif hasattr(source, "getConstrs"):  # gurobipy.Model
        if mode == "variables":
            g, pseudo_mps = _gurobi_to_variable_graph(source, exclude=exclude, groups=groups)
            _visualize_variables(g, model=pseudo_mps, output=output,
                                 max_neighbors=max_neighbors, label_nodes=label_nodes,
                                 node_categories=node_categories)
        else:
            g = _gurobi_to_constraint_graph(source, exclude=exclude, groups=groups)
            _visualize_constraints(g, output=output,
                                   max_neighbors=max_neighbors, label_nodes=label_nodes,
                                   node_categories=node_categories)

    elif hasattr(source, "getLp"):  # highspy.Highs
        if mode == "variables":
            g, pseudo_mps = _highs_to_variable_graph(source, exclude=exclude, groups=groups)
            _visualize_variables(g, model=pseudo_mps, output=output,
                                 max_neighbors=max_neighbors, label_nodes=label_nodes,
                                 node_categories=node_categories)
        else:
            g = _highs_to_constraint_graph(source, exclude=exclude, groups=groups)
            _visualize_constraints(g, output=output,
                                   max_neighbors=max_neighbors, label_nodes=label_nodes,
                                   node_categories=node_categories)

    elif hasattr(source, "NumVariables") and hasattr(source, "NumConstraints"):  # ortools pywraplp.Solver
        if mode == "variables":
            g, pseudo_mps = _ortools_to_variable_graph(source, exclude=exclude, groups=groups)
            _visualize_variables(g, model=pseudo_mps, output=output,
                                 max_neighbors=max_neighbors, label_nodes=label_nodes,
                                 node_categories=node_categories)
        else:
            g = _ortools_to_constraint_graph(source, exclude=exclude, groups=groups)
            _visualize_constraints(g, output=output,
                                   max_neighbors=max_neighbors, label_nodes=label_nodes,
                                   node_categories=node_categories)

    else:
        raise TypeError(
            f"unsupported source type {type(source).__name__!r}; "
            "expected a file path (str/Path) or a supported solver model "
            "(gurobipy.Model, highspy.Highs, ortools.linear_solver.pywraplp.Solver)"
        )
