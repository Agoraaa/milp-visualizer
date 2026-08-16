"""Internal rendering orchestrators for variable and constraint graphs."""

from __future__ import annotations

import numpy as np
import matplotlib.patches as mpatches

from .graph import VariableGraph, ConstraintGraph
from .embedding import embed, _top_neighbors
from .colors import color_by_prefix, color_by_category, _extract_prefix, _MAX_PREFIX_COLORS
from .render import (
    _VAR_COLORS,
    _CONSTRAINT_COLORS,
    _CONSTRAINT_TYPE_LABELS,
    render_png,
    render_plotly,
)

_FIGSIZE = (12, 10)


def _var_type(node: str, model) -> str:
    if model is None:
        return "continuous"
    if node in model.binary_vars:
        return "binary"
    if node in model.integer_vars:
        return "integer"
    return "continuous"


def _legend_from_colors(nodes: list[str], color_map: dict[str, str]) -> tuple[list, list[tuple[str, str]]]:
    seen: dict[str, str] = {}
    for n in nodes:
        seen[_extract_prefix(n)] = color_map[n]
    patches = [mpatches.Patch(color=c, label=p) for p, c in sorted(seen.items())]
    items = [(p, c) for p, c in sorted(seen.items())]
    return patches, items


def _legend_from_type_colors(label_color_map: dict[str, str]) -> tuple[list, list[tuple[str, str]]]:
    patches = [mpatches.Patch(color=c, label=k) for k, c in label_color_map.items()]
    items = list(label_color_map.items())
    return patches, items


def _resolve_colors_variables(nodes, model, node_categories):
    if node_categories is not None:
        color_map, cat_colors = color_by_category(nodes, node_categories)
        return color_map, *_legend_from_type_colors(cat_colors)
    n_prefixes = len({_extract_prefix(n) for n in nodes})
    if n_prefixes <= _MAX_PREFIX_COLORS:
        color_map = color_by_prefix(nodes)
        return color_map, *_legend_from_colors(nodes, color_map)
    color_map = {
        n: _VAR_COLORS["binary"] if model and n in model.binary_vars
        else _VAR_COLORS["integer"] if model and n in model.integer_vars
        else _VAR_COLORS["continuous"]
        for n in nodes
    }
    return color_map, *_legend_from_type_colors({k.capitalize(): v for k, v in _VAR_COLORS.items()})


def _resolve_colors_constraints(nodes, graph, node_categories):
    if node_categories is not None:
        color_map, cat_colors = color_by_category(nodes, node_categories)
        return color_map, *_legend_from_type_colors(cat_colors)
    n_prefixes = len({_extract_prefix(n) for n in nodes})
    if n_prefixes <= _MAX_PREFIX_COLORS:
        color_map = color_by_prefix(nodes)
        return color_map, *_legend_from_colors(nodes, color_map)
    color_map = {
        n: _CONSTRAINT_COLORS.get(graph.constraint_types.get(n, "L"), _CONSTRAINT_COLORS["L"])
        for n in nodes
    }
    return color_map, *_legend_from_type_colors(
        {f"{_CONSTRAINT_TYPE_LABELS[k]} ({k})": v for k, v in _CONSTRAINT_COLORS.items()}
    )


def _visualize_variables(
    graph: VariableGraph,
    model=None,
    output: str = "variable_graph.html",
    max_neighbors: int | None = None,
    label_nodes: bool | int | None = None,
    node_categories: dict[str, str] | None = None,
) -> None:
    if graph.num_nodes == 0:
        raise ValueError("graph has no nodes")

    coords, nodes = embed(graph)
    xs, ys = coords[:, 0], coords[:, 1]

    color_map, legend_patches, legend_items = _resolve_colors_variables(nodes, model, node_categories)
    colors = [color_map[n] for n in nodes]

    if graph.constraint_count:
        counts = np.array([graph.constraint_count.get(n, 0) for n in nodes], dtype=float)
    else:
        counts = np.asarray(graph.A.sum(axis=1)).flatten().astype(float)
    max_count = counts.max() if counts.max() > 0 else 1
    sizes = 30 + 120 * (counts / max_count)

    name = (model.name if model else None) or "MILP"
    title = f"{name}"
    draw_adj = _top_neighbors(graph, max_neighbors) if max_neighbors is not None else graph.to_adj()
    auto_label = label_nodes if label_nodes is not None else (True if graph.num_nodes <= 50 else False)

    if output.endswith(".html"):
        degree = np.diff(graph.A.indptr)
        node_degree = dict(zip(graph.node_list, degree))
        hover = [
            (
                f"<b>{n}</b><br>"
                f"type: {_var_type(n, model)}<br>"
                f"constraints: {int(counts[i])}<br>"
                f"neighbors: {node_degree.get(n, 0)}"
            )
            for i, n in enumerate(nodes)
        ]
        render_plotly(nodes, xs, ys, colors, sizes, hover, draw_adj, title,
                      output, legend_items, show_edges=True)
    else:
        render_png(nodes, xs, ys, colors, sizes, draw_adj, title, output,
                   _FIGSIZE, show_edges=True, label_nodes=auto_label,
                   legend_patches=legend_patches)
        print(f"saved → {output}  ({graph.num_nodes} nodes, {graph.num_edges} edges)")


def _visualize_constraints(
    graph: ConstraintGraph,
    output: str = "constraint_graph.html",
    max_neighbors: int | None = None,
    label_nodes: bool | int | None = None,
    node_categories: dict[str, str] | None = None,
) -> None:
    if graph.num_nodes == 0:
        raise ValueError("graph has no nodes")

    coords, nodes = embed(graph)
    xs, ys = coords[:, 0], coords[:, 1]

    color_map, legend_patches, legend_items = _resolve_colors_constraints(nodes, graph, node_categories)
    colors = [color_map[n] for n in nodes]

    counts = np.array([graph.variable_count.get(n, 0) for n in nodes], dtype=float)
    max_count = counts.max() if counts.max() > 0 else 1
    sizes = 30 + 120 * (counts / max_count)

    name = getattr(graph, "name", None) or "MILP"
    title = f"{name}"
    draw_adj = _top_neighbors(graph, max_neighbors) if max_neighbors is not None else graph.to_adj()
    auto_label = label_nodes if label_nodes is not None else (True if graph.num_nodes <= 50 else False)

    if output.endswith(".html"):
        degree = np.diff(graph.A.indptr)
        node_degree = dict(zip(graph.node_list, degree))
        hover = [
            (
                f"<b>{n}</b><br>"
                f"type: {_CONSTRAINT_TYPE_LABELS.get(graph.constraint_types.get(n, ''), n)}<br>"
                f"variables: {int(counts[i])}<br>"
                f"neighbors: {node_degree.get(n, 0)}"
            )
            for i, n in enumerate(nodes)
        ]
        render_plotly(nodes, xs, ys, colors, sizes, hover, draw_adj, title,
                      output, legend_items, show_edges=True)
    else:
        render_png(nodes, xs, ys, colors, sizes, draw_adj, title, output,
                   _FIGSIZE, show_edges=True, label_nodes=auto_label,
                   legend_patches=legend_patches)
        print(f"saved → {output}  ({graph.num_nodes} nodes, {graph.num_edges} edges)")
