"""Matplotlib and Plotly rendering functions."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


_VAR_COLORS = {
    "binary": "#e74c3c",
    "integer": "#f39c12",
    "continuous": "#2980b9",
}

_CONSTRAINT_COLORS = {
    "L": "#27ae60",
    "G": "#8e44ad",
    "E": "#e67e22",
}

_CONSTRAINT_TYPE_LABELS = {"L": "≤", "G": "≥", "E": "="}


def render_png(
    nodes: list[str],
    xs: np.ndarray,
    ys: np.ndarray,
    colors: list[str],
    sizes: np.ndarray,
    draw_adj: dict,
    title: str,
    output: str,
    figsize: tuple[int, int],
    show_edges: bool,
    label_nodes: bool | int,
    legend_patches: list,
) -> None:
    pos = {n: (xs[i], ys[i]) for i, n in enumerate(nodes)}
    fig, ax = plt.subplots(figsize=figsize)

    if show_edges and draw_adj:
        all_weights = [w for nbrs in draw_adj.values() for w in nbrs.values()]
        max_w = max(all_weights) if all_weights else 1
        seen: set[tuple[str, str]] = set()
        for u, nbrs in draw_adj.items():
            for v, w in nbrs.items():
                if (v, u) in seen:
                    continue
                seen.add((u, v))
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                ax.plot([x0, x1], [y0, y1], color="#cccccc",
                        alpha=float(0.15 + 0.6 * w / max_w), linewidth=0.5, zorder=0)

    ax.scatter(xs, ys, c=colors, s=sizes, zorder=2, edgecolors="white", linewidths=0.4)

    if label_nodes:
        if isinstance(label_nodes, int) and not isinstance(label_nodes, bool):
            rng = np.random.default_rng()
            k = min(label_nodes, len(nodes))
            label_set = {nodes[i] for i in rng.choice(len(nodes), size=k, replace=False)}
        else:
            label_set = set(nodes)
        for n, (x, y) in pos.items():
            if n in label_set:
                ax.annotate(n, (x, y), fontsize=6, ha="center", va="bottom",
                            xytext=(0, 4), textcoords="offset points", zorder=3)

    if legend_patches:
        ax.legend(handles=legend_patches, loc="best", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Embedding dim 1")
    ax.set_ylabel("Embedding dim 2")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def render_plotly(
    nodes: list[str],
    xs: np.ndarray,
    ys: np.ndarray,
    colors: list[str],
    sizes: np.ndarray,
    hover_texts: list[str],
    draw_adj: dict,
    title: str,
    output_html: str,
    legend_items: list[tuple[str, str]],
    show_edges: bool,
) -> None:
    import plotly.graph_objects as go

    traces: list[go.BaseTraceType] = []
    pos = dict(zip(nodes, zip(xs.tolist(), ys.tolist())))

    if show_edges and draw_adj:
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        seen: set[tuple[str, str]] = set()
        for u, nbrs in draw_adj.items():
            for v in nbrs:
                if (v, u) in seen:
                    continue
                seen.add((u, v))
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
        traces.append(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(color="#dddddd", width=0.5),
            hoverinfo="none",
            showlegend=False,
        ))

    px_sizes = 6 + 14 * (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1)
    traces.append(go.Scatter(
        x=xs.tolist(), y=ys.tolist(),
        mode="markers",
        marker=dict(
            color=colors,
            size=px_sizes.tolist(),
            line=dict(color="white", width=0.5),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    for label, color in legend_items:
        traces.append(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=color, size=10),
            name=label,
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        hovermode="closest",
        plot_bgcolor="white",
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(output_html)
    print(f"saved → {output_html}  (interactive)")
