"""Color utilities for node coloring."""

from __future__ import annotations

import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

_MAX_PREFIX_COLORS = 12


def _extract_prefix(name: str) -> str:
    # everything before the first '[' or first digit, trailing underscores stripped
    m = re.match(r"^([^\[0-9]+)", name)
    return m.group(1).rstrip("_") if m else name


def _generate_palette(n: int) -> list[str]:
    """Generate n visually distinct hex colors from a qualitative colormap."""
    cmap = plt.colormaps["tab20" if n > 10 else "tab10"]
    return [mcolors.to_hex(cmap(i / max(n, 1))) for i in range(n)]


def color_by_prefix(
    nodes: list[str] | set[str],
    palette: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assign colors to nodes based on their name prefix.

    Colors are generated to span the full colormap range for the number of
    unique prefixes found — 3 prefixes get 3 evenly-spaced colors, 10 get 10, etc.

    Prefix extraction rules:
    - everything before the first '[' (Gurobi array syntax: "x[0,1]" → "x")
    - or everything before the first digit ("cap_1" → "cap")

    Args:
        nodes: node names from a VariableGraph or ConstraintGraph
        palette: explicit {prefix: hex_color} overrides; remaining prefixes
                 are auto-assigned from a generated palette

    Returns:
        {node_name: hex_color} dict ready to pass as node_colors to visualize()
    """
    palette = dict(palette) if palette else {}

    # collect ordered unique prefixes
    seen_prefixes: list[str] = []
    node_prefix: dict[str, str] = {}
    for node in nodes:
        p = _extract_prefix(node)
        node_prefix[node] = p
        if p not in seen_prefixes:
            seen_prefixes.append(p)

    # generate palette sized to number of auto-assigned prefixes
    auto_prefixes = [p for p in seen_prefixes if p not in palette]
    auto_colors = _generate_palette(len(auto_prefixes))
    auto_map = dict(zip(auto_prefixes, auto_colors))

    prefix_to_color = {**auto_map, **palette}
    return {node: prefix_to_color[node_prefix[node]] for node in nodes}
