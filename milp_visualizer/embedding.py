"""Graph embedding utilities: csrgraph conversion, dimensionality reduction, edge filtering."""

from __future__ import annotations

import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]

import scipy.sparse as sp
import csrgraph as cg
import nodevectors


def _to_csrgraph(graph) -> tuple[cg.csrgraph, list[str]]:
    nodes = sorted(graph.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    rows, cols, data = [], [], []
    for u, nbrs in graph.adj.items():
        for v, w in nbrs.items():
            rows.append(idx[u])
            cols.append(idx[v])
            data.append(float(w))

    mat = sp.csr_matrix(
        (data, (rows, cols)) if data else ([], ([], [])),
        shape=(n, n),
        dtype=np.float32,
    )
    return cg.csrgraph(mat, nodenames=nodes), nodes


def _top_neighbors(adj: dict, k: int) -> dict:
    """Keep only the k heaviest neighbors per node."""
    result = {}
    for u, nbrs in adj.items():
        if len(nbrs) <= k:
            result[u] = nbrs
        else:
            result[u] = dict(sorted(nbrs.items(), key=lambda x: x[1], reverse=True)[:k])
    return result


def embed_raw(graph, n_components: int = 16) -> tuple[np.ndarray, list[str]]:
    """Embed graph nodes via GGVec only — returns high-dimensional coordinates.

    Returns (coords array of shape (n_nodes, n_components), sorted node name list).
    Useful for custom downstream analysis (clustering, custom projection, etc.).
    """
    G, nodes = _to_csrgraph(graph)
    g2v = nodevectors.GGVec(n_components=n_components, verbose=False)
    return g2v.fit_transform(G), nodes


def embed(graph, n_components: int = 16) -> tuple[np.ndarray, list[str]]:
    """Embed graph nodes via GGVec + UMAP — returns 2-D coordinates.

    Returns (coords array of shape (n_nodes, 2), sorted node name list).
    """
    import umap

    coords, nodes = embed_raw(graph, n_components=n_components)
    if n_components > 2:
        coords = umap.UMAP(n_components=2).fit_transform(coords)
    return coords, nodes
