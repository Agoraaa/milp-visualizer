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


_SPECTRAL_THRESHOLD = 100


def _spectral_embed(graph) -> tuple[np.ndarray, list[str]]:
    """Normalized Laplacian spectral embedding → 2-D coordinates.

    Used for small or near-complete graphs where GGVec random walks don't converge.
    """
    nodes = sorted(graph.nodes)
    n = len(nodes)
    idx = {nd: i for i, nd in enumerate(nodes)}

    rows, cols, data = [], [], []
    for u, nbrs in graph.adj.items():
        for v, w in nbrs.items():
            rows.append(idx[u])
            cols.append(idx[v])
            data.append(float(w))

    A = sp.csr_matrix(
        (data, (rows, cols)) if data else ([], ([], [])),
        shape=(n, n), dtype=np.float64,
    )
    degree = np.array(A.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degree))
    L = sp.eye(n, dtype=np.float64) - D_inv_sqrt @ A @ D_inv_sqrt

    k = min(3, n - 1)
    vals, vecs = sp.linalg.eigsh(L, k=k, which="SM")
    order = np.argsort(vals)
    vecs = vecs[:, order]

    if vecs.shape[1] >= 3:
        coords = vecs[:, 1:3]
    elif vecs.shape[1] == 2:
        coords = np.column_stack([vecs[:, 1], np.zeros(n)])
    else:
        coords = np.column_stack([np.zeros(n), np.zeros(n)])

    return coords.astype(np.float32), nodes


def embed(graph, n_components: int = 16) -> tuple[np.ndarray, list[str]]:
    """Embed graph nodes → 2-D coordinates.

    Uses spectral embedding for small graphs (< _SPECTRAL_THRESHOLD nodes),
    GGVec + UMAP otherwise.
    Returns (coords array of shape (n_nodes, 2), sorted node name list).
    """
    if graph.num_nodes < _SPECTRAL_THRESHOLD:
        return _spectral_embed(graph)

    import umap

    coords, nodes = embed_raw(graph, n_components=n_components)
    if n_components > 2:
        coords = umap.UMAP(n_components=2).fit_transform(coords)
    return coords, nodes
