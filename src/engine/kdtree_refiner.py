"""
kdtree_refiner.py
-----------------
Tier 2 of the two-tier spatial index.

Given a small candidate set (from the Spatial Hash), builds a local KD-Tree
and finds the k nearest neighbours using the anisotropic scaled distance:

    d(i,j) = sqrt( (pT_i - pT_j)^2 + 100*(y_i - y_j)^2 )

By embedding sqrt(100)*y into the coordinate space the standard Euclidean
KD-Tree reproduces the anisotropic metric exactly.

Complexity
----------
Build  : O(K log K)   K = |candidate set| ≪ N
Query  : O(log K)     per nearest-neighbour lookup
"""

import numpy as np
from scipy.spatial import KDTree

from src.utils.metrics import scaled_coordinates


class KDTreeRefiner:
    """
    Local KD-Tree built over a small candidate set.

    Parameters
    ----------
    pT_cands : pT values of candidate events,   shape (K,)
    y_cands  : y  values of candidate events,   shape (K,)
    orig_idx : original indices of candidates in the full dataset, shape (K,)
    """

    def __init__(
        self,
        pT_cands: np.ndarray,
        y_cands:  np.ndarray,
        orig_idx: np.ndarray,
    ) -> None:
        self.orig_idx = np.asarray(orig_idx, dtype=np.intp)
        coords = scaled_coordinates(
            np.asarray(pT_cands, dtype=np.float64),
            np.asarray(y_cands,  dtype=np.float64),
        )
        self._tree  = KDTree(coords)
        self._coords = coords

    # ── Public API ────────────────────────────────────────────────────────────

    def query_neighbors(
        self,
        pT_seed: float,
        y_seed:  float,
        k:       int,
        exclude_self: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the k nearest neighbours of the seed in the candidate set.

        Parameters
        ----------
        pT_seed, y_seed : seed coordinates
        k               : number of neighbours requested
        exclude_self    : if True, skip the point at distance ≈ 0

        Returns
        -------
        distances : scaled distances to neighbours, shape (k',)
        indices   : indices in the *original* dataset,  shape (k',)
                    (k' ≤ k after potential self-exclusion)
        """
        query_pt = scaled_coordinates(
            np.array([pT_seed]),
            np.array([y_seed]),
        )
        k_ask = min(k + int(exclude_self), len(self.orig_idx))
        dists, local_idx = self._tree.query(query_pt, k=k_ask)
        dists     = dists.ravel()
        local_idx = local_idx.ravel()

        if exclude_self:
            mask  = dists > 1e-10
            dists     = dists[mask]
            local_idx = local_idx[mask]

        return dists[:k], self.orig_idx[local_idx[:k]]

    def __len__(self) -> int:
        return len(self.orig_idx)


def build_local_refiner(
    pT_all:     np.ndarray,
    y_all:      np.ndarray,
    candidates: np.ndarray,
) -> KDTreeRefiner:
    """
    Convenience factory: extract candidate coordinates and build a refiner.

    Parameters
    ----------
    pT_all, y_all : full-dataset coordinate arrays
    candidates    : integer indices into pT_all / y_all

    Returns
    -------
    KDTreeRefiner over the candidate subset
    """
    return KDTreeRefiner(
        pT_cands=pT_all[candidates],
        y_cands=y_all[candidates],
        orig_idx=candidates,
    )