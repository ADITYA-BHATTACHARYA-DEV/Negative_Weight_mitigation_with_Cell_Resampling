"""
metrics.py
----------
Distance metric used throughout the spatial search.

The (pT, y) plane is anisotropic: pT spans O(10–100) GeV while
rapidity y spans O(1).  Without rescaling, the KD-Tree would be
dominated by pT separations and become blind to rapidity differences.

The chosen metric is:
    d(i,j) = sqrt( (pT_i - pT_j)^2 + 100*(y_i - y_j)^2 )

The factor of 100 gives rapidity differences roughly the same numerical
weight as momentum differences, consistent with the paper prescription.
"""

import numpy as np


RAPIDITY_SCALE: float = 100.0   # dimensionless rescaling factor


def scaled_distance(
    pT_i: float | np.ndarray,
    y_i:  float | np.ndarray,
    pT_j: float | np.ndarray,
    y_j:  float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute the anisotropic distance between event(s) i and event(s) j.

    Works element-wise on scalars or NumPy arrays of the same shape.

    Returns
    -------
    d : same shape as inputs
    """
    d_pT = pT_i - pT_j
    d_y  = y_i  - y_j
    return np.sqrt(d_pT**2 + RAPIDITY_SCALE * d_y**2)


def scaled_coordinates(pT: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Stack (pT, sqrt(100)*y) into a 2-column coordinate matrix suitable
    for `scipy.spatial.KDTree`.  The tree's Euclidean distance then
    reproduces `scaled_distance` exactly.

    Parameters
    ----------
    pT : shape (N,)
    y  : shape (N,)

    Returns
    -------
    coords : shape (N, 2)
    """
    return np.column_stack([pT, np.sqrt(RAPIDITY_SCALE) * y])