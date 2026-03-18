"""
spatial_hash.py
---------------
Tier 1 of the two-tier spatial index.

Divides the (pT, y) plane into a regular grid and stores event indices
in a hash map keyed by bin ID.  For a query point, the 9-cell neighbourhood
(3×3 window centred on the query bin) is returned in O(1).

Complexity
----------
Build  : O(N)
Query  : O(1) to retrieve candidate index lists
"""

import numpy as np
from collections import defaultdict
from typing import Iterator


# Default bin widths (tunable via constructor)
DEFAULT_DELTA_PT: float = 5.0   # GeV
DEFAULT_DELTA_Y:  float = 0.3   # rapidity units


class SpatialHashGrid:
    """
    Hash-based spatial grid for fast neighbourhood lookups.

    Parameters
    ----------
    pT     : array of transverse momenta, shape (N,)
    y      : array of rapidities,          shape (N,)
    delta_pT : bin width in pT  [GeV]
    delta_y  : bin width in y   [dimensionless]
    """

    def __init__(
        self,
        pT:       np.ndarray,
        y:        np.ndarray,
        delta_pT: float = DEFAULT_DELTA_PT,
        delta_y:  float = DEFAULT_DELTA_Y,
    ) -> None:
        self.pT       = pT
        self.y        = y
        self.delta_pT = delta_pT
        self.delta_y  = delta_y

        # Build the hash map: bin_id → list[event_index]
        self._grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._build()

    # ── Private ──────────────────────────────────────────────────────────────

    def _bin_id(self, pT_val: float, y_val: float) -> tuple[int, int]:
        """Return the integer bin coordinates for a single point."""
        return (
            int(np.floor(pT_val / self.delta_pT)),
            int(np.floor(y_val  / self.delta_y)),
        )

    def _build(self) -> None:
        """Populate the grid in one O(N) pass."""
        for idx in range(len(self.pT)):
            bid = self._bin_id(self.pT[idx], self.y[idx])
            self._grid[bid].append(idx)

    # ── Public API ────────────────────────────────────────────────────────────

    def candidate_indices(
        self,
        pT_query: float,
        y_query:  float,
        window:   int = 1,
    ) -> np.ndarray:
        """
        Return event indices in the (2*window+1)² neighbourhood of the query.

        Parameters
        ----------
        pT_query, y_query : query coordinates
        window            : half-width of the search window in bin units
                            (default 1 → 3×3 = 9 bins)

        Returns
        -------
        indices : 1-D int array (may be empty)
        """
        cx, cy = self._bin_id(pT_query, y_query)
        indices: list[int] = []
        for dx in range(-window, window + 1):
            for dy in range(-window, window + 1):
                indices.extend(self._grid.get((cx + dx, cy + dy), []))
        return np.array(indices, dtype=np.intp)

    def __len__(self) -> int:
        return len(self.pT)

    def num_bins_occupied(self) -> int:
        return len(self._grid)

    def iter_bins(self) -> Iterator[tuple[tuple[int, int], list[int]]]:
        """Iterate over (bin_id, index_list) pairs."""
        yield from self._grid.items()