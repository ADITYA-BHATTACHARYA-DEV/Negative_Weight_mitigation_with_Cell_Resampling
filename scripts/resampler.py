"""
src/engine/resampler.py
-----------------------
Negative-weight mitigation via Cell Resampling.

Algorithm (arXiv:2109.07851):
    For each negative-weight seed event i:
        1.  Build a cell C by greedily adding nearest neighbours in the
            scaled (pT, y) space until Σ_{j∈C} w_j ≥ 0.
        2.  Once the cell weight sum is positive, redistribute weights:

                w'_i = |w_i| / Σ_{j∈C} |w_j|  ×  Σ_{j∈C} w_j

            This preserves the total cell weight while making every
            event weight strictly positive.

Distance metric (as specified):
    d(i,j) = √( (pT_i − pT_j)² + 100·(y_i − y_j)² )

which is implemented by constructing the scaled coordinate array
    coords[:, 0] = pT
    coords[:, 1] = √100 · y  =  10 · y
and using the standard Euclidean KD-Tree.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree

log = logging.getLogger(__name__)


@dataclass
class ResamplingStats:
    """Scalar summary statistics returned by NegativeWeightResampler.run()."""
    n_seeds_total:       int   = 0
    n_seeds_processed:   int   = 0
    n_seeds_skipped:     int   = 0   # could not find enough neighbours
    avg_cell_size:       float = 0.0
    max_cell_size:       int   = 0
    total_weight_before: float = 0.0
    total_weight_after:  float = 0.0
    # Efficiency metrics
    neg_fraction_before: float = 0.0
    neg_fraction_after:  float = 0.0
    neg_count_before:    int   = 0
    neg_count_after:     int   = 0
    # Per-cell sizes for distribution analysis
    cell_sizes: list[int] = field(default_factory=list)


class NegativeWeightResampler:
    """
    Greedy Cell Resampling for negative-weight event mitigation.

    Parameters
    ----------
    pT, y, w : 1-D float64 arrays (same length N)
               w is modified **in-place**.
    delta_pT  : (unused — kept for API compatibility; distance is KD-Tree based)
    delta_y   : (unused — kept for API compatibility)
    max_neighbours : upper bound on neighbours to consider per seed
    """

    RAPIDITY_SCALE: float = 100.0   # coefficient in distance metric

    def __init__(
        self,
        pT: np.ndarray,
        y:  np.ndarray,
        w:  np.ndarray,
        delta_pT: float = 5.0,
        delta_y:  float = 0.3,
        max_neighbours: int = 200,
    ) -> None:
        self.pT  = np.asarray(pT, dtype=np.float64)
        self.y   = np.asarray(y,  dtype=np.float64)
        self.w   = np.asarray(w,  dtype=np.float64)   # modified in-place
        self.max_neighbours = max_neighbours

        # Build scaled coordinate matrix for the distance metric
        # d(i,j) = sqrt( ΔpT² + 100·Δy² )  ⟺  L2 on [pT, 10·y]
        coords = np.column_stack([
            self.pT,
            np.sqrt(self.RAPIDITY_SCALE) * self.y,
        ])
        self.tree = KDTree(coords)
        self._coords = coords

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> ResamplingStats:
        """
        Execute the greedy cell-building loop over all negative-weight seeds.

        Returns a ResamplingStats dataclass with efficiency metrics.
        """
        stats = ResamplingStats()
        stats.total_weight_before = float(self.w.sum())
        stats.neg_count_before    = int((self.w < 0).sum())
        stats.neg_fraction_before = (
            stats.neg_count_before / len(self.w) if len(self.w) > 0 else 0.0
        )

        # Identify all negative-weight seeds (indices)
        seeds = np.where(self.w < 0)[0]
        stats.n_seeds_total = len(seeds)

        cell_sizes: list[int] = []

        for seed_idx in seeds:
            # Skip if this event was already fixed while processing a prior seed
            if self.w[seed_idx] >= 0:
                continue

            cell_indices = self._build_cell(seed_idx)

            if cell_indices is None:
                # Could not assemble a positive-sum cell
                stats.n_seeds_skipped += 1
                log.debug("Seed %d skipped (insufficient neighbours)", seed_idx)
                continue

            self._redistribute(cell_indices)
            cell_sizes.append(len(cell_indices))
            stats.n_seeds_processed += 1

        # ── Summary statistics ─────────────────────────────────────────
        stats.total_weight_after  = float(self.w.sum())
        stats.neg_count_after     = int((self.w < 0).sum())
        stats.neg_fraction_after  = (
            stats.neg_count_after / len(self.w) if len(self.w) > 0 else 0.0
        )
        stats.cell_sizes = cell_sizes
        if cell_sizes:
            stats.avg_cell_size = float(np.mean(cell_sizes))
            stats.max_cell_size = int(np.max(cell_sizes))

        log.info(
            "Resampling complete: %d/%d seeds processed, %d skipped. "
            "Neg fraction: %.3f → %.3f. ΔΣw = %.3e",
            stats.n_seeds_processed, stats.n_seeds_total,
            stats.n_seeds_skipped,
            stats.neg_fraction_before, stats.neg_fraction_after,
            abs(stats.total_weight_after - stats.total_weight_before),
        )
        return stats

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_cell(self, seed_idx: int) -> list[int] | None:
        """
        Grow a cell starting from seed_idx by adding the nearest neighbours
        one at a time until the total cell weight sum ≥ 0.

        Returns a list of all event indices in the cell (seed included),
        or None if the cell weight sum never reaches ≥ 0.
        """
        k = min(self.max_neighbours, len(self.pT) - 1)
        # Query includes seed itself at distance 0
        _, nbr_indices = self.tree.query(self._coords[seed_idx], k=k + 1)
        # Remove seed from neighbour list; we track it separately
        nbr_indices = nbr_indices[nbr_indices != seed_idx]

        cell: list[int] = [seed_idx]
        w_sum = float(self.w[seed_idx])

        for ni in nbr_indices:
            cell.append(int(ni))
            w_sum += float(self.w[ni])
            if w_sum >= 0.0:
                return cell

        # Exhausted all neighbours without reaching w_sum ≥ 0
        return None

    def _redistribute(self, cell: list[int]) -> None:
        """
        Apply the weight redistribution formula to all events in the cell:

            w'_i = |w_i| / Σ_{j∈C} |w_j|  ×  Σ_{j∈C} w_j

        This conserves the total cell weight and makes all weights positive.
        """
        idx = np.array(cell, dtype=np.intp)
        w_cell = self.w[idx]

        sum_w     = float(w_cell.sum())        # Σ w_j  (must be ≥ 0)
        sum_abs_w = float(np.abs(w_cell).sum()) # Σ |w_j|

        if sum_abs_w == 0.0:
            # Degenerate cell — all weights zero; nothing to do
            return

        # w'_i = |w_i| / Σ|w_j| × Σw_j
        self.w[idx] = (np.abs(w_cell) / sum_abs_w) * sum_w