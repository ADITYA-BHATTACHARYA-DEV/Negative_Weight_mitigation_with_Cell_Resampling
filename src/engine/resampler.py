"""
resampler.py
------------
Task 2 – Cell Resampling (the "Brain")

Implements the greedy negative-weight elimination algorithm:

1.  For every event with w < 0 (the *seed*):
    a.  Grow a cell by adding the nearest available neighbour in (pT, y).
    b.  Repeat until Σ w_cell ≥ 0.
    c.  Redistribute weights:
            w_i_new = w_i_old * (1 + w_seed / Σ w_neighbours)
        The seed's weight is set to 0 (neutralised).
2.  Continue until no negative weights remain.

Two-tier spatial index (Spatial Hash → local KD-Tree) makes the inner
search sub-linear in the total event count.

Complexity
----------
O(M · log K)  where M = number of negative-weight seeds
                     K = size of local KD-Tree candidate set  (≪ N)
"""

import numpy as np
import logging
from dataclasses import dataclass, field

from src.engine.spatial_hash   import SpatialHashGrid
from src.engine.kdtree_refiner import build_local_refiner

log = logging.getLogger(__name__)


@dataclass
class ResamplerStats:
    """Diagnostic counters collected during a resampling run."""
    n_seeds_processed:  int = 0
    n_seeds_skipped:    int = 0   # seeds already neutralised by earlier cells
    n_cells_built:      int = 0
    avg_cell_size:      float = 0.0
    max_cell_size:      int = 0
    total_weight_before: float = 0.0
    total_weight_after:  float = 0.0
    cell_sizes: list[int] = field(default_factory=list)


class NegativeWeightResampler:
    """
    Greedy cell-building resampler.

    Parameters
    ----------
    pT        : transverse momenta,  shape (N,)
    y         : rapidities,          shape (N,)
    w         : event weights,       shape (N,)  – modified **in-place**
    delta_pT  : Spatial Hash bin width [GeV]
    delta_y   : Spatial Hash bin width [rapidity]
    window    : hash grid window half-width (default 1 → 3×3 bins)
    k_init    : initial k for KD-Tree queries
    """

    def __init__(
        self,
        pT:       np.ndarray,
        y:        np.ndarray,
        w:        np.ndarray,
        delta_pT: float = 5.0,
        delta_y:  float = 0.3,
        window:   int   = 1,
        k_init:   int   = 20,
    ) -> None:
        self.pT     = np.asarray(pT, dtype=np.float64)
        self.y      = np.asarray(y,  dtype=np.float64)
        self.w      = np.asarray(w,  dtype=np.float64)   # live weight array
        self.window = window
        self.k_init = k_init

        # Build the Spatial Hash once; it stores indices, not weights
        self._grid = SpatialHashGrid(self.pT, self.y, delta_pT, delta_y)
        log.info(
            "SpatialHashGrid built: %d events in %d occupied bins.",
            len(self.pT), self._grid.num_bins_occupied(),
        )

    # ── Core Algorithm ────────────────────────────────────────────────────────

    def run(self) -> ResamplerStats:
        """
        Execute the greedy resampling loop.

        Returns
        -------
        ResamplerStats with diagnostic information.
        """
        stats = ResamplerStats()
        stats.total_weight_before = float(self.w.sum())

        # Identify initial seeds (all events with w < 0)
        seed_indices = np.where(self.w < 0)[0]
        log.info("Starting resampling: %d negative-weight seeds.", len(seed_indices))

        for seed_idx in seed_indices:
            # Seeds can be neutralised by earlier cells – skip them
            if self.w[seed_idx] >= 0:
                stats.n_seeds_skipped += 1
                continue

            stats.n_seeds_processed += 1
            cell = self._grow_cell(seed_idx)
            self._redistribute(seed_idx, cell)

            stats.n_cells_built += 1
            stats.cell_sizes.append(len(cell))

        # Final diagnostics
        if stats.cell_sizes:
            stats.avg_cell_size = float(np.mean(stats.cell_sizes))
            stats.max_cell_size = int(np.max(stats.cell_sizes))

        stats.total_weight_after = float(self.w.sum())
        n_remaining = int((self.w < 0).sum())

        log.info(
            "Resampling complete: %d cells built, %d seeds skipped.",
            stats.n_cells_built, stats.n_seeds_skipped,
        )
        log.info(
            "  Avg cell size: %.1f  |  Max: %d",
            stats.avg_cell_size, stats.max_cell_size,
        )
        log.info(
            "  Total weight: %.4f → %.4f  (Δ = %.2e)",
            stats.total_weight_before, stats.total_weight_after,
            abs(stats.total_weight_after - stats.total_weight_before),
        )
        if n_remaining:
            log.warning(
                "%d events still have w < 0 (isolated seeds with no neighbours).",
                n_remaining,
            )
        return stats

    # ── Private helpers ───────────────────────────────────────────────────────

    def _grow_cell(self, seed_idx: int) -> list[int]:
        """
        Grow a cell around seed_idx until Σ w_cell ≥ 0.

        Returns the list of *neighbour* indices (seed excluded).
        """
        pT_s = float(self.pT[seed_idx])
        y_s  = float(self.y[seed_idx])
        w_s  = float(self.w[seed_idx])   # negative

        # Candidate pool from Spatial Hash (Tier 1)
        candidates = self._grid.candidate_indices(pT_s, y_s, self.window)
        # Remove the seed itself
        candidates = candidates[candidates != seed_idx]

        if len(candidates) == 0:
            # Expand search window as fallback
            candidates = self._grid.candidate_indices(pT_s, y_s, self.window + 1)
            candidates = candidates[candidates != seed_idx]

        if len(candidates) == 0:
            return []   # isolated event; cannot be neutralised

        # Build local KD-Tree (Tier 2)
        refiner = build_local_refiner(self.pT, self.y, candidates)

        cell:      list[int] = []
        w_sum:     float     = w_s
        k:         int       = min(self.k_init, len(candidates))
        exhausted: bool      = False

        while w_sum < 0 and not exhausted:
            _, nbr_indices = refiner.query_neighbors(pT_s, y_s, k=k)

            for nbr_idx in nbr_indices:
                if nbr_idx not in cell:
                    cell.append(int(nbr_idx))
                    w_sum += float(self.w[nbr_idx])
                    if w_sum >= 0:
                        break

            # If we've consumed all candidates and still negative – give up
            if len(cell) >= len(candidates):
                exhausted = True

            k = min(k * 2, len(candidates))   # double search radius

        return cell

    def _redistribute(self, seed_idx: int, cell: list[int]) -> None:
        """
        Apply the weight redistribution formula once a cell is finalised.

        w_i_new = w_i_old * (1 + w_seed / Σ w_neighbours)

        The seed itself is set to 0.
        """
        if not cell:
            return   # cannot neutralise; leave seed weight unchanged

        w_seed      = float(self.w[seed_idx])
        w_neighbours = float(self.w[cell].sum())

        if w_neighbours <= 0:
            # Pathological: entire cell is negative – skip redistribution
            log.debug(
                "Seed %d: Σw_neighbours = %.4f ≤ 0; skipping redistribution.",
                seed_idx, w_neighbours,
            )
            return

        scale_factor = 1.0 + w_seed / w_neighbours
        self.w[cell] *= scale_factor
        self.w[seed_idx] = 0.0   # neutralise the seed