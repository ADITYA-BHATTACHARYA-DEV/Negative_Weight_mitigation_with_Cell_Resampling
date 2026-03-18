"""
run_pipeline.py
---------------
Entry point for the full ATLAS Negative-Weight Mitigation pipeline.

Usage
-----
    cd ATLAS-NegWeight-Project
    python scripts/run_pipeline.py [--n-real 2000] [--n-virtual 1000] [--seed 42]

Steps
-----
    1. Ingest  – generate (or load) synthetic MC events
    2. Project – Born projection: merge Real + Virtual into (pT, y, w)
    3. Grid    – build Spatial Hash Grid
    4. Resample– greedy cell-building with KD-Tree refinement
    5. Prune   – remove zero-weight events
    6. Validate– overlay histograms + ratio plots
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── allow running from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.data_loader  import generate_synthetic_events
from src.projector          import combine_datasets
from src.engine.resampler   import NegativeWeightResampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def weighted_histogram(
    values: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """Return normalised weighted histogram counts."""
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    return counts


def make_validation_plots(
    combined_before: np.ndarray,
    w_after:         np.ndarray,
    output_dir:      Path,
) -> None:
    """
    Generate 1-D histograms (pT and y) with overlay + ratio panels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pT = combined_before["pT"]
    y  = combined_before["y"]
    w_before = combined_before["w"]

    specs = [
        ("pT", pT, np.linspace(pT.min(), min(pT.max(), 200), 51), r"$p_T$ [GeV]"),
        ("y",  y,  np.linspace(-3, 3, 41),                         r"Rapidity $y$"),
    ]

    for tag, vals, bins, xlabel in specs:
        h_before = weighted_histogram(vals, w_before, bins)
        h_after  = weighted_histogram(vals, w_after,  bins)

        # Ratio (guard against division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(h_before != 0, h_after / h_before, np.nan)

        fig = plt.figure(figsize=(8, 6))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

        ax_main  = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

        centres = 0.5 * (bins[:-1] + bins[1:])

        ax_main.step(centres, h_before, where="mid",
                     color="#2196F3", lw=1.8, label="Original")
        ax_main.step(centres, h_after,  where="mid",
                     color="#FF5722", lw=1.4, ls="--", label="Mitigated")
        ax_main.set_ylabel("Weighted counts")
        ax_main.legend(framealpha=0.9)
        ax_main.set_title(f"ATLAS Neg-Weight Mitigation – {xlabel} closure test")
        plt.setp(ax_main.get_xticklabels(), visible=False)

        ax_ratio.axhline(1.0, color="grey", lw=0.8, ls=":")
        ax_ratio.step(centres, ratio, where="mid", color="#4CAF50", lw=1.5)
        ax_ratio.set_ylim(0.9, 1.1)
        ax_ratio.set_ylabel("Mitigated / Original")
        ax_ratio.set_xlabel(xlabel)

        fname = output_dir / f"closure_{tag}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved validation plot → %s", fname)


def make_3d_phase_space_plot(
    combined_before: np.ndarray,
    w_after:         np.ndarray,
    output_dir:      Path,
) -> None:
    """
    3-D scatter: (pT, y, w) before and after resampling.
    Negative-weight events are highlighted.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pT = combined_before["pT"]
    y  = combined_before["y"]
    w_b = combined_before["w"]

    # Subsample for readability
    rng  = np.random.default_rng(0)
    idx  = rng.choice(len(pT), size=min(800, len(pT)), replace=False)

    fig  = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")

    for col, (w_plot, title) in enumerate(
        [(w_b, "Before Resampling"), (w_after, "After Resampling")]
    ):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.set_facecolor("#0d1117")

        pos_mask = w_plot[idx] >= 0
        neg_mask = ~pos_mask

        ax.scatter(
            pT[idx][pos_mask], y[idx][pos_mask], w_plot[idx][pos_mask],
            c="#00e5ff", s=4, alpha=0.6, label="w ≥ 0",
        )
        ax.scatter(
            pT[idx][neg_mask], y[idx][neg_mask], w_plot[idx][neg_mask],
            c="#ff1744", s=12, alpha=0.9, label="w < 0",
        )

        ax.set_xlabel(r"$p_T$ [GeV]", color="white", labelpad=6)
        ax.set_ylabel(r"$y$",         color="white", labelpad=6)
        ax.set_zlabel(r"$w$",         color="white", labelpad=6)
        ax.set_title(title, color="white", pad=8)
        ax.tick_params(colors="white")
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.legend(loc="upper left", fontsize=7,
                  facecolor="#1a1a2e", labelcolor="white")

    fig.suptitle("Phase-Space Weight Distribution", color="white",
                 fontsize=14, y=1.01)
    fname = output_dir / "phase_space_3d.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Saved 3-D phase space plot → %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path("output")

    # ── 1. Ingest ─────────────────────────────────────────────────────────────
    log.info("=== Step 1 / Ingest ===")
    real_events, virtual_events = generate_synthetic_events(
        n_real=args.n_real, n_virtual=args.n_virtual,
        neg_fraction=args.neg_fraction, seed=args.seed,
    )

    # ── 2. Project ────────────────────────────────────────────────────────────
    log.info("=== Step 2 / Born Projection ===")
    combined = combine_datasets(real_events, virtual_events)

    # Deep-copy weights for later comparison
    w_before = combined["w"].copy()

    # ── 3 & 4. Grid + Resample ────────────────────────────────────────────────
    log.info("=== Step 3+4 / Spatial Hash + Greedy Resampling ===")
    t0 = time.perf_counter()
    resampler = NegativeWeightResampler(
        pT=combined["pT"],
        y=combined["y"],
        w=combined["w"],   # modified in-place
        delta_pT=5.0,
        delta_y=0.3,
    )
    stats = resampler.run()
    elapsed = time.perf_counter() - t0
    log.info("Resampling wall-time: %.2f s", elapsed)

    # ── 5. Prune ──────────────────────────────────────────────────────────────
    log.info("=== Step 5 / Prune zero-weight events ===")
    keep_mask = combined["w"] != 0.0
    n_pruned  = int((~keep_mask).sum())
    log.info("Pruned %d zero-weight events (%d remaining).", n_pruned, keep_mask.sum())

    # ── 6. Validate ───────────────────────────────────────────────────────────
    log.info("=== Step 6 / Validation ===")

    # Reconstruct 'before' view for plotting (same events, original weights)
    combined_before = combined.copy()
    combined_before["w"] = w_before

    make_validation_plots(combined_before, combined["w"], output_dir)
    make_3d_phase_space_plot(combined_before, combined["w"], output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("PIPELINE COMPLETE")
    log.info("  Seeds processed  : %d", stats.n_seeds_processed)
    log.info("  Seeds skipped    : %d", stats.n_seeds_skipped)
    log.info("  Avg cell size    : %.1f", stats.avg_cell_size)
    log.info("  Max cell size    : %d",  stats.max_cell_size)
    log.info("  ΔΣw (weight cons): %.2e",
             abs(stats.total_weight_after - stats.total_weight_before))
    log.info("  Remaining w<0    : %d", int((combined["w"] < 0).sum()))
    log.info("  Output directory : %s", output_dir.resolve())
    log.info("=" * 55)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ATLAS Negative-Weight Mitigation pipeline")
    p.add_argument("--n-real",       type=int,   default=2000,  help="# real events")
    p.add_argument("--n-virtual",    type=int,   default=1000,  help="# virtual events")
    p.add_argument("--neg-fraction", type=float, default=0.20,  help="Fraction of virtual events with w<0")
    p.add_argument("--seed",         type=int,   default=42,    help="RNG seed")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())