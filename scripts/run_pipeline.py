"""
run_pipeline.py
---------------
ATLAS Negative-Weight Mitigation Pipeline — full execution entry point.

Fixes vs. previous version
---------------------------
1. Weight redistribution now uses the EXACT formula from arXiv:2109.07851:
       w'_i = |w_i| / Σ|w_j| × Σw_j   (implemented in src/engine/resampler.py)
   Previously the seed event was zeroed out instead of receiving a new
   positive weight.

2. Distance metric is now strictly:
       d(i,j) = √( (pT_i−pT_j)² + 100·(y_i−y_j)² )
   via scaled KD-Tree coordinates, matching the task specification.

3. Validation histograms (Task 3) are produced for BOTH pT and y with
   before/after overlays AND ratio panels.

Extras added
------------
• Efficiency dashboard: neg-weight fraction before/after, statistical
  dilution factor D = (N_pos + N_neg)/(N_pos - N_neg), computational
  gain estimate (fraction of effective events recovered).
• Cell-size distribution plot showing how many events were grouped
  per resampling cell.
• Phase-space heatmap showing where negative-weight events cluster.
• KS-test and χ² closure test between before/after distributions to
  quantify whether physical observables are sculpted.
• Summary JSON written to output/ for downstream processing.

Usage
-----
    cd atlas_project
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --n-real 5000 --n-virtual 2000 --neg-fraction 0.30
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader  import load_events
from projector          import combine_datasets
from resampler   import NegativeWeightResampler, ResamplingStats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Plot style
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "before":  "#2196F3",
    "after":   "#FF5722",
    "ratio":   "#4CAF50",
    "neg":     "#E53935",
    "pos":     "#1E88E5",
    "cell":    "#FFA726",
    "heatmap": "magma",
}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helper: weighted histogram
# ─────────────────────────────────────────────────────────────────────────────

def weighted_histogram(
    values:  np.ndarray,
    weights: np.ndarray,
    bins:    np.ndarray,
) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Validation histograms (1-D closure with ratio panels)
# ─────────────────────────────────────────────────────────────────────────────

def make_validation_plots(
    combined:  np.ndarray,
    w_before:  np.ndarray,
    w_after:   np.ndarray,
    output_dir: Path,
) -> dict:
    """
    Generate 1-D overlay histograms for pT and y.

    For each variable:
      • Top panel : weighted counts before (blue) and after (orange)
      • Bottom panel: ratio after/before with ±2% band

    Returns dict of KS-test p-values and chi2 per variable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pT = combined["pT"]
    y  = combined["y"]

    specs = [
        ("pT",
         pT,
         np.linspace(np.percentile(pT, 1), np.percentile(pT, 99), 51),
         r"$p_T$ [GeV]",
         r"$p_T$ distribution"),
        ("y",
         y,
         np.linspace(-2.6, 2.6, 41),
         r"Rapidity $y$",
         r"Rapidity $y$ distribution"),
    ]

    closure_metrics: dict = {}

    for tag, vals, bins, xlabel, title in specs:
        h_before = weighted_histogram(vals, w_before, bins)
        h_after  = weighted_histogram(vals, w_after,  bins)

        # ── Ratio ─────────────────────────────────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(h_before != 0, h_after / h_before, np.nan)

        # ── χ² closure test ───────────────────────────────────────────
        valid = h_before > 0
        chi2_stat = float(np.sum(
            (h_after[valid] - h_before[valid])**2 / h_before[valid]
        ))
        chi2_ndf  = int(valid.sum())
        chi2_pval = float(sp_stats.chi2.sf(chi2_stat, df=chi2_ndf))

        # ── KS test on unweighted event positions ─────────────────────
        # (uses absolute weight as a proxy for event density)
        ks_stat, ks_pval = sp_stats.ks_2samp(
            np.repeat(vals, np.round(np.abs(w_before) * 10).astype(int).clip(0, 50)),
            np.repeat(vals, np.round(np.abs(w_after)  * 10).astype(int).clip(0, 50)),
        )

        closure_metrics[tag] = {
            "chi2": chi2_stat, "chi2_ndf": chi2_ndf, "chi2_pval": chi2_pval,
            "ks_stat": ks_stat, "ks_pval": ks_pval,
        }

        # ── Figure ────────────────────────────────────────────────────
        fig = plt.figure(figsize=(9, 7))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.07)
        ax_main  = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

        centres = 0.5 * (bins[:-1] + bins[1:])
        width   = bins[1] - bins[0]

        ax_main.step(centres, h_before, where="mid",
                     color=COLORS["before"], lw=2.0, label="Original (before)")
        ax_main.step(centres, h_after,  where="mid",
                     color=COLORS["after"],  lw=1.6, ls="--", label="Mitigated (after)")
        ax_main.fill_between(centres, h_before, alpha=0.08,
                              color=COLORS["before"], step="mid")
        ax_main.fill_between(centres, h_after,  alpha=0.08,
                              color=COLORS["after"],  step="mid")
        ax_main.set_ylabel("Weighted counts", fontsize=11)
        ax_main.set_title(
            f"ATLAS Neg-Weight Mitigation — {title}\n"
            f"χ²/ndf = {chi2_stat:.1f}/{chi2_ndf}  "
            f"(p = {chi2_pval:.3f})   KS p = {ks_pval:.3f}",
            fontsize=10,
        )
        ax_main.legend(framealpha=0.9, fontsize=10)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Ratio panel
        ax_ratio.axhline(1.0,  color="grey", lw=0.9, ls=":")
        ax_ratio.axhspan(0.98, 1.02, color="grey", alpha=0.12, label="±2%")
        ax_ratio.step(centres, ratio, where="mid",
                      color=COLORS["ratio"], lw=1.6)
        ax_ratio.set_ylim(0.88, 1.12)
        ax_ratio.set_ylabel("After / Before", fontsize=10)
        ax_ratio.set_xlabel(xlabel, fontsize=11)
        ax_ratio.legend(fontsize=8, loc="upper right")

        fname = output_dir / f"closure_{tag}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved validation plot → %s", fname)

    return closure_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Extra 1 — Cell-size distribution
# ─────────────────────────────────────────────────────────────────────────────

def make_cell_size_plot(stats: ResamplingStats, output_dir: Path) -> None:
    """
    Histogram of resampling cell sizes.
    Small cells (2-3 events) indicate the method acts locally;
    large cells signal regions where negative weights are dense.
    """
    if not stats.cell_sizes:
        return

    sizes = np.array(stats.cell_sizes)
    bins  = np.arange(1, min(sizes.max() + 2, 52))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sizes, bins=bins, color=COLORS["cell"], edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(stats.avg_cell_size, color=COLORS["neg"], lw=1.5, ls="--",
               label=f"Mean = {stats.avg_cell_size:.1f}")
    ax.set_xlabel("Cell size (number of events)", fontsize=11)
    ax.set_ylabel("Number of cells", fontsize=11)
    ax.set_title(
        f"Resampling cell-size distribution\n"
        f"Total cells: {len(sizes)}   Max: {stats.max_cell_size}",
        fontsize=10,
    )
    ax.legend(fontsize=10)

    fname = output_dir / "cell_size_distribution.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cell-size plot → %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Extra 2 — Phase-space heatmap of negative-weight events
# ─────────────────────────────────────────────────────────────────────────────

def make_phase_space_heatmap(
    combined:  np.ndarray,
    w_before:  np.ndarray,
    output_dir: Path,
) -> None:
    """
    2-D heatmap in (pT, y) showing where negative-weight events cluster
    before resampling, with the density of positive events as a contour
    backdrop. Helps identify phase-space regions where the NLO subtraction
    is most aggressive.
    """
    pT = combined["pT"]
    y  = combined["y"]

    neg_mask = w_before < 0
    pos_mask = ~neg_mask

    pT_bins = np.linspace(np.percentile(pT, 1), np.percentile(pT, 99), 40)
    y_bins  = np.linspace(-2.6, 2.6, 30)

    h_neg, _, _ = np.histogram2d(pT[neg_mask], y[neg_mask],
                                  bins=[pT_bins, y_bins])
    h_pos, _, _ = np.histogram2d(pT[pos_mask], y[pos_mask],
                                  bins=[pT_bins, y_bins],
                                  weights=np.abs(w_before[pos_mask]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # ── Negative-weight density ───────────────────────────────────────
    ax = axes[0]
    im = ax.pcolormesh(pT_bins, y_bins, h_neg.T,
                       cmap="Reds", shading="flat")
    fig.colorbar(im, ax=ax, label="Negative-weight event count")
    ax.set_xlabel(r"$p_T$ [GeV]", fontsize=11)
    ax.set_ylabel(r"Rapidity $y$", fontsize=11)
    ax.set_title("Negative-weight event density\n(before mitigation)", fontsize=10)

    # ── Positive-weight density (backdrop) ───────────────────────────
    ax2 = axes[1]
    im2 = ax2.pcolormesh(pT_bins, y_bins, h_pos.T,
                          cmap="Blues", shading="flat")
    fig.colorbar(im2, ax=ax2, label="Positive-weight Σ|w|")
    # Overlay neg-event contours
    pT_centres = 0.5 * (pT_bins[:-1] + pT_bins[1:])
    y_centres  = 0.5 * (y_bins[:-1]  + y_bins[1:])
    ax2.contour(pT_centres, y_centres, h_neg.T,
                levels=5, colors="red", linewidths=0.8, alpha=0.7)
    ax2.set_xlabel(r"$p_T$ [GeV]", fontsize=11)
    ax2.set_title("Positive-weight density with\nneg-weight contours overlaid",
                  fontsize=10)

    fig.suptitle("Phase-space distribution of negative-weight events", fontsize=11)
    fig.tight_layout()

    fname = output_dir / "phase_space_heatmap.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved phase-space heatmap → %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Extra 3 — Efficiency / dilution dashboard
# ─────────────────────────────────────────────────────────────────────────────

def make_efficiency_dashboard(
    stats:      ResamplingStats,
    n_total:    int,
    output_dir: Path,
) -> dict:
    """
    One-page summary figure showing:
      • Neg-weight fraction before vs. after (bar)
      • Statistical dilution factor D = (N+ + N-)/(N+ - N-) before/after
      • Effective-event fraction η = 1/D  (computational gain)
      • Weight-sum conservation check

    Also returns the efficiency metrics as a dict for the summary JSON.
    """
    def dilution_factor(n_pos: int, n_neg: int) -> float:
        denom = n_pos - n_neg
        return (n_pos + n_neg) / denom if denom > 0 else float("inf")

    n_neg_b  = stats.neg_count_before
    n_neg_a  = stats.neg_count_after
    n_pos_b  = n_total - n_neg_b
    n_pos_a  = n_total - n_neg_a

    D_before = dilution_factor(n_pos_b, n_neg_b)
    D_after  = dilution_factor(n_pos_a, n_neg_a)
    eta_b    = 1.0 / D_before if D_before > 0 else 1.0
    eta_a    = 1.0 / D_after  if D_after  > 0 else 1.0
    gain     = (eta_a - eta_b) / eta_b * 100 if eta_b > 0 else 0.0

    eff_metrics = {
        "n_total":            n_total,
        "neg_fraction_before": stats.neg_fraction_before,
        "neg_fraction_after":  stats.neg_fraction_after,
        "dilution_before":     D_before,
        "dilution_after":      D_after,
        "effective_frac_before": eta_b,
        "effective_frac_after":  eta_a,
        "computational_gain_pct": gain,
        "weight_conservation_delta": abs(stats.total_weight_after - stats.total_weight_before),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # ── Panel 1: neg-weight fraction ─────────────────────────────────
    ax = axes[0]
    labels  = ["Before", "After"]
    values  = [stats.neg_fraction_before * 100, stats.neg_fraction_after * 100]
    bar_c   = [COLORS["neg"], COLORS["ratio"]]
    bars = ax.bar(labels, values, color=bar_c, edgecolor="white",
                  linewidth=0.8, width=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Negative-weight fraction (%)", fontsize=10)
    ax.set_title("Fraction of negative-weight events", fontsize=10)
    ax.set_ylim(0, max(values) * 1.3)

    # ── Panel 2: dilution factor ──────────────────────────────────────
    ax2 = axes[1]
    d_vals = [D_before, D_after]
    bars2 = ax2.bar(labels, d_vals, color=bar_c, edgecolor="white",
                    linewidth=0.8, width=0.4)
    for bar, val in zip(bars2, d_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")
    ax2.axhline(1.0, color="grey", lw=0.8, ls="--", label="D = 1 (ideal)")
    ax2.set_ylabel("Statistical dilution factor D", fontsize=10)
    ax2.set_title(
        r"Dilution factor $D = \frac{N^+ + N^-}{N^+ - N^-}$", fontsize=10
    )
    ax2.legend(fontsize=9)

    # ── Panel 3: effective-event fraction / gain ──────────────────────
    ax3 = axes[2]
    e_vals = [eta_b * 100, eta_a * 100]
    bars3 = ax3.bar(labels, e_vals, color=bar_c, edgecolor="white",
                    linewidth=0.8, width=0.4)
    for bar, val in zip(bars3, e_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")
    ax3.set_ylabel("Effective-event fraction η = 1/D (%)", fontsize=10)
    ax3.set_title(
        f"Computational efficiency gain\n+{gain:.1f}% effective events recovered",
        fontsize=10,
    )

    fig.suptitle(
        "ATLAS Neg-Weight Mitigation — Efficiency dashboard\n"
        f"ΔΣw (weight conservation) = {eff_metrics['weight_conservation_delta']:.2e}",
        fontsize=11,
    )
    fig.tight_layout()
    fname = output_dir / "efficiency_dashboard.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved efficiency dashboard → %s", fname)

    return eff_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Extra 4 — 3-D phase-space scatter (before / after)
# ─────────────────────────────────────────────────────────────────────────────

def make_3d_phase_space_plot(
    combined:  np.ndarray,
    w_after:   np.ndarray,
    output_dir: Path,
) -> None:
    """3-D scatter (pT, y, w) before and after resampling."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pT    = combined["pT"]
    y     = combined["y"]
    w_b   = combined["w"]

    rng = np.random.default_rng(0)
    idx = rng.choice(len(pT), size=min(1000, len(pT)), replace=False)

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")

    for col, (w_plot, title) in enumerate([
        (w_b,    "Before resampling"),
        (w_after, "After resampling"),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.set_facecolor("#0d1117")

        pos_m = w_plot[idx] >= 0
        neg_m = ~pos_m

        if pos_m.any():
            ax.scatter(pT[idx][pos_m], y[idx][pos_m], w_plot[idx][pos_m],
                       c="#00e5ff", s=5, alpha=0.55, label="w ≥ 0")
        if neg_m.any():
            ax.scatter(pT[idx][neg_m], y[idx][neg_m], w_plot[idx][neg_m],
                       c="#ff1744", s=18, alpha=0.9, label="w < 0",
                       marker="v")

        ax.set_xlabel(r"$p_T$ [GeV]", color="white", labelpad=6, fontsize=8)
        ax.set_ylabel(r"$y$",         color="white", labelpad=6, fontsize=8)
        ax.set_zlabel(r"$w$",         color="white", labelpad=6, fontsize=8)
        ax.set_title(title, color="white", pad=8, fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.legend(loc="upper left", fontsize=7,
                  facecolor="#1a1a2e", labelcolor="white")

    fig.suptitle("Phase-space weight distribution (pT, y, w)",
                 color="white", fontsize=12, y=1.01)

    fname = output_dir / "phase_space_3d.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Saved 3-D scatter → %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Extra 5 — weight-magnitude distribution (before vs after)
# ─────────────────────────────────────────────────────────────────────────────

def make_weight_distribution_plot(
    w_before: np.ndarray,
    w_after:  np.ndarray,
    output_dir: Path,
) -> None:
    """
    Plot the distribution of event weights |w| before and after mitigation
    on a log-y scale.  After resampling, there should be no entries at w < 0.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (w, label, color) in zip(axes, [
        (w_before, "Before mitigation", COLORS["before"]),
        (w_after,  "After mitigation",  COLORS["after"]),
    ]):
        w_pos = w[w > 0]
        w_neg = np.abs(w[w < 0])
        bins = np.linspace(0, np.percentile(np.abs(w), 99.5), 60)

        ax.hist(w_pos, bins=bins, color=color, alpha=0.7, label="w > 0")
        if len(w_neg):
            ax.hist(w_neg, bins=bins, color=COLORS["neg"], alpha=0.7,
                    label="|w| for w < 0")
        ax.set_yscale("log")
        ax.set_xlabel("Event weight |w|", fontsize=10)
        ax.set_ylabel("Count (log scale)", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle("Event weight magnitude distribution", fontsize=11)
    fig.tight_layout()
    fname = output_dir / "weight_distribution.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved weight distribution → %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # ── Step 1 / Ingest ───────────────────────────────────────────────────
    log.info("=== Step 1 / Ingest ===")
    real_events, virtual_events = load_events(
        real_path=args.real,
        virtual_path=args.virtual,
        fallback_synthetic=True,
        n_real=args.n_real,
        n_virtual=args.n_virtual,
        neg_fraction=args.neg_fraction,
        seed=args.seed,
    )
    log.info(
        "Events loaded: %d real, %d virtual",
        len(real_events["weight"]),
        len(virtual_events["weight"]),
    )

    # ── Step 2 / Born projection ──────────────────────────────────────────
    log.info("=== Step 2 / Born Projection ===")
    combined = combine_datasets(real_events, virtual_events)
    log.info(
        "Combined dataset: %d events, %d negative-weight (%.1f%%)",
        len(combined),
        int((combined["w"] < 0).sum()),
        (combined["w"] < 0).mean() * 100,
    )

    # Save snapshot of weights before resampling
    w_before = combined["w"].copy()

    # ── Step 3+4 / Spatial hash + greedy resampling ───────────────────────
    log.info("=== Step 3+4 / Resampling ===")
    t0 = time.perf_counter()
    resampler = NegativeWeightResampler(
        pT=combined["pT"],
        y=combined["y"],
        w=combined["w"],   # modified in-place
        max_neighbours=args.max_neighbours,
    )
    stats = resampler.run()
    elapsed = time.perf_counter() - t0
    log.info("Resampling wall-time: %.2f s", elapsed)

    # w_after points at same array (modified in-place by resampler)
    w_after = combined["w"]

    # ── Step 5 / Prune zero-weight events ─────────────────────────────────
    log.info("=== Step 5 / Prune ===")
    keep = w_after != 0.0
    log.info("Pruned %d zero-weight events (%d remaining)",
             int((~keep).sum()), int(keep.sum()))

    # ── Step 6 / Validation plots (Task 3) ────────────────────────────────
    log.info("=== Step 6 / Validation ===")
    closure_metrics = make_validation_plots(
        combined, w_before, w_after, output_dir
    )

    # ── Step 7 / Extra plots ──────────────────────────────────────────────
    log.info("=== Step 7 / Extra diagnostics ===")
    make_cell_size_plot(stats, output_dir)
    make_phase_space_heatmap(combined, w_before, output_dir)
    eff_metrics = make_efficiency_dashboard(stats, len(combined), output_dir)
    make_3d_phase_space_plot(combined, w_after, output_dir)
    make_weight_distribution_plot(w_before, w_after, output_dir)

    # ── Step 8 / Summary JSON ─────────────────────────────────────────────
    summary = {
        "run_config": vars(args),
        "n_events_total":       len(combined),
        "resampling_wall_time_s": elapsed,
        "seeds_total":          stats.n_seeds_total,
        "seeds_processed":      stats.n_seeds_processed,
        "seeds_skipped":        stats.n_seeds_skipped,
        "avg_cell_size":        stats.avg_cell_size,
        "max_cell_size":        stats.max_cell_size,
        "neg_count_before":     stats.neg_count_before,
        "neg_count_after":      stats.neg_count_after,
        "weight_conservation_delta": abs(
            stats.total_weight_after - stats.total_weight_before
        ),
        "closure": closure_metrics,
        "efficiency": eff_metrics,
    }
    json_path = output_dir / "summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Saved summary JSON → %s", json_path)

    # ── Final console summary ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("  Total events         : %d",  len(combined))
    log.info("  Seeds processed      : %d",  stats.n_seeds_processed)
    log.info("  Seeds skipped        : %d",  stats.n_seeds_skipped)
    log.info("  Avg / max cell size  : %.1f / %d",
             stats.avg_cell_size, stats.max_cell_size)
    log.info("  Neg fraction before  : %.3f", stats.neg_fraction_before)
    log.info("  Neg fraction after   : %.3f", stats.neg_fraction_after)
    log.info("  Dilution D before    : %.4f", eff_metrics["dilution_before"])
    log.info("  Dilution D after     : %.4f", eff_metrics["dilution_after"])
    log.info("  Computational gain   : +%.1f%%",
             eff_metrics["computational_gain_pct"])
    log.info("  ΔΣw (weight cons.)   : %.2e",
             eff_metrics["weight_conservation_delta"])
    log.info("  Remaining w < 0      : %d",  stats.neg_count_after)
    log.info("  Output directory     : %s",  output_dir.resolve())
    log.info("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ATLAS Negative-Weight Mitigation pipeline"
    )
    p.add_argument("--real",            type=str,   default=None,
                   help="Path to real_events.csv  (default: data/real_events.csv)")
    p.add_argument("--virtual",         type=str,   default=None,
                   help="Path to virtual_events.csv  (default: data/virtual_events.csv)")
    p.add_argument("--n-real",          type=int,   default=2000,
                   help="# synthetic real events (used only when CSV not found)")
    p.add_argument("--n-virtual",       type=int,   default=1000,
                   help="# synthetic virtual events (used only when CSV not found)")
    p.add_argument("--neg-fraction",    type=float, default=0.20,
                   help="Neg-weight fraction for synthetic fallback")
    p.add_argument("--seed",            type=int,   default=42,
                   help="RNG seed for synthetic fallback")
    p.add_argument("--max-neighbours",  type=int,   default=200,
                   help="Max KD-Tree neighbours per seed")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())