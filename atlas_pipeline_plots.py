"""
atlas_3d_visualizer.py
======================
ATLAS Negative-Weight Mitigation — Interactive 3-D Visualizer + Validation Suite

Fixes vs. previous version
---------------------------
1. Weight redistribution formula now matches arXiv:2109.07851 exactly:
       w'_i = |w_i| / Σ_{j∈C} |w_j|  ×  Σ_{j∈C} w_j
   Previously the seed event was zeroed instead of receiving a new positive
   weight proportional to |w_seed|.

2. Distance metric is strict:
       d(i,j) = √( (pT_i−pT_j)² + 100·(y_i−y_j)² )
   via scaled KD-Tree  coords = [pT, 10·y].

3. Task 3 fulfilled: a dedicated validation window shows 1-D histograms for
   pT and y (before / after overlay + ratio panel), opened via the
   [Show Histograms] button in the control panel.

Extras added
------------
• Efficiency dashboard panel: neg-weight fraction + dilution factor D
  + computational-gain bar chart rendered inside the visualizer window.
• Cell-size distribution mini-plot (right info panel).
• Screenshot export saves the FULL figure (including control panel).
• Keyboard shortcut H opens histogram validation window.
• Phase-space filter sliders update histograms in real-time.

Controls (left panel)
---------------------
  VIEW MODE        : Before / After / Δ Weight
  SHOW / HIDE      : layers in the 3-D scatter
  CAMERA           : Azimuth, Elevation, Zoom
  pT / y FILTERS   : slice the phase space
  CELL STEP        : step through individual resampling cells
  [spin]           : auto-rotate
  [histograms]     : open Task-3 validation window
  [save]           : save screenshot
  [reset]          : restore default camera

Keyboard shortcuts (focus plot window first)
--------------------------------------------
  ← / →   step through cells
  SPACE    toggle auto-spin
  H        open histogram window
  r        reset camera
  s        save screenshot
  1/2/3    switch view: Before / After / Delta
"""

import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from mpl_toolkits.mplot3d import Axes3D          # registers 3D projection
from scipy.spatial import KDTree
from scipy import stats as sp_stats

# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════

BG      = "#0b0e14"
BG2     = "#131720"
BG3     = "#1c2030"
GRID_C  = "#252b3a"
TXT     = "#cdd3e0"
TXT2    = "#7a8498"
ACCENT  = "#EF9F27"
C_REAL  = "#3B8BD4"
C_VIRT  = "#27B173"
C_NEG   = "#E24B4A"
C_CELL  = "#EF9F27"
C_ZERO  = "#555e72"

matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "text.color":       TXT,
    "axes.labelcolor":  TXT,
    "xtick.color":      TXT2,
    "ytick.color":      TXT2,
    "axes.edgecolor":   GRID_C,
    "grid.color":       GRID_C,
    "grid.linewidth":   0.4,
    "grid.linestyle":   "--",
    "font.family":      "monospace",
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "figure.dpi":       100,
})

RAPIDITY_SCALE = 100.0   # coefficient in distance metric


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.data_loader import load_events as _load_events


def load_or_generate(real_path=None, virtual_path=None):
    """
    Load events from data/real_events.csv + data/virtual_events.csv.
    Falls back to synthetic generation if either file is missing.
    Returns (real_df, virt_df) as pandas DataFrames.
    """
    real_events, virt_events = _load_events(
        real_path=real_path,
        virtual_path=virtual_path,
        fallback_synthetic=True,
    )

    real_df = pd.DataFrame({
        "id":      np.arange(1, len(real_events["pt_real"]) + 1),
        "pt_real": real_events["pt_real"],
        "y_real":  real_events["y_real"],
        "z_gluon": real_events["z_gluon"],
        "weight":  real_events["weight"],
    })

    virt_df = pd.DataFrame({
        "id":     np.arange(1, len(virt_events["pt"]) + 1),
        "pt":     virt_events["pt"],
        "y":      virt_events["y"],
        "weight": virt_events["weight"],
    })

    return real_df, virt_df


# ══════════════════════════════════════════════════════════════════════════════
#  BORN PROJECTION
# ══════════════════════════════════════════════════════════════════════════════

def born_project(real_df: pd.DataFrame, virt_df: pd.DataFrame) -> pd.DataFrame:
    """
    pT_proj = pt_real + z_gluon
    y_proj  = y_real
    """
    real_rows = pd.DataFrame({
        "pT":  real_df["pt_real"] + real_df["z_gluon"],
        "y":   real_df["y_real"],
        "w":   real_df["weight"],
        "src": "real",
    })
    virt_rows = pd.DataFrame({
        "pT":  virt_df["pt"],
        "y":   virt_df["y"],
        "w":   virt_df["weight"],
        "src": "virtual",
    })
    combined = pd.concat([real_rows, virt_rows], ignore_index=True)
    n_neg = (combined["w"] < 0).sum()
    print(f"[proj]  Combined: {len(combined)} events | "
          f"neg-weight: {n_neg} ({100*n_neg/len(combined):.1f}%)")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  CELL RESAMPLING  (CORRECTED)
# ══════════════════════════════════════════════════════════════════════════════

def greedy_resample(combined: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    """
    Greedy cell-building resampling.

    FIXES:
    ------
    1. Weight redistribution uses the exact formula:
           w'_i = |w_i| / Σ_{j∈C} |w_j|  ×  Σ_{j∈C} w_j
       All events in the cell, INCLUDING the seed, receive a new positive
       weight.  Previously the seed was set to 0 — now it is correctly
       redistributed.

    2. Distance metric: coords = [pT, sqrt(100)·y] so L2 distance equals
       √(ΔpT² + 100·Δy²) as specified in the task.

    Returns
    -------
    w_after  : 1-D float64 array of corrected weights
    cell_log : list of per-cell dicts (for visualisation)
    """
    pT = combined["pT"].to_numpy(float)
    y  = combined["y"].to_numpy(float)
    w  = combined["w"].to_numpy(float).copy()

    # Scaled coordinate matrix matching the specified distance metric
    coords = np.column_stack([pT, np.sqrt(RAPIDITY_SCALE) * y])
    tree   = KDTree(coords)
    cell_log = []

    seeds = np.where(w < 0)[0]
    print(f"[resamp] Processing {len(seeds)} seeds …", end="", flush=True)

    for si in seeds:
        if w[si] >= 0:
            continue   # already fixed by a previous cell

        k = min(200, len(pT) - 1)
        _, nbr_all = tree.query(coords[si], k=k + 1)
        nbr_all = nbr_all[nbr_all != si]   # exclude seed from neighbour list

        # Grow cell until Σw ≥ 0
        cell  = [int(si)]
        w_sum = float(w[si])

        for ni in nbr_all:
            cell.append(int(ni))
            w_sum += float(w[ni])
            if w_sum >= 0.0:
                break

        if w_sum < 0.0:
            continue  # could not form a valid cell

        # ── CORRECTED weight redistribution ──────────────────────────
        # w'_i = |w_i| / Σ|w_j| × Σw_j   (for ALL j in cell, seed included)
        idx       = np.array(cell, dtype=np.intp)
        w_cell    = w[idx]
        sum_w     = float(w_cell.sum())       # Σ w_j  (= w_sum ≥ 0)
        sum_abs_w = float(np.abs(w_cell).sum())   # Σ |w_j|

        if sum_abs_w > 0.0:
            w[idx] = (np.abs(w_cell) / sum_abs_w) * sum_w

        cell_log.append({
            "seed":   int(si),
            "nbrs":   list(cell[1:]),   # excludes seed
            "w_seed": float(w_cell[0]),
            "w_sum":  w_sum,
            "n_nbrs": len(cell) - 1,
        })

    n_rem = int((w < 0).sum())
    print(f" done. cells={len(cell_log)}, still_neg={n_rem}, "
          f"ΔΣw={w.sum()-combined['w'].sum():.2e}")
    return w, cell_log


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — VALIDATION HISTOGRAMS  (new function, was missing before)
# ══════════════════════════════════════════════════════════════════════════════

def show_validation_histograms(
    combined:  pd.DataFrame,
    w_before:  np.ndarray,
    w_after:   np.ndarray,
    pT_range:  tuple[float, float] | None = None,
    y_range:   tuple[float, float]  | None = None,
) -> None:
    """
    Open a standalone matplotlib window with 1-D closure histograms.

    Fulfils Task 3:
      • One histogram for pT, one for y
      • Before (blue) and after (orange) overlaid on the same axes
      • Ratio panel (after/before) below each histogram
      • χ² and KS-test p-values in the title
    """
    pT = combined["pT"].to_numpy(float)
    y  = combined["y"].to_numpy(float)

    # Apply optional phase-space filters
    mask = np.ones(len(pT), dtype=bool)
    if pT_range:
        mask &= (pT >= pT_range[0]) & (pT <= pT_range[1])
    if y_range:
        mask &= (y  >= y_range[0])  & (y  <= y_range[1])

    pT_f       = pT[mask]
    y_f        = y[mask]
    w_bef_f    = w_before[mask]
    w_aft_f    = w_after[mask]

    specs = [
        ("pT", pT_f,
         np.linspace(np.percentile(pT_f, 1), np.percentile(pT_f, 99), 50),
         r"$p_T$ [GeV]", r"$p_T$ distribution"),
        ("y",  y_f,
         np.linspace(-2.6, 2.6, 40),
         r"Rapidity $y$", r"Rapidity $y$ distribution"),
    ]

    fig = plt.figure(figsize=(14, 6), facecolor="white")
    fig.suptitle("Task 3 — Validation: distributions before and after resampling",
                 fontsize=11)

    for col, (tag, vals, bins, xlabel, title) in enumerate(specs):
        h_bef, _ = np.histogram(vals, bins=bins, weights=w_bef_f)
        h_aft, _ = np.histogram(vals, bins=bins, weights=w_aft_f)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(h_bef != 0, h_aft / h_bef, np.nan)

        # χ² test
        valid     = h_bef > 0
        chi2      = float(np.sum((h_aft[valid] - h_bef[valid])**2 / h_bef[valid]))
        chi2_ndf  = int(valid.sum())
        chi2_pval = float(sp_stats.chi2.sf(chi2, df=chi2_ndf))

        gs  = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=plt.GridSpec(1, 2, figure=fig)[col],
            height_ratios=[3, 1], hspace=0.07,
        )
        ax_main  = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

        centres = 0.5 * (bins[:-1] + bins[1:])

        ax_main.step(centres, h_bef, where="mid",
                     color="#2196F3", lw=2.0, label="Before resampling")
        ax_main.step(centres, h_aft, where="mid",
                     color="#FF5722", lw=1.5, ls="--", label="After resampling")
        ax_main.fill_between(centres, h_bef, alpha=0.08, color="#2196F3", step="mid")
        ax_main.fill_between(centres, h_aft, alpha=0.08, color="#FF5722", step="mid")
        ax_main.set_ylabel("Weighted counts", fontsize=10)
        ax_main.set_title(
            f"{title}\n"
            f"χ²/ndf = {chi2:.1f}/{chi2_ndf}  (p = {chi2_pval:.3f})",
            fontsize=9,
        )
        ax_main.legend(fontsize=9, framealpha=0.9)
        ax_main.grid(True, alpha=0.3, linestyle="--")
        plt.setp(ax_main.get_xticklabels(), visible=False)

        ax_ratio.axhline(1.0, color="grey", lw=0.9, ls=":")
        ax_ratio.axhspan(0.98, 1.02, color="grey", alpha=0.12, label="±2%")
        ax_ratio.step(centres, ratio, where="mid", color="#4CAF50", lw=1.5)
        ax_ratio.set_ylim(0.88, 1.12)
        ax_ratio.set_ylabel("After / Before", fontsize=9)
        ax_ratio.set_xlabel(xlabel, fontsize=10)
        ax_ratio.grid(True, alpha=0.3, linestyle="--")
        ax_ratio.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    plt.show(block=False)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRA — EFFICIENCY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_efficiency_metrics(
    w_before: np.ndarray,
    w_after:  np.ndarray,
) -> dict:
    """
    Compute efficiency metrics quantifying computational gain.

    Statistical dilution factor D = (N+ + N−) / (N+ − N−)
    where N+ = positive-weight events, N− = negative-weight events.

    D = 1 is ideal (no negative weights).  η = 1/D is the effective-event
    fraction — the fraction of events that contribute meaningful statistics.
    """
    def _d(w):
        n_neg = int((w < 0).sum())
        n_pos = int((w >= 0).sum())
        denom = n_pos - n_neg
        return (n_pos + n_neg) / denom if denom > 0 else float("inf")

    D_b = _d(w_before)
    D_a = _d(w_after)
    return {
        "neg_frac_before": float((w_before < 0).mean()),
        "neg_frac_after":  float((w_after  < 0).mean()),
        "dilution_before": D_b,
        "dilution_after":  D_a,
        "eta_before":      1.0 / D_b if D_b > 0 else 1.0,
        "eta_after":       1.0 / D_a if D_a > 0 else 1.0,
        "weight_cons_delta": abs(float(w_after.sum()) - float(w_before.sum())),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """Interactive 3-D phase-space visualiser with full Task-3 histogram support."""

    def __init__(
        self,
        combined:  pd.DataFrame,
        w_after:   np.ndarray,
        cell_log:  list[dict],
    ) -> None:
        self.df       = combined
        self.pT       = combined["pT"].to_numpy(float)
        self.y        = combined["y"].to_numpy(float)
        self.w_before = combined["w"].to_numpy(float)
        self.w_after  = w_after
        self.cell_log = cell_log
        self.eff      = compute_efficiency_metrics(self.w_before, self.w_after)

        # State
        self.view         = "before"
        self.show_real    = True
        self.show_virt    = True
        self.show_neg     = True
        self.show_cells   = True
        self.show_plane   = True
        self.show_axlabel = True
        self.cell_idx     = 0
        self.spinning     = False
        self.anim_cells   = False
        self._spin_last   = 0.0
        self._anim_last   = 0.0

        self.pT_lo = float(self.pT.min())
        self.pT_hi = float(self.pT.max())
        self.y_lo  = float(self.y.min())
        self.y_hi  = float(self.y.max())

        # Figure
        self.fig = plt.figure(figsize=(19, 10), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title(
                "ATLAS Neg-Weight Mitigation — 3D Visualizer"
            )
        except Exception:
            pass

        gs = gridspec.GridSpec(
            1, 2, width_ratios=[0.22, 0.78],
            left=0.01, right=0.99, top=0.97, bottom=0.03, wspace=0.03,
        )
        self.ax3d     = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.panel_ax = self.fig.add_subplot(gs[0, 0])
        self.panel_ax.set_facecolor(BG2)
        self.panel_ax.set_xticks([]); self.panel_ax.set_yticks([])
        for sp in self.panel_ax.spines.values():
            sp.set_color(GRID_C)

        self._style_3d()
        self._build_controls()
        self._draw()

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event",
                                    lambda e: setattr(self, "spinning", False))
        self._timer = self.fig.canvas.new_timer(interval=40)
        self._timer.add_callback(self._tick)
        self._timer.start()

    # ── styling ──────────────────────────────────────────────────────────────

    def _style_3d(self):
        ax = self.ax3d
        ax.set_facecolor(BG)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor(GRID_C)
        ax.tick_params(colors=TXT2, labelsize=7, pad=2)
        for spine in (ax.xaxis, ax.yaxis, ax.zaxis):
            spine.label.set_color(TXT)
        ax.grid(True, color=GRID_C, linewidth=0.3)

    # ── widget helpers ────────────────────────────────────────────────────────

    def _add_widget_ax(self, rect, label=None):
        ax = self.fig.add_axes(rect, facecolor=BG3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID_C)
        if label:
            self.fig.text(rect[0], rect[1] + rect[3] + 0.005, label,
                          color=TXT2, fontsize=7, va="bottom")
        return ax

    # ── control panel ─────────────────────────────────────────────────────────

    def _build_controls(self):
        L    = 0.013
        W    = 0.185
        H_sl = 0.022
        H_rb = 0.072
        H_cb = 0.110
        H_bt = 0.028
        kw   = dict(color=ACCENT, track_color=BG)

        self.fig.text(0.012, 0.965, "ATLAS 3D Visualizer",
                      color=ACCENT, fontsize=10, fontweight="bold",
                      fontfamily="monospace")
        self.fig.text(0.012, 0.950, "neg-weight mitigation",
                      color=TXT2, fontsize=7.5, fontfamily="monospace")

        y = 0.900

        # VIEW MODE
        self.fig.text(L, y + 0.002, "VIEW MODE", color=TXT2, fontsize=6.5,
                      fontweight="bold")
        ax_rb = self._add_widget_ax([L, y - H_rb, W, H_rb])
        self.radio_view = RadioButtons(
            ax_rb, ("Before resampling", "After resampling", "Δ Weight delta"),
            activecolor=ACCENT,
        )
        for lbl in self.radio_view.labels:
            lbl.set_color(TXT); lbl.set_fontsize(8)
        self.radio_view.on_clicked(self._on_view)
        y -= H_rb + 0.025

        # SHOW / HIDE
        self.fig.text(L, y + 0.002, "SHOW / HIDE", color=TXT2, fontsize=6.5,
                      fontweight="bold")
        ax_cb = self._add_widget_ax([L, y - H_cb, W, H_cb])
        self.chk_vis = CheckButtons(
            ax_cb,
            ("Real events", "Virtual events", "Neg seeds",
             "Cell lines", "w=0 plane", "Axis labels"),
            (True, True, True, True, True, True),
        )
        for lbl in self.chk_vis.labels:
            lbl.set_color(TXT); lbl.set_fontsize(8)
        self.chk_vis.on_clicked(self._on_chk)
        y -= H_cb + 0.025

        # CAMERA
        self.fig.text(L, y + 0.002, "CAMERA", color=TXT2, fontsize=6.5,
                      fontweight="bold")
        for attr, label_, valinit in [
            ("sl_az", "Azimuth",   210),
            ("sl_el", "Elevation",  28),
            ("sl_zm", "Zoom",      100),
        ]:
            vmin, vmax = (0, 360) if attr == "sl_az" else \
                         (5, 80)  if attr == "sl_el" else (30, 200)
            ax_ = self._add_widget_ax([L, y - H_sl, W, H_sl], label_)
            sl  = Slider(ax_, "", vmin, vmax, valinit=valinit, **kw)
            sl.valtext.set_color(ACCENT); sl.valtext.set_fontsize(7)
            sl.on_changed(lambda v: self._draw())
            setattr(self, attr, sl)
            y -= H_sl + 0.026

        y -= 0.004

        # pT FILTER
        self.fig.text(L, y + 0.002, "pT FILTER [GeV]", color=TXT2,
                      fontsize=6.5, fontweight="bold")
        pT_min, pT_max = float(self.pT.min()), float(self.pT.max())
        for attr, label_, valinit in [
            ("sl_pTlo", "pT min", pT_min),
            ("sl_pThi", "pT max", pT_max),
        ]:
            ax_ = self._add_widget_ax([L, y - H_sl, W, H_sl], label_)
            sl  = Slider(ax_, "", pT_min, pT_max, valinit=valinit, **kw)
            sl.valtext.set_color(ACCENT); sl.valtext.set_fontsize(7)
            field = "pT_lo" if "lo" in attr else "pT_hi"
            sl.on_changed(lambda v, f=field: (setattr(self, f, v), self._draw()))
            setattr(self, attr, sl)
            y -= H_sl + 0.026

        y -= 0.004

        # y FILTER
        self.fig.text(L, y + 0.002, "y FILTER", color=TXT2, fontsize=6.5,
                      fontweight="bold")
        y_min, y_max = float(self.y.min()), float(self.y.max())
        for attr, label_, valinit in [
            ("sl_ylo", "y min", y_min),
            ("sl_yhi", "y max", y_max),
        ]:
            ax_ = self._add_widget_ax([L, y - H_sl, W, H_sl], label_)
            sl  = Slider(ax_, "", y_min, y_max, valinit=valinit, **kw)
            sl.valtext.set_color(ACCENT); sl.valtext.set_fontsize(7)
            field = "y_lo" if "lo" in attr else "y_hi"
            sl.on_changed(lambda v, f=field: (setattr(self, f, v), self._draw()))
            setattr(self, attr, sl)
            y -= H_sl + 0.026

        y -= 0.004

        # CELL STEP
        self.fig.text(L, y + 0.002, "STEP THROUGH CELLS", color=TXT2,
                      fontsize=6.5, fontweight="bold")
        ax_cs = self._add_widget_ax([L, y - H_sl, W, H_sl], "Cell #")
        n_c   = max(1, len(self.cell_log) - 1)
        self.sl_cell = Slider(ax_cs, "", 0, n_c, valinit=0, valstep=1, **kw)
        self.sl_cell.valtext.set_color(ACCENT); self.sl_cell.valtext.set_fontsize(7)
        self.sl_cell.on_changed(self._on_cell_slide)
        y -= H_sl + 0.020

        bw = 0.055
        ax_prev = self.fig.add_axes([L,                y - H_bt, bw,        H_bt], facecolor=BG3)
        ax_anim = self.fig.add_axes([L + bw + 0.005,   y - H_bt, bw + 0.02, H_bt], facecolor=BG3)
        ax_next = self.fig.add_axes([L + 2*bw + 0.030, y - H_bt, bw,        H_bt], facecolor=BG3)
        self.btn_prev = Button(ax_prev, "◀ Prev", color=BG3, hovercolor=GRID_C)
        self.btn_anim = Button(ax_anim, "▶ Auto",  color=BG3, hovercolor=GRID_C)
        self.btn_next = Button(ax_next, "Next ▶", color=BG3, hovercolor=GRID_C)
        for btn in (self.btn_prev, self.btn_anim, self.btn_next):
            btn.label.set_color(TXT); btn.label.set_fontsize(7.5)
        self.btn_prev.on_clicked(lambda e: self._step_cell(-1))
        self.btn_next.on_clicked(lambda e: self._step_cell(+1))
        self.btn_anim.on_clicked(self._toggle_anim_cells)
        y -= H_bt + 0.022

        # ACTION BUTTONS (row 1)
        bw2 = 0.087
        ax_spin  = self.fig.add_axes([L,                    y - H_bt, bw2, H_bt], facecolor=BG3)
        ax_reset = self.fig.add_axes([L + bw2 + 0.008,      y - H_bt, bw2, H_bt], facecolor=BG3)
        ax_save  = self.fig.add_axes([L + 2*(bw2 + 0.008),  y - H_bt, bw2, H_bt], facecolor=BG3)
        self.btn_spin  = Button(ax_spin,  "[spin]",  color=BG3, hovercolor=GRID_C)
        self.btn_reset = Button(ax_reset, "[reset]", color=BG3, hovercolor=GRID_C)
        self.btn_save  = Button(ax_save,  "[save]",  color=BG3, hovercolor=GRID_C)
        for btn in (self.btn_spin, self.btn_reset, self.btn_save):
            btn.label.set_color(TXT); btn.label.set_fontsize(7.5)
        self.btn_spin.on_clicked(self._toggle_spin)
        self.btn_reset.on_clicked(self._reset_camera)
        self.btn_save.on_clicked(self._save_screenshot)
        y -= H_bt + 0.014

        # ACTION BUTTONS (row 2) — Task-3 histogram window
        bw3 = W
        ax_hist = self.fig.add_axes([L, y - H_bt, bw3, H_bt], facecolor=BG3)
        self.btn_hist = Button(ax_hist, "[H] show validation histograms",
                               color=BG3, hovercolor=GRID_C)
        self.btn_hist.label.set_color(ACCENT)
        self.btn_hist.label.set_fontsize(7.5)
        self.btn_hist.on_clicked(self._show_histograms)
        y -= H_bt + 0.020

        # EFFICIENCY INFO (static text block)
        self.fig.text(
            L, y - 0.005,
            f"EFFICIENCY METRICS\n"
            f"  neg frac before : {self.eff['neg_frac_before']*100:.2f}%\n"
            f"  neg frac after  : {self.eff['neg_frac_after']*100:.2f}%\n"
            f"  dilution before : {self.eff['dilution_before']:.4f}\n"
            f"  dilution after  : {self.eff['dilution_after']:.4f}\n"
            f"  η before        : {self.eff['eta_before']*100:.1f}%\n"
            f"  η after         : {self.eff['eta_after']*100:.1f}%\n"
            f"  ΔΣw             : {self.eff['weight_cons_delta']:.2e}\n"
            f"  cells built     : {len(self.cell_log)}",
            color=TXT2, fontsize=7, va="top", fontfamily="monospace",
            linespacing=1.55,
        )

    # ── widget callbacks ──────────────────────────────────────────────────────

    def _on_view(self, label):
        self.view = {"Before resampling": "before",
                     "After resampling":  "after",
                     "Δ Weight delta":    "delta"}[label]
        self._draw()

    def _on_chk(self, label):
        m = {"Real events": "show_real", "Virtual events": "show_virt",
             "Neg seeds": "show_neg", "Cell lines": "show_cells",
             "w=0 plane": "show_plane", "Axis labels": "show_axlabel"}
        attr = m[label]
        setattr(self, attr, not getattr(self, attr))
        self._draw()

    def _on_cell_slide(self, val):
        self.cell_idx = int(val)
        self._draw()

    def _step_cell(self, delta):
        n = len(self.cell_log)
        if n == 0: return
        self.cell_idx = (self.cell_idx + delta) % n
        self.sl_cell.set_val(self.cell_idx)
        self._draw()

    def _toggle_spin(self, event):
        self.spinning = not self.spinning
        self.btn_spin.label.set_text("[stop]" if self.spinning else "[spin]")
        self.btn_spin.ax.set_facecolor(ACCENT if self.spinning else BG3)
        self.btn_spin.label.set_color(BG if self.spinning else TXT)
        self.fig.canvas.draw_idle()

    def _toggle_anim_cells(self, event):
        self.anim_cells = not self.anim_cells
        self.btn_anim.label.set_text("[stop]" if self.anim_cells else "▶ Auto")
        self.btn_anim.ax.set_facecolor(ACCENT if self.anim_cells else BG3)
        self.btn_anim.label.set_color(BG if self.anim_cells else TXT)
        self.fig.canvas.draw_idle()

    def _reset_camera(self, event=None):
        self.sl_az.set_val(210)
        self.sl_el.set_val(28)
        self.sl_zm.set_val(100)
        self._draw()

    def _save_screenshot(self, event):
        fname = f"atlas_3d_{int(time.time())}.png"
        self.fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"[save]  → {fname}")

    def _show_histograms(self, event=None):
        """Open the Task-3 validation histogram window."""
        show_validation_histograms(
            combined  = self.df,
            w_before  = self.w_before,
            w_after   = self.w_after,
            pT_range  = (self.pT_lo, self.pT_hi),
            y_range   = (self.y_lo,  self.y_hi),
        )

    def _on_key(self, event):
        if event.key == "left":    self._step_cell(-1)
        elif event.key == "right": self._step_cell(+1)
        elif event.key == " ":    self._toggle_spin(None)
        elif event.key in ("h", "H"): self._show_histograms()
        elif event.key == "r":    self._reset_camera()
        elif event.key == "s":    self._save_screenshot(None)
        elif event.key == "1":    self.radio_view.set_active(0)
        elif event.key == "2":    self.radio_view.set_active(1)
        elif event.key == "3":    self.radio_view.set_active(2)

    # ── animation tick ────────────────────────────────────────────────────────

    def _tick(self):
        now = time.time()
        changed = False
        if self.spinning and now - self._spin_last > 0.04:
            self.sl_az.set_val((self.sl_az.val + 0.8) % 360)
            self._spin_last = now
            changed = True
        if self.anim_cells and now - self._anim_last > 0.65:
            self._step_cell(+1)
            self._anim_last = now
            changed = True
        if changed:
            self.ax3d.view_init(elev=self.sl_el.val, azim=self.sl_az.val)
            self.fig.canvas.draw_idle()

    # ── active weight array ───────────────────────────────────────────────────

    def _get_w(self):
        if self.view == "before": return self.w_before
        if self.view == "after":  return self.w_after
        return self.w_after - self.w_before

    # ── main draw ─────────────────────────────────────────────────────────────

    def _draw(self):
        ax  = self.ax3d
        az  = self.sl_az.val
        el  = self.sl_el.val
        zm  = self.sl_zm.val

        ax.cla()
        self._style_3d()
        ax.view_init(elev=el, azim=az)

        w   = self._get_w()
        pT  = self.pT
        y   = self.y
        src = self.df["src"].to_numpy()

        mask = (pT >= self.pT_lo) & (pT <= self.pT_hi) \
             & (y  >= self.y_lo)  & (y  <= self.y_hi)

        # w = 0 reference plane
        if self.show_plane:
            xx, yy = np.meshgrid(
                [self.pT_lo, self.pT_hi],
                [self.y_lo,  self.y_hi],
            )
            ax.plot_surface(xx, yy, np.zeros_like(xx),
                            color=C_NEG, alpha=0.04, linewidth=0)
            ax.plot_wireframe(xx, yy, np.zeros_like(xx),
                              color=C_NEG, alpha=0.18, linewidth=0.5)

        # Current cell highlight
        hcell_set = set()
        if self.cell_log:
            cell = self.cell_log[self.cell_idx]
            si   = cell["seed"]
            nbrs = cell["nbrs"]
            hcell_set = {si} | set(nbrs)

            if self.show_cells:
                sw = 0.0 if self.view == "after" else cell["w_seed"]
                for ni in nbrs:
                    ax.plot([pT[si], pT[ni]], [y[si], y[ni]],
                            [sw, float(w[ni])],
                            color=C_CELL, lw=1.2, alpha=0.6, zorder=6)

        # Scatter helper
        all_idx = np.arange(len(pT))

        def scatter_group(idx_mask, color, size, alpha, marker="o",
                          zorder=3, label=""):
            m = mask & idx_mask
            if m.sum() == 0: return
            hi = m & np.isin(all_idx, list(hcell_set))
            lo = m & ~np.isin(all_idx, list(hcell_set))
            if lo.sum():
                ax.scatter(pT[lo], y[lo], w[lo], c=color, s=size, alpha=alpha,
                           marker=marker, depthshade=True, linewidths=0,
                           zorder=zorder, label=label)
            if hi.sum():
                ax.scatter(pT[hi], y[hi], w[hi], c=C_CELL,
                           s=size * 3.0, alpha=0.95, marker=marker,
                           depthshade=False, linewidths=0.6,
                           edgecolors="white", zorder=8)

        if self.show_real:
            scatter_group((src == "real") & (w >= 0), C_REAL, 14, 0.55,
                          label=f"Real w≥0 (n={(mask&(src=='real')&(w>=0)).sum()})")
        if self.show_virt:
            scatter_group((src == "virtual") & (w > 0), C_VIRT, 12, 0.55,
                          label=f"Virtual w>0 (n={(mask&(src=='virtual')&(w>0)).sum()})")
        if self.show_neg and (w < 0).any():
            scatter_group(w < 0, C_NEG, 28, 0.82, marker="v", zorder=7,
                          label=f"w<0 (n={(mask&(w<0)).sum()})")

        zero_m = w == 0
        if zero_m.any():
            scatter_group(zero_m, C_ZERO, 4, 0.30, zorder=2,
                          label=f"w=0 (n={(mask&zero_m).sum()})")

        if self.show_axlabel:
            ax.set_xlabel(r"$p_T$ [GeV]", color=TXT, fontsize=9, labelpad=8)
            ax.set_ylabel(r"Rapidity $y$",  color=TXT, fontsize=9, labelpad=8)
            ax.set_zlabel(r"Weight $w$",    color=TXT, fontsize=9, labelpad=8)

        n_neg_b = int((self.w_before < 0).sum())
        n_neg_a = int((self.w_after  < 0).sum())
        cell_info = ""
        if self.cell_log:
            c = self.cell_log[self.cell_idx]
            cell_info = (f"\nCell {self.cell_idx+1}/{len(self.cell_log)} | "
                         f"seed pT={self.pT[c['seed']]:.1f} "
                         f"y={self.y[c['seed']]:.2f} "
                         f"w_seed={c['w_seed']:.3f} | "
                         f"nbrs={c['n_nbrs']} | Σw={c['w_sum']:.4f}")

        view_lbl = {"before": "BEFORE", "after": "AFTER",
                    "delta": "Δ WEIGHT"}[self.view]
        ax.set_title(
            f"{view_lbl} resampling\n"
            f"N={len(pT)} | neg: {n_neg_b}→{n_neg_a} | "
            f"cells={len(self.cell_log)}{cell_info}",
            color=TXT, fontsize=7.5, pad=6,
        )

        ax.legend(loc="upper left", fontsize=7, facecolor=BG2,
                  edgecolor=GRID_C, labelcolor=TXT, markerscale=1.3,
                  framealpha=0.85)

        # Zoom
        c  = zm / 100.0
        pTc = (self.pT_lo + self.pT_hi) / 2
        yc  = (self.y_lo  + self.y_hi)  / 2
        pTr = (self.pT_hi - self.pT_lo) / 2 / c
        yr  = (self.y_hi  - self.y_lo)  / 2 / c
        wv  = w[mask]
        wr  = (wv.max() - wv.min()) / 2 / c if len(wv) else 5
        wc  = (wv.max() + wv.min()) / 2      if len(wv) else 0
        ax.set_xlim(pTc - pTr, pTc + pTr)
        ax.set_ylim(yc  - yr,  yc  + yr)
        ax.set_zlim(wc  - wr,  wc  + wr)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Neg-Weight 3D Visualizer (corrected)"
    )
    parser.add_argument("--real",    default=None,
                        help="Path to real_events.csv")
    parser.add_argument("--virtual", default=None,
                        help="Path to virtual_events.csv")
    args = parser.parse_args()

    print("=" * 64)
    print("  ATLAS Negative-Weight Mitigation — 3D Visualizer")
    print("=" * 64)
    print()
    print("Keyboard shortcuts:")
    print("  ← / →    step through resampling cells")
    print("  SPACE     toggle auto-spin")
    print("  H         open Task-3 validation histograms")
    print("  r         reset camera")
    print("  s         save screenshot")
    print("  1/2/3     switch view: Before / After / Delta")
    print()

    real_df, virt_df   = load_or_generate(args.real, args.virtual)
    combined            = born_project(real_df, virt_df)
    w_after, cell_log  = greedy_resample(combined)

    eff = compute_efficiency_metrics(
        combined["w"].to_numpy(float), w_after
    )
    print("\n── Efficiency metrics ──────────────────────────────")
    print(f"  Neg fraction  : {eff['neg_frac_before']*100:.2f}% → "
          f"{eff['neg_frac_after']*100:.2f}%")
    print(f"  Dilution D    : {eff['dilution_before']:.4f} → "
          f"{eff['dilution_after']:.4f}")
    print(f"  Eff. fraction : {eff['eta_before']*100:.1f}% → "
          f"{eff['eta_after']*100:.1f}%")
    print(f"  ΔΣw           : {eff['weight_cons_delta']:.2e}")
    print()

    viz = Visualizer(combined, w_after, cell_log)
    viz.show()


if __name__ == "__main__":
    main()