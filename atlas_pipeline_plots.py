"""
ATLAS Negative-Weight Mitigation — Standalone 3D Interactive Visualizer
========================================================================

Run:
    pip install numpy scipy matplotlib pandas
    python atlas_3d_visualizer.py

Or pass your own CSV files:
    python atlas_3d_visualizer.py --real real_events.csv --virtual virtual_events.csv

CSV column requirements
  real_events.csv    : id, pt_real, y_real, z_gluon, weight
  virtual_events.csv : id, pt, y, weight

Controls (live panel on the left)
  • View mode     — Before / After / Δ Weight
  • Show/Hide     — Real, Virtual, Seeds, Cell lines, w=0 plane, Axis labels
  • Camera        — Azimuth, Elevation, Zoom sliders
  • Filters       — pT range, y range
  • Cell step     — step through individual resampling cells with Prev/Next/Auto
  • Spin toggle   — continuous auto-rotation
  • Screenshot    — save current view to PNG
  • Reset         — restore default camera
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import (
    Slider, Button, RadioButtons, CheckButtons
)
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D          # registers 3D projection
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy.spatial import KDTree

# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════
BG       = "#0b0e14"      # main background
BG2      = "#131720"      # panel background
BG3      = "#1c2030"      # widget slots
GRID_C   = "#252b3a"      # grid / pane lines
TXT      = "#cdd3e0"      # primary text
TXT2     = "#7a8498"      # secondary text
ACCENT   = "#EF9F27"      # amber accent
C_REAL   = "#3B8BD4"      # real positive
C_VIRT   = "#27B173"      # virtual positive
C_NEG    = "#E24B4A"      # negative seed
C_CELL   = "#EF9F27"      # cell / neighbour
C_ZERO   = "#555e72"      # neutralised
C_PROJ   = "#B48BDA"      # born-projected highlight
C_DELTA_P= "#27B173"
C_DELTA_N= "#E24B4A"

matplotlib.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : BG,
    "text.color"       : TXT,
    "axes.labelcolor"  : TXT,
    "xtick.color"      : TXT2,
    "ytick.color"      : TXT2,
    "axes.edgecolor"   : GRID_C,
    "grid.color"       : GRID_C,
    "grid.linewidth"   : 0.4,
    "grid.linestyle"   : "--",
    "font.family"      : "monospace",
    "font.size"        : 8,
    "axes.titlesize"   : 9,
    "axes.labelsize"   : 9,
    "figure.dpi"       : 100,
})

RAPIDITY_SCALE = 100.0

# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_or_generate(real_path=None, virtual_path=None):
    """Load CSV files if provided, otherwise generate synthetic data anchored
    to the user's actual values (pt_real=30.60, y_real=1.188, etc.)."""

    # ── anchor values from the user's sample rows ─────────────────────────
    PT_ANCHOR = 30.60180862
    Y_ANCHOR  = 1.188217111
    Z_ANCHOR  = 1.262930928
    W_ANCHOR  = 7.56818116
    PT_VIRT   = 70.23653667
    Y_VIRT    = -1.323280281
    W_VIRT    = -1.056124491

    rng = np.random.default_rng(seed=int(PT_ANCHOR * 1000) % (2**31))

    if real_path and os.path.exists(real_path):
        real_df = pd.read_csv(real_path, sep=None, engine="python")
        real_df.columns = [c.strip() for c in real_df.columns]
        print(f"[data]  Loaded real_events: {len(real_df)} rows")
    else:
        print("[data]  Generating synthetic real events anchored to sample row …")
        N = 900
        pt  = rng.exponential(scale=PT_ANCHOR * 0.8, size=N) + PT_ANCHOR * 0.25
        y   = np.clip(rng.normal(Y_ANCHOR,  1.6, N), -5, 5)
        zg  = np.abs(rng.exponential(Z_ANCHOR * 1.3, N))
        w   = np.abs(rng.gamma(shape=2.2, scale=W_ANCHOR / 2.2, size=N)) + 0.1
        real_df = pd.DataFrame({"id": np.arange(1, N+1),
                                 "pt_real": np.r_[PT_ANCHOR, pt[:N-1]],
                                 "y_real":  np.r_[Y_ANCHOR,  y[:N-1]],
                                 "z_gluon": np.r_[Z_ANCHOR,  zg[:N-1]],
                                 "weight":  np.r_[W_ANCHOR,  w[:N-1]]})

    if virtual_path and os.path.exists(virtual_path):
        virt_df = pd.read_csv(virtual_path, sep=None, engine="python")
        virt_df.columns = [c.strip() for c in virt_df.columns]
        print(f"[data]  Loaded virtual_events: {len(virt_df)} rows")
    else:
        print("[data]  Generating synthetic virtual events anchored to sample row …")
        N = 700
        pt  = rng.exponential(scale=abs(PT_VIRT) * 0.75, size=N) + abs(PT_VIRT) * 0.25
        y   = np.clip(rng.normal(Y_VIRT, 1.9, N), -5, 5)
        w   = np.abs(rng.gamma(shape=1.8, scale=abs(W_VIRT) * 1.4, size=N)) + 0.05
        neg = rng.random(N) < 0.23
        w[neg] *= -1
        virt_df = pd.DataFrame({"id": np.arange(1, N+1),
                                 "pt":     np.r_[PT_VIRT,    pt[:N-1]],
                                 "y":      np.r_[Y_VIRT,     y[:N-1]],
                                 "weight": np.r_[W_VIRT,     w[:N-1]]})

    return real_df, virt_df


def born_project(real_df, virt_df):
    """pT_proj = pt_real + z_gluon ; y_proj = y_real"""
    real_rows = pd.DataFrame({
        "pT"  : real_df["pt_real"] + real_df["z_gluon"],
        "y"   : real_df["y_real"],
        "w"   : real_df["weight"],
        "src" : "real",
    })
    virt_rows = pd.DataFrame({
        "pT"  : virt_df["pt"],
        "y"   : virt_df["y"],
        "w"   : virt_df["weight"],
        "src" : "virtual",
    })
    combined = pd.concat([real_rows, virt_rows], ignore_index=True)
    n_neg = (combined["w"] < 0).sum()
    print(f"[proj]  Combined: {len(combined)} events | neg-weight: {n_neg} ({100*n_neg/len(combined):.1f}%)")
    return combined


def greedy_resample(combined):
    """Greedy cell-building algorithm. Returns (w_after, cell_log)."""
    pT = combined["pT"].to_numpy(float)
    y  = combined["y"].to_numpy(float)
    w  = combined["w"].to_numpy(float).copy()

    coords = np.column_stack([pT, np.sqrt(RAPIDITY_SCALE) * y])
    tree   = KDTree(coords)
    cell_log = []   # list of dicts per cell

    seeds = np.where(w < 0)[0]
    print(f"[resamp] {len(seeds)} seeds …", end="", flush=True)

    for si in seeds:
        if w[si] >= 0:
            continue
        w_seed = float(w[si])
        k      = min(60, len(pT) - 1)
        _, nbrs = tree.query(coords[si], k=k+1)
        nbrs    = nbrs[nbrs != si]

        cell, w_sum = [], w_seed
        for ni in nbrs:
            cell.append(int(ni))
            w_sum += float(w[ni])
            if w_sum >= 0:
                break

        if w_sum < 0 or not cell:
            continue

        w_pos  = float(w[np.array(cell)].sum())
        if w_pos <= 0:
            continue

        cell_log.append({
            "seed"   : int(si),
            "nbrs"   : list(cell),
            "w_seed" : w_seed,
            "w_sum"  : w_sum,
            "n_nbrs" : len(cell),
        })
        scale       = 1.0 + w_seed / w_pos
        w[cell]    *= scale
        w[si]       = 0.0

    n_rem = int((w < 0).sum())
    print(f" done. cells={len(cell_log)}, still_neg={n_rem}, ΔΣw={w.sum()-combined['w'].sum():.2e}")
    return w, cell_log


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """Full interactive 3-D visualiser with a rich left-panel control set."""

    def __init__(self, combined, w_after, cell_log, real_df):
        self.df       = combined
        self.pT       = combined["pT"].to_numpy(float)
        self.y        = combined["y"].to_numpy(float)
        self.w_before = combined["w"].to_numpy(float)
        self.w_after  = w_after
        self.cell_log = cell_log
        self.real_df  = real_df

        # ── state ─────────────────────────────────────────────────────────
        self.view         = "before"   # before | after | delta
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

        # ── figure layout ─────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(18, 10), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            "ATLAS Neg-Weight Mitigation — 3D Phase Space Visualizer"
        )

        # left panel (controls) width = 0.22, right = 3D canvas
        gs = gridspec.GridSpec(1, 2, width_ratios=[0.22, 0.78],
                               left=0.01, right=0.99,
                               top=0.97, bottom=0.03, wspace=0.03)

        self.ax3d = self.fig.add_subplot(gs[0, 1], projection="3d")
        self._style_3d()

        # control panel background
        self.panel_ax = self.fig.add_subplot(gs[0, 0])
        self.panel_ax.set_facecolor(BG2)
        self.panel_ax.set_xticks([]); self.panel_ax.set_yticks([])
        for sp in self.panel_ax.spines.values():
            sp.set_color(GRID_C)

        self._build_controls()
        self._draw()

        # connect events
        self.fig.canvas.mpl_connect("key_press_event",  self._on_key)
        self.fig.canvas.mpl_connect("close_event",      lambda e: setattr(self, "spinning", False))
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

    # ── control panel builder ─────────────────────────────────────────────────

    def _add_widget_ax(self, rect, label=None):
        """Add a widget axes slot at [left, bottom, width, height] in figure coords."""
        ax = self.fig.add_axes(rect, facecolor=BG3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(GRID_C)
        if label:
            self.fig.text(rect[0], rect[1] + rect[3] + 0.005, label,
                          color=TXT2, fontsize=7, va="bottom")
        return ax

    def _build_controls(self):
        L = 0.013   # left edge of widget slots
        W = 0.185   # width of widget slots
        H_sl = 0.022  # slider height
        H_rb = 0.072  # radio height
        H_cb = 0.110  # checkbox height
        H_bt = 0.028  # button height

        # ── title ─────────────────────────────────────────────────────────
        self.fig.text(0.012, 0.965, "ATLAS 3D Visualizer",
                      color=ACCENT, fontsize=10, fontweight="bold",
                      fontfamily="monospace")
        self.fig.text(0.012, 0.950, "neg-weight mitigation",
                      color=TXT2, fontsize=7.5, fontfamily="monospace")

        y = 0.900

        # ── VIEW MODE radio ───────────────────────────────────────────────
        self.fig.text(L, y + 0.002, "VIEW MODE", color=TXT2, fontsize=6.5,
                      fontweight="bold", fontfamily="monospace")
        ax_rb = self._add_widget_ax([L, y - H_rb, W, H_rb])
        self.radio_view = RadioButtons(
            ax_rb, ("Before resampling", "After resampling", "Δ Weight delta"),
            activecolor=ACCENT
        )
        for lbl in self.radio_view.labels:
            lbl.set_color(TXT); lbl.set_fontsize(8)
        self.radio_view.on_clicked(self._on_view)
        y -= H_rb + 0.025

        # ── SHOW / HIDE checkboxes ─────────────────────────────────────────
        self.fig.text(L, y + 0.002, "SHOW / HIDE", color=TXT2, fontsize=6.5,
                      fontweight="bold", fontfamily="monospace")
        ax_cb = self._add_widget_ax([L, y - H_cb, W, H_cb])
        chk_labels  = ("Real events", "Virtual events",
                        "Neg seeds", "Cell lines", "w=0 plane", "Axis labels")
        chk_actives = (True, True, True, True, True, True)
        self.chk_vis = CheckButtons(ax_cb, chk_labels, chk_actives)
        for lbl in self.chk_vis.labels:
            lbl.set_color(TXT); lbl.set_fontsize(8)
        self.chk_vis.on_clicked(self._on_chk)
        y -= H_cb + 0.025

        # ── CAMERA sliders ─────────────────────────────────────────────────
        self.fig.text(L, y + 0.002, "CAMERA", color=TXT2, fontsize=6.5,
                      fontweight="bold", fontfamily="monospace")
        kw = dict(color=ACCENT, track_color=BG)

        ax_az = self._add_widget_ax([L, y - H_sl, W, H_sl], "Azimuth")
        self.sl_az = Slider(ax_az, "", 0, 360, valinit=210, **kw)
        self.sl_az.valtext.set_color(ACCENT); self.sl_az.valtext.set_fontsize(7)
        self.sl_az.on_changed(lambda v: self._draw())
        y -= H_sl + 0.026

        ax_el = self._add_widget_ax([L, y - H_sl, W, H_sl], "Elevation")
        self.sl_el = Slider(ax_el, "", 5, 80, valinit=28, **kw)
        self.sl_el.valtext.set_color(ACCENT); self.sl_el.valtext.set_fontsize(7)
        self.sl_el.on_changed(lambda v: self._draw())
        y -= H_sl + 0.026

        ax_zm = self._add_widget_ax([L, y - H_sl, W, H_sl], "Zoom")
        self.sl_zm = Slider(ax_zm, "", 30, 200, valinit=100, **kw)
        self.sl_zm.valtext.set_color(ACCENT); self.sl_zm.valtext.set_fontsize(7)
        self.sl_zm.on_changed(lambda v: self._draw())
        y -= H_sl + 0.030

        # ── PT FILTER ─────────────────────────────────────────────────────
        self.fig.text(L, y + 0.002, "pT FILTER [GeV]", color=TXT2, fontsize=6.5,
                      fontweight="bold", fontfamily="monospace")
        pT_min, pT_max = float(self.pT.min()), float(self.pT.max())

        ax_pL = self._add_widget_ax([L, y - H_sl, W, H_sl], "pT min")
        self.sl_pTlo = Slider(ax_pL, "", pT_min, pT_max, valinit=pT_min, **kw)
        self.sl_pTlo.valtext.set_color(ACCENT); self.sl_pTlo.valtext.set_fontsize(7)
        self.sl_pTlo.on_changed(lambda v: (setattr(self, "pT_lo", v), self._draw()))
        y -= H_sl + 0.026

        ax_pH = self._add_widget_ax([L, y - H_sl, W, H_sl], "pT max")
        self.sl_pThi = Slider(ax_pH, "", pT_min, pT_max, valinit=pT_max, **kw)
        self.sl_pThi.valtext.set_color(ACCENT); self.sl_pThi.valtext.set_fontsize(7)
        self.sl_pThi.on_changed(lambda v: (setattr(self, "pT_hi", v), self._draw()))
        y -= H_sl + 0.030

        # ── Y FILTER ──────────────────────────────────────────────────────
        self.fig.text(L, y + 0.002, "y FILTER", color=TXT2, fontsize=6.5,
                      fontweight="bold", fontfamily="monospace")
        y_min, y_max = float(self.y.min()), float(self.y.max())

        ax_yL = self._add_widget_ax([L, y - H_sl, W, H_sl], "y min")
        self.sl_ylo = Slider(ax_yL, "", y_min, y_max, valinit=y_min, **kw)
        self.sl_ylo.valtext.set_color(ACCENT); self.sl_ylo.valtext.set_fontsize(7)
        self.sl_ylo.on_changed(lambda v: (setattr(self, "y_lo", v), self._draw()))
        y -= H_sl + 0.026

        ax_yH = self._add_widget_ax([L, y - H_sl, W, H_sl], "y max")
        self.sl_yhi = Slider(ax_yH, "", y_min, y_max, valinit=y_max, **kw)
        self.sl_yhi.valtext.set_color(ACCENT); self.sl_yhi.valtext.set_fontsize(7)
        self.sl_yhi.on_changed(lambda v: (setattr(self, "y_hi", v), self._draw()))
        y -= H_sl + 0.030

        # ── CELL STEP slider ───────────────────────────────────────────────
        self.fig.text(L, y + 0.002, "STEP THROUGH CELLS", color=TXT2, fontsize=6.5,
                      fontweight="bold", fontfamily="monospace")
        ax_cs = self._add_widget_ax([L, y - H_sl, W, H_sl], "Cell #")
        n_cells = max(1, len(self.cell_log) - 1)
        self.sl_cell = Slider(ax_cs, "", 0, n_cells, valinit=0,
                              valstep=1, **kw)
        self.sl_cell.valtext.set_color(ACCENT); self.sl_cell.valtext.set_fontsize(7)
        self.sl_cell.on_changed(self._on_cell_slide)
        y -= H_sl + 0.020

        # Prev / Auto / Next buttons
        bw = 0.055
        ax_prev = self.fig.add_axes([L,          y - H_bt, bw,      H_bt], facecolor=BG3)
        ax_anim = self.fig.add_axes([L + bw + 0.005, y - H_bt, bw + 0.02, H_bt], facecolor=BG3)
        ax_next = self.fig.add_axes([L + 2*bw + 0.030, y - H_bt, bw,   H_bt], facecolor=BG3)
        self.btn_prev = Button(ax_prev, "◀ Prev", color=BG3, hovercolor=GRID_C)
        self.btn_anim = Button(ax_anim, "▶ Auto",  color=BG3, hovercolor=GRID_C)
        self.btn_next = Button(ax_next, "Next ▶", color=BG3, hovercolor=GRID_C)
        for btn in (self.btn_prev, self.btn_anim, self.btn_next):
            btn.label.set_color(TXT); btn.label.set_fontsize(7.5)
        self.btn_prev.on_clicked(lambda e: self._step_cell(-1))
        self.btn_next.on_clicked(lambda e: self._step_cell(+1))
        self.btn_anim.on_clicked(self._toggle_anim_cells)
        y -= H_bt + 0.030

        # ── ACTION BUTTONS ────────────────────────────────────────────────
        bw2 = 0.087
        ax_spin  = self.fig.add_axes([L,          y - H_bt, bw2,      H_bt], facecolor=BG3)
        ax_reset = self.fig.add_axes([L + bw2 + 0.008, y - H_bt, bw2, H_bt], facecolor=BG3)
        ax_save  = self.fig.add_axes([L + 2*(bw2 + 0.008), y - H_bt, bw2, H_bt], facecolor=BG3)
        self.btn_spin  = Button(ax_spin,  "[spin]",      color=BG3, hovercolor=GRID_C)
        self.btn_reset = Button(ax_reset, "[reset]",      color=BG3, hovercolor=GRID_C)
        self.btn_save  = Button(ax_save,  "[save]",      color=BG3, hovercolor=GRID_C)
        for btn in (self.btn_spin, self.btn_reset, self.btn_save):
            btn.label.set_color(TXT); btn.label.set_fontsize(7.5)
        self.btn_spin.on_clicked(self._toggle_spin)
        self.btn_reset.on_clicked(self._reset_camera)
        self.btn_save.on_clicked(self._save_screenshot)

    # ── widget callbacks ──────────────────────────────────────────────────────

    def _on_view(self, label):
        mapping = {
            "Before resampling" : "before",
            "After resampling"  : "after",
            "Δ Weight delta"    : "delta",
        }
        self.view = mapping[label]
        self._draw()

    def _on_chk(self, label):
        mapping = {
            "Real events"  : "show_real",
            "Virtual events": "show_virt",
            "Neg seeds"    : "show_neg",
            "Cell lines"   : "show_cells",
            "w=0 plane"    : "show_plane",
            "Axis labels"  : "show_axlabel",
        }
        attr = mapping[label]
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
        self.ax3d.view_init(elev=28, azim=210)
        self._draw()

    def _save_screenshot(self, event):
        fname = f"atlas_3d_{int(time.time())}.png"
        self.fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"[save]  Screenshot → {fname}")

    def _on_key(self, event):
        if event.key == "left":  self._step_cell(-1)
        elif event.key == "right": self._step_cell(+1)
        elif event.key == " ":   self._toggle_spin(None)
        elif event.key == "r":   self._reset_camera()
        elif event.key == "s":   self._save_screenshot(None)
        elif event.key == "1":   self.radio_view.set_active(0)
        elif event.key == "2":   self.radio_view.set_active(1)
        elif event.key == "3":   self.radio_view.set_active(2)

    # ── animation tick ────────────────────────────────────────────────────────

    def _tick(self):
        now = time.time()
        changed = False
        if self.spinning and now - self._spin_last > 0.04:
            new_az = (self.sl_az.val + 0.8) % 360
            self.sl_az.set_val(new_az)
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
        ax = self.ax3d
        az, el, zm = self.sl_az.val, self.sl_el.val, self.sl_zm.val

        ax.cla()
        self._style_3d()
        ax.view_init(elev=el, azim=az)

        w = self._get_w()
        pT, y = self.pT, self.y
        src   = self.df["src"].to_numpy()

        # ── filter mask ───────────────────────────────────────────────────
        mask = (pT >= self.pT_lo) & (pT <= self.pT_hi) \
             & (y  >= self.y_lo)  & (y  <= self.y_hi)

        # ── w=0 reference plane ───────────────────────────────────────────
        if self.show_plane:
            pT_range = np.array([self.pT_lo, self.pT_hi])
            y_range  = np.array([self.y_lo,  self.y_hi])
            xx, yy = np.meshgrid(pT_range, y_range)
            ax.plot_surface(xx, yy, np.zeros_like(xx),
                            color=C_NEG, alpha=0.04, linewidth=0, zorder=0)
            ax.plot_wireframe(xx, yy, np.zeros_like(xx),
                              color=C_NEG, alpha=0.18, linewidth=0.5)

        # ── current cell highlight ─────────────────────────────────────────
        hcell_set = set()
        if self.cell_log:
            cell = self.cell_log[self.cell_idx]
            si   = cell["seed"]
            nbrs = cell["nbrs"]
            hcell_set = {si} | set(nbrs)

            if self.show_cells:
                sw = 0.0 if self.view == "after" else cell["w_seed"]
                for ni in nbrs:
                    nw = float(w[ni])
                    ax.plot(
                        [pT[si], pT[ni]], [y[si], y[ni]], [sw, nw],
                        color=C_CELL, lw=1.2, alpha=0.6, zorder=6
                    )

        # ── all cell lines (background, subtle) ──────────────────────────
        if self.show_cells and self.view != "before":
            for cell in self.cell_log:
                s = cell["seed"]
                sw = 0.0 if self.view == "after" else cell["w_seed"]
                for ni in cell["nbrs"][:2]:
                    ax.plot(
                        [pT[s], pT[ni]], [y[s], y[ni]], [sw, float(w[ni])],
                        color=C_CELL, lw=0.4, alpha=0.08, zorder=2
                    )

        # ── scatter layers ─────────────────────────────────────────────────
        def scatter_group(idx_mask, color, size, alpha, marker="o", zorder=3, label=""):
            m = mask & idx_mask
            if m.sum() == 0: return
            # highlight membership in current cell
            hi = m & np.isin(np.arange(len(pT)), list(hcell_set))
            lo = m & ~np.isin(np.arange(len(pT)), list(hcell_set))
            if lo.sum():
                ax.scatter(pT[lo], y[lo], w[lo],
                           c=color, s=size, alpha=alpha,
                           marker=marker, depthshade=True,
                           linewidths=0, zorder=zorder, label=label)
            if hi.sum():
                sz2 = size * (3.5 if (marker in ("v", "*")) else 2.5)
                ax.scatter(pT[hi], y[hi], w[hi],
                           c=C_CELL, s=sz2, alpha=0.95,
                           marker=marker, depthshade=False,
                           linewidths=0.6, edgecolors="white",
                           zorder=8)

        # real positive
        if self.show_real:
            if self.view == "delta":
                r_pos = (src == "real") & (w >= 0)
                r_neg = (src == "real") & (w < 0)
                scatter_group(r_pos, C_DELTA_P, 12, 0.55, label="Real Δ>0")
                scatter_group(r_neg, C_DELTA_N, 12, 0.55, label="Real Δ<0")
            else:
                scatter_group((src == "real") & (w >= 0), C_REAL, 14, 0.55,
                              label=f"Real w≥0 (n={(mask & (src=='real') & (w>=0)).sum()})")

        # virtual positive
        if self.show_virt:
            if self.view == "delta":
                v_pos = (src == "virtual") & (w >= 0)
                v_neg = (src == "virtual") & (w < 0)
                scatter_group(v_pos, C_DELTA_P, 10, 0.55, label="Virt Δ>0")
                scatter_group(v_neg, C_DELTA_N, 10, 0.55, label="Virt Δ<0")
            else:
                scatter_group((src == "virtual") & (w > 0), C_VIRT, 12, 0.55,
                              label=f"Virtual w>0 (n={(mask & (src=='virtual') & (w>0)).sum()})")

        # neg seeds
        if self.show_neg:
            neg_m = w < 0
            if neg_m.sum():
                scatter_group(neg_m, C_NEG, 28, 0.82, marker="v", zorder=7,
                              label=f"w<0 seed (n={(mask & neg_m).sum()})")

        # neutralised (w==0)
        zero_m = w == 0
        if zero_m.sum():
            scatter_group(zero_m, C_ZERO, 4, 0.30, zorder=2,
                          label=f"Neutralised (n={(mask & zero_m).sum()})")

        # ── axis labels ───────────────────────────────────────────────────
        if self.show_axlabel:
            ax.set_xlabel(r"$p_T$  [GeV]",    color=TXT, fontsize=9, labelpad=8)
            ax.set_ylabel(r"Rapidity  $y$",   color=TXT, fontsize=9, labelpad=8)
            ax.set_zlabel(r"Weight  $w$",     color=TXT, fontsize=9, labelpad=8)
        else:
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

        # ── title & info ──────────────────────────────────────────────────
        view_lbl = {"before": "BEFORE resampling",
                    "after" : "AFTER resampling",
                    "delta" : "Δ WEIGHT (after − before)"}[self.view]

        n_neg_b = int((self.w_before < 0).sum())
        n_neg_a = int((self.w_after  < 0).sum())

        cell_info = ""
        if self.cell_log:
            c  = self.cell_log[self.cell_idx]
            cell_info = (f"\nCell {self.cell_idx+1}/{len(self.cell_log)} | "
                         f"seed pT={self.pT[c['seed']]:.1f} y={self.y[c['seed']]:.2f} "
                         f"w_seed={c['w_seed']:.3f} | nbrs={c['n_nbrs']} | "
                         f"Σw={c['w_sum']:.4f}")

        ax.set_title(
            f"{view_lbl}\n"
            f"events={len(self.pT)} | neg: {n_neg_b}→{n_neg_a} | "
            f"cells={len(self.cell_log)}{cell_info}",
            color=TXT, fontsize=7.5, pad=6
        )

        # ── legend ────────────────────────────────────────────────────────
        legend = ax.legend(
            loc="upper left",
            fontsize=7,
            facecolor=BG2,
            edgecolor=GRID_C,
            labelcolor=TXT,
            markerscale=1.3,
            framealpha=0.85,
        )

        # ── zoom via axis limits ───────────────────────────────────────────
        center = zm / 100.0
        pT_c   = (self.pT_lo + self.pT_hi) / 2
        y_c    = (self.y_lo  + self.y_hi)  / 2
        pT_r   = (self.pT_hi - self.pT_lo) / 2 / center
        y_r    = (self.y_hi  - self.y_lo)  / 2 / center
        w_vis  = w[mask]
        w_r    = (w_vis.max() - w_vis.min()) / 2 / center if len(w_vis) else 5
        w_c    = (w_vis.max() + w_vis.min()) / 2           if len(w_vis) else 0

        ax.set_xlim(pT_c - pT_r, pT_c + pT_r)
        ax.set_ylim(y_c  - y_r,  y_c  + y_r)
        ax.set_zlim(w_c  - w_r,  w_c  + w_r)

        self.fig.canvas.draw_idle()

    # ── entry point ───────────────────────────────────────────────────────────

    def show(self):
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Neg-Weight 3D Visualizer"
    )
    parser.add_argument("--real",    default=None,
                        help="Path to real_events.csv  (cols: id,pt_real,y_real,z_gluon,weight)")
    parser.add_argument("--virtual", default=None,
                        help="Path to virtual_events.csv  (cols: id,pt,y,weight)")
    args = parser.parse_args()

    print("=" * 64)
    print("  ATLAS Negative-Weight Mitigation — 3D Visualizer")
    print("=" * 64)
    print()
    print("Keyboard shortcuts (focus the plot window first):")
    print("  ← / →   step through resampling cells")
    print("  SPACE    toggle auto-spin")
    print("  r        reset camera")
    print("  s        save screenshot")
    print("  1/2/3    switch view: Before / After / Delta")
    print()

    real_df,  virt_df  = load_or_generate(args.real, args.virtual)
    combined            = born_project(real_df, virt_df)
    w_after, cell_log  = greedy_resample(combined)

    viz = Visualizer(combined, w_after, cell_log, real_df)
    viz.show()


if __name__ == "__main__":
    main()