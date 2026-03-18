"""
src/utils/data_loader.py
------------------------
Data ingestion for ATLAS negative-weight mitigation pipeline.

Primary source — CSV files (real data):
    data/real_events.csv    : columns  id, pt_real, y_real, z_gluon, weight
    data/virtual_events.csv : columns  id, pt, y, weight

Fallback — synthetic NLO-like data generated on the fly when CSVs are
not found (useful for quick tests / CI).  Distributions mimic realistic
ATLAS Z-boson production kinematics at sqrt(s) = 13 TeV.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# Default CSV paths — resolved relative to wherever the script is run from
_DEFAULT_REAL_CSV    = Path("data/real_events.csv")
_DEFAULT_VIRTUAL_CSV = Path("data/virtual_events.csv")

# Required column names in each CSV
_REAL_COLS    = {"pt_real", "y_real", "z_gluon", "weight"}
_VIRTUAL_COLS = {"pt", "y", "weight"}


# ─────────────────────────────────────────────────────────────────────────────
# CSV loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path, required_cols, label):
    """Load a CSV, strip column-name whitespace, validate required columns."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{label}: CSV at '{path}' is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    log.info("Loaded %s: %d events from '%s'", label, len(df), path)
    return df


def load_real_events(path=None):
    """
    Load real-emission events from data/real_events.csv (or a custom path).

    Expected columns: id, pt_real, y_real, z_gluon, weight
    """
    p  = Path(path) if path else _DEFAULT_REAL_CSV
    df = _load_csv(p, _REAL_COLS, "real_events")
    return {
        "pt_real": df["pt_real"].to_numpy(np.float64),
        "y_real":  df["y_real"].to_numpy(np.float64),
        "z_gluon": df["z_gluon"].to_numpy(np.float64),
        "weight":  df["weight"].to_numpy(np.float64),
    }


def load_virtual_events(path=None):
    """
    Load virtual-correction events from data/virtual_events.csv (or a custom path).

    Expected columns: id, pt, y, weight
    """
    p  = Path(path) if path else _DEFAULT_VIRTUAL_CSV
    df = _load_csv(p, _VIRTUAL_COLS, "virtual_events")
    return {
        "pt":     df["pt"].to_numpy(np.float64),
        "y":      df["y"].to_numpy(np.float64),
        "weight": df["weight"].to_numpy(np.float64),
    }


def load_events(real_path=None, virtual_path=None, fallback_synthetic=True, **synth_kwargs):
    """
    Master loader: tries data/real_events.csv and data/virtual_events.csv first.
    Falls back to synthetic generation when files are missing.

    Parameters
    ----------
    real_path           : override path to real_events.csv
    virtual_path        : override path to virtual_events.csv
    fallback_synthetic  : if True, generate synthetic data when CSVs missing
    **synth_kwargs      : forwarded to generate_synthetic_events()
                          (n_real, n_virtual, neg_fraction, seed)

    Returns
    -------
    (real_events, virtual_events) dicts compatible with projector.combine_datasets
    """
    real_p    = Path(real_path)    if real_path    else _DEFAULT_REAL_CSV
    virtual_p = Path(virtual_path) if virtual_path else _DEFAULT_VIRTUAL_CSV

    real_ok    = real_p.exists()
    virtual_ok = virtual_p.exists()

    if real_ok and virtual_ok:
        log.info("Both CSV files found — loading real data.")
        return load_real_events(real_p), load_virtual_events(virtual_p)

    if not fallback_synthetic:
        missing = [str(p) for p, ok in [(real_p, real_ok), (virtual_p, virtual_ok)] if not ok]
        raise FileNotFoundError(
            f"CSV file(s) not found: {missing}. "
            "Set fallback_synthetic=True to use synthetic data instead."
        )

    if real_ok and not virtual_ok:
        log.warning("'%s' not found — using synthetic virtual events.", virtual_p)
        _, virtual_events = generate_synthetic_events(**synth_kwargs)
        return load_real_events(real_p), virtual_events

    if virtual_ok and not real_ok:
        log.warning("'%s' not found — using synthetic real events.", real_p)
        real_events, _ = generate_synthetic_events(**synth_kwargs)
        return real_events, load_virtual_events(virtual_p)

    log.warning(
        "CSV files not found at '%s' and '%s' — generating fully synthetic data.",
        real_p, virtual_p,
    )
    return generate_synthetic_events(**synth_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback generator
# ─────────────────────────────────────────────────────────────────────────────

def _sudakov_pT(rng, n, pT_peak=8.0):
    scale = pT_peak / 2.0
    raw   = rng.gamma(shape=2.0, scale=scale, size=n)
    hard  = rng.exponential(scale=40.0, size=n)
    blend = rng.random(n) < 0.08
    return (np.where(blend, hard, raw) + 1.0).astype(np.float64)


def _rapidity(rng, n):
    return np.clip(rng.normal(0.0, 1.3, size=n), -2.5, 2.5).astype(np.float64)


def _nlo_weights(rng, n, neg_fraction, scale=1.0):
    pos_w    = rng.gamma(shape=1.5, scale=scale, size=n)
    neg_w    = -rng.gamma(shape=1.2, scale=scale * 0.7, size=n)
    neg_mask = rng.random(n) < neg_fraction
    return np.where(neg_mask, neg_w, pos_w).astype(np.float64)


def generate_synthetic_events(n_real=2000, n_virtual=1000, neg_fraction=0.20, seed=42):
    """
    Generate fully synthetic NLO-like events (fallback when no CSV data).

    Returns
    -------
    real_events    : dict with keys pt_real, y_real, z_gluon, weight
    virtual_events : dict with keys pt, y, weight
    """
    rng = np.random.default_rng(seed)

    pt_real = _sudakov_pT(rng, n_real, pT_peak=10.0)
    y_real  = _rapidity(rng, n_real)
    z_gluon = np.clip(rng.exponential(scale=3.5, size=n_real), 0.05, 25.0)
    w_real  = _nlo_weights(rng, n_real, neg_fraction=0.05, scale=2.5)

    real_events = {
        "pt_real": pt_real,
        "y_real":  y_real,
        "z_gluon": z_gluon,
        "weight":  w_real,
    }

    pt_virt = _sudakov_pT(rng, n_virtual, pT_peak=12.0)
    y_virt  = _rapidity(rng, n_virtual)
    w_virt  = _nlo_weights(rng, n_virtual, neg_fraction=neg_fraction, scale=1.8)

    virtual_events = {
        "pt":     pt_virt,
        "y":      y_virt,
        "weight": w_virt,
    }

    return real_events, virtual_events