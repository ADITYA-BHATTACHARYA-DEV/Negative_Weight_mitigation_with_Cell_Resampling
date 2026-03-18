"""
data_loader.py
--------------
Reader for synthetic HepMC-style events stored as NumPy / Parquet.
Produces structured NumPy arrays with fields: pT, y, w, z_gluon (Real only).
"""

import numpy as np
from pathlib import Path


def generate_synthetic_events(
    n_real: int = 2000,
    n_virtual: int = 1000,
    neg_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic Real and Virtual MC events for testing.

    Real events carry an extra radiation variable z_gluon.
    Virtual events carry negative weights to simulate loop-level cancellations.

    Parameters
    ----------
    n_real       : number of real-emission events
    n_virtual    : number of virtual / counter-term events
    neg_fraction : fraction of virtual events with w < 0
    seed         : RNG seed for reproducibility

    Returns
    -------
    real_events    : structured array  (pT, y, w, z_gluon)
    virtual_events : structured array  (pT, y, w)          [z_gluon = 0]
    """
    rng = np.random.default_rng(seed)

    # ── Real events ──────────────────────────────────────────────────────────
    pT_real     = rng.exponential(scale=40.0, size=n_real) + 10.0   # GeV, > 10
    y_real      = rng.uniform(-3.0, 3.0, size=n_real)
    w_real      = rng.uniform(0.5, 2.0, size=n_real)                # all positive
    z_gluon     = rng.uniform(0.0, 5.0, size=n_real)                # radiation var

    real_dtype = np.dtype([
        ("pT",      np.float64),
        ("y",       np.float64),
        ("w",       np.float64),
        ("z_gluon", np.float64),
    ])
    real_events = np.zeros(n_real, dtype=real_dtype)
    real_events["pT"]      = pT_real
    real_events["y"]       = y_real
    real_events["w"]       = w_real
    real_events["z_gluon"] = z_gluon

    # ── Virtual events ────────────────────────────────────────────────────────
    pT_virt = rng.exponential(scale=40.0, size=n_virtual) + 10.0
    y_virt  = rng.uniform(-3.0, 3.0, size=n_virtual)
    w_virt  = rng.uniform(0.5, 2.0, size=n_virtual)

    # Flip a fraction to negative weights
    n_neg = int(neg_fraction * n_virtual)
    neg_idx = rng.choice(n_virtual, size=n_neg, replace=False)
    w_virt[neg_idx] *= -1.0

    virt_dtype = np.dtype([
        ("pT",      np.float64),
        ("y",       np.float64),
        ("w",       np.float64),
        ("z_gluon", np.float64),   # kept for schema consistency, set to 0
    ])
    virtual_events = np.zeros(n_virtual, dtype=virt_dtype)
    virtual_events["pT"]      = pT_virt
    virtual_events["y"]       = y_virt
    virtual_events["w"]       = w_virt
    virtual_events["z_gluon"] = 0.0

    return real_events, virtual_events


def load_events_numpy(path: str) -> np.ndarray:
    """Load a structured NumPy array saved with np.save."""
    return np.load(path, allow_pickle=False)


def save_events_numpy(events: np.ndarray, path: str) -> None:
    """Persist a structured array to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, events)