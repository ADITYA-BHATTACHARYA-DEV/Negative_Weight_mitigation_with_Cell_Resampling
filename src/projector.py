"""
projector.py
------------
Task 1 – Born Projection

Maps Real-emission events from the (N+1)-parton phase space into the
N-parton Born phase space by integrating out the gluon radiation variable.

Projection formulae (from the paper):
    pT_projected = pT_real + z_gluon
    y_projected  = y_real          (rapidity is unchanged)

After projection both Real and Virtual events live in the same
(pT, y) coordinate system and can be merged into a single dataset.
"""

import numpy as np
import logging

log = logging.getLogger(__name__)

# Output dtype for the combined (projected) dataset
COMBINED_DTYPE = np.dtype([
    ("pT", np.float64),
    ("y",  np.float64),
    ("w",  np.float64),
    ("source", "U7"),       # "real" or "virtual" – useful for debugging
])


def project_real_events(real_events: np.ndarray) -> np.ndarray:
    """
    Apply the Born projection to Real-emission events.

    Parameters
    ----------
    real_events : structured array with fields (pT, y, w, z_gluon)

    Returns
    -------
    projected : structured array with fields (pT, y, w, source="real")
    """
    n = len(real_events)
    projected = np.empty(n, dtype=COMBINED_DTYPE)

    projected["pT"]     = real_events["pT"] + real_events["z_gluon"]   # ← key step
    projected["y"]      = real_events["y"]                              # unchanged
    projected["w"]      = real_events["w"]
    projected["source"] = "real"

    log.info("Projected %d real events into Born phase space.", n)
    log.info(
        "  pT range after projection: [%.2f, %.2f] GeV",
        projected["pT"].min(), projected["pT"].max(),
    )
    return projected


def keep_virtual_events(virtual_events: np.ndarray) -> np.ndarray:
    """
    Wrap Virtual events in the same combined dtype (no coordinate change).

    Parameters
    ----------
    virtual_events : structured array with fields (pT, y, w, z_gluon)

    Returns
    -------
    wrapped : structured array with fields (pT, y, w, source="virtual")
    """
    n = len(virtual_events)
    wrapped = np.empty(n, dtype=COMBINED_DTYPE)

    wrapped["pT"]     = virtual_events["pT"]
    wrapped["y"]      = virtual_events["y"]
    wrapped["w"]      = virtual_events["w"]
    wrapped["source"] = "virtual"

    log.info("Kept %d virtual events unchanged.", n)
    return wrapped


def combine_datasets(
    real_events: np.ndarray,
    virtual_events: np.ndarray,
) -> np.ndarray:
    """
    Full pipeline: project Real, wrap Virtual, concatenate.

    Parameters
    ----------
    real_events    : structured array (pT, y, w, z_gluon)
    virtual_events : structured array (pT, y, w, z_gluon)

    Returns
    -------
    combined : structured array (pT, y, w, source)
               Length = len(real_events) + len(virtual_events)
    """
    projected = project_real_events(real_events)
    virtual   = keep_virtual_events(virtual_events)
    combined  = np.concatenate([projected, virtual])

    n_neg = int((combined["w"] < 0).sum())
    log.info(
        "Combined dataset: %d events total, %d negative-weight (%.1f%%).",
        len(combined), n_neg, 100.0 * n_neg / len(combined),
    )
    return combined