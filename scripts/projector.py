"""
src/projector.py
----------------
Born projection: maps (n+1)-body real-emission kinematics back into
the n-body Born phase space so that real and virtual events share the
same (pT, y) coordinate system.

Projection formulae (from arXiv:2109.07851 / task specification):
    pT_proj = pT_real + z_gluon
    y_proj  = y_real

The combined structured NumPy array has dtype:
    [("pT", f8), ("y", f8), ("w", f8)]
"""

import numpy as np
from numpy.typing import NDArray


_DTYPE = np.dtype([("pT", np.float64), ("y", np.float64), ("w", np.float64)])


def born_project_real(real_events: dict) -> NDArray:
    """Apply Born projection to real-emission events."""
    pT = real_events["pt_real"] + real_events["z_gluon"]
    y  = real_events["y_real"]
    w  = real_events["weight"]
    out = np.empty(len(pT), dtype=_DTYPE)
    out["pT"] = pT
    out["y"]  = y
    out["w"]  = w
    return out


def project_virtual(virtual_events: dict) -> NDArray:
    """Wrap virtual events into the shared structured array format."""
    pT = virtual_events["pt"]
    y  = virtual_events["y"]
    w  = virtual_events["weight"]
    out = np.empty(len(pT), dtype=_DTYPE)
    out["pT"] = pT
    out["y"]  = y
    out["w"]  = w
    return out


def combine_datasets(real_events: dict, virtual_events: dict) -> NDArray:
    """
    Project real events and concatenate with virtual events.

    Returns
    -------
    combined : structured ndarray with fields (pT, y, w)
               dtype float64 for all fields.
    """
    real_proj = born_project_real(real_events)
    virt_proj = project_virtual(virtual_events)
    combined  = np.concatenate([real_proj, virt_proj])
    return combined