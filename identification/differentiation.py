"""
Differentiation configuration (spec Section 6, differentiation half).

Wraps PySINDy's differentiation methods behind a config-driven interface,
and reports back the exact method + parameters used, for reproducibility
alongside every fitted model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pysindy.differentiation import FiniteDifference, SINDyDerivative

from sdmodeler.config import DifferentiationConfig, DifferentiationMethod


@dataclass
class DifferentiationResult:
    dx: np.ndarray
    method: DifferentiationMethod
    params: dict[str, Any]


def _median_dt(t: np.ndarray) -> float:
    return float(np.median(np.diff(t)))


def build_differentiator(config: DifferentiationConfig, t: np.ndarray):
    """Returns (differentiator_callable, params_used) for the configured method."""
    if config.method == DifferentiationMethod.FINITE_DIFFERENCE:
        return FiniteDifference(), {}

    if config.method == DifferentiationMethod.SAVITZKY_GOLAY:
        dt = _median_dt(t)
        half_window_seconds = ((config.savgol_window_length - 1) // 2) * dt
        params = {
            "kind": "savitzky_golay",
            "left": half_window_seconds,
            "right": half_window_seconds,
            "order": config.savgol_polyorder,
        }
        return SINDyDerivative(**params), params

    if config.method == DifferentiationMethod.SPLINE:
        params = {"kind": "spline", "s": 1.0}
        return SINDyDerivative(**params), params

    if config.method == DifferentiationMethod.TOTAL_VARIATION:
        params = {"kind": "trend_filtered", "order": 0, "alpha": 1.0}
        return SINDyDerivative(**params), params

    raise ValueError(f"Unknown differentiation method: {config.method}")


def differentiate(x: np.ndarray, t: np.ndarray, config: DifferentiationConfig) -> DifferentiationResult:
    """Differentiate each column of x (n_samples, n_states) w.r.t. t.

    Returns the derivative array alongside the exact method/params used,
    for reproducibility (spec Section 6/7 requirement).
    """
    differentiator, params = build_differentiator(config, t)

    if x.ndim == 1:
        dx = differentiator(x, t)
    else:
        dx = np.column_stack([differentiator(x[:, i], t) for i in range(x.shape[1])])

    return DifferentiationResult(dx=dx, method=config.method, params=params)
