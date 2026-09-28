"""
Validation (spec Section 8).

Forward-simulation validation is the primary acceptance criterion for a
fitted model, not in-sample R^2. For each fitted (segment, mode) candidate,
integrate the discovered ODE forward from an initial condition and compare
against the actual trajectory over a configurable horizon. Models whose
divergence exceeds a configurable acceptance threshold are flagged
unreliable and excluded from downstream use/reporting, not silently kept.

v1 limitation (explicitly flagged, not silently papered over): the fitting
workflow (Section 7 / Step 8) fits on the entire segment/mode window rather
than holding out a separate validation split. This module therefore
validates against a trailing window of the *same* segment/mode frame the
model was fit on. This is a legitimate check of whether the discovered ODE
reproduces trajectory dynamics (a materially different test than in-sample
derivative-fit R^2), but it is not a fully out-of-sample test. A true
held-out split is a natural v2 enhancement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sdmodeler.config import DivergenceMetric, FeatureLibraryConfig, FeatureLibraryTier, ValidationConfig
from sdmodeler.identification.feature_library import build_feature_row
from sdmodeler.identification.fitting import FittedModel

_DIVERGENCE_RELATIVE_THRESHOLD = 0.2


@dataclass
class ValidationResult:
    applicable: bool
    reason: Optional[str] = None
    predicted_trajectory: Optional[np.ndarray] = None
    actual_trajectory: Optional[np.ndarray] = None
    timestamps: Optional[pd.DatetimeIndex] = None
    metric: Optional[DivergenceMetric] = None
    score: Optional[float] = None
    is_reliable: Optional[bool] = None


def simulate_forward(
    model: FittedModel,
    initial_x: np.ndarray,
    u_trajectory: np.ndarray,
    dt: float,
    feature_library_config: FeatureLibraryConfig,
) -> np.ndarray:
    """Integrate dx/dt = Theta(x, u) @ coefficients forward via RK4, holding
    u at each step's zero-order-hold value (consistent with u being a
    control/exogenous input that is never linearly interpolated, spec
    Section 1).
    """
    n_steps = u_trajectory.shape[0]
    n_x = len(initial_x)
    trajectory = np.zeros((n_steps, n_x))
    trajectory[0] = initial_x

    def derivative(x_state: np.ndarray, u_row: np.ndarray) -> np.ndarray:
        row = build_feature_row(x_state, u_row, model.x_variables, model.u_variables, feature_library_config)
        return row @ model.coefficients.T

    for i in range(1, n_steps):
        x_prev = trajectory[i - 1]
        u_current = u_trajectory[i]

        k1 = derivative(x_prev, u_current)
        k2 = derivative(x_prev + 0.5 * dt * k1, u_current)
        k3 = derivative(x_prev + 0.5 * dt * k2, u_current)
        k4 = derivative(x_prev + dt * k3, u_current)

        trajectory[i] = x_prev + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return trajectory


def forecast_samples(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return trajectory samples after the observed initialization point."""
    return actual[1:], predicted[1:]


def normalized_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    scale = np.std(actual, axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    error = (actual - predicted) / scale
    return float(np.sqrt(np.mean(error ** 2)))


def _divergence_time_seconds(
    actual: np.ndarray, predicted: np.ndarray, dt: float, relative_threshold: float = _DIVERGENCE_RELATIVE_THRESHOLD
) -> float:
    scale = np.std(actual, axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    relative_error = np.abs((actual - predicted) / scale)
    exceeded = np.any(relative_error > relative_threshold, axis=1)
    exceeded_indices = np.flatnonzero(exceeded)
    if len(exceeded_indices) == 0:
        return float((len(actual) - 1) * dt)
    return float(exceeded_indices[0] * dt)


def validate_model(
    model: FittedModel,
    mode_segment_frame: pd.DataFrame,
    feature_library_config: FeatureLibraryConfig,
    config: ValidationConfig,
) -> ValidationResult:
    """Run forward-simulation validation for one fitted model against a
    trailing window of its own (segment, mode) frame. Mutates model.validation_score
    and model.is_reliable in place, and returns the full ValidationResult.
    """
    if FeatureLibraryTier.TIER_3_FOURIER in model.library_tiers:
        result = ValidationResult(
            applicable=False,
            reason="Tier 3 (Fourier) models are a seasonal function of time, "
            "not a state-feedback ODE, and are not forward-simulation validated (spec Section 5/8).",
        )
        model.validation_score = None
        model.is_reliable = None
        return result

    if len(mode_segment_frame) < 2:
        model.validation_score = None
        model.is_reliable = None
        return ValidationResult(applicable=False, reason="Validation requires at least two samples.")
    intervals = (mode_segment_frame.index[1:] - mode_segment_frame.index[:-1]).total_seconds()
    if np.any(intervals <= 0) or not np.allclose(intervals, intervals[0], rtol=1e-6, atol=1e-9):
        raise ValueError("Validation requires strictly increasing, uniform sampling intervals.")
    dt = float(intervals[0])
    horizon_seconds = config.horizon_minutes * 60.0
    horizon_samples = int(round(horizon_seconds / dt)) + 1

    if len(mode_segment_frame) < horizon_samples:
        result = ValidationResult(
            applicable=False,
            reason=f"Segment/mode has {len(mode_segment_frame)} samples, fewer than the "
            f"{horizon_samples} needed for a {config.horizon_minutes}-minute validation horizon.",
        )
        model.validation_score = None
        model.is_reliable = None
        return result

    start_idx = len(mode_segment_frame) - horizon_samples
    window = mode_segment_frame.iloc[start_idx:]

    actual_x = window[model.x_variables].to_numpy(dtype=float)
    u_trajectory = window[model.u_variables].to_numpy(dtype=float)
    timestamps = window.index

    predicted_x = simulate_forward(model, actual_x[0], u_trajectory, dt, feature_library_config)

    if config.divergence_metric == DivergenceMetric.NORMALIZED_RMSE:
        forecast_actual, forecast_predicted = forecast_samples(actual_x, predicted_x)
        score = normalized_rmse(forecast_actual, forecast_predicted)
        is_reliable = score <= config.acceptance_threshold
    elif config.divergence_metric == DivergenceMetric.TRAJECTORY_DIVERGENCE_TIME:
        score = _divergence_time_seconds(actual_x, predicted_x, dt)
        is_reliable = score >= config.acceptance_threshold
    else:
        raise ValueError(f"Unknown divergence metric: {config.divergence_metric}")

    model.validation_score = score
    model.is_reliable = is_reliable
    model.validation_metric = config.divergence_metric

    return ValidationResult(
        applicable=True,
        predicted_trajectory=predicted_x,
        actual_trajectory=actual_x,
        timestamps=timestamps,
        metric=config.divergence_metric,
        score=score,
        is_reliable=is_reliable,
    )
