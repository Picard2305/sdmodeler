"""
Separating plant from controller — SINDYc structure (spec Section 3).

Given a canonical/segmented frame and a `PlantControllerSplitConfig`, this
module builds the `x` (state to be identified) and `u` (control input passed
to SINDYc) matrices. The split is entirely config-driven — no variable name
is hardcoded here beyond the config's own defaults.

This module also implements the "hold Instances constant" filter for
controlled load-test data where autoscaling was deliberately disabled — the
cleanest identification setting per the spec — and documents (rather than
silently papers over) the known feedback-loop limitation of externalizing
control inputs that are actually endogenous (e.g. RequestRate via
client-side retry/backoff on Latency).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sdmodeler.config import PlantControllerSplitConfig


@dataclass
class PlantControllerData:
    x: np.ndarray
    u: np.ndarray
    x_variables: list[str]
    u_variables: list[str]
    timestamps: pd.DatetimeIndex


def _require_columns(frame: pd.DataFrame, columns: list[str], role: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Configured {role} variables missing from frame columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def build_plant_controller_data(
    frame: pd.DataFrame, config: PlantControllerSplitConfig
) -> PlantControllerData:
    """Extract (x, u) matrices from a frame per the configured split.

    Raises ValueError with an explicit list of missing columns if the
    configured x/u variable names aren't present in the frame — this is
    treated as a config error, not silently skipped.
    """
    _require_columns(frame, config.x_variables, role="x (state)")
    _require_columns(frame, config.u_variables, role="u (control)")

    for col in config.x_variables + config.u_variables:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(
                f"Variable '{col}' is non-numeric (dtype={frame[col].dtype}); "
                "encode categorical control variables (e.g. feature_flag_state, "
                "deploy_id) to numeric form before building plant/controller data."
            )

    x = frame[config.x_variables].to_numpy(dtype=float)
    u = frame[config.u_variables].to_numpy(dtype=float)

    return PlantControllerData(
        x=x,
        u=u,
        x_variables=list(config.x_variables),
        u_variables=list(config.u_variables),
        timestamps=frame.index,
    )


def autoscaling_frozen_mask(frame: pd.DataFrame, instances_column: str = "Instances") -> pd.Series:
    """Boolean mask marking rows where Instances has not changed from the
    previous row — a proxy for "autoscaling was not actively resizing" in
    the absence of explicit metadata. Prefer explicit metadata (spec
    Section 2's autoscaling_frozen_window_metadata_key) when available;
    this is a fallback heuristic for when only raw Instances data exists.
    """
    if instances_column not in frame.columns:
        raise ValueError(f"Column '{instances_column}' not found in frame.")
    diffs = frame[instances_column].diff()
    mask = diffs.fillna(0) == 0
    return mask


def apply_hold_instances_constant(
    frame: pd.DataFrame, config: PlantControllerSplitConfig, instances_column: str = "Instances"
) -> pd.DataFrame:
    """Require a constant Instances value throughout the fitting window
    when configured; transitions must be split into separate windows upstream.
    """
    if not config.hold_instances_constant:
        return frame
    if instances_column not in frame.columns:
        raise ValueError(f"Column '{instances_column}' not found in frame.")
    if frame[instances_column].nunique(dropna=False) > 1:
        raise ValueError(f"Instances column '{instances_column}' changes within the fitting window.")
    return frame


def prepare_frame_for_identification(
    frame: pd.DataFrame, config: PlantControllerSplitConfig, instances_column: str = "Instances"
) -> pd.DataFrame:
    """Verify the constant-Instances window, if configured, before fitting.
    Frozen-window selection is enforced upstream at segment selection time.
    """
    return apply_hold_instances_constant(frame, config, instances_column=instances_column)
