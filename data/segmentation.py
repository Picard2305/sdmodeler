"""
Segmentation strategy (spec Section 2).

Splits the canonical time series into fit-able windows along two axes:
  1. Deploy/version boundaries — hard splits by default (opt-in override to
     cross a boundary).
  2. Traffic/operating regime — v1 uses a documented time-of-day rule; a
     changepoint-detection hook is left for a later enhancement.

This is also where the `regime_label` column (reserved but left unassigned
by ingestion) is finally populated — regime/mode labeling is this step's
responsibility, not ingestion's.

Each resulting segment carries metadata (deploy_id, regime_label, start/end,
sample_count, data_quality_flag) and segments that are too short or too flat
are flagged and skipped rather than silently fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from sdmodeler.config import RegimeDetectionMethod, SegmentationConfig
from sdmodeler.data.ingestion import REGIME_LABEL_COLUMN

DEPLOY_ID_COLUMN = "deploy_id"


@dataclass(frozen=True)
class DeployBoundary:
    deploy_id: str
    start: pd.Timestamp


@dataclass
class SegmentMetadata:
    deploy_id: Optional[str]
    regime_label: Optional[str]
    start: pd.Timestamp
    end: pd.Timestamp
    sample_count: int
    data_quality_flag: Optional[str] = None
    dynamic_range_by_variable: dict = field(default_factory=dict)


@dataclass
class Segment:
    metadata: SegmentMetadata
    frame: pd.DataFrame


def assign_deploy_ids(df: pd.DataFrame, deploy_boundaries: list[DeployBoundary]) -> pd.DataFrame:
    """Assign a `deploy_id` column based on a sorted list of deploy boundary
    timestamps. Every row belongs to the deploy_id of the most recent
    boundary at or before its timestamp; rows before the first boundary get
    deploy_id=None.
    """
    df = df.copy()
    if not deploy_boundaries:
        df[DEPLOY_ID_COLUMN] = None
        return df

    sorted_boundaries = sorted(deploy_boundaries, key=lambda b: b.start)
    boundary_times = [b.start for b in sorted_boundaries]
    boundary_ids = [b.deploy_id for b in sorted_boundaries]

    idx = np.searchsorted(boundary_times, df.index, side="right") - 1
    deploy_ids = [boundary_ids[i] if i >= 0 else None for i in idx]
    df[DEPLOY_ID_COLUMN] = deploy_ids
    return df


def assign_regime_labels(df: pd.DataFrame, config: SegmentationConfig) -> pd.DataFrame:
    """Populate the regime_label column. v1 supports a time-of-day rule;
    changepoint detection is a documented hook for a future version, not
    implemented here.
    """
    df = df.copy()
    if config.regime_detection_method == RegimeDetectionMethod.TIME_OF_DAY_RULE:
        if config.business_hours_start_utc is None or config.business_hours_end_utc is None:
            raise ValueError(
                "RegimeDetectionMethod.TIME_OF_DAY_RULE requires "
                "business_hours_start_utc and business_hours_end_utc to be set."
            )
        hours = df.index.hour
        is_business_hours = (hours >= config.business_hours_start_utc) & (
            hours < config.business_hours_end_utc
        )
        df[REGIME_LABEL_COLUMN] = np.where(is_business_hours, "business_hours", "off_hours")
    elif config.regime_detection_method == RegimeDetectionMethod.CHANGEPOINT_DETECTION:
        raise NotImplementedError(
            "Changepoint-detection regime labeling is a documented hook for a "
            "future version (spec Section 2) and is not implemented in v1."
        )
    else:
        raise ValueError(f"Unknown regime_detection_method: {config.regime_detection_method}")
    return df


def _contiguous_runs(labels: pd.Series) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx_exclusive) for contiguous runs of
    identical (deploy_id, regime_label) combined label values, treating NaN
    consistently as its own group.
    """
    runs = []
    if len(labels) == 0:
        return runs
    values = labels.to_numpy()
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or not _labels_equal(values[i], values[start]):
            runs.append((start, i))
            start = i
    return runs


def _labels_equal(a, b) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return all(_labels_equal(x, y) for x, y in zip(a, b))
    a_is_nan = a is None or (isinstance(a, float) and np.isnan(a))
    b_is_nan = b is None or (isinstance(b, float) and np.isnan(b))
    if a_is_nan or b_is_nan:
        return a_is_nan and b_is_nan
    return a == b


def _compute_dynamic_range(frame: pd.DataFrame, state_variables: list[str]) -> dict:
    ranges = {}
    for var in state_variables:
        if var not in frame.columns:
            continue
        series = frame[var].dropna()
        if len(series) == 0:
            ranges[var] = 0.0
            continue
        mean_abs = abs(series.mean())
        spread = series.max() - series.min()
        ranges[var] = float(spread / mean_abs) if mean_abs > 1e-9 else float(spread)
    return ranges


def _quality_flag(
    sample_count: int,
    dynamic_range: dict,
    config: SegmentationConfig,
) -> Optional[str]:
    if sample_count < config.min_segment_length_samples:
        return "insufficient_length"
    for var, threshold in config.min_dynamic_range_threshold.items():
        if var in dynamic_range and dynamic_range[var] < threshold:
            return "insufficient_dynamic_range"
    return None


def segment(
    df: pd.DataFrame,
    config: SegmentationConfig,
    state_variables: list[str],
    deploy_boundaries: Optional[list[DeployBoundary]] = None,
) -> list[Segment]:
    """Full segmentation pipeline: assign deploy_id, assign regime_label,
    split into contiguous (deploy_id, regime_label) runs, compute metadata
    and quality flags. Segments below the minimum length/dynamic-range
    thresholds are still returned (with their flag set) so callers can
    report on skipped segments rather than have them silently vanish.
    """
    df = assign_deploy_ids(df, deploy_boundaries or [])
    df = assign_regime_labels(df, config)

    combined_labels = list(zip(df[DEPLOY_ID_COLUMN], df[REGIME_LABEL_COLUMN]))
    combined_series = pd.Series(combined_labels, index=df.index)

    segments: list[Segment] = []
    for start_idx, end_idx in _contiguous_runs(combined_series):
        sub_frame = df.iloc[start_idx:end_idx]
        deploy_id, regime_label = combined_labels[start_idx]
        dynamic_range = _compute_dynamic_range(sub_frame, state_variables)
        quality_flag = _quality_flag(len(sub_frame), dynamic_range, config)
        metadata = SegmentMetadata(
            deploy_id=deploy_id,
            regime_label=regime_label,
            start=sub_frame.index[0],
            end=sub_frame.index[-1],
            sample_count=len(sub_frame),
            data_quality_flag=quality_flag,
            dynamic_range_by_variable=dynamic_range,
        )
        segments.append(Segment(metadata=metadata, frame=sub_frame))

    return segments


def fittable_segments(segments: list[Segment]) -> list[Segment]:
    """Filter to segments with no data-quality flag (i.e. safe to fit)."""
    return [s for s in segments if s.metadata.data_quality_flag is None]
