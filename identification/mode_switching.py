"""
Hybrid / switching dynamics (spec Section 4).

Mode detection is independent of, and prior to, SINDy fitting: each segment
is further partitioned into discrete "modes" based on user-supplied
threshold conditions on state/control variables (e.g. CPU above/below the
known autoscaling trigger threshold). This is config-driven, not learned,
in v1.

This is the mechanism that replaces trying to get `max(0, CPU-70)`-style
hinge terms in a single global feature library to reconstruct switching
behavior implicitly: instead, a separate SINDy model is fit per mode per
segment, with mode boundaries stored alongside each fitted model so
downstream simulation can switch models when a trajectory crosses a
boundary.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import pandas as pd

from sdmodeler.config import ModeSwitchingConfig, ModeThreshold, ThresholdOperator

_OPERATORS = {
    ThresholdOperator.GREATER_THAN: operator.gt,
    ThresholdOperator.GREATER_EQUAL: operator.ge,
    ThresholdOperator.LESS_THAN: operator.lt,
    ThresholdOperator.LESS_EQUAL: operator.le,
}


@dataclass
class ModeSegment:
    mode_label: str
    start: pd.Timestamp
    end: pd.Timestamp
    sample_count: int
    frame: pd.DataFrame


def evaluate_thresholds(frame: pd.DataFrame, thresholds: list[ModeThreshold]) -> pd.DataFrame:
    """Return a DataFrame with one boolean column per threshold, named
    '<variable>_<operator>_<value>', indicating whether that threshold
    condition holds at each timestamp.
    """
    columns = {}
    for t in thresholds:
        if t.variable not in frame.columns:
            raise ValueError(
                f"Mode threshold references variable '{t.variable}' not present "
                f"in frame columns: {list(frame.columns)}"
            )
        op_fn = _OPERATORS[t.operator]
        col_name = f"{t.variable}_{t.operator.value}_{t.value}"
        columns[col_name] = op_fn(frame[t.variable], t.value)
    return pd.DataFrame(columns, index=frame.index)


def compute_mode_labels(frame: pd.DataFrame, thresholds: list[ModeThreshold]) -> pd.Series:
    """Combine per-threshold booleans into a single mode label string per row."""
    if not thresholds:
        return pd.Series(["default"] * len(frame), index=frame.index)
    bool_df = evaluate_thresholds(frame, thresholds)
    labels = bool_df.apply(
        lambda row: "|".join(f"{col}={val}" for col, val in row.items()), axis=1
    )
    return labels


def split_into_modes(frame: pd.DataFrame, config: ModeSwitchingConfig) -> list[ModeSegment]:
    """Partition a frame into contiguous runs of identical mode label,
    per the configured thresholds. If mode switching is disabled, returns
    a single ModeSegment covering the whole frame with mode_label='default'.
    """
    if not config.enabled or not config.thresholds:
        return [
            ModeSegment(
                mode_label="default",
                start=frame.index[0],
                end=frame.index[-1],
                sample_count=len(frame),
                frame=frame,
            )
        ] if len(frame) > 0 else []

    labels = compute_mode_labels(frame, config.thresholds)

    mode_segments: list[ModeSegment] = []
    start_idx = 0
    values = labels.to_numpy()
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start_idx]:
            sub_frame = frame.iloc[start_idx:i]
            mode_segments.append(
                ModeSegment(
                    mode_label=values[start_idx],
                    start=sub_frame.index[0],
                    end=sub_frame.index[-1],
                    sample_count=len(sub_frame),
                    frame=sub_frame,
                )
            )
            start_idx = i

    return mode_segments
