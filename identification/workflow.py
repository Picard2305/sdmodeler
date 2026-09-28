from __future__ import annotations

from typing import Optional

import pandas as pd

from sdmodeler.config import PipelineConfig
from sdmodeler.data.segmentation import DeployBoundary, Segment, fittable_segments, segment
from sdmodeler.identification.fitting import CandidateResult, FitResult, fit_mode_segment
from sdmodeler.identification.mode_switching import split_into_modes
from sdmodeler.validation import validate_model


def _prepare_fitting_frames(df: pd.DataFrame, config: PipelineConfig) -> list[pd.DataFrame]:
    frames = [df]
    if config.plant_controller_split.filter_to_autoscaling_frozen_windows_only:
        key = config.segmentation.autoscaling_frozen_window_metadata_key
        if not key or key not in df.columns:
            raise ValueError(f"Frozen-window selection requires a boolean metadata column: {key!r}")
        if not pd.api.types.is_bool_dtype(df[key]):
            raise ValueError(f"Frozen-window metadata column '{key}' must be boolean")
        frames = [
            group
            for _, group in df.groupby(df[key].ne(df[key].shift()).cumsum(), sort=False)
            if group[key].iloc[0]
        ]
    if config.plant_controller_split.hold_instances_constant:
        if "Instances" not in df.columns:
            raise ValueError("Instances column is required when hold_instances_constant is enabled")
        frames = [
            group
            for frame in frames
            for _, group in frame.groupby(frame["Instances"].ne(frame["Instances"].shift()).cumsum(), sort=False)
        ]
    return frames


def run_fitting_workflow(
    df: pd.DataFrame,
    config: PipelineConfig,
    deploy_boundaries: Optional[list[DeployBoundary]] = None,
) -> list[FitResult]:
    """Coordinate segmentation, fitting, and validation for a telemetry frame."""
    frames = _prepare_fitting_frames(df, config)
    segments: list[Segment] = fittable_segments([
        seg
        for frame in frames
        for seg in segment(
            frame,
            config.segmentation,
            state_variables=config.plant_controller_split.x_variables,
            deploy_boundaries=deploy_boundaries,
        )
    ])

    results: list[FitResult] = []
    for seg in segments:
        mode_segments = split_into_modes(seg.frame, config.mode_switching)
        for mode_seg in mode_segments:
            candidate_results = [
                CandidateResult(
                    model=candidate,
                    validation=validate_model(
                        candidate, mode_seg.frame, config.feature_library, config.validation
                    ),
                )
                for candidate in fit_mode_segment(mode_seg, seg.metadata, config)
            ]
            results.append(
                FitResult(
                    segment_metadata=seg.metadata,
                    mode_segment=mode_seg,
                    candidate_results=candidate_results,
                )
            )

    return results
