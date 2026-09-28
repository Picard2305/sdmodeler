"""
Fitting workflow (spec Section 7).

Wires the full pipeline order explicitly:

    ingest (external) -> segment (deploy + regime) -> split mode ->
    build feature library per tier -> check conditioning -> differentiate ->
    fit per (segment, mode) -> store model + metadata

Every fitted model artifact stores: discovered coefficients, which library
tier was used, the differentiation method/params, the optimizer/threshold
used, the segment and mode it was fit on, sample count, and fit diagnostics
(R^2 here; the simulation-based validation score from Section 8 is attached
later by the validation step and defaults to None here).

Because spec Section 6 requires the sparsity threshold to be swept over a
grid rather than fixed, this module does not pick a single "winning"
threshold — it returns one FittedModel candidate per threshold in the grid,
per (segment, mode). Downstream validation (Section 8) is the acceptance
mechanism that selects/keeps candidates, not this step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from sdmodeler.config import DivergenceMetric, FeatureLibraryConfig, FeatureLibraryTier, PipelineConfig
from sdmodeler.data.segmentation import DeployBoundary, SegmentMetadata
from sdmodeler.identification.differentiation import differentiate
from sdmodeler.identification.feature_library import build_feature_library
from sdmodeler.identification.mode_switching import ModeSegment
from sdmodeler.identification.optimizer import SparsitySweepPoint, sweep_sparsity_thresholds
from sdmodeler.identification.plant_controller import build_plant_controller_data, prepare_frame_for_identification

if TYPE_CHECKING:
    from sdmodeler.validation import ValidationResult


@dataclass
class FittedModel:
    coefficients: np.ndarray
    feature_names: list[str]
    x_variables: list[str]
    u_variables: list[str]

    library_tiers: list[FeatureLibraryTier]
    condition_number: float
    condition_number_warning: Optional[str]

    differentiation_method: str
    differentiation_params: dict[str, Any]

    optimizer_method: str
    sparsity_threshold: float
    n_nonzero_terms: int

    segment_metadata: SegmentMetadata
    mode_label: str
    sample_count: int

    r_squared: float
    validation_score: Optional[float] = None
    is_reliable: Optional[bool] = None
    feature_library_config: Optional[FeatureLibraryConfig] = None
    validation_metric: Optional[DivergenceMetric] = None


@dataclass
class CandidateResult:
    model: FittedModel
    validation: ValidationResult


@dataclass
class FitResult:
    segment_metadata: SegmentMetadata
    mode_segment: ModeSegment
    candidate_results: list[CandidateResult] = field(default_factory=list)

    @property
    def candidates(self) -> list[FittedModel]:
        return [result.model for result in self.candidate_results]

    @property
    def validation_results(self) -> list[ValidationResult]:
        return [result.validation for result in self.candidate_results]


def _r_squared(dx: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((dx - predicted) ** 2)
    ss_tot = np.sum((dx - np.mean(dx, axis=0)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def fit_mode_segment(
    mode_segment: ModeSegment,
    segment_metadata: SegmentMetadata,
    config: PipelineConfig,
) -> list[FittedModel]:
    """Fit one candidate model per sparsity-threshold grid point for a
    single (segment, mode) window. Returns an empty list if the mode
    segment is too short to fit.
    """
    if mode_segment.sample_count < config.segmentation.min_segment_length_samples:
        return []

    frame = prepare_frame_for_identification(mode_segment.frame, config.plant_controller_split)
    if len(frame) < config.segmentation.min_segment_length_samples:
        return []

    x_variables = config.plant_controller_split.x_variables
    u_variables = config.plant_controller_split.u_variables
    invalid = [
        name for name in x_variables + u_variables
        if name in frame and pd.api.types.is_numeric_dtype(frame[name])
        and not np.isfinite(frame[name].to_numpy(dtype=float)).all()
    ]
    if invalid:
        raise ValueError(f"Configured variables contain missing or non-finite observations: {invalid}")

    plant_controller_data = build_plant_controller_data(frame, config.plant_controller_split)

    library_result = build_feature_library(
        frame,
        x_variables,
        config.feature_library,
        additional_features=plant_controller_data.u,
    )

    theta = np.hstack([library_result.features.to_numpy(dtype=float), plant_controller_data.u])
    feature_names = list(library_result.features.columns) + [f"u:{name}" for name in u_variables]

    t_seconds = (frame.index - frame.index[0]).total_seconds().to_numpy()
    diff_result = differentiate(plant_controller_data.x, t_seconds, config.differentiation)

    sweep_points: list[SparsitySweepPoint] = sweep_sparsity_thresholds(theta, diff_result.dx, config.optimizer)

    candidates: list[FittedModel] = []
    for point in sweep_points:
        predicted = theta @ point.coefficients.T
        r2 = _r_squared(diff_result.dx, predicted)
        candidates.append(
            FittedModel(
                coefficients=point.coefficients,
                feature_names=feature_names,
                x_variables=list(x_variables),
                u_variables=list(u_variables),
                library_tiers=list(config.feature_library.enabled_tiers),
                condition_number=library_result.condition_number,
                condition_number_warning=library_result.condition_number_warning,
                differentiation_method=diff_result.method.value,
                differentiation_params=diff_result.params,
                optimizer_method=config.optimizer.method.value,
                sparsity_threshold=point.threshold,
                n_nonzero_terms=point.n_nonzero_terms,
                segment_metadata=segment_metadata,
                mode_label=mode_segment.mode_label,
                sample_count=mode_segment.sample_count,
                r_squared=r2,
                feature_library_config=config.feature_library.model_copy(deep=True),
            )
        )
    return candidates


def run_fitting_workflow(
    df: pd.DataFrame,
    config: PipelineConfig,
    deploy_boundaries: Optional[list[DeployBoundary]] = None,
) -> list[FitResult]:
    """Run the end-to-end fitting workflow through its orchestration module."""
    from sdmodeler.identification.workflow import run_fitting_workflow as run_workflow

    return run_workflow(df, config, deploy_boundaries)
