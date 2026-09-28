"""
Configuration schema for the PySINDy-based AWS telemetry system-identification
pipeline.

This module is the single source of truth for every config-driven decision
described in `.\\sindy-aws-telemetry-spec.md`. Nothing downstream (ingestion,
segmentation, x/u split, mode switching, feature library, differentiation,
optimizer, validation) should hardcode a choice that belongs here.

Several fields below are placeholders pending answers to the open design
questions in Section 11 of the spec. They default to `None` (or an explicit
"unknown"/"unset" sentinel) rather than a guessed value, and are flagged with
`# OPEN QUESTION` comments. Fill these in with real values from your AWS
environment before running the pipeline against production data.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Section 1: Data ingestion and schema
# ---------------------------------------------------------------------------

class TelemetrySourceType(str, Enum):
    CLOUDWATCH_GET_METRIC_DATA = "cloudwatch_get_metric_data"
    CLOUDWATCH_GET_METRIC_STATISTICS = "cloudwatch_get_metric_statistics"
    PROMETHEUS = "prometheus"
    TIMESTREAM = "timestream"
    CSV_EXPORT = "csv_export"


class ResamplingMethod(str, Enum):
    ZERO_ORDER_HOLD = "zero_order_hold"
    LINEAR_INTERPOLATION = "linear_interpolation"
    NONE = "none"


class VariableRole(str, Enum):
    STATE = "state"
    CONTROL = "control"


class VariableSpec(BaseModel):
    """One column in the canonical internal schema."""

    name: str
    role: VariableRole
    unit: Optional[str] = None
    resampling_method: ResamplingMethod

    @model_validator(mode="after")
    def _control_vars_use_zero_order_hold(self) -> "VariableSpec":
        if self.role == VariableRole.CONTROL and self.resampling_method != ResamplingMethod.ZERO_ORDER_HOLD:
            raise ValueError(
                f"Control variable '{self.name}' must use zero_order_hold resampling "
                "(control/exogenous inputs, including Instances, deploy_id, and "
                "feature flags, are never linearly interpolated per spec Section 1)."
            )
        return self


class TelemetrySourceConfig(BaseModel):
    """Where telemetry comes from and how to pull it."""

    source_type: TelemetrySourceType
    # OPEN QUESTION (spec Section 11): confirm actual telemetry access method
    # available in your environment (direct CloudWatch API vs. an
    # intermediate store like Prometheus/Timestream).
    region: Optional[str] = None
    namespace: Optional[str] = None
    metric_names: list[str] = Field(default_factory=list)
    metric_dimensions: dict[str, dict[str, str]] = Field(default_factory=dict)
    # OPEN QUESTION (spec Section 11): native sampling interval, and whether
    # it differs across metrics. Populate per-metric below once known.
    native_sampling_interval_seconds: Optional[dict[str, float]] = None


class NonAWSSourceConfig(BaseModel):
    """At least one non-AWS-native source (application logs/APM)."""

    source_name: str
    endpoint_or_path: Optional[str] = None
    metric_names: list[str] = Field(default_factory=list)
    native_sampling_interval_seconds: Optional[dict[str, float]] = None


class DataQualityConfig(BaseModel):
    max_clock_skew_seconds: float = 5.0
    missing_data_strategy: str = "forward_fill_state_gap_flag_control"
    unit_normalization: dict[str, str] = Field(default_factory=dict)


class IngestionConfig(BaseModel):
    telemetry_source: TelemetrySourceConfig
    non_aws_sources: list[NonAWSSourceConfig] = Field(default_factory=list)
    variables: list[VariableSpec]
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)

    @field_validator("variables")
    @classmethod
    def _instances_and_events_are_control_never_differentiated(cls, v: list[VariableSpec]) -> list[VariableSpec]:
        for var in v:
            if var.name.lower() in {"instances", "deploy_id", "deploy_flag", "feature_flag_state"}:
                if var.role != VariableRole.CONTROL:
                    raise ValueError(
                        f"'{var.name}' must be role=control per spec Section 1 "
                        "(exogenous input, not smooth state to be differentiated)."
                    )
        return v


# ---------------------------------------------------------------------------
# Section 2: Segmentation strategy
# ---------------------------------------------------------------------------

class DeployBoundarySource(str, Enum):
    CI_CD = "ci_cd"
    CODEDEPLOY_API = "codedeploy_api"
    ECS_DEPLOYMENT_EVENTS = "ecs_deployment_events"
    MANUAL_TIMESTAMPS = "manual_timestamps"
    UNKNOWN = "unknown"


class RegimeDetectionMethod(str, Enum):
    TIME_OF_DAY_RULE = "time_of_day_rule"
    CHANGEPOINT_DETECTION = "changepoint_detection"  # hook for v2, not required for v1


class SegmentationConfig(BaseModel):
    # OPEN QUESTION (spec Section 11): confirm actual source of deploy
    # boundary timestamps available in your environment.
    deploy_boundary_source: DeployBoundarySource = DeployBoundarySource.UNKNOWN
    allow_fit_across_deploy_boundary: bool = False  # explicit opt-in only

    regime_detection_method: RegimeDetectionMethod = RegimeDetectionMethod.TIME_OF_DAY_RULE
    business_hours_start_utc: Optional[int] = None
    business_hours_end_utc: Optional[int] = None

    min_segment_length_samples: int = 30
    min_dynamic_range_threshold: dict[str, float] = Field(
        default_factory=dict,
        description="Per-state-variable minimum (max-min)/mean or std threshold "
        "below which a segment is flagged insufficient dynamic range and skipped.",
    )

    # OPEN QUESTION (spec Section 11): are autoscaler-disabled load-test /
    # chaos-engineering traces available? If so, flag/prioritize them here.
    autoscaling_frozen_window_metadata_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Section 3: Separating plant from controller (SINDYc structure)
# ---------------------------------------------------------------------------

class PlantControllerSplitConfig(BaseModel):
    x_variables: list[str] = Field(
        default_factory=lambda: [
            "CPU", "Memory", "Latency", "QueueDepth", "RequestRate", "ErrorRate",
        ]
    )
    u_variables: list[str] = Field(
        default_factory=lambda: ["Instances", "deploy_flag", "feature_flag_vector"]
    )
    hold_instances_constant: bool = False
    filter_to_autoscaling_frozen_windows_only: bool = False

    known_feedback_limitations: list[str] = Field(
        default_factory=lambda: [
            "RequestRate may be partly a function of Latency via client-side "
            "retry/backoff; this is a real feedback loop inside x and is not "
            "externalized by this split (spec Section 3)."
        ]
    )

    @model_validator(mode="after")
    def _x_u_disjoint(self) -> "PlantControllerSplitConfig":
        overlap = set(self.x_variables) & set(self.u_variables)
        if overlap:
            raise ValueError(f"Variables cannot be in both x and u: {overlap}")
        return self


# ---------------------------------------------------------------------------
# Section 4: Hybrid / switching dynamics
# ---------------------------------------------------------------------------

class ThresholdOperator(str, Enum):
    GREATER_THAN = "gt"
    GREATER_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_EQUAL = "le"


class ModeThreshold(BaseModel):
    variable: str
    operator: ThresholdOperator
    value: float
    description: Optional[str] = None


class ModeSwitchingConfig(BaseModel):
    enabled: bool = True
    # OPEN QUESTION (spec Section 11): known operational thresholds
    # (autoscaling trigger %, alerting thresholds) to seed mode switching.
    thresholds: list[ModeThreshold] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Section 5: Feature library
# ---------------------------------------------------------------------------

class FeatureLibraryTier(str, Enum):
    TIER_1_LINEAR_DOMAIN = "tier_1_linear_domain"
    TIER_2_POLYNOMIAL = "tier_2_polynomial"
    TIER_3_FOURIER = "tier_3_fourier"


class Tier1Config(BaseModel):
    domain_terms: list[str] = Field(
        default_factory=list,
        description="User-supplied, per-system domain terms, e.g. "
        "'queueing_ratio', '1 - cache_hit_rate', 'retry_x_error'. "
        "Not auto-generated (spec Section 5).",
    )


class Tier2Config(BaseModel):
    max_polynomial_degree: int = Field(default=2, ge=1, le=2)
    include_pairwise_interactions: bool = True


class Tier3Config(BaseModel):
    fourier_periods: list[str] = Field(default_factory=lambda: ["daily", "weekly"])
    min_segment_duration_days: float = 7.0
    fit_separately_from_fast_timescale_terms: bool = True


class FeatureLibraryConfig(BaseModel):
    enabled_tiers: list[FeatureLibraryTier] = Field(
        default_factory=lambda: [FeatureLibraryTier.TIER_1_LINEAR_DOMAIN]
    )
    tier1: Tier1Config = Field(default_factory=Tier1Config)
    tier2: Optional[Tier2Config] = None
    tier3: Optional[Tier3Config] = None
    condition_number_warning_threshold: float = 1e10

    @model_validator(mode="after")
    def _tier3_never_combined_with_others(self) -> "FeatureLibraryConfig":
        if (
            FeatureLibraryTier.TIER_3_FOURIER in self.enabled_tiers
            and len(self.enabled_tiers) > 1
        ):
            raise ValueError(
                "Tier 3 (Fourier/seasonal) must be fit separately, never combined "
                "in the same regression as Tier 1/2 fast-timescale terms (spec Section 5)."
            )
        return self


# ---------------------------------------------------------------------------
# Section 6: Differentiation and optimizer configuration
# ---------------------------------------------------------------------------

class DifferentiationMethod(str, Enum):
    FINITE_DIFFERENCE = "finite_difference"
    SAVITZKY_GOLAY = "savitzky_golay"
    SPLINE = "spline"
    TOTAL_VARIATION = "total_variation"


class DifferentiationConfig(BaseModel):
    method: DifferentiationMethod = DifferentiationMethod.SAVITZKY_GOLAY
    savgol_window_length: int = 11
    savgol_polyorder: int = 3


class OptimizerMethod(str, Enum):
    STLSQ = "stlsq"
    SR3 = "sr3"
    ELASTIC_NET = "elastic_net"


class OptimizerConfig(BaseModel):
    method: OptimizerMethod = OptimizerMethod.STLSQ
    sparsity_threshold_grid: list[float] = Field(
        default_factory=lambda: [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    )
    guidance_note: str = (
        "STLSQ is the default. Switch to SR3 for noisy/correlated telemetry "
        "(likely the common case for AWS CloudWatch data) — this is guidance "
        "text, not an automatic decision in v1 (spec Section 6)."
    )


# ---------------------------------------------------------------------------
# Section 8: Validation
# ---------------------------------------------------------------------------

class DivergenceMetric(str, Enum):
    NORMALIZED_RMSE = "normalized_rmse"
    TRAJECTORY_DIVERGENCE_TIME = "trajectory_divergence_time"


class ValidationConfig(BaseModel):
    horizon_minutes: float = 10.0
    divergence_metric: DivergenceMetric = DivergenceMetric.NORMALIZED_RMSE
    acceptance_threshold: float = 0.2


# ---------------------------------------------------------------------------
# Section 11: Open design questions (explicit placeholders)
# ---------------------------------------------------------------------------

class OpenDesignQuestions(BaseModel):
    """
    Explicit tracker for the open design questions from spec Section 11.
    All fields default to None/unknown. The pipeline should refuse to run
    in "production mode" (as opposed to a dry-run/schema-validation mode)
    while any of these remain unanswered, and should surface which ones
    are still open.
    """

    telemetry_access_method_confirmed: Optional[str] = None
    native_sampling_interval_confirmed: Optional[bool] = None
    deploy_boundary_source_confirmed: Optional[bool] = None
    autoscaler_disabled_loadtest_data_available: Optional[bool] = None
    known_operational_thresholds_seeded: Optional[bool] = None

    def unresolved(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if v is None]


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    ingestion: IngestionConfig
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    plant_controller_split: PlantControllerSplitConfig = Field(
        default_factory=PlantControllerSplitConfig
    )
    mode_switching: ModeSwitchingConfig = Field(default_factory=ModeSwitchingConfig)
    feature_library: FeatureLibraryConfig = Field(default_factory=FeatureLibraryConfig)
    differentiation: DifferentiationConfig = Field(default_factory=DifferentiationConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    open_design_questions: OpenDesignQuestions = Field(default_factory=OpenDesignQuestions)

    def assert_ready_for_production_run(self) -> None:
        unresolved = self.open_design_questions.unresolved()
        if unresolved:
            raise ValueError(
                "Cannot run against production data: open design questions "
                f"unresolved: {unresolved}. See spec Section 11."
            )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load a PipelineConfig from a JSON or YAML file. YAML support
        requires PyYAML to be installed; JSON has no extra dependency.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # lazy import: not a hard dependency for JSON-only users
            except ImportError as exc:
                raise ImportError(
                    "Reading a YAML config requires PyYAML. Install it with "
                    "'pip install pyyaml', or use a .json config file instead."
                ) from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.model_validate(data)
