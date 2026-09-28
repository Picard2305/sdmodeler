"""
Output / reporting (spec Section 9).

Three responsibilities:
  1. Render a fitted model's discovered ODE in human-readable mathematical
     notation (not just a raw coefficient array), alongside its segment/mode
     and validation score.
  2. Cross-deploy comparison: given two fitted models for the same regime
     across two different deploy_ids, report which coefficients changed and
     by how much.
  3. Serialize a fitted model set (coefficients + metadata) to JSON for later
     use in simulation/what-if tooling.

Per spec Section 8, models flagged unreliable by validation are excluded
from downstream use/reporting by default (not silently kept) — the
reliable-only filtering is applied at the report/export entry points here,
not by discarding data (candidates remain queryable via `fitting.FitResult`
directly if a caller wants to inspect a rejected model).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from sdmodeler.config import DivergenceMetric, FeatureLibraryConfig, FeatureLibraryTier
from sdmodeler.data.segmentation import SegmentMetadata
from sdmodeler.identification.fitting import FitResult, FittedModel

_ZERO_COEFFICIENT_TOLERANCE = 1e-12


def format_equations(model: FittedModel, coefficient_precision: int = 4) -> list[str]:
    """Render one human-readable ODE per state variable, e.g.
    'dCPU/dt = -0.0200*CPU + 0.1500*u:Instances'. Zero (thresholded-out)
    terms are omitted; an all-zero row renders as 'dX/dt = 0'.
    """
    coefficients = np.atleast_2d(model.coefficients)
    equations: list[str] = []
    for i, x_var in enumerate(model.x_variables):
        terms: list[str] = []
        for j, feature_name in enumerate(model.feature_names):
            coeff = coefficients[i, j]
            if abs(coeff) < _ZERO_COEFFICIENT_TOLERANCE:
                continue
            sign = "-" if coeff < 0 else "+"
            terms.append(f"{sign} {abs(coeff):.{coefficient_precision}f}*{feature_name}")
        if not terms:
            equations.append(f"d{x_var}/dt = 0")
            continue
        rhs = " ".join(terms)
        if rhs.startswith("+ "):
            rhs = rhs[2:]
        equations.append(f"d{x_var}/dt = {rhs}")
    return equations


@dataclass
class ModelReport:
    equations: list[str]
    deploy_id: Optional[str]
    regime_label: Optional[str]
    mode_label: str
    segment_start: Any
    segment_end: Any
    sample_count: int
    library_tiers: list[str]
    differentiation_method: str
    optimizer_method: str
    sparsity_threshold: float
    n_nonzero_terms: int
    r_squared: float
    validation_score: Optional[float]
    is_reliable: Optional[bool]
    validation_metric: Optional[str] = None

    def render(self) -> str:
        lines = [
            f"Segment: deploy_id={self.deploy_id} regime={self.regime_label} "
            f"mode={self.mode_label} [{self.segment_start} -> {self.segment_end}] "
            f"(n={self.sample_count})",
            f"Library tiers: {', '.join(self.library_tiers)}",
            f"Differentiation: {self.differentiation_method}; "
            f"Optimizer: {self.optimizer_method} (threshold={self.sparsity_threshold}, "
            f"{self.n_nonzero_terms} nonzero terms)",
            f"In-sample R^2: {self.r_squared:.4f}",
            f"Validation score ({self.validation_metric}): {self.validation_score} "
            f"(reliable={self.is_reliable})",
            "Discovered dynamics:",
        ]
        for eq in self.equations:
            lines.append(f"  {eq}")
        return "\n".join(lines)


def build_model_report(model: FittedModel) -> ModelReport:
    return ModelReport(
        equations=format_equations(model),
        deploy_id=model.segment_metadata.deploy_id,
        regime_label=model.segment_metadata.regime_label,
        mode_label=model.mode_label,
        segment_start=model.segment_metadata.start,
        segment_end=model.segment_metadata.end,
        sample_count=model.sample_count,
        library_tiers=[t.value for t in model.library_tiers],
        differentiation_method=model.differentiation_method,
        optimizer_method=model.optimizer_method,
        sparsity_threshold=model.sparsity_threshold,
        n_nonzero_terms=model.n_nonzero_terms,
        r_squared=model.r_squared,
        validation_score=model.validation_score,
        is_reliable=model.is_reliable,
        validation_metric=model.validation_metric.value if model.validation_metric else None,
    )


def reliable_candidates(fit_results: list[FitResult]) -> list[FittedModel]:
    """Flatten fit results to only the candidates validation marked
    reliable (spec Section 8: unreliable models are excluded from
    downstream use/reporting, not silently kept).
    """
    models: list[FittedModel] = []
    for result in fit_results:
        for candidate_result in result.candidate_results:
            if candidate_result.model.is_reliable:
                models.append(candidate_result.model)
    return models


def reportable_candidates(fit_results: list[FitResult]) -> list[FittedModel]:
    models: list[FittedModel] = []
    for result in fit_results:
        for candidate_result in result.candidate_results:
            model = candidate_result.model
            is_unvalidated_fourier = (
                model.is_reliable is None
                and FeatureLibraryTier.TIER_3_FOURIER in model.library_tiers
            )
            if model.is_reliable or is_unvalidated_fourier:
                models.append(model)
    return models


@dataclass
class CoefficientDelta:
    x_variable: str
    feature_name: str
    coefficient_a: float
    coefficient_b: float
    delta: float
    appeared: bool
    disappeared: bool


@dataclass
class ModelComparison:
    deploy_id_a: Optional[str]
    deploy_id_b: Optional[str]
    regime_label: Optional[str]
    mode_label: str
    warnings: list[str] = field(default_factory=list)
    deltas: list[CoefficientDelta] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Comparing deploy '{self.deploy_id_a}' vs '{self.deploy_id_b}' "
            f"(regime={self.regime_label}, mode={self.mode_label})",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        changed = [d for d in self.deltas if abs(d.delta) > _ZERO_COEFFICIENT_TOLERANCE]
        if not changed:
            lines.append("  No coefficient changes detected.")
        for d in changed:
            tag = " (new term)" if d.appeared else " (term removed)" if d.disappeared else ""
            lines.append(
                f"  d{d.x_variable}/dt : {d.feature_name}: "
                f"{d.coefficient_a:.4f} -> {d.coefficient_b:.4f} "
                f"(delta={d.delta:+.4f}){tag}"
            )
        return "\n".join(lines)


def compare_models(model_a: FittedModel, model_b: FittedModel) -> ModelComparison:
    """Cross-deploy comparison for two fitted models. Per spec Section 9,
    this is meant to be applied to two models fit on the *same regime*
    across *different deploy_ids*; mismatches are reported as warnings
    rather than raised, since a caller may still want the diff.
    """
    warnings: list[str] = []
    if model_a.segment_metadata.regime_label != model_b.segment_metadata.regime_label:
        warnings.append(
            f"Regime labels differ ({model_a.segment_metadata.regime_label!r} vs "
            f"{model_b.segment_metadata.regime_label!r}); comparison may not be apples-to-apples."
        )
    if model_a.mode_label != model_b.mode_label:
        warnings.append(
            f"Mode labels differ ({model_a.mode_label!r} vs {model_b.mode_label!r}); "
            "comparison may not be apples-to-apples."
        )
    if model_a.segment_metadata.deploy_id == model_b.segment_metadata.deploy_id:
        warnings.append(
            "Both models share the same deploy_id; this is not a cross-deploy comparison."
        )
    if model_a.x_variables != model_b.x_variables:
        raise ValueError(
            f"Cannot compare models with different x_variables: "
            f"{model_a.x_variables} vs {model_b.x_variables}"
        )

    coeffs_a = np.atleast_2d(model_a.coefficients)
    coeffs_b = np.atleast_2d(model_b.coefficients)

    features_a = model_a.feature_names
    features_b = model_b.feature_names
    all_features = list(dict.fromkeys(features_a + features_b))

    deltas: list[CoefficientDelta] = []
    for i, x_var in enumerate(model_a.x_variables):
        for feature_name in all_features:
            idx_a = features_a.index(feature_name) if feature_name in features_a else None
            idx_b = features_b.index(feature_name) if feature_name in features_b else None
            coeff_a = float(coeffs_a[i, idx_a]) if idx_a is not None else 0.0
            coeff_b = float(coeffs_b[i, idx_b]) if idx_b is not None else 0.0
            deltas.append(
                CoefficientDelta(
                    x_variable=x_var,
                    feature_name=feature_name,
                    coefficient_a=coeff_a,
                    coefficient_b=coeff_b,
                    delta=coeff_b - coeff_a,
                    appeared=idx_a is None and idx_b is not None,
                    disappeared=idx_a is not None and idx_b is None,
                )
            )

    return ModelComparison(
        deploy_id_a=model_a.segment_metadata.deploy_id,
        deploy_id_b=model_b.segment_metadata.deploy_id,
        regime_label=model_a.segment_metadata.regime_label,
        mode_label=model_a.mode_label,
        warnings=warnings,
        deltas=deltas,
    )


def model_to_dict(model: FittedModel) -> dict:
    """JSON-serializable representation of a fitted model: coefficients +
    all metadata required to reproduce/interpret it later (spec Section 9:
    serialized export suitable for downstream simulation/what-if tooling).
    """
    if model.feature_library_config is None:
        raise ValueError("Cannot export a model without feature_library_config")
    return {
        "coefficients": np.atleast_2d(model.coefficients).tolist(),
        "feature_names": model.feature_names,
        "x_variables": model.x_variables,
        "u_variables": model.u_variables,
        "equations": format_equations(model),
        "library_tiers": [t.value for t in model.library_tiers],
        "feature_library_config": model.feature_library_config.model_dump(mode="json"),
        "condition_number": model.condition_number,
        "condition_number_warning": model.condition_number_warning,
        "differentiation_method": model.differentiation_method,
        "differentiation_params": model.differentiation_params,
        "optimizer_method": model.optimizer_method,
        "sparsity_threshold": model.sparsity_threshold,
        "n_nonzero_terms": model.n_nonzero_terms,
        "segment_metadata": {
            "deploy_id": model.segment_metadata.deploy_id,
            "regime_label": model.segment_metadata.regime_label,
            "start": model.segment_metadata.start.isoformat(),
            "end": model.segment_metadata.end.isoformat(),
            "sample_count": model.segment_metadata.sample_count,
            "data_quality_flag": model.segment_metadata.data_quality_flag,
            "dynamic_range_by_variable": model.segment_metadata.dynamic_range_by_variable,
        },
        "mode_label": model.mode_label,
        "sample_count": model.sample_count,
        "r_squared": model.r_squared,
        "validation_score": model.validation_score,
        "validation_metric": model.validation_metric.value if model.validation_metric else None,
        "is_reliable": model.is_reliable,
    }


def model_from_dict(data: dict) -> FittedModel:
    metadata = data["segment_metadata"]
    return FittedModel(
        coefficients=np.asarray(data["coefficients"], dtype=float),
        feature_names=data["feature_names"],
        x_variables=data["x_variables"],
        u_variables=data["u_variables"],
        library_tiers=[FeatureLibraryTier(tier) for tier in data["library_tiers"]],
        feature_library_config=FeatureLibraryConfig.model_validate(data["feature_library_config"]),
        condition_number=data["condition_number"],
        condition_number_warning=data["condition_number_warning"],
        differentiation_method=data["differentiation_method"],
        differentiation_params=data["differentiation_params"],
        optimizer_method=data["optimizer_method"],
        sparsity_threshold=data["sparsity_threshold"],
        n_nonzero_terms=data["n_nonzero_terms"],
        segment_metadata=SegmentMetadata(
            deploy_id=metadata["deploy_id"],
            regime_label=metadata["regime_label"],
            start=pd.Timestamp(metadata["start"]),
            end=pd.Timestamp(metadata["end"]),
            sample_count=metadata["sample_count"],
            data_quality_flag=metadata["data_quality_flag"],
            dynamic_range_by_variable=metadata["dynamic_range_by_variable"],
        ),
        mode_label=data["mode_label"],
        sample_count=data["sample_count"],
        r_squared=data["r_squared"],
        validation_score=data["validation_score"],
        validation_metric=DivergenceMetric(data["validation_metric"]) if data["validation_metric"] else None,
        is_reliable=data["is_reliable"],
    )


def export_fit_results(
    fit_results: list[FitResult],
    path: str,
    *,
    reliable_only: bool = True,
) -> None:
    """Serialize a fitted model set to JSON. By default, validation-reliable
    candidates and Tier 3 candidates (which are not forward-validated by
    design) are exported; unreliable candidates are excluded. Pass
    reliable_only=False to export every candidate for inspection.
    """
    if reliable_only:
        models = reportable_candidates(fit_results)
    else:
        models = [c for r in fit_results for c in r.candidates]

    payload = {"models": [model_to_dict(m) for m in models]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
