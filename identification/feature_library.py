"""
Feature library (spec Section 5).

Builds a tiered feature library rather than one large combined library:
  - Tier 1 (default): linear terms in state variables plus user-supplied
    domain-knowledge terms (pandas-eval expressions over the frame's
    columns, e.g. "1 - cache_hit_rate"). Not auto-generated.
  - Tier 2 (opt-in): low-order polynomial (degree <= 2) and pairwise
    interaction terms, combined with Tier 1 in the same regression.
  - Tier 3 (opt-in, separate fit): Fourier terms for diurnal/weekly
    structure. Never combined with Tier 1/2 in the same regression
    (enforced already by FeatureLibraryConfig's validator).

Also performs the required collinearity/condition-number check on the
resulting design matrix, returning a warning rather than silently failing.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sdmodeler.config import FeatureLibraryConfig, FeatureLibraryTier, Tier1Config, Tier2Config, Tier3Config

_PERIOD_SECONDS = {
    "daily": 24 * 3600,
    "weekly": 7 * 24 * 3600,
}


@dataclass
class FeatureLibraryResult:
    features: pd.DataFrame
    condition_number: float
    condition_number_warning: str | None = None


def build_tier1_features(frame: pd.DataFrame, x_variables: list[str], config: Tier1Config) -> pd.DataFrame:
    columns = {var: frame[var] for var in x_variables}
    for expr in config.domain_terms:
        columns[expr] = frame.eval(expr)
    return pd.DataFrame(columns, index=frame.index)


def build_tier2_features(frame: pd.DataFrame, x_variables: list[str], config: Tier2Config) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    if config.max_polynomial_degree >= 2:
        for var in x_variables:
            columns[f"{var}^2"] = frame[var] ** 2
    if config.max_polynomial_degree >= 2 and config.include_pairwise_interactions:
        for a, b in itertools.combinations(x_variables, 2):
            columns[f"{a}*{b}"] = frame[a] * frame[b]
    return pd.DataFrame(columns, index=frame.index)


def build_tier3_features(frame: pd.DataFrame, config: Tier3Config) -> pd.DataFrame:
    duration_days = (frame.index[-1] - frame.index[0]).total_seconds() / 86400.0
    if duration_days < config.min_segment_duration_days:
        raise ValueError(
            f"Tier 3 (Fourier) features require a segment spanning at least "
            f"{config.min_segment_duration_days} days; this segment spans "
            f"{duration_days:.2f} days (spec Section 5)."
        )

    t_seconds = (frame.index - frame.index[0]).total_seconds().to_numpy()
    columns: dict[str, np.ndarray] = {}
    for period_name in config.fourier_periods:
        if period_name not in _PERIOD_SECONDS:
            raise ValueError(f"Unknown fourier period '{period_name}'; supported: {list(_PERIOD_SECONDS)}")
        period_seconds = _PERIOD_SECONDS[period_name]
        omega = 2 * np.pi / period_seconds
        columns[f"sin_{period_name}"] = np.sin(omega * t_seconds)
        columns[f"cos_{period_name}"] = np.cos(omega * t_seconds)
    return pd.DataFrame(columns, index=frame.index)


def check_condition_number(features: pd.DataFrame | np.ndarray, threshold: float) -> tuple[float, str | None]:
    matrix = features.to_numpy(dtype=float) if isinstance(features, pd.DataFrame) else np.asarray(features, dtype=float)
    cond = float(np.linalg.cond(matrix))
    warning = None
    if cond > threshold:
        warning = (
            f"Regression design matrix condition number ({cond:.3e}) exceeds "
            f"configured threshold ({threshold:.3e}); collinear features detected "
            "(e.g. CPU/RequestRate) — consider dropping redundant terms (spec Section 5)."
        )
    return cond, warning


def _build_feature_frames(
    frame: pd.DataFrame,
    x_variables: list[str],
    config: FeatureLibraryConfig,
) -> list[pd.DataFrame]:
    tiers = set(config.enabled_tiers)
    if FeatureLibraryTier.TIER_3_FOURIER in tiers:
        if config.tier3 is None:
            raise ValueError("Tier 3 enabled but no tier3 config provided.")
        return [build_tier3_features(frame, config.tier3)]

    feature_frames = []
    if FeatureLibraryTier.TIER_1_LINEAR_DOMAIN in tiers:
        feature_frames.append(build_tier1_features(frame, x_variables, config.tier1))
    if FeatureLibraryTier.TIER_2_POLYNOMIAL in tiers:
        if config.tier2 is None:
            raise ValueError("Tier 2 enabled but no tier2 config provided.")
        feature_frames.append(build_tier2_features(frame, x_variables, config.tier2))
    return feature_frames


def build_feature_row(
    x: np.ndarray,
    u: np.ndarray,
    x_variables: list[str],
    u_variables: list[str],
    config: FeatureLibraryConfig,
) -> np.ndarray:
    """Evaluate the Tier 1/2 feature library at a single (x, u) point, for
    use during forward-simulation validation (spec Section 8). Returns a
    1D array whose column order matches build_feature_library's output
    concatenated with u (i.e. the same order used when building the
    training design matrix in the fitting workflow).

    Not applicable to Tier 3 (Fourier) models, which are a function of
    time rather than state and are not integrated forward as an ODE.
    """
    if FeatureLibraryTier.TIER_3_FOURIER in set(config.enabled_tiers):
        raise ValueError(
            "build_feature_row is not applicable to Tier 3 (Fourier) models; "
            "these are a seasonal function of time, not a state-feedback ODE "
            "(spec Section 5)."
        )

    row_frame = pd.DataFrame(
        {**{v: [x[i]] for i, v in enumerate(x_variables)}, **{v: [u[i]] for i, v in enumerate(u_variables)}}
    )
    features_row = pd.concat(_build_feature_frames(row_frame, x_variables, config), axis=1)
    features_row = features_row.to_numpy(dtype=float).flatten()

    return np.concatenate([features_row, u])


def build_feature_library(
    frame: pd.DataFrame,
    x_variables: list[str],
    config: FeatureLibraryConfig,
    *,
    additional_features: np.ndarray | None = None,
) -> FeatureLibraryResult:
    feature_frames = _build_feature_frames(frame, x_variables, config)
    if not feature_frames:
        raise ValueError("No feature library tiers enabled.")
    features = pd.concat(feature_frames, axis=1)

    design_matrix = features.to_numpy(dtype=float)
    if additional_features is not None:
        additional_matrix = np.asarray(additional_features, dtype=float)
        if additional_matrix.ndim != 2 or additional_matrix.shape[0] != len(features):
            raise ValueError("additional_features must be a 2D matrix with one row per frame sample.")
        design_matrix = np.hstack([design_matrix, additional_matrix])

    cond, warning = check_condition_number(design_matrix, config.condition_number_warning_threshold)
    return FeatureLibraryResult(features=features, condition_number=cond, condition_number_warning=warning)
