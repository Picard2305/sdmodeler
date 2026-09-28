"""
Optimizer configuration and sparsity sweep (spec Section 6, optimizer half).

Builds the configured PySINDy optimizer (STLSQ default, SR3, or an
elastic-net variant via sklearn), and sweeps the sparsity threshold over a
small grid rather than trusting a single fixed value, reporting a
sparsity-vs-fit-error curve so a human can sanity-check the choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pysindy.optimizers import SR3, STLSQ, WrappedOptimizer
from sklearn.linear_model import ElasticNet

from sdmodeler.config import OptimizerConfig, OptimizerMethod


def build_optimizer(config: OptimizerConfig, threshold: float):
    if config.method == OptimizerMethod.STLSQ:
        return STLSQ(threshold=threshold)
    if config.method == OptimizerMethod.SR3:
        return SR3(reg_weight_lam=threshold)
    if config.method == OptimizerMethod.ELASTIC_NET:
        return WrappedOptimizer(ElasticNet(alpha=threshold, fit_intercept=False))
    raise ValueError(f"Unknown optimizer method: {config.method}")


@dataclass
class SparsitySweepPoint:
    threshold: float
    n_nonzero_terms: int
    fit_rmse: float
    coefficients: np.ndarray


def sweep_sparsity_thresholds(
    theta: np.ndarray, dx: np.ndarray, config: OptimizerConfig
) -> list[SparsitySweepPoint]:
    """Fit the configured optimizer at every threshold in the grid and
    report (threshold, sparsity, fit error) so a human can inspect the
    trade-off curve rather than trusting a default blindly.
    """
    results = []
    for threshold in config.sparsity_threshold_grid:
        optimizer = build_optimizer(config, threshold)
        optimizer.fit(theta, dx)
        coef = np.asarray(optimizer.coef_)
        predicted = theta @ coef.T
        residual = dx - predicted
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        n_nonzero = int(np.count_nonzero(coef))
        results.append(
            SparsitySweepPoint(
                threshold=threshold,
                n_nonzero_terms=n_nonzero,
                fit_rmse=rmse,
                coefficients=coef,
            )
        )
    return results
