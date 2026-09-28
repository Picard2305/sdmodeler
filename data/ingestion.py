"""
Data ingestion and canonical schema (spec Section 1).

Responsibilities:
  - Define the canonical internal multivariate time-series schema.
  - Pull telemetry from CloudWatch (GetMetricData / GetMetricStatistics) and
    at least one non-AWS-native source.
  - Align mismatched sampling intervals per variable role:
      * zero-order hold (forward fill) for control variables
      * linear interpolation only for genuinely continuous state variables
      * Instances (and any control variable) is NEVER linearly interpolated
  - Handle missing data, clock skew across sources, and unit normalization.

The canonical frame always carries a `regime_label` column, but ingestion
never assigns values to it — per spec Section 1, regime/mode labeling is the
responsibility of the segmentation step, not ingestion.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd

from sdmodeler.config import (
    DataQualityConfig,
    IngestionConfig,
    NonAWSSourceConfig,
    ResamplingMethod,
    TelemetrySourceConfig,
    TelemetrySourceType,
    VariableRole,
    VariableSpec,
)

logger = logging.getLogger(__name__)

REGIME_LABEL_COLUMN = "regime_label"


def _to_utc_timestamp(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts if ts.tzinfo is not None else ts.tz_localize("UTC")


class TelemetryFetcher(Protocol):
    def fetch(self, metric_name: str, start: datetime, end: datetime) -> pd.Series:
        """Return a pd.Series indexed by tz-aware UTC DatetimeIndex, named metric_name."""
        ...


class CloudWatchClient(Protocol):
    def get_metric_data(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def get_metric_statistics(self, **kwargs: Any) -> dict[str, Any]:
        ...


class CloudWatchFetcher:
    """Fetches metrics via CloudWatch GetMetricData or GetMetricStatistics.

    boto3 is imported lazily so this module (and the rest of the package)
    can be used/tested without boto3 installed when CloudWatch is not the
    active telemetry source.
    """

    def __init__(
        self,
        source_config: TelemetrySourceConfig,
        period_seconds: int = 60,
        client: Optional[CloudWatchClient] = None,
    ):
        self.source_config = source_config
        self.period_seconds = period_seconds
        self._client = client

    def _get_client(self) -> CloudWatchClient:
        if self._client is None:
            import boto3  # lazy import

            self._client = boto3.client("cloudwatch", region_name=self.source_config.region)
        return self._client

    def fetch(self, metric_name: str, start: datetime, end: datetime) -> pd.Series:
        client = self._get_client()
        if self.source_config.source_type == TelemetrySourceType.CLOUDWATCH_GET_METRIC_DATA:
            return self._fetch_get_metric_data(client, metric_name, start, end)
        elif self.source_config.source_type == TelemetrySourceType.CLOUDWATCH_GET_METRIC_STATISTICS:
            return self._fetch_get_metric_statistics(client, metric_name, start, end)
        raise ValueError(
            f"CloudWatchFetcher cannot handle source_type={self.source_config.source_type}"
        )

    def _dimensions(self, metric_name: str) -> list[dict[str, str]]:
        return [
            {"Name": name, "Value": value}
            for name, value in self.source_config.metric_dimensions.get(metric_name, {}).items()
        ]

    def _fetch_get_metric_data(self, client, metric_name: str, start: datetime, end: datetime) -> pd.Series:
        dimensions = self._dimensions(metric_name)
        metric = {"Namespace": self.source_config.namespace, "MetricName": metric_name}
        if dimensions:
            metric["Dimensions"] = dimensions
        request = {
            "MetricDataQueries": [{
                "Id": "m1",
                "MetricStat": {"Metric": metric, "Period": self.period_seconds, "Stat": "Average"},
                "ReturnData": True,
            }],
            "StartTime": start,
            "EndTime": end,
        }
        timestamps = []
        values = []
        while True:
            response = client.get_metric_data(**request)
            for result in response["MetricDataResults"]:
                timestamps.extend(result["Timestamps"])
                values.extend(result["Values"])
            if not response.get("NextToken"):
                break
            request["NextToken"] = response["NextToken"]
        return pd.Series(data=values, index=pd.DatetimeIndex(timestamps, tz="UTC"), name=metric_name).sort_index()

    def _fetch_get_metric_statistics(self, client, metric_name: str, start: datetime, end: datetime) -> pd.Series:
        dimensions = self._dimensions(metric_name)
        response = client.get_metric_statistics(
            Namespace=self.source_config.namespace,
            MetricName=metric_name,
            StartTime=start,
            EndTime=end,
            Period=self.period_seconds,
            Statistics=["Average"],
            **({"Dimensions": dimensions} if dimensions else {}),
        )
        datapoints = sorted(response["Datapoints"], key=lambda d: d["Timestamp"])
        return pd.Series(
            data=[d["Average"] for d in datapoints],
            index=pd.DatetimeIndex([d["Timestamp"] for d in datapoints], tz="UTC"),
            name=metric_name,
        )


class CSVSourceFetcher:
    """Generic non-AWS-native source fetcher (e.g. APM/log export as CSV).

    Expects a CSV with a `timestamp` column and one column per metric name.
    This satisfies spec Section 1's requirement for "at least one non-AWS
    source" without assuming a specific APM vendor.
    """

    def __init__(self, source_config: NonAWSSourceConfig):
        self.source_config = source_config

    def fetch(self, metric_name: str, start: datetime, end: datetime) -> pd.Series:
        if self.source_config.endpoint_or_path is None:
            raise ValueError(
                f"NonAWSSourceConfig '{self.source_config.source_name}' has no endpoint_or_path set."
            )
        df = pd.read_csv(self.source_config.endpoint_or_path, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        series = df[metric_name]
        series.index = pd.DatetimeIndex(series.index, tz="UTC") if series.index.tz is None else series.index
        series = series.sort_index()
        return series.loc[(series.index >= _to_utc_timestamp(start)) & (series.index <= _to_utc_timestamp(end))]


def check_clock_skew(
    series_by_source: dict[str, pd.Series], max_skew_seconds: float
) -> list[str]:
    """Compare the first timestamp of each source's series; warn if they
    disagree by more than max_skew_seconds. Returns list of warning strings.
    """
    warnings: list[str] = []
    firsts = {name: s.index[0] for name, s in series_by_source.items() if len(s) > 0}
    if len(firsts) < 2:
        return warnings
    reference_name, reference_ts = next(iter(firsts.items()))
    for name, ts in firsts.items():
        skew = abs((ts - reference_ts).total_seconds())
        if skew > max_skew_seconds:
            msg = (
                f"Clock skew detected: source '{name}' starts at {ts}, "
                f"source '{reference_name}' starts at {reference_ts} "
                f"(skew={skew:.1f}s > threshold={max_skew_seconds}s)"
            )
            logger.warning(msg)
            warnings.append(msg)
    return warnings


def _resample_variable(series: pd.Series, var: VariableSpec, target_index: pd.DatetimeIndex) -> pd.Series:
    reindexed = series.reindex(series.index.union(target_index)).sort_index()

    if var.resampling_method == ResamplingMethod.ZERO_ORDER_HOLD:
        filled = reindexed.ffill()
    elif var.resampling_method == ResamplingMethod.LINEAR_INTERPOLATION:
        if var.role == VariableRole.CONTROL:
            raise ValueError(
                f"Refusing to linearly interpolate control variable '{var.name}' "
                "(spec Section 1: control inputs are never linearly interpolated)."
            )
        filled = reindexed.interpolate(method="time")
    elif var.resampling_method == ResamplingMethod.NONE:
        filled = reindexed
    else:
        raise ValueError(f"Unknown resampling method: {var.resampling_method}")

    return filled.reindex(target_index)


def apply_missing_data_strategy(
    df: pd.DataFrame, variables: list[VariableSpec], data_quality: DataQualityConfig
) -> pd.DataFrame:
    df = df.copy()
    strategy = data_quality.missing_data_strategy
    if strategy == "forward_fill_state_gap_flag_control":
        for var in variables:
            if var.name not in df.columns:
                continue
            gap_col = f"{var.name}_was_missing"
            df[gap_col] = df[var.name].isna()
            if var.role == VariableRole.CONTROL:
                df[var.name] = df[var.name].ffill()
            else:
                df[var.name] = df[var.name].ffill().bfill()
    else:
        raise ValueError(f"Unknown missing_data_strategy: {strategy}")
    return df


def apply_unit_normalization(df: pd.DataFrame, unit_normalization: dict[str, str]) -> pd.DataFrame:
    """Applies simple named unit conversions. v1 supports a small fixed set
    of conversions; extend as new units are encountered.
    """
    df = df.copy()
    converters = {
        "bytes_to_gb": lambda s: s / (1024 ** 3),
        "millis_to_seconds": lambda s: s / 1000.0,
        "percent_to_fraction": lambda s: s / 100.0,
    }
    for var_name, conversion_name in unit_normalization.items():
        if var_name not in df.columns:
            continue
        if conversion_name not in converters:
            raise ValueError(f"Unknown unit conversion '{conversion_name}' for variable '{var_name}'")
        df[var_name] = converters[conversion_name](df[var_name])
    return df


def build_canonical_frame(
    raw_series_by_variable: dict[str, pd.Series],
    ingestion_config: IngestionConfig,
    target_interval_seconds: float,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Combine raw per-variable series into the canonical internal schema:
    a DataFrame indexed by a uniform UTC DatetimeIndex, with one column per
    configured variable plus an unassigned `regime_label` column.
    """
    check_clock_skew(raw_series_by_variable, ingestion_config.data_quality.max_clock_skew_seconds)

    target_index = pd.date_range(
        start=_to_utc_timestamp(start),
        end=_to_utc_timestamp(end),
        freq=pd.Timedelta(seconds=target_interval_seconds),
    )

    columns = {}
    for var in ingestion_config.variables:
        if var.name not in raw_series_by_variable:
            logger.warning("No raw data found for configured variable '%s'; filling with NaN.", var.name)
            columns[var.name] = pd.Series(np.nan, index=target_index)
            continue
        columns[var.name] = _resample_variable(raw_series_by_variable[var.name], var, target_index)

    df = pd.DataFrame(columns, index=target_index)
    df = apply_missing_data_strategy(df, ingestion_config.variables, ingestion_config.data_quality)
    df = apply_unit_normalization(df, ingestion_config.data_quality.unit_normalization)

    df[REGIME_LABEL_COLUMN] = pd.Series(np.nan, index=target_index, dtype=object)

    return df


def build_fetcher(
    telemetry_source: Optional[TelemetrySourceConfig] = None,
    non_aws_source: Optional[NonAWSSourceConfig] = None,
) -> TelemetryFetcher:
    if telemetry_source is not None:
        if telemetry_source.source_type in (
            TelemetrySourceType.CLOUDWATCH_GET_METRIC_DATA,
            TelemetrySourceType.CLOUDWATCH_GET_METRIC_STATISTICS,
        ):
            return CloudWatchFetcher(telemetry_source)
        raise ValueError(f"No fetcher implemented for source_type={telemetry_source.source_type}")
    if non_aws_source is not None:
        return CSVSourceFetcher(non_aws_source)
    raise ValueError("Must provide either telemetry_source or non_aws_source")


def fetch_all_raw_series(
    ingestion_config: IngestionConfig, start: datetime, end: datetime
) -> dict[str, pd.Series]:
    """Fetch raw (pre-resampling) series for every configured variable,
    routing each variable name to whichever configured source (CloudWatch
    or a non-AWS source) actually lists it in its metric_names. Variables
    not found in any source are omitted (build_canonical_frame fills them
    with NaN and logs a warning, rather than failing the whole run).
    """
    raw_series: dict[str, pd.Series] = {}

    telemetry_source = ingestion_config.telemetry_source
    cloudwatch_metric_names = set(telemetry_source.metric_names)
    cloudwatch_fetcher = build_fetcher(telemetry_source=telemetry_source)

    non_aws_fetchers = [
        (set(source.metric_names), build_fetcher(non_aws_source=source))
        for source in ingestion_config.non_aws_sources
    ]

    for var in ingestion_config.variables:
        if var.name in cloudwatch_metric_names and cloudwatch_fetcher is not None:
            raw_series[var.name] = cloudwatch_fetcher.fetch(var.name, start, end)
            continue
        for metric_names, fetcher in non_aws_fetchers:
            if var.name in metric_names:
                raw_series[var.name] = fetcher.fetch(var.name, start, end)
                break
        else:
            if var.name not in cloudwatch_metric_names:
                logger.warning(
                    "Variable '%s' is not listed in any configured source's "
                    "metric_names; it will be filled with NaN.",
                    var.name,
                )
    return raw_series
