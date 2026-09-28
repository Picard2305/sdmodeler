"""
Command-line entry point.

Usage:
    python -m sdmodeler run --config config.json --start 2024-01-01T00:00:00Z \\
        --end 2024-01-02T00:00:00Z --target-interval-seconds 60 \\
        --output models.json

    python -m sdmodeler run --config config.json --input-csv telemetry.csv \\
        --output models.json

See `.\\README.md` Section 3 for the full walkthrough and config shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from sdmodeler.config import PipelineConfig
from sdmodeler.data.ingestion import build_canonical_frame, fetch_all_raw_series
from sdmodeler.data.segmentation import DeployBoundary
from sdmodeler.identification.fitting import FitResult, FittedModel
from sdmodeler.identification.workflow import run_fitting_workflow
from sdmodeler.plotting import save_validation_plot
from sdmodeler.reporting import build_model_report, export_fit_results, reportable_candidates


def _parse_timestamp(value: str) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def _load_deploy_boundaries(path: Optional[str]) -> list[DeployBoundary]:
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DeployBoundary(deploy_id=entry["deploy_id"], start=pd.Timestamp(entry["start"])) for entry in data]


def _load_input_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def _build_frame(config: PipelineConfig, args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv:
        return _load_input_csv(args.input_csv)

    if not args.start or not args.end or not args.target_interval_seconds:
        raise ValueError(
            "--start, --end, and --target-interval-seconds are required "
            "when --input-csv is not provided (they define the ingestion window)."
        )
    start = _parse_timestamp(args.start)
    end = _parse_timestamp(args.end)
    raw_series = fetch_all_raw_series(config.ingestion, start, end)
    return build_canonical_frame(raw_series, config.ingestion, args.target_interval_seconds, start, end)


def _plot_first_model_validation(models: list[FittedModel], fit_results: list[FitResult]) -> None:
    if not models:
        return

    model = models[0]
    validation = next(
        candidate_result.validation
        for result in fit_results
        for candidate_result in result.candidate_results
        if candidate_result.model is model
    )
    save_validation_plot(model, validation)


def _present_fit_results(
    fit_results: list[FitResult], include_unreliable: bool, output_path: Optional[str]
) -> None:
    models = (
        [candidate_result.model for result in fit_results for candidate_result in result.candidate_results]
        if include_unreliable
        else reportable_candidates(fit_results)
    )
    if not models:
        print("No models to report (no fittable segments, or none passed validation).", file=sys.stderr)
    for model in models:
        print(build_model_report(model).render())
        print()

    if output_path:
        export_fit_results(fit_results, output_path, reliable_only=not include_unreliable)
        print(f"Exported {len(models)} model(s) to {output_path}")

    _plot_first_model_validation(models, fit_results)


def run_command(args: argparse.Namespace) -> int:
    config = PipelineConfig.from_file(args.config)

    if not args.allow_unresolved_design_questions:
        config.assert_ready_for_production_run()

    frame = _build_frame(config, args)
    deploy_boundaries = _load_deploy_boundaries(args.deploy_boundaries)
    fit_results = run_fitting_workflow(frame, config, deploy_boundaries=deploy_boundaries)
    _present_fit_results(fit_results, args.include_unreliable, args.output)
    return 0


def _configure_run_parser(run_parser: argparse.ArgumentParser) -> None:
    run_parser.add_argument("--config", required=True, help="Path to a JSON or YAML PipelineConfig file.")
    run_parser.add_argument(
        "--input-csv",
        help="Path to a pre-built canonical CSV (must have a 'timestamp' column) to use instead of live ingestion.",
    )
    run_parser.add_argument("--start", help="Ingestion window start (ISO 8601). Required unless --input-csv is set.")
    run_parser.add_argument("--end", help="Ingestion window end (ISO 8601). Required unless --input-csv is set.")
    run_parser.add_argument(
        "--target-interval-seconds",
        type=float,
        help="Uniform resampling interval in seconds. Required unless --input-csv is set.",
    )
    run_parser.add_argument(
        "--deploy-boundaries",
        help="Path to a JSON file: a list of {\"deploy_id\": ..., \"start\": ISO8601} objects.",
    )
    run_parser.add_argument("--output", help="Path to write the exported model set as JSON.")
    run_parser.add_argument(
        "--include-unreliable",
        action="store_true",
        help="Include validation-unreliable candidates in report/export output (default: reliable-only).",
    )
    run_parser.add_argument(
        "--allow-unresolved-design-questions",
        action="store_true",
        help="Skip the production-readiness check for spec Section 11 open design questions.",
    )
    run_parser.set_defaults(func=run_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sdmodeler", description="PySINDy-based AWS telemetry system identification.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full ingest -> fit -> validate -> report pipeline.")
    _configure_run_parser(run_parser)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
