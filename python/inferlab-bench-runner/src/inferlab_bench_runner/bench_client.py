"""Thin protocol entrypoint for release-owned Bench operations."""

import importlib.metadata
import json
import sys
import traceback
from pathlib import Path

from inferlab_measurement_sdk import (
    BenchClientRequest,
    BenchClientResult,
    BenchPopulationPreparationRequest,
    BenchPopulationPreparationResult,
    CaseBudgetExpired,
    CaseDeadline,
    ClientStatus,
    parse_args,
)

from inferlab_bench_runner.execution import execute
from inferlab_bench_runner.population import prepare_population
from inferlab_bench_runner.result_metrics import NORMALIZATION_SCHEMA


def handle_population_preparation(input_text: str) -> BenchPopulationPreparationResult:
    request = BenchPopulationPreparationRequest.model_validate_json(input_text)
    return prepare_population(request)


def handle_bench_execution(input_text: str) -> BenchClientResult:
    request = BenchClientRequest.model_validate_json(input_text)
    deadline = CaseDeadline(request.case_budget_seconds)
    result = execute(request, deadline)
    try:
        deadline.remaining()
    except CaseBudgetExpired as deadline_error:
        error = str(deadline_error)
        if result.error is not None:
            error = f"{result.error}; {error}"
        return result.model_copy(update={"status": ClientStatus.failed, "error": error})
    return result


def main() -> int:
    args = parse_args()
    if args.handshake:
        print(
            json.dumps(
                {
                    "aiperf_version": importlib.metadata.version("aiperf"),
                    "transformers_version": importlib.metadata.version("transformers"),
                }
            )
        )
        return 0
    if args.input is None or args.output is None:
        raise ValueError("--input and --output are required")
    output = Path(args.output)
    result: BenchPopulationPreparationResult | BenchClientResult
    try:
        input_text = Path(args.input).read_text(encoding="utf-8")
        if args.prepare:
            result = handle_population_preparation(input_text)
        else:
            result = handle_bench_execution(input_text)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        if args.prepare:
            result = BenchPopulationPreparationResult(
                schema_version=1,
                status=ClientStatus.failed,
                materialization_identity="unknown",
                requested_entries=0,
                candidate_entries=0,
                admitted_entries=0,
                ineligible_entries=0,
                ineligible_reasons={},
                population=None,
                input_tokens=None,
                output_tokens=None,
                evidence_path=None,
                error=str(error),
            )
        else:
            result = BenchClientResult(
                schema_version=1,
                status=ClientStatus.failed,
                completed_requests=0,
                failed_requests=0,
                normalization_schema=NORMALIZATION_SCHEMA,
                metrics={},
                native_command=[],
                native_exit_code=None,
                raw_artifacts=[],
                error=str(error),
            )
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
