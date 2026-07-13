"""Layer-1 eval runner: score the deterministic comparison engine against labelled cases.

Runs paired_verifier._evaluate_fact (the same code the live pipeline uses once the LLM has
extracted a fact) on each labelled case and compares its verdict — and optionally its computed
value — against the expected label. No LLM, no API keys, fully reproducible.

Usage:
    python -m eval.run_comparison_eval
    python -m eval.run_comparison_eval --cases eval/cases/comparison --json out.json --fail-under 1.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from eval.dataset import ComparisonCase, build_fact, build_source, load_cases
from eval.metrics import compute_metrics
from eval.report import CaseResult, render_console, write_json
from paired_verifier import MATCH_TOLERANCE, _evaluate_fact

_DEFAULT_CASES_DIR = Path(__file__).parent / "cases" / "comparison"


def _discover_case_files(cases_arg: Path) -> List[Path]:
    if cases_arg.is_dir():
        return sorted(cases_arg.glob("*.yaml")) + sorted(cases_arg.glob("*.yml"))
    return [cases_arg]


def _value_matches(computed: Optional[float], expected: Optional[float]) -> Optional[bool]:
    if expected is None:
        return None
    if computed is None:
        return False
    return abs(computed - expected) <= MATCH_TOLERANCE


def evaluate_case(case: ComparisonCase) -> CaseResult:
    source = build_source(case)
    fact = build_fact(case)
    result = _evaluate_fact(fact, [source])

    verdict_ok = result.verdict == case.expected.verdict
    value_ok = _value_matches(result.computed_value, case.expected.computed_value)
    return CaseResult(
        id=case.id,
        operation=case.fact.operation,
        expected_verdict=case.expected.verdict,
        predicted_verdict=result.verdict,
        verdict_ok=verdict_ok,
        value_ok=value_ok,
        computed_value=result.computed_value,
        reasoning=result.reasoning,
    )


def run(cases: List[ComparisonCase]) -> Tuple[List[CaseResult], object]:
    results = [evaluate_case(c) for c in cases]
    pairs = [(r.expected_verdict, r.predicted_verdict) for r in results]
    metrics = compute_metrics(pairs)
    return results, metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Layer-1 comparison-engine accuracy eval.")
    parser.add_argument(
        "--cases", type=Path, default=_DEFAULT_CASES_DIR,
        help="YAML file or directory of case files (default: eval/cases/comparison).",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write a machine-readable report here.")
    parser.add_argument(
        "--fail-under", type=float, default=None,
        help="Exit non-zero if verdict accuracy is below this fraction (e.g. 1.0 for a CI gate).",
    )
    args = parser.parse_args(argv)

    # The engine's reasoning strings contain non-ASCII (→, Δ, ≤); force UTF-8 so printing
    # the report doesn't crash on a legacy Windows console codepage (cp1252).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    case_files = _discover_case_files(args.cases)
    if not case_files:
        print(f"No case files found at {args.cases}", file=sys.stderr)
        return 2

    cases = load_cases(case_files)
    results, metrics = run(cases)

    print(render_console(metrics, results))
    if args.json:
        write_json(args.json, metrics, results)
        print(f"Wrote JSON report to {args.json}")

    # Non-zero on ANY failing case (including value_ok-only failures, which don't lower
    # verdict accuracy) so CI can't go green past a broken case; --fail-under adds an
    # explicit accuracy threshold on top.
    exit_code = 0
    if not all(r.passed for r in results):
        exit_code = 1
    if args.fail_under is not None and metrics.accuracy < args.fail_under:
        print(f"FAIL: accuracy {metrics.accuracy:.3f} < fail-under {args.fail_under}", file=sys.stderr)
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
