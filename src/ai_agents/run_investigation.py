"""Command-line entry point for data-quality investigations.

From the project root:

    python -m src.ai_agents.run_investigation
    python -m src.ai_agents.run_investigation --rule-id DQ001
    python -m src.ai_agents.run_investigation --all-failed
    python -m src.ai_agents.run_investigation --all-failed --dry-run

With no options, this investigates DQ004. A dry run lists findings and does
not call a model. Unit tests do not run this module as a program.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .findings_loader import PROJECT_ROOT, FindingSelectionError, FindingsFileError
from .investigator import (
    InvestigationAttempt,
    InvestigationOutputError,
    InvestigatorConfigError,
    run_investigation,
    run_investigations,
)

DQ004 = "DQ004"


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Read the investigation options.

    --rule-id and --all-failed cannot be combined. Neither option means DQ004.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.ai_agents.run_investigation",
        description="Investigate validated data-quality findings with one CrewAI agent.",
    )
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument(
        "--rule-id",
        dest="rule_id",
        help="Investigate one finding, for example DQ001.",
    )
    choice.add_argument(
        "--all-failed",
        action="store_true",
        help="Investigate every finding whose status is FAIL, one at a time.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected findings without calling a model.",
    )
    return parser.parse_args(argv)


def display_path(path: Path) -> str:
    """Show a project-relative path when the file lives in this project."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def print_dry_run(attempts: list[InvestigationAttempt]) -> None:
    """Print the findings a real run would investigate."""
    print("Dry run. No model call will be made.")
    for attempt in attempts:
        print(
            f"{attempt.rule_id} {attempt.status} {attempt.dataset} -> "
            f"{display_path(attempt.output_path)}"
        )
    noun = "finding" if len(attempts) == 1 else "findings"
    print(f"{len(attempts)} {noun} selected.")


def print_batch_summary(attempts: list[InvestigationAttempt]) -> int:
    """Print one line per finding and return how many failed."""
    failed = 0
    for attempt in attempts:
        if attempt.error is None:
            print(f"{attempt.rule_id}: wrote {display_path(attempt.output_path)}")
            continue
        failed += 1
        print(f"{attempt.rule_id}: failed: {attempt.error}")
    written = len(attempts) - failed
    print(f"Investigated {len(attempts)} findings: {written} written, {failed} failed.")
    return failed


def main(argv: list[str] | None = None) -> None:
    """Investigate the selected findings and set a failing exit code when needed.

    Args:
        argv: Options to parse. When omitted, the process arguments are used.
            Tests pass an empty list to keep the DQ004 default.
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.dry_run or args.all_failed:
        try:
            attempts = run_investigations(
                rule_id=args.rule_id,
                all_failed=args.all_failed,
                dry_run=args.dry_run,
            )
        except (
            FindingsFileError,
            FindingSelectionError,
            InvestigatorConfigError,
        ) as exc:
            print(f"Investigation failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if args.dry_run:
            print_dry_run(attempts)
            return
        if print_batch_summary(attempts) > 0:
            raise SystemExit(1)
        return

    rule_id = args.rule_id or DQ004
    try:
        path = run_investigation(rule_id)
    except (
        FindingsFileError,
        FindingSelectionError,
        InvestigatorConfigError,
        InvestigationOutputError,
    ) as exc:
        print(f"Investigation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Wrote investigation report to {path}")


if __name__ == "__main__":
    main()
