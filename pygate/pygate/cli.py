# The `pygate` command. `check` exits 0 when nothing is found, 1 when something is, and 2 when a
# path or a configuration cannot be read -- ruff's convention, so a script that reads ruff's exit
# status reads this one. `--fix` and `--diff` behave as ruff's do, for the one fixable rule.

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pygate.check import CheckResult, FixMode, check_paths
from pygate.registry import RULES, RULES_BY_CODE
from pygate.report import render_concise, render_json
from pygate.settings import ConfigError, Overrides

EXIT_CLEAN = 0

EXIT_FINDINGS = 1

EXIT_ERROR = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: Arguments, defaulting to the process's own.

    Returns:
        Process exit status.
    """
    args = _parser().parse_args(argv)
    if args.command == "rule":
        return _explain(args.code, show_all=args.all)
    overrides = Overrides(select=args.select, ignore=args.ignore)
    mode = FixMode.DIFF if args.diff else FixMode.WRITE if args.fix else FixMode.OFF
    try:
        result = check_paths(args.paths, overrides, Path.cwd(), mode)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"pygate: {exc}", file=sys.stderr)
        return EXIT_ERROR
    for refusal in result.refusals:
        print(f"pygate: fix refused: {refusal}", file=sys.stderr)
    if mode is FixMode.DIFF:
        return _report_diff(result, exit_zero=args.exit_zero)
    render = render_json if args.output_format == "json" else render_concise
    print(render(result.diagnostics))
    if result.fixed_files:
        print(
            f"Moved {result.moved} definition(s) in {result.fixed_files} file(s).", file=sys.stderr
        )
    return EXIT_FINDINGS if result.diagnostics and not args.exit_zero else EXIT_CLEAN


def _explain(code: str | None, show_all: bool) -> int:
    """Print one rule's explanation, or every rule's summary.

    Args:
        code: Rule code to explain.
        show_all: Whether to list every rule instead.

    Returns:
        Process exit status.
    """
    if show_all:
        for rule in RULES:
            marker = "" if rule.default else "  (not selected by default)"
            print(f"{rule.code}  {rule.name:<34} {rule.summary}{marker}")
        return EXIT_CLEAN
    rule = RULES_BY_CODE.get((code or "").upper())
    if rule is None:
        print(f"pygate rule: name a rule code, or pass --all (got {code!r})", file=sys.stderr)
        return EXIT_ERROR
    print(f"{rule.code} ({rule.name})\n\n{rule.summary}\n\n{rule.explanation}")
    return EXIT_CLEAN


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="pygate", description="House-style rules ruff cannot express."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check files and directories")
    check.add_argument("paths", nargs="*", type=Path, default=[Path(".")])
    check.add_argument("--output-format", choices=("concise", "json"), default="concise")
    check.add_argument(
        "--select", type=_selectors, default=None, help="comma-separated; replaces `select`"
    )
    check.add_argument(
        "--ignore", type=_selectors, default=(), help="comma-separated; added to `ignore`"
    )
    check.add_argument(
        "--exit-zero", action="store_true", help="exit 0 even when something is found"
    )
    fixing = check.add_mutually_exclusive_group()
    fixing.add_argument("--fix", action="store_true", help="apply the PG105 zone reorder")
    fixing.add_argument(
        "--diff", action="store_true", help="print the PG105 reorder, write nothing"
    )
    rule = commands.add_parser("rule", help="explain a rule")
    rule.add_argument("code", nargs="?")
    rule.add_argument("--all", action="store_true", help="list every rule")
    return parser


def _report_diff(result: CheckResult, exit_zero: bool) -> int:
    """Print the zone reorder as diffs, as ruff's ``--diff`` does.

    Args:
        result: The check, run in diff mode.
        exit_zero: Whether to exit 0 even when a file would change.

    Returns:
        1 when any file would change, 0 otherwise.
    """
    print("".join(result.diffs), end="")
    print(
        f"Would move {result.moved} definition(s) in {result.fixed_files} file(s).",
        file=sys.stderr,
    )
    return EXIT_FINDINGS if result.diffs and not exit_zero else EXIT_CLEAN


def _selectors(value: str) -> tuple[str, ...]:
    """Split a comma-separated selector list.

    Args:
        value: The option's value.

    Returns:
        The selectors.
    """
    return tuple(part.strip() for part in value.split(",") if part.strip())
