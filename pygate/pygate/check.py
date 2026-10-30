# Running the rules: read each file once, apply the zone fix when asked, run every family over it,
# keep what its configuration enables, then apply its noqa directives.

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pygate.diagnostic import Diagnostic
from pygate.files import python_files
from pygate.fix.apply import fix_source
from pygate.noqa import directives, suppress
from pygate.rules import cohesion, composition, layout, shared_state
from pygate.settings import Configurations, Overrides, Settings
from pygate.source import SourceModule, read_module

SYNTAX_ERROR = "PG000"

#: The one rule with a fix. Its fix runs only where the rule itself is enabled.
ZONE_ORDER = "PG105"


class FixMode(StrEnum):
    """What a check does with the findings it knows how to fix."""

    OFF = "off"
    WRITE = "write"
    DIFF = "diff"


@dataclass
class CheckResult:
    """Findings and fixes from one file, or accumulated over many.

    Attributes:
        diagnostics: Findings left standing.
        moved: Top-level statements the zone fix moved, or would move.
        fixed_files: Files the zone fix changed, or would change.
        diffs: One unified diff per changed file, in diff mode.
        refusals: Why the zone fix left a region or a file alone.
    """

    diagnostics: list[Diagnostic] = field(default_factory=list)
    moved: int = 0
    fixed_files: int = 0
    diffs: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)


def check_paths(
    paths: Sequence[Path], overrides: Overrides, cwd: Path, fix: FixMode = FixMode.OFF
) -> CheckResult:
    """Check every Python file the paths cover.

    Args:
        paths: Files and directories.
        overrides: Command-line selection applied over each file's configuration.
        cwd: Root for default settings, when no configuration governs a file.
        fix: Whether to write, preview, or skip the zone fix.

    Returns:
        Every finding, sorted by file and position, with what the fix did.

    Raises:
        ConfigError: When a governing configuration is invalid.
        FileNotFoundError: When a named path does not exist.
    """
    configurations = Configurations(overrides, cwd)
    total = CheckResult()
    for path in python_files(paths, configurations):
        result = _check_file(path, configurations.for_path(path), fix)
        total.diagnostics.extend(result.diagnostics)
        total.moved += result.moved
        total.fixed_files += result.fixed_files
        total.diffs.extend(result.diffs)
        total.refusals.extend(result.refusals)
    total.diagnostics.sort()
    return total


def _check_file(path: Path, settings: Settings, fix: FixMode = FixMode.OFF) -> CheckResult:
    """Check one file under the settings that govern it.

    Args:
        path: The file.
        settings: Its settings.
        fix: Whether to write, preview, or skip the zone fix.

    Returns:
        The findings its configuration enables and its directives leave standing. After a written
        fix, those are the findings of the rewritten file.
    """
    enabled = settings.enabled_for(path)
    try:
        module = read_module(path)
    except (SyntaxError, UnicodeDecodeError) as exc:
        line = getattr(exc, "lineno", None) or 1
        found = Diagnostic(path, line, 1, SYNTAX_ERROR, f"The module cannot be read: {exc}")
        return CheckResult(diagnostics=[found] if SYNTAX_ERROR in enabled else [])
    result = CheckResult()
    # As with ruff, a fix touches only a file with a finding it fixes: the reorder also lifts setup
    # beside the imports, which is not itself a PG105 finding and must not rewrite a clean file.
    if fix is not FixMode.OFF and ZONE_ORDER in enabled and layout.zone_offenders(module.tree):
        module = _apply_fix(module, fix, result)
    findings = [finding for finding in _findings(module, settings) if finding.code in enabled]
    standing = suppress(findings, directives(module.text), path)
    result.diagnostics = [finding for finding in standing if finding.code in enabled]
    return result


def _apply_fix(module: SourceModule, fix: FixMode, result: CheckResult) -> SourceModule:
    """Run the zone fix over one module, recording what it did.

    Args:
        module: The module as read.
        fix: Write or diff.
        result: The file's result, updated in place.

    Returns:
        The module the rules should read: rewritten after a written fix, the original otherwise.
    """
    fixed = fix_source(module.text, module.tree, str(module.path))
    result.refusals.extend(fixed.refusals)
    if not fixed.moved:
        return module
    result.moved += fixed.moved
    result.fixed_files += 1
    if fix is FixMode.DIFF:
        result.diffs.append(fixed.diff)
        return module
    module.path.write_text(fixed.text, encoding="utf-8")
    return SourceModule(module.path, fixed.text, ast.parse(fixed.text, filename=str(module.path)))


def _findings(module: SourceModule, settings: Settings) -> list[Diagnostic]:
    """Run every rule family over one module.

    Args:
        module: The parsed module.
        settings: Thresholds per family.

    Returns:
        Every finding, before selection and suppression.
    """
    return [
        *layout.check(module, settings.layout),
        *shared_state.check(module),
        *composition.check(module, settings.composition),
        *cohesion.check(module, settings.cohesion),
    ]
