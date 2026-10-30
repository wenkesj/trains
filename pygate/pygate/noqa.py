# Suppression directives: the per-line noqa comment ruff also reads, codes after a colon, and the
# per-file `pygate: noqa` comment for rules that report a whole module. A directive that suppresses
# nothing is itself a finding, which is what stops an exemption outliving the defect that earned it.

from __future__ import annotations

import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

from pygate.diagnostic import Diagnostic
from pygate.registry import RULES_BY_CODE

CODE_LIST = r"(?P<codes>[A-Z]+[0-9]+(?:[\s,]+[A-Z]+[0-9]+)*)"

LINE_DIRECTIVE = re.compile(rf"#\s*noqa(?::\s*{CODE_LIST})?", re.IGNORECASE)

FILE_DIRECTIVE = re.compile(rf"^#\s*pygate\s*:\s*noqa(?::\s*{CODE_LIST})?", re.IGNORECASE)

#: A code this checker owns. Anchored and exact, because ruff's own prefixes include `PGH`, and a
#: noqa naming PGH003 on the same line is ruff's business, not an unused directive of ours.
OWN_CODE = re.compile(r"PG[0-9]{3}")

UNUSED_CODE = "PG001"


@dataclass(frozen=True)
class Directive:
    """One suppression comment.

    Attributes:
        line: Line the comment sits on.
        column: 0-based column of the comment.
        file_level: Whether it covers the whole file rather than its own line.
        codes: Upper-cased codes it names, or ``None`` for a blanket directive.
    """

    line: int
    column: int
    file_level: bool
    codes: frozenset[str] | None

    def covers(self, diagnostic: Diagnostic) -> bool:
        """Decide whether this directive suppresses a finding.

        Args:
            diagnostic: The finding.

        Returns:
            Whether the finding is on this directive's line, or the directive is file-level, and
            the directive is blanket or names the finding's code.
        """
        if not self.file_level and diagnostic.line != self.line:
            return False
        return self.codes is None or diagnostic.code in self.codes


def directives(text: str) -> list[Directive]:
    """Find every suppression comment in a module.

    Args:
        text: Module source.

    Returns:
        The directives, in source order. Text that merely looks like one inside a string is not
        a comment and is not returned.
    """
    lines = text.splitlines()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return []
    found: list[Directive] = []
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        row, column = token.start
        own_line = not lines[row - 1][:column].strip()
        directive = _parse(token.string, row, column, own_line)
        if directive is not None:
            found.append(directive)
    return found


def suppress(diagnostics: list[Diagnostic], found: list[Directive], path: Path) -> list[Diagnostic]:
    """Drop the findings a directive covers, and report the directives that covered none.

    Args:
        diagnostics: Findings of the rules enabled for the file.
        found: The file's directives.
        path: The file, for the unused-directive findings.

    Returns:
        Surviving findings, then one ``PG001`` per pygate code a directive names in vain.
    """
    kept: list[Diagnostic] = []
    used: set[tuple[Directive, str]] = set()
    for diagnostic in diagnostics:
        covering = [directive for directive in found if directive.covers(diagnostic)]
        if not covering:
            kept.append(diagnostic)
        used.update((directive, diagnostic.code) for directive in covering)
    return kept + _unused(found, used, path)


def _parse(comment: str, row: int, column: int, own_line: bool) -> Directive | None:
    """Read one comment as a directive, if it is one.

    Args:
        comment: The comment token, ``#`` included.
        row: Its line.
        column: Its column.
        own_line: Whether nothing but whitespace precedes it on its line.

    Returns:
        The directive, or ``None`` when the comment is not one.
    """
    file_match = FILE_DIRECTIVE.match(comment) if own_line else None
    match = file_match or LINE_DIRECTIVE.search(comment)
    if match is None:
        return None
    listed = match.group("codes")
    codes = None if listed is None else frozenset(re.split(r"[\s,]+", listed.upper()))
    return Directive(line=row, column=column, file_level=file_match is not None, codes=codes)


def _unused(
    found: list[Directive], used: set[tuple[Directive, str]], path: Path
) -> list[Diagnostic]:
    """Report every pygate code a directive names without suppressing it.

    Args:
        found: The file's directives.
        used: Each directive paired with a code it actually suppressed.
        path: The file.

    Returns:
        One ``PG001`` per unused code.
    """
    return [
        Diagnostic(
            path,
            directive.line,
            directive.column + 1,
            UNUSED_CODE,
            f"Unused `noqa` directive: {code} "
            + ("suppresses nothing here" if code in RULES_BY_CODE else "is not a pygate rule"),
        )
        for directive in found
        if directive.codes is not None
        for code in sorted(directive.codes)
        if OWN_CODE.fullmatch(code) and (directive, code) not in used
    ]
