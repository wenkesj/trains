# Rendering findings. The concise format is ruff's, so an editor or a script that reads one reads
# the other; the JSON format carries the same fields ruff's does under the same names.

from __future__ import annotations

import json
from collections.abc import Sequence

from pygate.diagnostic import Diagnostic


def render_concise(diagnostics: Sequence[Diagnostic]) -> str:
    """Render one line per finding and a closing count.

    Args:
        diagnostics: Findings, already sorted.

    Returns:
        The report.
    """
    if not diagnostics:
        return "All checks passed!"
    lines = [f"{d.path}:{d.line}:{d.column}: {d.code} {d.message}" for d in diagnostics]
    noun = "error" if len(diagnostics) == 1 else "errors"
    return "\n".join([*lines, f"Found {len(diagnostics)} {noun}."])


def render_json(diagnostics: Sequence[Diagnostic]) -> str:
    """Render findings as a JSON array.

    Args:
        diagnostics: Findings, already sorted.

    Returns:
        The array, one object per finding.
    """
    return json.dumps(
        [
            {
                "code": d.code,
                "message": d.message,
                "filename": str(d.path),
                "location": {"row": d.line, "column": d.column},
            }
            for d in diagnostics
        ],
        indent=2,
    )
