# pygate

A Python style checker for structural rules that complement Ruff.
Opinionated rules for declaration order, shared state, composition, and cohesion.

## Usage

After the [workspace setup](../README.md#contributing), run:

```bash
.venv/bin/pygate check <name>
```

Configure rules under `[tool.pygate]` in a member's `pyproject.toml`. Unknown configuration keys are errors. Use `# noqa: PG...` for a targeted suppression; `PG001` reports unused suppressions.

`--fix` can reorder top-level declarations for `PG105` when the change is a pure permutation. Review generated changes before committing them.

## Development

From the checkout root:

```bash
.venv/bin/ruff check pygate
.venv/bin/pygate check pygate
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest pygate/tests -q
CUDA_VISIBLE_DEVICES="" .venv/bin/ty check --project pygate --python .venv
```

For a new rule, register its code in `registry.py`, add its check to the corresponding rule family, and add focused tests under `tests/rules/`. Preserve existing rule numbers. Return diagnostics at the location where a suppression belongs, and put positive integer thresholds in the family settings.

## Architecture

The checker uses only the Python standard library. It reads files and settings, optionally reorders declarations, runs the enabled rules, applies suppressions, and reports diagnostics.

Rule families live in `rules/`: layout (`PG1xx`), shared state (`PG2xx`), composition (`PG3xx`), and cohesion (`PG4xx`). `PG001` handles unused suppressions and cannot itself be suppressed.
