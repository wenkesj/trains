"""Generate code-reference pages from public Trains modules without importing SDKs."""

from pathlib import Path

import mkdocs_gen_files

NAV = []
ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "trains/trains").rglob("*.py")):
    parts = path.relative_to(ROOT / "trains").with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts[-1].startswith("_"):
        continue
    module = ".".join(parts)
    page = Path("modules", *parts).with_suffix(".md")
    with mkdocs_gen_files.open(page, "w") as output:
        output.write(f"# {module}\n\n::: {module}\n")
    NAV.append((module, page.as_posix()))
    mkdocs_gen_files.set_edit_path(page, str(path.relative_to(ROOT)))

with mkdocs_gen_files.open("index.md", "a") as output:
    output.write("\n## Modules\n\n")
    for module, page in NAV:
        output.write(f"- [`{module}`]({page})\n")
