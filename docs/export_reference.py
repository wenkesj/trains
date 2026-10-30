"""Export static mkdocstrings content for the shared developer docs shell."""

import argparse
import html
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonConsoleLexer, PythonLexer

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "var/docs-reference"



class Outline(HTMLParser):
    """Extract public object headings without loading provider SDKs."""

    def __init__(self) -> None:
        super().__init__()
        self.objects: list[dict] = []
        self.current: dict | None = None
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record object heading IDs and their hierarchy."""
        values = dict(attrs)
        if tag in {"h2", "h3", "h4", "h5", "h6"} and values.get("id"):
            self.current = {"id": values["id"], "level": int(tag[1]), "name": ""}

    def handle_data(self, data: str) -> None:
        """Collect readable heading labels and searchable content."""
        if self.current is not None:
            self.current["name"] += data
        self.text.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Finish a public object heading."""
        if self.current is not None and tag == f"h{self.current['level']}":
            self.current["name"] = self.current["name"].strip()
            self.objects.append(self.current)
            self.current = None


def route(module: str, object_id: str = "") -> str:
    """Return an application route for a Python module or object."""
    return f"#/docs/reference/python/{module}" + (f"/{object_id}" if object_id else "")


def highlighted_code(match: re.Match[str]) -> str:
    """Highlight generated Python code with the shared reference color scheme."""
    classes = match[1] or ""
    text = html.unescape(re.sub(r"<[^>]*>", "", match[2]))
    lexer = (
        PythonConsoleLexer()
        if "pycon" in classes or text.lstrip().startswith(">>>")
        else PythonLexer()
    )
    tokens = highlight(text, lexer, HtmlFormatter(nowrap=True))
    signature = " doc-signature" if "doc-signature" in classes else ""
    return f'<div class="highlight{signature}"><pre><code>{tokens}</code></pre></div>'


def export(output: Path) -> None:
    """Write a navigation manifest and independently loadable module content."""
    destination_root = output / "python-reference"
    destination_root.mkdir(parents=True, exist_ok=True)
    pages = sorted((GENERATED / "modules").rglob("index.html"))
    if not pages:
        raise ValueError("No Python reference pages were generated")
    modules = {
        ".".join(page.relative_to(GENERATED / "modules").parts[:-1]): page
        for page in pages
    }
    targets = {
        page.relative_to(GENERATED).as_posix(): module
        for module, page in modules.items()
    }
    manifest = []
    for module, page in modules.items():
        match = re.search(
            r"<article data-python-reference>(.*?)</article>",
            page.read_text(),
            re.DOTALL,
        )
        if match is None:
            raise ValueError(f"Missing generated content for {module}")
        content = re.sub(r"<h1\b[^>]*>.*?</h1>", "", match[1], count=1, flags=re.DOTALL)

        def link(match: re.Match[str], page: Path = page) -> str:
            """Keep generated cross-references inside the documentation shell."""
            href = html.unescape(match[1])
            target = urlsplit(
                urljoin(
                    "https://reference.invalid/"
                    + page.relative_to(GENERATED).as_posix(),
                    href,
                )
            )
            if target.netloc != "reference.invalid":
                return match[0]
            path = unquote(target.path).lstrip("/")
            if path.endswith("/"):
                path += "index.html"
            owner = targets.get(path)
            if owner is None:
                return match[0]
            return (
                'href="'
                + html.escape(route(owner, unquote(target.fragment)), quote=True)
                + '"'
            )

        content = re.sub(r'href="([^"]*)"', link, content)
        content = re.sub(
            r'<pre><code(?: class="([^"]*)")?>(.*?)</code></pre>',
            highlighted_code,
            content,
            flags=re.DOTALL,
        )
        parser = Outline()
        parser.feed(content)
        objects = [item for item in parser.objects if item["id"] != module]
        first_paragraph = re.search(r"<p>(.*?)</p>", content, re.DOTALL)
        summary = (
            html.unescape(re.sub(r"<[^>]+>", "", first_paragraph[1])).strip()
            if first_paragraph
            else ""
        )
        source = ROOT / "trains" / Path(*module.split("."))
        source = (
            source.with_suffix(".py")
            if source.with_suffix(".py").is_file()
            else source / "__init__.py"
        )
        manifest.append(
            {
                "id": module,
                "summary": summary,
                "objects": objects,
                "source": source.relative_to(ROOT).as_posix(),
            }
        )
        (destination_root / f"{module}.json").write_text(
            json.dumps({"id": module, "html": content})
        )
    (destination_root / "index.json").write_text(json.dumps({"modules": manifest}))
    # Preserve the previous reference entry point without shipping a second theme.
    legacy = output / "reference"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "index.html").write_text(
        '<!doctype html><meta http-equiv="refresh" content="0;url=../#/docs/reference/python"><a href="../#/docs/reference/python">Python code reference</a>'
    )
    for module, page in modules.items():
        destination = legacy / page.relative_to(GENERATED)
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = (
            os.path.relpath(output, destination.parent)
            + "/"
            + route(module)
        )
        destination.write_text(
            f'<!doctype html><meta http-equiv="refresh" content="0;url={target}"><a href="{target}">Python code reference</a>'
        )
    print(f"Exported {len(manifest)} modules into the developer docs shell.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs/dist")
    export(parser.parse_args().output.resolve())
