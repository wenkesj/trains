"""Reject development API content in public documentation artifacts."""
from pathlib import Path


def main() -> None:
    """Check the complete public build before serving or uploading it."""
    root = Path(__file__).parent / "dist"
    if not (root / "python-reference/index.json").is_file():
        raise SystemExit("Missing public Python reference")
    forbidden = ("127.0.0.1:8382", "/api/v1/create_session", "#/docs/reference/train", "local inference API", "local training API")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "openapi.json" or "ApiDocumentation-" in path.name:
            raise SystemExit(f"Development artifact in public build: {path}")
        if path.suffix in {".js", ".json", ".html", ".md"}:
            content = path.read_text()
            for value in forbidden:
                if value in content:
                    raise SystemExit(f"Development content {value!r} in {path}")
    print("Public documentation contains no development API artifacts.")


if __name__ == "__main__":
    main()
