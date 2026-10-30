import argparse
import json
from pathlib import Path

from train.api.api import create_app


def main():
    parser = argparse.ArgumentParser(description="Export the local training API schema.")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    path = Path(__file__).parent / "development" / "openapi.json"
    schema = create_app().openapi()
    if arguments.check:
        if not path.exists() or json.loads(path.read_text()) != schema:
            raise SystemExit("Training OpenAPI schema differs from the documentation snapshot")
        print("Training OpenAPI snapshot matches")
    else:
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
