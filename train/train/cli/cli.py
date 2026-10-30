from pathlib import Path

import typer
import uvicorn

from train.api.api import create_app
from train.runtime import Runtime

app = typer.Typer()


@app.callback()
def main() -> None:
    """Run local LoRA training services."""


@app.command()
def serve(
    model: list[str] = typer.Option([], help="Registered name=/local/model/path"),
    root: Path = Path("var/train"),
    host: str = "127.0.0.1",
    port: int = 8382,
) -> None:
    """Run the training API with registered local model snapshots."""
    models = {}
    for entry in model:
        name, separator, path = entry.partition("=")
        if not separator or not name or not Path(path).is_dir():
            raise typer.BadParameter("Use --model name=/existing/local/model/path")
        models[name] = Path(path)
    uvicorn.run(create_app(models, root, Runtime()), host=host, port=port)


if __name__ == "__main__":
    app()
