import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field

from train.backend import LocalTrainer
from train.models import Model, ModelSpec


class Checkpoint(Model):
    """A named checkpoint alias pointing at an immutable physical snapshot."""

    uri: str
    directory: Path
    step: int
    implementation: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def publish(
        cls,
        root: Path,
        model_id: str,
        name: str,
        operation_id: str,
        engine: LocalTrainer,
        sampler: bool,
        overwrite: bool,
    ) -> "Checkpoint":
        """Save an immutable sampler snapshot and its named alias."""
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
            raise ValueError("Checkpoint name must be a single path component")
        folder = "sampler_weights" if sampler else "weights"
        parent = root / model_id / folder
        if (parent / name / "model.json").is_file():
            raise ValueError("Checkpoint name conflicts with an immutable snapshot")
        alias = parent / f"{name}.json"
        if alias.exists() and not overwrite:
            raise ValueError("Checkpoint name already exists")
        directory = parent / f"{name}-{operation_id}"
        engine.save(directory, sampler=sampler)
        record = cls(
            uri=f"tinker://{model_id}/{folder}/{name}",
            directory=directory,
            step=engine.step,
            implementation=engine.kernel_report,
        )
        record.save(directory / "checkpoint.json")
        record.save(alias)
        return record

    @classmethod
    def resolve(cls, root: Path, uri: str, models: dict[str, Path]) -> Path:
        """Resolve a native checkpoint reference to an immutable snapshot."""
        if uri.startswith("local/run-"):
            _, owner, name = uri.split("/")
            if not re.fullmatch(r"run-[0-9a-f]{32}", owner) or not re.fullmatch(
                r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name
            ):
                raise ValueError("Invalid Firetitan checkpoint identity")
            owner = owner.removeprefix("run-")
            for folder in ("weights", "sampler_weights"):
                path = root / owner / folder / name
                if (path / "model.json").is_file():
                    uri = f"tinker://{owner}/{folder}/{name}"
                    break
        parsed = urlparse(uri)
        owner = parsed.netloc
        parts = parsed.path.strip("/").split("/")
        if parsed.scheme == "bt" and owner.startswith("loops:"):
            owner = owner.removeprefix("loops:")
            parts = ["weights" if part == "trainer_weights" else part for part in parts]
        elif parsed.scheme != "tinker":
            raise ValueError("Expected a local tinker:// or bt://loops: checkpoint URI")
        if len(parts) != 2 or parts[0] not in {"weights", "sampler_weights"}:
            raise ValueError("Invalid checkpoint URI")
        if any(not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", part) for part in [owner, *parts]):
            raise ValueError("Invalid checkpoint path")
        path = root / owner / parts[0] / parts[1]
        alias = path.with_name(f"{path.name}.json")
        if alias.is_file():
            path = cls.load(alias).directory
        path = path.resolve()
        if not path.is_relative_to(root) or not (path / "model.json").is_file():
            raise KeyError("Checkpoint not found")
        spec = ModelSpec.load(path / "model.json")
        if models.get(spec.base_model) != spec.base_path:
            raise ValueError("Checkpoint base model is not registered")
        return path

    @classmethod
    def listing(
        cls, root: Path, models: dict[str, Path], run_id: str | None = None
    ) -> list["Checkpoint"]:
        """List saved checkpoint identities and their metadata."""
        if run_id and not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", run_id):
            raise ValueError("Invalid run ID")
        records = []
        for folder in ("weights", "sampler_weights"):
            for path in sorted(root.glob(f"{run_id or '*'}/{folder}/*.json")):
                record = cls.load(path)
                record.directory = cls.resolve(root, record.uri, models)
                records.append(record)
        return records

    def metadata(self) -> dict:
        """Read checkpoint metadata without loading model weights."""
        spec = ModelSpec.load(self.directory / "model.json")
        owner, folder, _ = self.directory.parts[-3:]
        return {
            "id": self.directory.name,
            "run_id": owner,
            "checkpoint_id": self.uri.rsplit("/", 1)[-1],
            "checkpoint_type": "lora",
            "created_at": datetime.fromtimestamp(self.directory.stat().st_mtime, UTC).isoformat(),
            "base_model": spec.base_model,
            "lora_adapter_config": {"r": spec.rank},
            "target": "trainer" if folder == "weights" else "sampler",
            "size_bytes": sum(
                path.stat().st_size for path in self.directory.iterdir() if path.is_file()
            ),
        }
