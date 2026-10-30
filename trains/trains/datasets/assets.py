import base64
import hashlib
import mimetypes
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, IterableDataset, Value

ASSET_FEATURE = [{"id": Value("string"), "mime_type": Value("string"), "data": Value("binary")}]


def read_asset(uri: str) -> tuple[bytes, str]:
    """Read explicit local/data assets; applications resolve authenticated remote URIs."""
    if uri.startswith("data:"):
        header, encoded = uri.split(",", 1)
        if not header.endswith(";base64"):
            raise ValueError("Media data URIs must use base64")
        return base64.b64decode(encoded, validate=True), header[5:-7]
    if "://" in uri and not uri.startswith("file://"):
        raise ValueError("Remote media requires an application-supplied resolver")
    path = Path(uri.removeprefix("file://"))
    return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def materialize_assets(
    dataset: Dataset | IterableDataset, *, resolver: Callable[[str], tuple[bytes, str]] = read_asset
) -> Dataset | IterableDataset:
    """Embed content-addressed media bytes in native HF storage for portable snapshots.

    Args:
        dataset: HF dataset or iterable of complete model-visible conversation records.
        resolver: Callback resolving asset identities into portable media bytes.

    Returns:
        A dataset whose media can be saved and replayed without the original files.
    """
    if dataset.features is None:
        raise ValueError("Asset materialization requires explicit HF dataset features")
    features = Features({**dataset.features, "assets": ASSET_FEATURE})
    return dataset.map(lambda row: _snapshot(row, resolver), features=features)


def resolve_assets(record: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve every media reference in a conversation before serialization."""
    assets = {asset["id"]: asset for asset in record.get("assets") or []}

    def visit(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("asset://"):
            identifier = value.removeprefix("asset://")
            if identifier not in assets:
                raise ValueError("Missing materialized media asset: " + identifier)
            asset = assets[identifier]
            if hashlib.sha256(asset["data"]).hexdigest() != identifier:
                raise ValueError("Media asset hash mismatch")
            return (
                "data:" + asset["mime_type"] + ";base64," + base64.b64encode(asset["data"]).decode()
            )
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        return value

    return visit(record)


def _snapshot(
    row: Mapping[str, Any], resolver: Callable[[str], tuple[bytes, str]]
) -> dict[str, Any]:
    """Embed a referenced media asset as portable bytes and metadata."""
    assets = {asset["id"]: asset for asset in row.get("assets") or []}

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: visit(item) for key, item in value.items()}
        if value.get("type") in {
            "image",
            "image_url",
            "input_image",
            "audio",
            "input_audio",
            "video",
            "video_url",
            "input_video",
        }:
            for key in ["url", "image", "audio", "video", "image_url", "video_url"]:
                uri = value.get(key)
                if isinstance(uri, dict):
                    uri = uri.get("url")
                if isinstance(uri, str) and not uri.startswith("asset://"):
                    data, mime = resolver(uri)
                    identifier = hashlib.sha256(data).hexdigest()
                    assets[identifier] = {"id": identifier, "mime_type": mime, "data": data}
                    result[key] = (
                        {"url": "asset://" + identifier}
                        if isinstance(value.get(key), dict)
                        else "asset://" + identifier
                    )
        return result

    output = {key: visit(value) for key, value in row.items() if key != "assets"}
    return {**output, "assets": list(assets.values())}
