import base64

import pytest

from datasets import load_from_disk
from trains.datasets import chat_dataset_from_records, materialize_assets
from trains.datasets.assets import resolve_assets


def test_snapshot_is_portable_and_hash_verified(tmp_path):
    content = b"original-media-bytes"
    source = tmp_path / "image.png"
    source.write_bytes(content)
    chats = chat_dataset_from_records(
        [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image", "image": str(source)}]},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ]
    )
    saved = materialize_assets(chats)
    saved.save_to_disk(tmp_path / "saved")
    source.unlink()
    row = load_from_disk(tmp_path / "saved")[0]
    resolved = resolve_assets(row)
    assert resolved["messages"][:-1][0]["content"][0]["image"].endswith(
        base64.b64encode(content).decode()
    )
    row["assets"][0]["data"] = b"changed"
    with pytest.raises(ValueError, match="hash"):
        resolve_assets(row)


def test_remote_assets_require_explicit_resolver():
    chats = chat_dataset_from_records(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "https://private/image.png"}}
                        ],
                    },
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ]
    )
    with pytest.raises(ValueError, match="resolver"):
        materialize_assets(chats)
    saved = materialize_assets(chats, resolver=lambda uri: (b"png", "image/png"))
    assert len(saved[0]["assets"]) == 1
