"""Verify native cached datasets are usable without contacting Hugging Face."""

import socket

import pytest
from datasets import Dataset

import run


def test_cached_split_survives_hf_unavailability(tmp_path, monkeypatch):
    original = Dataset.from_dict({"messages": [[{"role": "user", "content": "Question"}]]})
    downloads = []

    def download(*args, **kwargs):
        downloads.append(kwargs["revision"])
        return original

    monkeypatch.setattr(run, "load_dataset", download)
    first = run._cached_split("train_sft", tmp_path)

    def reject(*args, **kwargs):
        raise AssertionError("Cached split must not contact the network or HF loader")

    monkeypatch.setattr(run, "load_dataset", reject)
    monkeypatch.setattr(socket.socket, "connect", reject)
    cached = run._cached_split("train_sft", tmp_path, offline=True)
    assert cached.to_list() == first.to_list() == original.to_list()
    assert downloads == [run.REVISION]


def test_offline_cache_miss_fails_before_download(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Offline cache misses must not download")

    monkeypatch.setattr(run, "load_dataset", reject)
    with pytest.raises(ValueError, match="run.*cache"):
        run._cached_split("test_sft", tmp_path, offline=True)
