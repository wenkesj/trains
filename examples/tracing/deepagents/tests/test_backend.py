import os

import pytest

from deepagents_tracing.agent import DockerBackend
from deepagents_tracing.sandbox import Sandbox


@pytest.mark.skipif(os.getenv("TRAINS_TEST_DOCKER") != "1", reason="requires Docker")
def test_native_filesystem_tools_edit_only_the_container_workspace(tmp_path):
    # Exercise the real Deep Agents implementation, including its transfer hooks.
    with Sandbox(tmp_path) as sandbox:
        backend = DockerBackend(sandbox)
        path = "/workspace/quoted' name.py"
        assert backend.upload_files([(path, b"answer = 42\n")])[0].error is None
        assert backend.download_files([path])[0].content == b"answer = 42\n"
        read = backend.read(path)
        assert read.error is None
        assert read.file_data is not None
        assert read.file_data["content"] == "answer = 42"
        assert backend.execute("python3 -c 'print(6 * 7)'").output.strip() == "42"
        assert backend.execute("printf forbidden > /etc/forbidden").exit_code != 0
    assert (tmp_path / "quoted' name.py").read_text() == "answer = 42\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["quoted' name.py"]
