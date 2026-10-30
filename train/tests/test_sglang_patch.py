import shutil

import pytest

from train._sglang_patch import _apply_patch

pytestmark = pytest.mark.skipif(shutil.which("patch") is None, reason="GNU patch is unavailable")


def test_patch_is_idempotent_and_rejects_incompatible_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    module = source / "module.py"
    module.write_text("before\n")
    patch = tmp_path / "compat.patch"
    patch.write_text("--- a/module.py\n+++ b/module.py\n@@ -1 +1 @@\n-before\n+after\n")
    _apply_patch(source, patch)
    assert module.read_text() == "after\n"
    _apply_patch(source, patch)
    assert module.read_text() == "after\n"
    module.write_text("incompatible\n")
    with pytest.raises(RuntimeError, match="pinned Compose image"):
        _apply_patch(source, patch)
    assert module.read_text() == "incompatible\n"


def test_full_patch_rejects_partial_application_before_mutation(tmp_path):
    for name, content in [("first.py", "before\n"), ("second.py", "after\n")]:
        (tmp_path / name).write_text(content)
    patch = tmp_path / "compat.patch"
    patch.write_text(
        "".join(
            f"--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-before\n+after\n"
            for name in ["first.py", "second.py"]
        )
    )
    with pytest.raises(RuntimeError):
        _apply_patch(tmp_path, patch)
    assert (tmp_path / "first.py").read_text() == "before\n"
    assert (tmp_path / "second.py").read_text() == "after\n"
