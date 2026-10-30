from fastapi.testclient import TestClient

from train.api.api import create_app


def test_only_native_training_contracts_are_exposed(tmp_path):
    app = create_app(root=tmp_path)
    removed = {"/updates", "/batches/prepare", "/checkpoints/resolve"}
    paths = app.openapi()["paths"]
    assert not removed.intersection(paths)
    assert "/api/v1/create_session" in paths
    assert "/v1/loops/sessions" in paths
    with TestClient(app) as client:
        for path in removed:
            assert client.post(path, json={}).status_code in {404, 501}
