import httpx


def test_sampling_retries_preserve_operation_and_sequence_ids(training_server, poll):
    with httpx.Client(base_url=training_server) as client:
        session = client.post(
            "/api/v1/create_sampling_session", json={"base_model": "tiny"}
        ).json()["sampling_session_id"]
        body = {
            "sampling_session_id": session,
            "seq_id": 0,
            "num_samples": 1,
            "prompt": {"chunks": [{"tokens": [1, 4]}]},
            "sampling_params": {"max_tokens": 2, "seed": 42},
        }
        first = client.post("/api/v1/asample", json=body).json()
        assert client.post("/api/v1/asample", json=body).json() == first
        assert client.post("/api/v1/asample", json={**body, "num_samples": 2}).status_code == 422
        assert len(poll(client, first["request_id"])["sequences"]) == 1
        headers = {"Idempotency-Key": "one-sample"}
        request = {"model": "tiny", "prompt": [1, 4], "n": 1, "max_tokens": 2}
        queued = client.post("/sample", json=request, headers=headers)
        assert queued.status_code == 202
        assert client.post("/sample", json=request, headers=headers).json() == queued.json()
        assert client.post("/sample", json={**request, "n": 2}, headers=headers).status_code == 422
