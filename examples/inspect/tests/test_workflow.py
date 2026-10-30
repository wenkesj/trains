"""Exercise Inspect's native Tinker-compatible provider and log writer against local HTTP."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate
from trains.datasets.inspect import _records_from_log
from trains.datum.adapters.tinker import TinkerDatumAdapter


def test_native_chat_call_and_tinker_datum_share_environment(tmp_path, monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "yes"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("TINKER_API_KEY", "local")
    try:
        logs = eval(
            Task(
                dataset=[Sample(input="Say yes", target="yes")],
                solver=generate(),
                scorer=match(),
            ),
            model="openai-api/tinker/test-model",
            model_base_url=f"http://127.0.0.1:{server.server_port}/v1",
            log_dir=str(tmp_path),
            log_model_api=True,
            display="none",
        )
        assert logs[0].status == "success"
        rows = list(_records_from_log(Path(logs[0].location)))
        assert rows[0]["messages"][-1]["content"] == "yes"
        assert requests[0]["messages"][-1]["content"] == "Say yes"
        datum = TinkerDatumAdapter().from_tokens([1, 2], [3, 4])
        assert datum.loss_fn_inputs["target_tokens"].data == [2, 3, 4]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
