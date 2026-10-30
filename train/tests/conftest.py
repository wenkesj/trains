import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch
import uvicorn
from transformers import LlamaConfig, LlamaForCausalLM

from train.api.api import create_app
from train.runtime import Runtime
from train.serving import Serving


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "base"
    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    model.save_pretrained(path)
    return path


@pytest.fixture
def training_server(model_path, tmp_path, vllm):
    app = create_app({"tiny": model_path}, tmp_path / "rpc", Runtime(serving=Serving(url=vllm[0])))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
    assert not thread.is_alive()


@pytest.fixture
def vllm():
    state = {"loaded": {}, "calls": [], "fail_load": False, "fail_inference": False}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200, content_type="application/json"):
            data = json.dumps(value).encode() if content_type == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state["calls"].append((self.path, None, self.headers.get("Authorization")))
            self.reply({"data": [{"id": "tiny"}, *({"id": name} for name in state["loaded"])]})

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.loads(data) if data else {}
            state["calls"].append((self.path, body, self.headers.get("Authorization")))
            if self.path.startswith(("/sleep", "/wake_up")):
                self.reply({})
            elif self.path == "/v1/load_lora_adapter":
                if state["fail_load"]:
                    self.reply({"error": "load failed"}, 500)
                else:
                    state["loaded"][body["lora_name"]] = body["lora_path"]
                    self.reply({})
            elif self.path == "/v1/unload_lora_adapter":
                state["loaded"].pop(body["lora_name"])
                self.reply({})
            elif state["fail_inference"]:
                self.reply({"error": {"message": "invalid request"}}, 400)
            elif body.get("stream"):
                self.reply(
                    b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\ndata: [DONE]\n\n',
                    content_type="text/event-stream",
                )
            else:
                self.reply(
                    {
                        "id": "test",
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "text": "ok",
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
