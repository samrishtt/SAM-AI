"""Unit tests for the OpenAI-Compatible REST API Server."""

import io
import json
import pytest
from api_server import OpenAICompatibleHandler, repl


class MockSocket:
    def __init__(self, data: bytes):
        self.rfile = io.BytesIO(data)
        self.wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if "r" in mode:
            return self.rfile
        return self.wfile


class DummyHandler(OpenAICompatibleHandler):
    def __init__(self, method: str, path: str, body: dict = None):
        self.command = method
        self.path = path
        self.headers = {}
        payload = json.dumps(body).encode("utf-8") if body else b""
        self.headers["Content-Length"] = str(len(payload))
        self.rfile = io.BytesIO(payload)
        self.wfile = io.BytesIO()

    def send_response(self, code, message=None):
        self.response_code = code

    def send_header(self, keyword, value):
        pass

    def end_headers(self):
        pass


def test_api_server_models_endpoint():
    handler = DummyHandler("GET", "/v1/models")
    handler.do_GET()
    assert handler.response_code == 200

    raw = handler.wfile.getvalue().decode("utf-8")
    data = json.loads(raw)
    assert data["object"] == "list"
    assert any(m["id"] == "hyper-astra-3.5" for m in data["data"])


def test_api_server_chat_completions_endpoint():
    body = {
        "model": "hyper-astra-3.5",
        "messages": [{"role": "user", "content": "hello"}],
    }
    handler = DummyHandler("POST", "/v1/chat/completions", body=body)
    handler.do_POST()
    assert handler.response_code == 200

    raw = handler.wfile.getvalue().decode("utf-8")
    data = json.loads(raw)
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "content" in data["choices"][0]["message"]
    assert "usage" in data


def test_api_server_embeddings_endpoint():
    body = {"input": "test embedding representation"}
    handler = DummyHandler("POST", "/v1/embeddings", body=body)
    handler.do_POST()
    assert handler.response_code == 200

    raw = handler.wfile.getvalue().decode("utf-8")
    data = json.loads(raw)
    assert data["object"] == "list"
    assert len(data["data"][0]["embedding"]) == 64
