"""OpenAI-Compatible REST API Server for Sovereign AI Platform.

Exposes standard OpenAI v1 endpoints enabling drop-in integration with:
- Cursor / VS Code (Continue.dev)
- LangChain, LlamaIndex, LiteLLM
- Custom enterprise applications and client SDKs

Endpoints:
- POST /v1/chat/completions
- GET /v1/models
- POST /v1/embeddings
"""

from __future__ import annotations
import http.server
import json
import os
import socketserver
import sys
import time
import uuid
from pathlib import Path

# Ensure root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.chat.repl import InteractiveAgiREPL

PORT = 8000
repl = InteractiveAgiREPL()


class OpenAICompatibleHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Quiet logging

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models" or self.path == "/models":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "object": "list",
                "data": [
                    {
                        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "sam-ai-frontier",
                        "permission": [],
                        "root": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        "parent": None,
                    },
                    {
                        "id": "sam-ai-r1-reasoning",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "sam-ai-frontier",
                        "permission": [],
                        "root": "sam-ai-r1-reasoning",
                        "parent": None,
                    },
                    {
                        "id": "hyper-astra-3.5",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "sovereign-ai-corp",
                        "permission": [],
                        "root": "hyper-astra-3.5",
                        "parent": None,
                    },
                    {
                        "id": "sovereign-transformer-120k",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "sovereign-ai-corp",
                        "permission": [],
                        "root": "sovereign-transformer-120k",
                        "parent": None,
                    },
                ],
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif self.path == "/" or self.path == "/health":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "active", "service": "Sovereign AI Engine API"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            req_data = json.loads(body) if body else {}
        except Exception:
            req_data = {}

        if self.path == "/v1/chat/completions" or self.path == "/chat/completions":
            messages = req_data.get("messages", [])
            user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                    break

            model_name = req_data.get("model", "hyper-astra-3.5")
            stream = req_data.get("stream", False)

            # Process command through cognitive engine
            reply_text = repl.process_command(user_msg) if user_msg else "Hello! How can I assist you today?"
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created_ts = int(time.time())

            response_payload = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": reply_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_msg.split()) * 2,
                    "completion_tokens": len(reply_text.split()) * 2,
                    "total_tokens": (len(user_msg.split()) + len(reply_text.split())) * 2,
                },
            }

            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode("utf-8"))

        elif self.path == "/v1/embeddings" or self.path == "/embeddings":
            input_text = req_data.get("input", "")
            if isinstance(input_text, list):
                input_text = " ".join(input_text)
            vec = repl.engine.episodic_store.embedder.embed(input_text).tolist()

            resp = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": vec,
                        "index": 0,
                    }
                ],
                "model": "sovereign-embed-64",
                "usage": {"prompt_tokens": len(input_text.split()), "total_tokens": len(input_text.split())},
            }
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()


class ThreadedAPIServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_api_server():
    server = ThreadedAPIServer(("127.0.0.1", PORT), OpenAICompatibleHandler)
    print(f"[*] OpenAI-Compatible REST API Server active at: http://127.0.0.1:{PORT}")
    print("[*] Compatible with Cursor, Continue.dev, LangChain, and OpenAI Python SDK.")
    server.serve_forever()


if __name__ == "__main__":
    start_api_server()
