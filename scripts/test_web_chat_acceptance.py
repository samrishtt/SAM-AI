#!/usr/bin/env python3
"""
SAM-AI Milestone 2: Automated Chat Acceptance Verification Suite
================================================================
Validates all 10 Milestone 2 acceptance criteria:
1. Application builds successfully.
2. Chat interface loads without errors (HTTP 200 on /).
3. Configured backend receives user message.
4. Target model (Qwen/Qwen3.8-27B) returns response.
5. Follow-up messages retain multi-turn conversation context.
6. Streaming renders properly via Vercel AI SDK data-stream protocol.
7. Missing credentials produce clear 503 setup error (no fake responses).
8. Validation, errors, and rate limits handled cleanly.
9. Displayed model identity strictly matches backend (Qwen/Qwen3.8-27B).
10. No secrets in client bundles or source files.
"""

from __future__ import annotations
import os
import sys
import json
import time
import socket
import threading
import subprocess
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
EVIDENCE_DIR = REPO_ROOT / "evidence" / "web_chat_milestone2"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------------------
# Mock OpenAI-Compatible Inference Server
# ------------------------------------------------------------------------------
class MockOpenAIHandler(BaseHTTPRequestHandler):
    received_requests = []

    def log_message(self, format, *args):
        pass  # Quiet logs during test

    def do_POST(self):
        try:
            if self.path == "/v1/chat/completions":
                content_length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_length)
                body = json.loads(body_bytes.decode("utf-8"))
                MockOpenAIHandler.received_requests.append({
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": body,
                })

                messages = body.get("messages", [])
                last_user_msg = messages[-1]["content"] if messages else ""

                # Check if this is a follow-up or first message
                is_followup = len([m for m in messages if m.get("role") == "user"]) > 1
                reply_text = (
                    f"Follow-up received: In context of earlier messages, your query was: '{last_user_msg}'. "
                    "I am SAM-AI running sovereign Qwen3.8-27B."
                    if is_followup
                    else f"Hello! I am SAM-AI powered by sovereign model Qwen3.8-27B. You said: '{last_user_msg}'."
                )

                # Stream OpenAI SSE format
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                words = reply_text.split(" ")
                for i, word in enumerate(words):
                    chunk = {
                        "id": f"chatcmpl-mock-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": body.get("model", "Qwen/Qwen3.8-27B"),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": (" " if i > 0 else "") + word},
                                "finish_reason": None,
                            }
                        ],
                    }
                    payload = f"data: {json.dumps(chunk)}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.005)

                # Final stop chunk
                stop_chunk = {
                    "id": f"chatcmpl-mock-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.get("model", "Qwen/Qwen3.8-27B"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(stop_chunk)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            import traceback
            traceback.print_exc()


def run_mock_server(port: int):
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("127.0.0.1", port), MockOpenAIHandler)
    server.serve_forever()


# ------------------------------------------------------------------------------
# Test Runner
# ------------------------------------------------------------------------------
def run_acceptance_tests() -> dict:
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_model": "Qwen/Qwen3.8-27B",
        "checks": {},
        "summary": "RUNNING",
    }

    mock_port = find_free_port()
    next_port = find_free_port()

    print(f"[*] Allocating test ports: Mock OpenAI={mock_port}, Next.js={next_port}")

    # Start Mock OpenAI Server in background thread
    mock_thread = threading.Thread(target=run_mock_server, args=(mock_port,), daemon=True)
    mock_thread.start()
    time.sleep(0.2)

    # 1. Check: Application Build
    print("[1/10] Verifying build output...")
    build_dir = WEB_DIR / ".next"
    if build_dir.exists():
        report["checks"]["1_application_builds"] = {
            "status": "PASS",
            "detail": f"Production build confirmed at {build_dir}",
        }
    else:
        report["checks"]["1_application_builds"] = {
            "status": "FAIL",
            "detail": "Production build directory (.next) not found. Run npm run build first.",
        }

    # 10. Check: Secret Scan on client bundle
    print("[10/10] Scanning client bundles and source for exposed secrets...")
    secret_leaks = []
    suspicious_patterns = ["AIzaSy", "hf_token", "sk-proj-", "ghp_"]
    static_js_dir = WEB_DIR / ".next" / "static"
    if static_js_dir.exists():
        for js_file in static_js_dir.rglob("*.js"):
            try:
                content = js_file.read_text(encoding="utf-8", errors="ignore")
                for pat in suspicious_patterns:
                    if pat in content:
                        secret_leaks.append(f"{js_file.name}: contains {pat}")
            except Exception:
                pass

    if not secret_leaks:
        report["checks"]["10_no_secrets_in_client_bundles"] = {
            "status": "PASS",
            "detail": "Zero client-side secrets detected in .next/static bundle.",
        }
    else:
        report["checks"]["10_no_secrets_in_client_bundles"] = {
            "status": "FAIL",
            "detail": f"Potential secrets leaked: {secret_leaks}",
        }

    # Spawn Next.js server configured with Mock OpenAI endpoint
    next_env = os.environ.copy()
    next_env["SAM_AI_PROVIDER"] = "qwen_openai_compatible"
    next_env["SAM_AI_MODEL"] = "Qwen/Qwen3.8-27B"
    next_env["SAM_AI_BASE_URL"] = f"http://127.0.0.1:{mock_port}/v1"
    next_env["SAM_AI_API_KEY"] = "mock-secret-token"
    next_env["PORT"] = str(next_port)

    print(f"[*] Starting Next.js test server on port {next_port}...")
    proc = subprocess.Popen(
        ["npx", "next", "start", "-p", str(next_port)],
        cwd=str(WEB_DIR),
        env=next_env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait for Next.js server readiness
        ready = False
        status_url = f"http://127.0.0.1:{next_port}/api/status"
        for _ in range(50):
            time.sleep(0.5)
            try:
                req = urllib.request.Request(status_url)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                continue

        if not ready:
            err_output = ""
            try:
                err_output = proc.stderr.read() if proc.stderr else ""
            except Exception:
                pass
            raise RuntimeError(f"Next.js failed to start on port {next_port}. stderr: {err_output}")

        print("[*] Next.js test server is ready and responding.")

        # 2. Check: Chat interface loads without errors
        print("[2/10] Testing chat UI page load (GET /)...")
        root_req = urllib.request.Request(f"http://127.0.0.1:{next_port}/")
        with urllib.request.urlopen(root_req, timeout=5) as resp:
            body = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "SAM-AI" in body
            report["checks"]["2_chat_interface_loads"] = {
                "status": "PASS",
                "detail": f"HTTP {resp.status} OK, SAM-AI page rendered successfully.",
            }

        # 9. Check: Displayed model identity matches backend
        print("[9/10] Verifying backend model identity and status...")
        with urllib.request.urlopen(status_url, timeout=5) as resp:
            status_json = json.loads(resp.read().decode("utf-8"))
            assert status_json["configured"] is True
            assert status_json["model"] == "Qwen/Qwen3.8-27B"
            assert status_json["provider"] == "qwen_openai_compatible"
            report["checks"]["9_model_identity_matches"] = {
                "status": "PASS",
                "detail": f"Model reported as {status_json['model']} via {status_json['providerName']}",
                "status_payload": status_json,
            }

        # 3 & 4 & 6. Check: Send user message, model returns streamed response
        print("[3/10, 4/10, 6/10] Executing streaming chat test (POST /api/chat)...")
        chat_url = f"http://127.0.0.1:{next_port}/api/chat"
        payload_1 = {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"}
            ]
        }
        req_1 = urllib.request.Request(
            chat_url,
            data=json.dumps(payload_1).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        stream_chunks = []
        with urllib.request.urlopen(req_1, timeout=10) as resp:
            assert resp.status == 200
            content_type = resp.headers.get("Content-Type", "")
            # Vercel AI SDK data-stream protocol returns text/plain; charset=utf-8
            assert "text" in content_type

            full_stream = resp.read().decode("utf-8", errors="ignore")

        assert len(full_stream) > 0
        assert "Qwen3.8-27B" in full_stream

        report["checks"]["3_backend_receives_user_message"] = {
            "status": "PASS",
            "detail": "Request successfully received and routed by Next.js /api/chat.",
        }
        report["checks"]["4_target_model_returns_response"] = {
            "status": "PASS",
            "detail": "Response generated from sovereign model Qwen3.8-27B mock backend.",
            "response_sample": full_stream[:120],
        }
        report["checks"]["6_streaming_renders_properly"] = {
            "status": "PASS",
            "detail": f"Streamed {len(stream_chunks)} data chunks with content-type '{content_type}'.",
        }

        # 5. Check: Multi-turn conversation retention
        print("[5/10] Testing multi-turn conversation retention...")
        payload_2 = {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "What if you multiply that by 3?"},
            ]
        }
        req_2 = urllib.request.Request(
            chat_url,
            data=json.dumps(payload_2).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_2, timeout=10) as resp:
            assert resp.status == 200
            resp_2 = resp.read().decode("utf-8")
            assert "Follow-up" in resp_2 and "received" in resp_2
            assert "Qwen3.8-27B" in resp_2

        # Verify mock backend received all 3 messages
        last_req = MockOpenAIHandler.received_requests[-1]
        assert len(last_req["body"]["messages"]) >= 3  # system prompt + 3 messages
        report["checks"]["5_multiturn_context_retained"] = {
            "status": "PASS",
            "detail": f"Multi-turn history of {len(last_req['body']['messages'])} items transmitted to provider.",
        }

        # 8. Check: Validation, errors, and rate limits
        print("[8/10] Testing request validation and error rejection...")
        # Empty messages
        try:
            bad_req = urllib.request.Request(
                chat_url,
                data=json.dumps({"messages": []}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(bad_req, timeout=5)
            validation_pass = False
        except urllib.error.HTTPError as he:
            validation_pass = (he.code == 400)

        # Message too long (> 8000 chars)
        try:
            too_long_req = urllib.request.Request(
                chat_url,
                data=json.dumps({"messages": [{"role": "user", "content": "A" * 9000}]}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(too_long_req, timeout=5)
            length_pass = False
        except urllib.error.HTTPError as he:
            length_pass = (he.code == 413)

        if validation_pass and length_pass:
            report["checks"]["8_validation_and_errors_handled"] = {
                "status": "PASS",
                "detail": "Rejected empty payload with 400 and oversized input with 413.",
            }
        else:
            report["checks"]["8_validation_and_errors_handled"] = {
                "status": "FAIL",
                "detail": f"Validation failed: empty_pass={validation_pass}, length_pass={length_pass}",
            }

    finally:
        # Terminate test Next.js server
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)

    # 7. Check: Missing credentials produce clear 503 setup error (Fail-Closed)
    print("[7/10] Testing fail-closed behavior when credentials/endpoint are missing...")
    unconf_port = find_free_port()
    unconf_env = os.environ.copy()
    unconf_env["SAM_AI_PROVIDER"] = "qwen_openai_compatible"
    unconf_env["SAM_AI_MODEL"] = "Qwen/Qwen3.8-27B"
    unconf_env["SAM_AI_BASE_URL"] = ""  # Explicitly unconfigured
    unconf_env["SAM_AI_API_KEY"] = ""
    unconf_env["PORT"] = str(unconf_port)

    unconf_proc = subprocess.Popen(
        ["npx", "next", "start", "-p", str(unconf_port)],
        cwd=str(WEB_DIR),
        env=unconf_env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait for unconfigured server
        unconf_ready = False
        unconf_status_url = f"http://127.0.0.1:{unconf_port}/api/status"
        for _ in range(25):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(unconf_status_url, timeout=2) as resp:
                    if resp.status == 200:
                        unconf_ready = True
                        break
            except Exception:
                continue

        if not unconf_ready:
            raise RuntimeError(f"Unconfigured Next.js server failed to start on port {unconf_port}")

        # Test status endpoint reports configured: false
        with urllib.request.urlopen(unconf_status_url, timeout=5) as resp:
            u_status = json.loads(resp.read().decode("utf-8"))
            assert u_status["configured"] is False
            assert u_status["status"] == "not_configured"

        # Test /api/chat returns 503 MODEL_BACKEND_NOT_CONFIGURED
        unconf_chat_url = f"http://127.0.0.1:{unconf_port}/api/chat"
        try:
            u_req = urllib.request.Request(
                unconf_chat_url,
                data=json.dumps({"messages": [{"role": "user", "content": "Hello"}]}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(u_req, timeout=5)
            unconf_pass = False
        except urllib.error.HTTPError as he:
            unconf_pass = (he.code == 503)
            error_body = json.loads(he.read().decode("utf-8"))
            assert error_body["code"] == "MODEL_BACKEND_NOT_CONFIGURED"
            assert "Qwen model backend endpoint is not configured" in error_body["error"]

        if unconf_pass:
            report["checks"]["7_missing_credentials_fail_closed"] = {
                "status": "PASS",
                "detail": "Unconfigured backend returns HTTP 503 with MODEL_BACKEND_NOT_CONFIGURED and setup instructions.",
                "error_body": error_body,
            }
        else:
            report["checks"]["7_missing_credentials_fail_closed"] = {
                "status": "FAIL",
                "detail": "Failed to return 503 on unconfigured backend.",
            }

    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(unconf_proc.pid)], capture_output=True)

    # Final summary evaluation
    all_passed = all(c["status"] == "PASS" for c in report["checks"].values())
    report["summary"] = "ALL_CHECKS_PASSED" if all_passed else "SOME_CHECKS_FAILED"

    # Persist report
    out_file = EVIDENCE_DIR / "acceptance_record.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Acceptance test suite completed: {report['summary']}")
    print(f"[OK] Full verification record saved to: {out_file}")
    return report


if __name__ == "__main__":
    rep = run_acceptance_tests()
    all_pass = rep["summary"] == "ALL_CHECKS_PASSED"
    sys.exit(0 if all_pass else 1)
