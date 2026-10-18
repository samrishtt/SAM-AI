"""
SAM-AI Phase 3B: Web Research Engine (Stage 1 & Stage 2) Test Suite
===================================================================
Verifies:
1. Stage 1: Web search provider abstraction, MockSearchProvider, SearXNGSearchProvider, fail-closed handling.
2. Stage 2: Server-side SafeWebFetcher, SSRF firewall, DNS resolution validation, pinned transport, streaming byte bounding, HTML sanitization, and untrusted research envelopes.
3. Execution of full 14-test JavaScript/TypeScript suite via Node.js --experimental-strip-types.
4. Clean separation of server-only research utilities from browser bundles.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
RESEARCH_DIR = WEB_DIR / "src" / "lib" / "research"
TEST_SCRIPT = WEB_DIR / "tests" / "test_research_engine.mjs"


def test_research_module_files_exist():
    """Verify that all required Phase 3B Stage 1 and Stage 2 modules exist."""
    required_files = [
        RESEARCH_DIR / "types.ts",
        RESEARCH_DIR / "ssrf-guard.ts",
        RESEARCH_DIR / "search-provider.ts",
        RESEARCH_DIR / "safe-fetcher.ts",
        RESEARCH_DIR / "index.ts",
        TEST_SCRIPT,
    ]
    for file_path in required_files:
        assert file_path.exists(), f"Required Phase 3B file missing: {file_path}"


def test_research_engine_node_acceptance_suite():
    """Execute the full 14-test acceptance suite via Node.js and verify all tests pass."""
    cmd = ["node", "--experimental-strip-types", str(TEST_SCRIPT)]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print("Node test stdout:\n", result.stdout)
    if result.stderr:
        print("Node test stderr:\n", result.stderr)

    assert result.returncode == 0, f"Node acceptance suite exited with code {result.returncode}"
    assert "Results: 14 passed out of 14 tests." in result.stdout, "Not all 14 tests passed"
    assert "FAILED" not in result.stdout, "Failures reported in Node acceptance suite"


def test_server_only_research_security_invariants():
    """Inspect research source files to verify security invariants."""
    safe_fetcher_content = (RESEARCH_DIR / "safe-fetcher.ts").read_text(encoding="utf-8")
    ssrf_guard_content = (RESEARCH_DIR / "ssrf-guard.ts").read_text(encoding="utf-8")
    search_provider_content = (RESEARCH_DIR / "search-provider.ts").read_text(encoding="utf-8")
    types_content = (RESEARCH_DIR / "types.ts").read_text(encoding="utf-8")

    # 1. Untrusted context tagging
    assert "isUntrustedContent" in types_content
    assert "isUntrustedContent: true" in safe_fetcher_content

    # 2. Max bytes bounds (500 KB)
    assert "500 * 1024" in safe_fetcher_content or "512000" in safe_fetcher_content

    # 3. Transport IP pinning
    assert "lookup" in safe_fetcher_content
    assert "pinnedIp" in safe_fetcher_content

    # 4. Strict SSRF CIDR blocklist
    for blocked_cidr in ["127.0.0.0", "10.0.0.0", "172.16.0.0", "192.168.0.0", "169.254.0.0"]:
        assert blocked_cidr in ssrf_guard_content

    # 5. Metadata IP blocking
    assert "169.254.169.254" in ssrf_guard_content
    assert "metadata.google.internal" in ssrf_guard_content

    # 6. Active content stripping
    for tag in ["script", "style", "nav", "form", "iframe"]:
        assert tag in safe_fetcher_content

    # 7. Fail-closed SearchNotConfiguredError
    assert "SearchNotConfiguredError" in types_content
    assert "SEARCH_NOT_CONFIGURED" in types_content
