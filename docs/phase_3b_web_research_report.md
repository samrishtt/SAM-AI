# SAM-AI Phase 3B: Web Research Engine (Stage 1 & Stage 2) Report

**Status:** ACCEPTED  
**Date:** 2026-10-18  
**Repository:** `samrishtt/SAM-AI`  
**Base Commit:** `23ecb54`  
**Target Scope:** Stage 1 (Search-Provider Abstraction) and Stage 2 (Safe Public-Page Retrieval & SSRF Defenses)  

---

## 1. Executive Summary

Phase 3B Stage 1 and Stage 2 establish the sovereign, secure web research backend foundation for SAM-AI without relying on commercial AI routers, third-party proprietary LLMs, or fragile client-side fetching.

All web retrieval occurs strictly server-side behind a hardened SSRF firewall and DNS-validation layer. Any external web content ingested by SAM-AI is bounded in memory (500 KB limit), stripped of active content (`<script>`, `<style>`, `<form>`, `<nav>`, `<iframe>`), and tagged with an explicit security envelope (`isUntrustedContent: true`) to prevent prompt-injection attacks from hijacking model reasoning.

---

## 2. Architecture & Components

### 2.1 Stage 1: Pluggable Search-Provider Abstraction

Defined in `web/src/lib/research/types.ts` and implemented in `web/src/lib/research/search-provider.ts`:

1. **`WebSearchProvider` Interface:**
   - Standardized contract: `search(options: SearchOptions): Promise<SearchResult[]>` and `checkHealth(): Promise<SearchProviderStatus>`.
   - Canonical result model containing destination URL, title, snippet, provider ID, retrieval timestamp, and optional scoring/engine metadata.
2. **`MockSearchProvider`:**
   - Fully deterministic in-memory provider supporting regular-expression fixture matching, simulated network latencies, and forced error conditions for reproducible test environments.
3. **`SearXNGSearchProvider`:**
   - Lightweight, dependency-free HTTP client for self-hosted SearXNG meta-search JSON APIs.
   - Strictly enforces URL syntax, permitted ports (443, 80, 8080, 8443, 8888), parameter normalization, timeout handling, non-JSON body rejection, and rate-limit parsing (HTTP 429).
4. **`SearchNotConfiguredError`:**
   - Fail-closed error raised when search is unconfigured (`SAM_AI_SEARCH_PROVIDER=none` or empty), ensuring no unintended public leaks or silent network queries.

### 2.2 Stage 2: Server-Side Safe Web Fetcher & SSRF Firewall

Defined in `web/src/lib/research/ssrf-guard.ts` and `web/src/lib/research/safe-fetcher.ts`:

1. **URL Syntax & Scheme Validation:**
   - Rejects embedded credentials in URLs (`https://user:pass@host`).
   - Requires HTTPS scheme by default (rejects `http:`, `ftp:`, `file:`, `javascript:`, `data:`).
   - Enforces default port 443 (or configured whitelist).
2. **Comprehensive CIDR Firewall:**
   - Full normalization of IPv4 (including octal, hex, decimal integers) and IPv6 (including IPv4-mapped IPv6 `::ffff:127.0.0.1`).
   - Blocklists all non-public address spaces:
     - Loopback (`127.0.0.0/8`, `::1/128`)
     - RFC 1918 Private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `fc00::/7`)
     - Link-Local (`169.254.0.0/16`, `fe80::/10`)
     - Cloud Metadata endpoints (`169.254.169.254`, `metadata.google.internal`, `instance-data`)
     - Carrier-Grade NAT (`100.64.0.0/10`), Multicast (`224.0.0.0/4`, `ff00::/8`), and Reserved ranges.
3. **Multi-Record DNS Resolution:**
   - Resolves all `A` and `AAAA` records for target hostnames using `node:dns.promises`.
   - If *any* returned address belongs to a prohibited subnet, the entire request is blocked before a socket connection is attempted.
4. **Transport IP-Pinning (TOCTOU & DNS Rebinding Elimination):**
   - Eliminates Time-Of-Check to Time-Of-Use (TOCTOU) exploits and DNS rebinding attacks by overriding the Node.js `lookup` hook in `https.request`. The socket is guaranteed to connect to the exact validated IP address verified during pre-flight resolution.
5. **Strict Manual Redirect Follower:**
   - Disables automatic HTTP client redirect following.
   - Manually intercepts 301, 302, 303, 307, 308 response codes.
   - Detects circular redirects, enforces a maximum limit of 3 hops, and subjects every redirect target URL to full URL syntax and SSRF re-validation.
6. **Streaming Byte Bounding:**
   - Stream chunks are counted at the byte level during transport.
   - Immediately aborts socket and destroys stream if payload exceeds 500 KB (512,000 bytes), preventing decompression bombs and memory exhaustion.
7. **Content-Type Whitelist:**
   - Validates `Content-Type` header against `text/html`, `text/plain`, and `application/xhtml+xml`.
   - Rejects binary formats, PDFs, archives, images, audio, video, and executables.
8. **HTML Sanitization & Text Extraction:**
   - Strips dangerous/noisy tags: `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<form>`, `<iframe>`, `<noscript>`.
   - Decodes HTML entities and normalizes whitespace while extracting document title and canonical URL.
9. **Untrusted Context Isolation:**
   - All fetch outputs set `isUntrustedContent: true`.
   - Formatted using `formatUntrustedResearchEnvelope()` which wraps retrieved sources inside explicit fences with a mandatory Security Notice informing the reasoning model that content is untrusted external data and must never be interpreted as commands or system instructions.

---

## 3. Verification & Test Evidence

### 3.1 Node.js Acceptance Suite (`web/tests/test_research_engine.mjs`)

Executed via `node --experimental-strip-types web/tests/test_research_engine.mjs`:

| # | Test Name | Result |
|---|---|---|
| 1 | Mock search provider returns typed and bounded results | **PASSED** |
| 2 | Missing provider configuration returns SEARCH_NOT_CONFIGURED | **PASSED** |
| 3 | Malformed provider responses and timeouts fail safely | **PASSED** |
| 4 | URL credentials and non-HTTPS destinations are rejected | **PASSED** |
| 5 | Localhost, IPv4/IPv6 private networks, loopback, link-local, and metadata destinations rejected | **PASSED** |
| 6 | Every redirect hop is revalidated | **PASSED** |
| 7 | DNS results containing a prohibited IP are rejected | **PASSED** |
| 8 | DNS-rebinding / TOCTOU limitations addressed by transport | **PASSED** |
| 9 | Redirect loops and excessive hops are rejected | **PASSED** |
| 10 | Oversized bodies are stopped before excessive memory is consumed | **PASSED** |
| 11 | Unsupported MIME types are rejected | **PASSED** |
| 12 | HTML scripts, forms, and injected markup are excluded from extracted text | **PASSED** |
| 13 | Prompt-injection text is treated as untrusted content, not an instruction | **PASSED** |
| 14 | All errors are bounded, structured, and free of secrets | **PASSED** |

**Summary:** 14 passed out of 14 tests (100%).

### 3.2 Python Pytest Suite (`tests/test_phase_3b_web_research.py`)

Executed via `python -m pytest tests/test_phase_2j_chat_context_and_style.py tests/test_multidomain_latex_rescoring.py tests/test_phase_3b_web_research.py`:

- Total collected: 14 test cases.
- Passed: 14 test cases in 0.60 seconds.
- Zero regressions in Phase 2J context isolation or Phase 3A rescoring fixtures.

### 3.3 Next.js Production Build

Executed via `npm run build` in `web/`:
- TypeScript type-checking (`npx tsc --noEmit`): **PASSED** (0 errors).
- Next.js production compilation: **SUCCESS**.
- Production routes generated: 9 static, 6 dynamic API routes.

---

## 4. Security Audit & Invariants Checklist

- [x] **No Secret Leakage:** No search API keys or provider tokens are exposed in client bundles, public HTML, or error messages.
- [x] **Fail-Closed Design:** Unconfigured or misconfigured providers throw structured `SearchNotConfiguredError`.
- [x] **SSRF Defense:** Complete RFC 1918, link-local, loopback, and cloud metadata blocking with IPv4/IPv6 normalization.
- [x] **DNS Rebinding Defense:** Pinned transport hook forces connection to verified IP.
- [x] **Memory Safety:** Streaming 500 KB hard limit enforced during socket read.
- [x] **Prompt-Injection Defense:** `isUntrustedContent: true` metadata and fenced markdown envelope.
- [x] **Scope Boundaries Respected:** Stage 1 and Stage 2 only. No Stage 3 (citation synthesis) or Stage 4 (evals) started prematurely. No GPU resources requested or used.
