# SAM-AI Self-Hosting Readiness Checklist

**Company:** Parallax Intelligence Lab  
**Project:** SAM-AI  
**Document Status:** AUDITED & ACTIONABLE  
**Date:** October 18, 2026  
**Repository:** `samrishtt/SAM-AI`  

---

## 1. Overview & Checklist Status Legend

This checklist defines the concrete technical gates, verification commands, and evidence requirements before migrating the SAM-AI web application from Vercel to a self-hosted server or containerized environment.

### Status Legend
- **`[VERIFIED]`**: Implemented in code, tested, and validated in the current repository.
- **`[PROPOSED]`**: Architecturally designed and ready for implementation upon server provisioning.
- **`[NOT YET CONFIGURED]`**: Depends on external assets (e.g., custom domain ownership, server IP).
- **`[BLOCKED]`**: Requires dedicated hardware resources (e.g., GPU VRAM for local inference).

---

## 2. Technical Readiness Matrix

### 2.1 Next.js Runtime & Application Build

- [x] **`[VERIFIED]` Node.js Version Compatibility:** Codebase verified compatible with Node.js v20 LTS and Node.js v22/v26 (`web/package.json` specifies `"node": ">=20.17"`).
- [x] **`[VERIFIED]` Clean Production Build:** `npm run build` in `web/` completes successfully with 0 errors, generating 9 static routes and 6 dynamic serverless API routes.
- [x] **`[VERIFIED]` Type Safety:** `npx tsc --noEmit` completes with 0 errors under `"strict": true`.
- [ ] **`[PROPOSED]` Standalone Output Generation:** Add `output: 'standalone'` to `web/next.config.mjs` when building Docker container to bundle only minimal production dependencies.
  - *Evidence Required:* `.next/standalone/server.js` generated and boots independently via `node server.js`.
- [x] **`[VERIFIED]` Dynamic Route Enforcement:** Dynamic API routes (`/api/chat`, `/api/status`) are explicitly marked `force-dynamic` to ensure they execute server-side on every request.

### 2.2 Streaming, Keepalives & Reverse Proxy Compatibility

- [x] **`[VERIFIED]` Vercel AI SDK SSE Data Stream:** `/api/chat` correctly emits Vercel AI SDK Data Stream protocol (`0:"text"\n`) over chunked transfer encoding.
- [x] **`[VERIFIED]` Periodic Keepalive Heartbeats:** `streamGradioChat` emits 3.5s interval keepalive frames (`0:""\n`) to prevent upstream proxies from closing idle connections while the model deliberates.
- [ ] **`[PROPOSED]` Reverse Proxy Unbuffered Streaming:** Upstream reverse proxy (Nginx or Caddy) must have proxy buffering disabled.
  - *Evidence Required for Nginx:* `proxy_buffering off; proxy_cache off; proxy_read_timeout 120s;` present in Nginx vhost config.
  - *Evidence Required for Caddy:* Caddyfile configured with `reverse_proxy localhost:3000`.
- [x] **`[VERIFIED]` Gradio SSE Completion Block Parsing:** Multi-line SSE parser in `web/src/lib/gradio-connector.ts` verified to parse completed event payloads without dropping trailing turns.

### 2.3 Environment Configuration & Secret Isolation

- [x] **`[VERIFIED]` Template Documentation:** `web/.env.example` documents all required `SAM_AI_*` variables with security annotations.
- [x] **`[VERIFIED]` Server-Side Secret Protection:** Secrets (`SAM_AI_API_KEY`, `SAM_AI_SITE_PASSWORD`, internal IPs) are strictly referenced in server routes and never exported in client bundles (`process.env.NEXT_PUBLIC_*`).
- [ ] **`[NOT YET CONFIGURED]` Production Environment File:** `.env.production` provisioned on target server with restrictive filesystem permissions (`chmod 600 .env.production`).
- [x] **`[VERIFIED]` Reverse Proxy Trust Flag:** `SAM_AI_TRUST_PROXY=true` implemented in `web/src/lib/security.ts` to allow correct client IP resolution from `X-Forwarded-For` when behind Caddy/Nginx.

### 2.4 Server Hardware, OS & Networking Prerequisites

- [ ] **`[NOT YET CONFIGURED]` Dedicated Server Host:** Linux VPS or dedicated server provisioned (Ubuntu 24.04 LTS or Debian 12 recommended).
  - *Minimum Specs (Web Gateway):* 2 vCPUs, 2 GB RAM, 20 GB SSD storage.
  - *Recommended Specs (Web Gateway):* 2–4 vCPUs, 4 GB RAM, 40 GB SSD.
- [ ] **`[NOT YET CONFIGURED]` Static Public IPv4 Address:** Server assigned a persistent public IP for DNS `A` record routing.
- [ ] **`[NOT YET CONFIGURED]` Network Bandwidth & Stability:** Redundant network uplink with 99.9% uptime SLA.
- [x] **`[VERIFIED]` Workstation Protection Invariant:** The user's Windows development machine must **never** be exposed directly to the public internet.

### 2.5 Security, Firewall & TLS Hardening

- [ ] **`[PROPOSED]` Host Firewall (UFW):** Firewall configured to permit strictly necessary inbound ports:
  - Port 22 (SSH — key authentication only, password auth disabled, fail2ban active).
  - Port 80 (HTTP — auto-redirected to HTTPS).
  - Port 443 (HTTPS — TLS termination).
  - All other inbound ports blocked by default (`ufw default deny incoming`).
- [ ] **`[PROPOSED]` Automated TLS/SSL Certificates:** Let's Encrypt certificates managed automatically via Caddy or Certbot with automated renewal.
- [x] **`[VERIFIED]` In-Memory IP Rate Limiting:** `checkRateLimit()` in `web/src/lib/rate-limiter.ts` limits requests to 20 req/min per IP with `Retry-After` headers.
- [x] **`[VERIFIED]` Path Traversal Firewall:** Workspace explorer (`/api/workspace/*`) enforces strict path canonicalization, symlink escape rejection, and secret file pattern exclusion.
- [x] **`[VERIFIED]` Web Research SSRF Firewall:** `web/src/lib/research/ssrf-guard.ts` blocks loopback, private subnets (RFC 1918), link-local, and cloud metadata (`169.254.169.254`).

### 2.6 Health Probes, Process Supervision & Monitoring

- [x] **`[VERIFIED]` System Health Endpoint:** `/api/status` actively verifies backend connectivity, model identity, and security configuration.
- [ ] **`[PROPOSED]` Process Supervisor:** Node.js process managed via `systemd` service or `docker compose` with `restart: unless-stopped`.
  - *Evidence Required:* System service unit file `/etc/systemd/system/sam-ai.service` configured.
- [ ] **`[PROPOSED]` Health Check Probe:** Uptime monitor (e.g. Uptime Kuma or external HTTP probe) hitting `/api/status` every 60 seconds.
- [ ] **`[PROPOSED]` Log Rotation:** Application logs piped to `journald` or Docker logging with size limits (e.g. 50MB max per file, 3 rotated files).

### 2.7 Inference Decoupling & Latency Verification

- [x] **`[VERIFIED]` Decoupled Architecture:** The Next.js website communicates with the model purely over HTTPS/SSE via `SAM_AI_BASE_URL`.
- [x] **`[VERIFIED]` ZeroGPU Space Integration:** Live Space `Samrish2009/sam-ai-assistant` running `DeepSeek-R1-Distill-Qwen-7B` with adapters detached.
- [ ] **`[BLOCKED]` Self-Hosted 7B GPU Inference:** Self-hosting the model locally requires dedicated GPU hardware (>=16GB VRAM, e.g. RTX 4080/3090, A10G) running vLLM or SGLang. Currently blocked by hardware availability; free ZeroGPU hosting is retained.

---

## 3. Pre-Flight Verification Commands

Before pointing DNS records to a newly provisioned self-hosted server, execute this verification runbook:

```bash
# 1. Verify Node.js runtime version (must be >= 20.17)
node --version

# 2. Verify all research engine unit/integration tests
node --experimental-strip-types web/tests/test_research_engine.mjs

# 3. Verify Python regression test suites
python -m pytest tests/test_phase_3b_web_research.py tests/test_phase_2j_chat_context_and_style.py tests/test_multidomain_latex_rescoring.py tests/test_website_content_and_status.py

# 4. Verify TypeScript compilation
cd web && npx tsc --noEmit

# 5. Build optimized production bundle
npm run build

# 6. Start local production server on port 3000
npm run start

# 7. In a separate terminal, probe the live endpoints locally:
curl -I http://127.0.0.1:3000/
curl -I http://127.0.0.1:3000/research
curl -s http://127.0.0.1:3000/api/status | jq .

# 8. Test chat streaming locally:
curl -N -X POST http://127.0.0.1:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

All 8 verification checks must pass cleanly before updating authoritative DNS records.
