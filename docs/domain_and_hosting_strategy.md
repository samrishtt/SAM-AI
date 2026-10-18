# SAM-AI Domain Strategy and Future Self-Hosting Readiness Architecture

**Company:** Parallax Intelligence Lab  
**Project:** SAM-AI  
**Document Status:** PROPOSED & AUDITED  
**Date:** October 18, 2026  
**Repository:** `samrishtt/SAM-AI`  
**Current Production Website:** [https://sam-ai-lilac-six.vercel.app](https://sam-ai-lilac-six.vercel.app)  
**Current Foundation Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`  
**Current Inference Backend:** Hugging Face ZeroGPU Space (`Samrish2009/sam-ai-assistant`)  

---

## 1. Executive Summary & Status Matrix

This document defines the architectural strategy for transitioning SAM-AI to a custom domain (e.g., purchased via Hostinger, Cloudflare Registrar, or another ICANN-accredited registrar) and outlines the technical readiness roadmap for migrating the web application from Vercel to an owner-controlled self-hosted server in the future.

### Key Audit Findings:
1. **The application cannot be converted to a static HTML export:** The Next.js application requires a live Node.js server runtime to execute dynamic API routes (`/api/chat`, `/api/chat/explain`, `/api/status`, `/api/workspace/*`), streaming SSE response transformations, in-memory rate limiting, and server-side secret isolation.
2. **Decoupling of Web Hosting and Model Inference:** The Next.js web application and the 7B parameter reasoning model run on two completely separate infrastructure layers. Moving the website to a CPU server does **not** move the 7B model off Hugging Face ZeroGPU.
3. **Recommended Phased Path:** Retain Vercel for website hosting while mapping the new custom domain in Phase 1 (zero downtime, automatic SSL, global CDN). Prepare self-hosting on a containerized Linux server (Docker + Caddy/Nginx) for Phase 2 once server infrastructure is procured.

### Strategic Status Matrix

| Component / Layer | Current Provider | Status Label | Notes |
| :--- | :--- | :--- | :--- |
| **Domain Registration** | None (User considering Hostinger/Cloudflare) | **Not Yet Configured** | No domain purchased; awaiting user registrar selection |
| **Authoritative DNS** | Vercel Default (`*.vercel.app`) | **Not Yet Configured** | DNS records will be configured after domain purchase |
| **Next.js Web Gateway** | Vercel Serverless (Free Tier) | **Deployed and verified** | Live at `sam-ai-lilac-six.vercel.app` (commit `190175c`) |
| **Model Inference Backend** | Hugging Face Space (`zero-a10g`) | **Deployed and verified** | Running pure `DeepSeek-R1-Distill-Qwen-7B` on ZeroGPU |
| **Web Research Engine** | Server-side TypeScript modules | **Implemented locally** | Stage 1 & 2 pass 14/14 tests; not yet live in user chat |
| **Self-Hosted Web Gateway** | Proposed Linux Docker / Node.js | **Proposed** | Code is ready; awaiting server provisioning |
| **Self-Hosted 7B Inference** | Proposed local vLLM / SGLang GPU | **Blocked** | Requires dedicated GPU hardware (>=16GB VRAM) |

---

## 2. Four Layers of Internet Infrastructure Explained

To make informed architectural decisions, it is essential to distinguish the four distinct technical layers involved in serving SAM-AI:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. DOMAIN REGISTRAR (e.g., Hostinger, Cloudflare Registrar, Namecheap)  │
│    - Legal ownership of the domain name (WHOIS/RDAP record).           │
│    - Delegates authoritative nameservers (NS records).                  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Nameserver Delegation
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. AUTHORITATIVE DNS (e.g., Cloudflare DNS, Hostinger DNS, Vercel DNS)  │
│    - Translates names to addresses: A, AAAA, CNAME, TXT records.        │
│    - Directs traffic to the Web Gateway IP or CNAME alias.              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTPS Requests (Port 443)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. WEB & GATEWAY HOSTING (Currently Vercel; Future: Self-Hosted Docker) │
│    - Serves Next.js UI, React Server Components, and static assets.     │
│    - Executes Serverless/Node.js API routes (/api/chat, /api/status).    │
│    - Enforces rate limiting, password protection, and stream keepalives.│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Server-to-Server Private API (SSE)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. GPU MODEL INFERENCE (Currently Hugging Face ZeroGPU; Future: vLLM)   │
│    - Dedicated GPU compute (NVIDIA A10G / RTX 6000 / T4).              │
│    - Executes DeepSeek-R1-Distill-Qwen-7B weights.                      │
│    - Emits raw reasoning traces and answers.                            │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Domain Registrar:** The company from which you lease the domain name. The registrar manages ownership, annual renewals, and nameserver delegation.
2. **Authoritative DNS:** The service that responds to DNS queries from visitors' browsers. It maps your human-readable domain (e.g. `yourdomain.com`) to specific server IP addresses (`A` records) or canonical hostnames (`CNAME` records).
3. **Website & App Gateway Hosting:** The computing infrastructure running the Next.js web application. It serves HTML/JS to visitors, validates site passwords, enforces rate limits, and proxies chat requests to the inference model.
4. **Model Inference Hosting:** The specialized high-performance GPU server that executes the billions of mathematical matrix multiplications required to run `DeepSeek-R1-Distill-Qwen-7B`.

---

## 3. Current Architecture Audit

### 3.1 Codebase & Runtime Requirements

Inspection of `web/package.json`, `web/src/app/api/`, and `web/src/lib/` establishes the following runtime requirements:

```
+-----------------------------+-------------------------------+------------------------------------------+
| Feature / Route             | Runtime Classification        | Technical Dependencies                   |
+-----------------------------+-------------------------------+------------------------------------------+
| Homepage (/)                | Static-page compatible        | Next.js App Router RSC, Tailwind CSS     |
| Research Archive (/research)| Static-page compatible        | Static content, Markdown tables          |
| Status Shell (/status)      | Static-page compatible        | Client component shell fetching API      |
| Chat Shell (/chat)          | Static-page compatible        | Vercel AI SDK useChat, local storage     |
| /api/chat                   | Node.js server required       | ReadableStream, SSE keepalives, fetch    |
| /api/chat/explain           | Node.js server required       | JSON body parsing, rate limiter          |
| /api/status                 | Node.js server required       | Force-dynamic, backend HTTP probe        |
| /api/workspace/files        | Node.js server required       | Node.js fs.readdir, path traversal checks|
| /api/workspace/file         | Node.js server required       | Node.js fs.readFile, 512KB limit         |
| /api/workspace/status       | Node.js server required       | Node.js process.cwd() inspection         |
| Web Research Engine         | Node.js server required       | node:https, node:dns, IP-pinned transport|
+-----------------------------+-------------------------------+------------------------------------------+
```

### 3.2 Incompatibility with Static Export
Next.js supports `output: 'export'` for purely static sites. However, attempting static export on SAM-AI fails because:
- `/api/chat` uses `maxDuration = 60` and returns dynamic streaming `Response(gradioStream)` objects.
- `/api/status` is marked `export const dynamic = "force-dynamic"` to inspect the live backend at request time.
- `/api/workspace/*` accesses the live server filesystem via Node.js `fs` APIs.
- In-memory rate limiting (`web/src/lib/rate-limiter.ts`) and password verification (`web/src/lib/security.ts`) require a live server environment to prevent credential exposure.

**Conclusion:** The application must be deployed on an active Node.js runtime environment (such as Vercel Serverless Functions, a long-running Node.js process, or a Docker container).

---

## 4. Recommended Phase 1: Domain-First on Vercel

### 4.1 Why the "Domain-First" Approach is Recommended
Before investing time, hardware, or money into self-hosting servers, the user should attach the custom domain to the **existing working Vercel deployment**:

1. **Zero Downtime & Zero Risk:** The current deployment is verified, live, and responsive. Attaching a domain to Vercel requires zero code changes and zero server provisioning.
2. **Automatic TLS/SSL:** Vercel automatically provisions and renews free Let's Encrypt SSL/TLS certificates for the custom domain.
3. **Global Edge CDN:** Static assets (JS, CSS, fonts) are cached at edge locations worldwide with HTTP/3 and Brotli compression.
4. **Preserves Free Tier:** Vercel's hobby tier provides generous serverless bandwidth, while Hugging Face ZeroGPU provides free GPU inference.
5. **Clean Migration Abstraction:** Once the custom domain is active on Vercel, migrating to a self-hosted server in the future is as simple as updating DNS records—users will not notice any URL change.

### 4.2 Registrar Options Comparison

| Registrar | Estimated Cost | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Cloudflare Registrar** | At-cost wholesale ($9–$11/yr for `.com`) | Zero markup, built-in world-class DNS, free DDoS protection, free WHOIS privacy. | Requires nameservers pointing to Cloudflare; cannot buy non-standard TLDs. | **Highly Recommended** (Best value and technical control). |
| **Hostinger** | First year promo cheap ($6–$10), renews higher ($14–$18) | Easy payment methods (UPI/cards in India), beginner-friendly dashboard. | Renewal prices jump after year 1; upsells web hosting packages. | **Acceptable** (Good if already using Hostinger services). |
| **Namecheap / Porkbun** | Wholesale pricing ($10–$12/yr) | Transparent pricing, free WHOIS privacy, reliable support. | Separate dashboard if using Cloudflare DNS later. | **Good Alternative**. |

### 4.3 Proposed Subdomain Architecture

When a domain is acquired (e.g., `example.com`), the recommended subdomain mapping is:

```
┌─────────────────────────┬───────────────────────────────┬──────────────────────────────────────────┐
│ FQDN                    │ Target Service                │ Current Recommended Implementation       │
+─────────────────────────+───────────────────────────────+──────────────────────────────────────────+
│ example.com             │ Apex / Root Domain            │ CNAME or ALIAS pointing to Vercel        │
│ www.example.com         │ Primary Website & Chat UI     │ CNAME pointing to cname.vercel-dns.com   │
│ api.example.com         │ Optional API Gateway Alias    │ (Optional Future) Currently part of www  │
│ models.example.com      │ Self-Hosted Inference Gateway │ (Blocked) Only when local GPU server exists│
└─────────────────────────┴───────────────────────────────┴──────────────────────────────────────────┘
```

> **Important Rule:** Do **not** manually create or guess DNS records before purchasing the domain and adding it to the Vercel project dashboard. Vercel will provide the exact `A` and `CNAME` records required for verification.

---

## 5. Recommended Phase 2: Future Self-Hosting Architecture

When the user decides to migrate the Next.js web application from Vercel to a self-hosted server (e.g., a dedicated Linux VPS or local homelab server), the target architecture is structured as follows:

```
                            INTERNET (PUBLIC)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ FIREWALL (UFW / Cloud Provider Security Group)                         │
│ - Port 80 (HTTP -> 301 Redirect to HTTPS)                              │
│ - Port 443 (HTTPS -> Reverse Proxy)                                    │
│ - Port 22 (SSH -> Key-based authentication only, rate-limited)         │
│ - All other ports BLOCKED                                               │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ REVERSE PROXY & TLS TERMINATION (Caddy or Nginx)                       │
│ - Automatic Let's Encrypt TLS renewal                                   │
│ - Disables proxy buffering (proxy_buffering off) for SSE streaming      │
│ - Proxy keepalive timeout >= 65 seconds                                │
│ - Client max body size limit = 10 MB                                   │
│ - Forwards X-Forwarded-For and X-Real-IP headers                        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ HTTP (Localhost / Docker network)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ NEXT.JS APPLICATION CONTAINER (Node.js 20 LTS Alpine)                   │
│ - Runs 'next start' or standalone server on port 3000                   │
│ - In-memory rate limiting and site password gate active                │
│ - Server-side environment variables loaded from secure .env.production │
│ - Read-only workspace bound to project root                            │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ Private HTTPS Outbound
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MODEL INFERENCE (Hugging Face ZeroGPU or Future vLLM GPU Server)       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Web Gateway Self-Hosting Requirements

#### 1. Server Hardware Specifications (Web Application Only):
- **CPU:** 2 vCPUs (x86_64 or ARM64).
- **RAM:** 2 GB minimum (4 GB recommended for comfortable `npm run build` headroom).
- **Disk:** 20 GB SSD storage.
- **Operating System:** Ubuntu 24.04 LTS or Debian 12.
- **Node.js Runtime:** Node.js v20 LTS or v22.

#### 2. Reverse Proxy & Streaming Configuration (Caddy vs Nginx):
Streaming chat requires unbuffered responses. A standard Nginx proxy buffers server responses by default, which breaks real-time token streaming.

**Option A: Caddy (Recommended for simplicity)**
Caddy handles automatic HTTPS and unbuffered proxying out of the box:
```caddy
yourdomain.com, www.yourdomain.com {
    reverse_proxy localhost:3000 {
        # Streaming SSE is preserved automatically by Caddy
    }
}
```

**Option B: Nginx**
Nginx requires explicit directives to preserve SSE stream responsiveness:
```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL Certificates
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # CRITICAL for SSE streaming:
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 120s;
    }
}
```

#### 3. Containerization Strategy (Dockerfile):
Next.js supports building a minimal standalone distribution. When deploying to Docker, setting `output: 'standalone'` in `next.config.mjs` reduces container size from ~800MB to ~150MB by copying only necessary node_modules.

---

## 6. Website Hosting vs. GPU Inference Hosting (Crucial Decoupling)

A common point of confusion is assuming that "self-hosting SAM-AI" means hosting both the website and the AI model on the same machine.

### Why They Must Remain Decoupled:
1. **Web Gateway (CPU only):**
   - Lightweight Node.js server.
   - Requires ~1–2 GB RAM, 2 CPU cores, and basic broadband.
   - Costs ~$4–$6/month on a standard cloud VPS (or free on Vercel).
2. **Reasoning Model Inference (Heavy GPU required):**
   - `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` has 7 billion parameters.
   - In 16-bit precision (bfloat16), model weights alone consume **14 GB of VRAM**.
   - With KV-cache allocation for 2,048–4,096 tokens, inference requires at least **16–20 GB of VRAM**.
   - Suitable hardware: NVIDIA RTX 4080 (16GB), RTX 3090/4090 (24GB), Tesla T4 (16GB), or A10G (24GB).
   - Commercial GPU cloud rental costs: ~$0.50–$1.20/hour ($350–$850/month if run 24/7).
   - Local hardware cost: ~$1,000–$2,000 for a dedicated desktop GPU workstation.

### Current Architectural Recommendation:
Keep model inference on **Hugging Face ZeroGPU** (which is completely free and provides dynamic A10G GPU allocation) while migrating only the Next.js web application to your own server or custom domain.

---

## 7. Migration & DNS Cutover Checklist

When the user is ready to execute domain attachment or self-hosting, follow this sequential procedure:

### Phase 1: Custom Domain on Vercel
- [ ] **Step 1:** Purchase domain from chosen registrar (Hostinger, Cloudflare Registrar, etc.).
- [ ] **Step 2:** Open the Vercel Dashboard -> SAM-AI Project -> Settings -> Domains.
- [ ] **Step 3:** Enter the apex domain (`example.com`) and `www.example.com`.
- [ ] **Step 4:** Copy the exact DNS records generated by Vercel (e.g. `A` record for apex, `CNAME` for `www`).
- [ ] **Step 5:** Enter those records in your registrar's DNS management panel.
- [ ] **Step 6:** Wait for DNS propagation (typically 2–30 minutes) and verify SSL certificate issuance.
- [ ] **Step 7:** Test `/`, `/chat`, `/research`, `/status`, and live streaming `/api/chat` on the new custom domain.

### Phase 2: Self-Hosted Server Migration (Future)
- [ ] **Step 1:** Provision Linux VPS (Ubuntu 24.04, 2GB+ RAM) with static IPv4.
- [ ] **Step 2:** Configure firewall (UFW): allow ports 22, 80, 443; deny all other inbound traffic.
- [ ] **Step 3:** Install Docker or Node.js v20 LTS.
- [ ] **Step 4:** Clone repository and copy `.env.production` with all required `SAM_AI_*` environment variables.
- [ ] **Step 5:** Set `SAM_AI_TRUST_PROXY=true` to properly resolve client IPs behind the reverse proxy.
- [ ] **Step 6:** Configure reverse proxy (Caddy or Nginx) with `proxy_buffering off;` and `proxy_read_timeout 120s;`.
- [ ] **Step 7:** Run acceptance tests locally on the server (`npm run build`, `npm run start`).
- [ ] **Step 8:** Test endpoints locally via `curl -I http://127.0.0.1:3000/api/status`.
- [ ] **Step 9:** Lower DNS TTL to 300 seconds (5 minutes) on the existing domain.
- [ ] **Step 10:** Update DNS `A` records to point to the new server's IP address.
- [ ] **Step 11:** Verify live chat streaming and monitoring logs.
- [ ] **Step 12:** Keep Vercel project available as a hot standby for emergency rollback.

---

## 8. Verification & Boundaries Confirmation

- **No domains purchased or renewed.**
- **No DNS records created or modified.**
- **No production settings or Vercel configurations altered.**
- **No model weights migrated or training runs initiated.**
- **No credentials, tokens, or private secrets exposed.**
