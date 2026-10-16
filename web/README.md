# SAM-AI Web Chat MVP

A responsive, streaming conversational AI web application built with **Next.js (App Router)**, **TypeScript**, **Tailwind CSS**, and the **Vercel AI SDK**, designed for deployment on **Vercel**.

---

## 1. Architectural Overview & Request Flow

```
User Browser
    │
    ▼ (1. Submits prompt with optional 'x-sam-ai-password' header)
Next.js Edge / Server Route: /api/chat
    │
    ├──► (2. Security Gate) Checks SAM_AI_SITE_PASSWORD (rejects if unauthorized)
    │
    ├──► (3. Rate Limiter) Best-effort in-memory sliding window (rejects if > 20 req/min)
    │
    ├──► (4. Payload Validator) Enforces max 50 messages, max 8,000 chars per message
    │
    ▼
SAM-AI Model Adapter: src/lib/model-adapter.ts
    │
    ├──► PRIMARY (Default): SAM_AI_PROVIDER=qwen_openai_compatible
    │    Connects to OpenAI-compatible inference server running Qwen/Qwen3.8-27B
    │    (e.g., vLLM, SGLang, Ollama) via SAM_AI_BASE_URL.
    │    If SAM_AI_BASE_URL is missing -> throws MODEL_BACKEND_NOT_CONFIGURED error.
    │    NEVER silently falls back to third-party APIs.
    │
    └──► OPTIONAL: SAM_AI_PROVIDER=google
         Connects to Google Gemini API using GOOGLE_GENERATIVE_AI_API_KEY.
    │
    ▼ (5. Token Stream via streamText())
Browser UI (Streaming markdown & syntax-highlighted code blocks)
    │
    ▼ (6. Persistence)
Local Browser Storage (localStorage)
```

### Sovereign Architecture Decoupling:
All provider-specific API calls are strictly encapsulated inside `src/lib/model-adapter.ts`. The primary default is configured for **`Qwen/Qwen3.8-27B`** served over an OpenAI-compatible gateway (`SAM_AI_BASE_URL`). The Google Gemini provider is retained strictly as an explicit opt-in provider. The client-side UI remains completely decoupled from model providers.

---

## 2. Environment Variables

Create a `.env.local` file inside the `web/` directory for local development, or add these variables in your **Vercel Project Settings → Environment Variables**:

| Variable | Required? | Default | Description |
|---|---|---|---|
| `SAM_AI_PROVIDER` | No | `qwen_openai_compatible` | Provider type: `qwen_openai_compatible` (primary default) or `google` (opt-in). |
| `SAM_AI_MODEL` | No | `Qwen/Qwen3.8-27B` | Active model identifier. Primary: `Qwen/Qwen3.8-27B`. (For Google: `gemini-2.0-flash`). |
| `SAM_AI_BASE_URL` | **Yes (for Qwen)** | *None* | HTTP URL of the OpenAI-compatible inference server (e.g., `http://localhost:8000/v1` or `https://inference.your-domain.org/v1`). |
| `SAM_AI_API_KEY` | No | `EMPTY` | Optional server-side API key for the inference endpoint. Never exposed to browser. |
| `GOOGLE_GENERATIVE_AI_API_KEY` | Only if `SAM_AI_PROVIDER=google` | *None* | Google AI Studio Gemini API key. |
| `SAM_AI_SITE_PASSWORD` | No | *None* (Open Preview) | If set, users must enter this password to unlock the UI and send chat requests. |
| `SAM_AI_MAX_OUTPUT_TOKENS` | No | `2048` | Maximum token length for assistant completions (1-8192). |
| `SAM_AI_RATE_LIMIT_PER_MINUTE` | No | `20` | In-memory sliding-window request limit per client IP. |

See `.env.example` for a clean template with empty placeholders.

---

## 3. Local Development

From the `web/` directory:

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.example .env.local
# Edit .env.local and configure SAM_AI_BASE_URL for your Qwen server

# 3. Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 4. Production Build & Verification

To verify that the application compiles and builds for production deployment:

```bash
# Run Next.js production build
npm run build

# Start production server locally
npm start
```

---

## 5. Deployment Instructions for Vercel

Vercel hosts the Next.js web application frontend and API routes, not the 27B parameter model weights. The Qwen model weights are served via a separate GPU inference server (e.g., vLLM/SGLang).

1. Push your repository to GitHub.
2. In Vercel, select **Add New Project** and choose the repository.
3. In **Root Directory**, click **Edit** and set it to: `web`.
4. In **Environment Variables**, configure:
   - `SAM_AI_PROVIDER`: `qwen_openai_compatible`
   - `SAM_AI_MODEL`: `Qwen/Qwen3.8-27B`
   - `SAM_AI_BASE_URL`: `https://your-gpu-server.example.com/v1`
   - `SAM_AI_API_KEY`: *(Your inference server key, if required)*
   - `SAM_AI_SITE_PASSWORD`: *(Optional password to restrict access)*
5. Click **Deploy**.

---

## 6. Honest Disclosures & Model Attribution

- **Separate Weight Serving:** Vercel does not host model weights. Qwen3.8-27B requires ~54 GB in BF16 (~14.5 GB in 4-bit) and must run on a dedicated inference server.
- **No Silent Fallback:** If `SAM_AI_BASE_URL` is unconfigured, the chat route returns HTTP 503 (`MODEL_BACKEND_NOT_CONFIGURED`). It will never secretly call a third-party model and misrepresent it as Qwen.
- **Local Storage:** All chat history is saved exclusively in the user's browser `localStorage`. No server-side database or user tracking is utilized in this MVP.
