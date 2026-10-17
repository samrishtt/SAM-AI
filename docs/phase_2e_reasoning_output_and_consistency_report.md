# SAM-AI Phase 2E: Unified Inference, Hidden Reasoning Handling, Duplicate-Response Fix, and Chat Consistency Report

## 1. Executive Summary

During Phase 2E, we investigated and resolved three critical production and developer-experience issues across the SAM-AI web application and Hugging Face ZeroGPU inference space (`https://samrish2009-sam-ai-assistant.hf.space`):

1. **Reasoning Trace Leakage**: DeepSeek-R1 series models emit step-by-step thinking traces wrapped in `<think>...</think>`. Raw token streaming previously transmitted these internal reasoning traces directly into user chat windows and exposed closing `</think>` tags.
2. **Duplicate Assistant Responses**:
   - In `ChatInterface.tsx` (`useChat.onFinish`), saving messages locally was previously doing `[...messages, message]`, causing the assistant message to be duplicated on reload or turn completion.
   - In `spaces/sam_ai_zerogpu/app.py` (`chat_respond`), simultaneous Enter-key press and Send-button clicks could append duplicate user/assistant turns.
   - In `web/src/lib/gradio-connector.ts`, mid-stream network dropouts could trigger a second full generation attempt via `/api/predict`, repeating previously streamed tokens.
3. **Cross-Platform Inconsistency**:
   - Divergent system prompts existed across `spaces/sam_ai_zerogpu/app.py`, `model_loader.py`, and `web/src/lib/model-adapter.ts`.
   - Distinct visual themes (dark navy in Space vs. warm ivory/terracotta on website) created an inconsistent product experience.

All three root causes have been systematically eliminated through dual-sided stream filtering (both at Space engine and web connector levels), strict deduplication guards, canonical system prompt unification, and light-theme CSS alignment.

---

## 2. Root Cause Analysis

### 2.1 Hidden Reasoning Leakage
DeepSeek-R1 models utilize specialized chat templates that append `<｜Assistant｜><think>\n` to prompt prefixes. When generating:
1. The model directly outputs reasoning tokens.
2. The model concludes reasoning with `</think>\n\n`.
3. The model subsequently outputs the user-facing response.

Neither `spaces/sam_ai_zerogpu/model_loader.py` nor `web/src/lib/gradio-connector.ts` previously inspected token deltas for reasoning delimiters. In addition, if a generation was cut off by token limits before emitting `</think>`, the unfinished reasoning trace was exposed without an explanation.

### 2.2 Response Duplication
Three distinct failure modes caused duplicate answers:
1. **Frontend Local Storage Duplication**: In `ChatInterface.tsx`, the `onFinish(message)` handler appended `message` to `messages` even though `useChat` already included `message` in its internal state array.
2. **Gradio Multi-Event Race**: In `app.py`, submitting via Enter key and Send button concurrently invoked `chat_respond` twice with the same history array, resulting in double assistant rows.
3. **Connector Fallback Duplication**: If Gradio 4 SSE streaming failed after emitting initial tokens, `streamGradioChat` fell back to `/api/predict` from token 0, restarting and duplicating the entire answer.

### 2.3 Cross-Interface Inconsistency
- **System Prompts**: The Space used a 4-line developer prompt, while the web adapter used a 5-point instruction set with differing wording.
- **Visual Design**: The Space used Gradio's dark navy palette (`#0b0f19`, `#111827`, `#0f172a`), contrasting with the deployed Next.js site's warm ivory (`#FAF8F5`) and terracotta (`#D97745`) palette.

---

## 3. Implemented Solutions

### 3.1 Dual-Sided Reasoning Stream Filters (`ReasoningStreamFilter`)
We implemented identical, zero-dependency streaming state machines in TypeScript (`web/src/lib/reasoning-filter.ts`) and Python (`spaces/sam_ai_zerogpu/model_loader.py`):
- **Chunk-Boundary Buffering**: Safely buffers arbitrary token slices across `</think>` boundaries (e.g. `"<"` + `"/th"` + `"ink>"`).
- **Whitespace Stripping**: Automatically cleans leading blank lines immediately following `</think>`.
- **Fail-Closed Truncation Notice**: If generation finishes or hits max tokens before `</think>` is reached, the internal thoughts are safely discarded and replaced with an honest recoverable notice:
  `*[SAM-AI Notice: The reasoning process was interrupted before a final answer could be generated. Please retry your request.]*`
- **Defense-in-Depth UI Sanitizer**: `web/src/components/ChatMessage.tsx` processes all displayed and copied message contents through `filterReasoningFromText()`, ensuring legacy stored messages in browser storage never render raw `<think>` tags.

### 3.2 Deduplication Protection
- **`ChatInterface.tsx`**:
  ```typescript
  const messageAlreadyPresent = messages.some((m) => m.id === message.id);
  const mergedMessages = messageAlreadyPresent ? messages : [...messages, message];
  ```
- **`app.py`**:
  ```python
  history_list = [list(turn) for turn in (chat_history or [])]
  if history_list and history_list[-1][0] == user_message and not history_list[-1][1]:
      new_history = history_list
  else:
      new_history = history_list + [[user_message, ""]]
  ```
- **`gradio-connector.ts`**:
  Tracks `hasEmittedTokens`. If tokens have already been transmitted, fallback to `/api/predict` is aborted.

### 3.3 Canonical System Prompt Unification
The canonical prompt is now shared across both backend and frontend:
```text
You are SAM-AI, an experimental sovereign artificial intelligence assistant developed by Parallax Intelligence Lab, powered by the open-weight foundation model DeepSeek-R1-Distill-Qwen-7B.

Core Operating Guidelines:
1. Provide accurate, mathematically sound, evidence-backed, and clear responses.
2. Offer helpful, step-by-step user-facing explanations when explaining calculations, code, or complex concepts.
3. Keep final answers organized, readable, and directly responsive to the user's prompt.
4. Do not falsely claim to be a different proprietary assistant or model.
5. Do not claim unsupported tools, live web browsing, external execution environments, or autonomous capabilities unless specifically provided.
6. Acknowledge limitations and uncertainties transparently.
```

### 3.4 Space Warm-Light Theme Alignment
`spaces/sam_ai_zerogpu/app.py` was styled to match the production light theme:
- Background: `#FAF8F5`
- Card/Block: `#FFFFFF`
- Borders: `#E8E2DA`
- Primary Button: Terracotta `#D97745` (hover `#C86332`)
- Body Text: Charcoal `#28231F`

---

## 4. Verification & Test Evidence

| Component / Test Suite | Scope | Result |
| :--- | :--- | :--- |
| `scripts/test_reasoning_filter.mjs` | Node.js stream splitting, chunk boundaries, truncation fail-close, pure function | **5/5 PASSED** |
| `spaces/sam_ai_zerogpu/tests/test_reasoning_filter.py` | Python stream filtering, split tokens, canonical prompt | **6/6 PASSED** |
| `spaces/sam_ai_zerogpu/tests/test_space_components.py` | Cryptographic SHA-256 verification, fail-closed configs, secret omission | **7/7 PASSED** |
| `web/` Typecheck (`npx tsc --noEmit`) | Next.js TypeScript strict verification | **0 Errors (PASSED)** |

---

## 5. Architectural Invariants Preserved
1. **Zero Third-Party Fallbacks**: No requests are routed to OpenAI, Gemini, Anthropic, or external router endpoints.
2. **Rejected Adapter Kept Out**: Experiment 001 adapter is not loaded or promoted.
3. **Credential Safety**: No API tokens, keys, or passwords were exposed, committed, or requested.
