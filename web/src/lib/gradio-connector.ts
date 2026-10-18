/**
 * SAM-AI Sovereign Gradio API Connector
 * =====================================
 * Server-side connector for interacting with Gradio Spaces (e.g. Hugging Face ZeroGPU)
 * without requiring third-party npm packages.
 *
 * Implements:
 * 1. Health inspection via Gradio `/config` and `/info` endpoints.
 * 2. Multi-turn conversation mapping to Gradio chat history pairs.
 * 3. Streaming token generation via Gradio 4 Server-Sent Events (SSE) `/call/...` protocol.
 * 4. Fallback to `/api/predict` for compatibility with older Gradio runtimes.
 * 5. Formatting of output into Vercel AI SDK Data Stream protocol (`0:"chunk"\n`).
 */

import { ReasoningStreamFilter, filterReasoningFromText } from "./reasoning-filter";

export interface GradioChatOptions {
  baseUrl: string;
  messages: Array<{ role: string; content: string }>;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  apiKey?: string;
  systemPrompt?: string;
}

export interface GradioHealthResult {
  isReady: boolean;
  status: "ready" | "loading" | "unreachable" | "error";
  modelName: string;
  detail: string;
}

/**
 * Checks whether a URL target indicates a Gradio Space or endpoint.
 */
export function isGradioEndpoint(urlStr: string): boolean {
  if (!urlStr) return false;
  const lower = urlStr.toLowerCase();
  if (lower.includes(".hf.space") || lower.includes("/spaces/")) {
    return true;
  }
  if (process.env.SAM_AI_BACKEND_TYPE?.toLowerCase() === "gradio") {
    return true;
  }
  return false;
}

/**
 * Normalizes a Gradio Space URL into its direct HTTP origin.
 * Converts 'https://huggingface.co/spaces/user/space' -> 'https://user-space.hf.space'.
 */
export function normalizeGradioBaseUrl(rawUrl: string): string {
  const trimmed = rawUrl.trim().replace(/\/+$/, "");
  const hfSpacesMatch = trimmed.match(/^https?:\/\/huggingface\.co\/spaces\/([^/]+)\/([^/]+)/i);
  if (hfSpacesMatch) {
    const user = hfSpacesMatch[1].toLowerCase();
    const space = hfSpacesMatch[2].toLowerCase();
    return `https://${user}-${space}.hf.space`;
  }
  return trimmed;
}

/**
 * Inspects the readiness and metadata of a Gradio backend.
 */
export async function checkGradioHealth(baseUrl: string, apiKey?: string): Promise<GradioHealthResult> {
  const normalized = normalizeGradioBaseUrl(baseUrl);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 3000);

  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  if (apiKey && apiKey !== "EMPTY") {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }

  try {
    // Attempt 1: Fetch Gradio app config
    const configResp = await fetch(`${normalized}/config`, {
      method: "GET",
      headers,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (configResp.ok) {
      const configData = (await configResp.json()) as any;
      const title = configData?.title || "SAM-AI Sovereign Assistant";
      return {
        isReady: true,
        status: "ready",
        modelName: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        detail: `Gradio Space active: '${title}'`,
      };
    }

    return {
      isReady: false,
      status: "unreachable",
      modelName: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
      detail: `Gradio backend returned HTTP ${configResp.status}`,
    };
  } catch (err: any) {
    clearTimeout(timeoutId);
    return {
      isReady: false,
      status: "unreachable",
      modelName: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
      detail: err?.name === "AbortError" ? "Gradio health check timed out (3.0s)" : err?.message || "Connection refused",
    };
  }
}

/**
 * Converts a standard message list into (lastUserMessage, chatHistoryPairs).
 * Handles:
 * - Empty, invalid, or malformed message arrays
 * - System message filtering
 * - Consecutive user messages (merged cleanly)
 * - Orphaned assistant messages (discarded safely)
 * - Zero history leaks for single-message / fresh conversations
 */
export function formatGradioChatHistory(
  messages: Array<{ role: string; content: string }>
): { userMessage: string; history: Array<[string, string]> } {
  if (!Array.isArray(messages) || messages.length === 0) {
    return { userMessage: "", history: [] };
  }

  // 1. Sanitize and retain only valid user and assistant messages
  const sanitized: Array<{ role: "user" | "assistant"; content: string }> = [];
  for (const m of messages) {
    if (!m || typeof m !== "object") continue;
    if (m.role !== "user" && m.role !== "assistant") continue;
    if (typeof m.content !== "string") continue;
    const trimmed = m.content.trim();
    if (!trimmed) continue;
    sanitized.push({ role: m.role, content: trimmed });
  }

  if (sanitized.length === 0) {
    return { userMessage: "", history: [] };
  }

  // 2. Find the index of the latest user message
  let lastUserIdx = -1;
  for (let i = sanitized.length - 1; i >= 0; i--) {
    if (sanitized[i].role === "user") {
      lastUserIdx = i;
      break;
    }
  }

  if (lastUserIdx === -1) {
    return { userMessage: "", history: [] };
  }

  let lastUserMessage = sanitized[lastUserIdx].content;

  // 3. Process preceding turns into strictly paired [user, assistant] history
  const history: Array<[string, string]> = [];
  let pendingUser = "";

  for (let i = 0; i < lastUserIdx; i++) {
    const turn = sanitized[i];
    if (turn.role === "user") {
      if (pendingUser) {
        // Merge consecutive user messages rather than silently dropping earlier content
        pendingUser = `${pendingUser}\n\n${turn.content}`;
      } else {
        pendingUser = turn.content;
      }
    } else if (turn.role === "assistant") {
      if (pendingUser) {
        history.push([pendingUser, turn.content]);
        pendingUser = "";
      }
      // If no pendingUser, this is an orphaned assistant turn; safely skip
    }
  }

  // 4. If there was a trailing unanswered user prompt prior to the last message, merge it
  if (pendingUser) {
    lastUserMessage = `${pendingUser}\n\n${lastUserMessage}`;
  }

  return { userMessage: lastUserMessage, history };
}

/**
 * Streams chat responses from a Gradio Space formatted in Vercel AI SDK data-stream protocol.
 */
export async function streamGradioChat(options: GradioChatOptions): Promise<ReadableStream<Uint8Array>> {
  const normalized = normalizeGradioBaseUrl(options.baseUrl);
  const { userMessage, history } = formatGradioChatHistory(options.messages);

  const temperature = options.temperature ?? 0.6;
  const topP = options.topP ?? 0.95;
  const maxTokens = options.maxTokens ?? 2048;

  const authHeaders: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (options.apiKey && options.apiKey !== "EMPTY") {
    authHeaders["Authorization"] = `Bearer ${options.apiKey}`;
  }

  const encoder = new TextEncoder();

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      let hasEmittedTokens = false;
      const reasoningFilter = new ReasoningStreamFilter();

      try {
        // Attempt Gradio 4 SSE protocol: POST /call/chat_respond -> GET /call/chat_respond/<event_id>
        const callPayload = {
          data: [userMessage, history, temperature, topP, maxTokens],
        };

        let eventId: string | null = null;
        let sseEndpoint = `${normalized}/call/chat_respond`;

        let callResp = await fetch(sseEndpoint, {
          method: "POST",
          headers: authHeaders,
          body: JSON.stringify(callPayload),
        });

        // Fallback to /gradio_api/call/chat_respond if 404
        if (callResp.status === 404) {
          sseEndpoint = `${normalized}/gradio_api/call/chat_respond`;
          callResp = await fetch(sseEndpoint, {
            method: "POST",
            headers: authHeaders,
            body: JSON.stringify(callPayload),
          });
        }

        if (callResp.ok) {
          const callData = (await callResp.json()) as any;
          eventId = callData?.event_id;
        }

        if (eventId) {
          // Stream from /call/chat_respond/<event_id>
          const streamUrl = `${sseEndpoint}/${eventId}`;
          const streamResp = await fetch(streamUrl, {
            method: "GET",
            headers: {
              ...authHeaders,
              Accept: "text/event-stream",
            },
          });

          if (!streamResp.ok || !streamResp.body) {
            throw new Error(`Failed to establish SSE stream from Gradio (HTTP ${streamResp.status})`);
          }

          const reader = streamResp.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          let lastFullText = "";
          let currentEventType = "";
          let streamCompleted = false;

          // Send keepalive heartbeats every 3.5s to prevent Vercel idle connection timeout
          const keepaliveTimer = setInterval(() => {
            try {
              controller.enqueue(encoder.encode(`0:""\n`));
            } catch {
              // Ignore if controller closed
            }
          }, 3500);

          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split("\n");
              buffer = lines.pop() || "";

              for (const line of lines) {
                const trimmedLine = line.trim();
                if (!trimmedLine) {
                  currentEventType = "";
                  continue;
                }

                if (trimmedLine.startsWith("event:")) {
                  currentEventType = trimmedLine.slice(6).trim();
                  if (currentEventType === "complete") {
                    streamCompleted = true;
                  }
                } else if (trimmedLine.startsWith("data:")) {
                  const jsonStr = trimmedLine.slice(5).trim();
                  if (!jsonStr) continue;

                  try {
                    const parsed = JSON.parse(jsonStr);
                    // Gradio chatbot output is [[ [user, assistant], ... ], ""]
                    if (Array.isArray(parsed) && parsed.length > 0) {
                      const chatHistory = parsed[0];
                      if (Array.isArray(chatHistory) && chatHistory.length > 0) {
                        const lastTurn = chatHistory[chatHistory.length - 1];
                        if (Array.isArray(lastTurn) && lastTurn.length > 1) {
                          const currentFullText = lastTurn[1] || "";
                          if (currentFullText.length > lastFullText.length) {
                            const delta = currentFullText.slice(lastFullText.length);
                            lastFullText = currentFullText;
                            const cleanChunk = reasoningFilter.processChunk(delta);
                            if (cleanChunk) {
                              hasEmittedTokens = true;
                              // Emit AI SDK token chunk: 0:"<cleanChunk>"\n
                              controller.enqueue(encoder.encode(`0:${JSON.stringify(cleanChunk)}\n`));
                            }
                          }
                        }
                      }
                    }
                  } catch {
                    // Ignore parse errors on malformed heartbeat events
                  }

                  if (streamCompleted) {
                    clearInterval(keepaliveTimer);
                    const finalNotice = reasoningFilter.finalize();
                    if (finalNotice) {
                      hasEmittedTokens = true;
                      controller.enqueue(encoder.encode(`0:${JSON.stringify(finalNotice)}\n`));
                    }
                    controller.enqueue(encoder.encode(`d:{"finishReason":"stop"}\n`));
                    controller.close();
                    return;
                  }
                }
              }
            }
          } finally {
            clearInterval(keepaliveTimer);
          }

          // Finish stream if loop terminates
          const finalNotice = reasoningFilter.finalize();
          if (finalNotice) {
            hasEmittedTokens = true;
            controller.enqueue(encoder.encode(`0:${JSON.stringify(finalNotice)}\n`));
          }
          controller.enqueue(encoder.encode(`d:{"finishReason":"stop"}\n`));
          controller.close();
          return;
        }

        // Only fall back to /api/predict if no tokens have been streamed yet
        if (hasEmittedTokens) {
          controller.enqueue(encoder.encode(`d:{"finishReason":"stop"}\n`));
          controller.close();
          return;
        }

        // Fallback for Gradio /api/predict or /run/predict
        const predictUrl = `${normalized}/api/predict`;
        const predictResp = await fetch(predictUrl, {
          method: "POST",
          headers: authHeaders,
          body: JSON.stringify(callPayload),
        });

        if (!predictResp.ok) {
          throw new Error(`Gradio backend returned HTTP ${predictResp.status}`);
        }

        const predictData = (await predictResp.json()) as any;
        let responseText = "";

        if (predictData?.data && Array.isArray(predictData.data) && predictData.data.length > 0) {
          const chatHistory = predictData.data[0];
          if (Array.isArray(chatHistory) && chatHistory.length > 0) {
            const lastTurn = chatHistory[chatHistory.length - 1];
            if (Array.isArray(lastTurn) && lastTurn.length > 1) {
              responseText = lastTurn[1] || "";
            }
          }
        }

        if (!responseText && typeof predictData?.data === "string") {
          responseText = predictData.data;
        }

        // Filter reasoning traces server-side from non-streaming response
        const cleanResponse = filterReasoningFromText(responseText);

        // Stream complete text in chunks
        const chunkSize = 16;
        for (let i = 0; i < cleanResponse.length; i += chunkSize) {
          const slice = cleanResponse.slice(i, i + chunkSize);
          controller.enqueue(encoder.encode(`0:${JSON.stringify(slice)}\n`));
        }

        controller.enqueue(encoder.encode(`d:{"finishReason":"stop"}\n`));
        controller.close();
      } catch (err: any) {
        if (!hasEmittedTokens) {
          const errorMsg = err?.message || "Failed to communicate with Gradio inference backend";
          controller.enqueue(encoder.encode(`0:${JSON.stringify(`\n\n[Inference Error]: ${errorMsg}`)}\n`));
        }
        controller.enqueue(encoder.encode(`d:{"finishReason":"error"}\n`));
        controller.close();
      }
    },
  });
}

/**
 * Executes an on-demand non-streaming completion from the configured Gradio Space.
 * Useful for secondary tasks (e.g. reasoning explanation generation) without altering chat turns.
 */
export async function generateGradioText(options: GradioChatOptions): Promise<string> {
  const stream = await streamGradioChat(options);
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let fullText = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunkStr = decoder.decode(value, { stream: true });
    const lines = chunkStr.split("\n");
    for (const line of lines) {
      if (line.startsWith("0:")) {
        try {
          const content = JSON.parse(line.slice(2));
          if (typeof content === "string") {
            fullText += content;
          }
        } catch {}
      }
    }
  }
  return filterReasoningFromText(fullText);
}
