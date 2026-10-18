import { streamText } from "ai";
import { getModelInstance, getGenerationTemperature, SAM_AI_SYSTEM_PROMPT, validateInferenceEndpoint } from "@/lib/model-adapter";
import { isGradioEndpoint, streamGradioChat } from "@/lib/gradio-connector";
import { validateSitePassword, getClientIp } from "@/lib/security";
import { checkRateLimit } from "@/lib/rate-limiter";

export const maxDuration = 60; // Max streaming execution duration in seconds

export async function POST(req: Request) {
  // 1. Password Gate Validation (if configured)
  const clientPassword = req.headers.get("x-sam-ai-password");
  const authCheck = validateSitePassword(clientPassword);
  if (!authCheck.authorized) {
    return new Response(
      JSON.stringify({
        error: authCheck.error || "Access password required. Please unlock with the site password.",
        code: "UNAUTHORIZED",
      }),
      { status: 401, headers: { "Content-Type": "application/json" } }
    );
  }

  // 2. Best-Effort In-Memory Rate Limiting
  const clientIp = getClientIp(req.headers);
  const maxReqs = parseInt(process.env.SAM_AI_RATE_LIMIT_PER_MINUTE || "20", 10);
  const rateLimit = checkRateLimit(clientIp, maxReqs);
  if (!rateLimit.allowed) {
    return new Response(
      JSON.stringify({
        error: `Rate limit reached (${maxReqs} req/min). Please wait ${rateLimit.resetSeconds}s before retrying.`,
        code: "RATE_LIMITED",
        resetSeconds: rateLimit.resetSeconds,
      }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "Retry-After": String(rateLimit.resetSeconds),
        },
      }
    );
  }

  // 3. Payload Extraction and Validation
  let body: any;
  try {
    body = await req.json();
  } catch {
    return new Response(
      JSON.stringify({ error: "Invalid JSON payload in request body", code: "BAD_REQUEST" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const { messages } = body;
  if (!Array.isArray(messages) || messages.length === 0) {
    return new Response(
      JSON.stringify({ error: "Request must include a non-empty 'messages' array", code: "BAD_REQUEST" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  // Limit conversation history depth
  if (messages.length > 50) {
    return new Response(
      JSON.stringify({
        error: "Conversation history exceeds maximum depth (50 messages). Please start a new conversation.",
        code: "PAYLOAD_TOO_LARGE",
      }),
      { status: 413, headers: { "Content-Type": "application/json" } }
    );
  }

  // Validate message contents & character length limits
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") {
      return new Response(
        JSON.stringify({ error: "Malformed message object inside 'messages' array", code: "BAD_REQUEST" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }
    if (typeof msg.content === "string" && msg.content.length > 8000) {
      return new Response(
        JSON.stringify({
          error: "Message exceeds maximum allowed length (8,000 characters). Please condense your input.",
          code: "MESSAGE_TOO_LONG",
        }),
        { status: 413, headers: { "Content-Type": "application/json" } }
      );
    }
  }

  // 4. Check for Gradio Space Backend vs. OpenAI-compatible backend
  const configuredBaseUrl = process.env.SAM_AI_BASE_URL?.trim();
  if (configuredBaseUrl && isGradioEndpoint(configuredBaseUrl)) {
    const validation = validateInferenceEndpoint(configuredBaseUrl);
    if (!validation.isValid) {
      const isProhibited = validation.errorCode === "PROHIBITED_THIRD_PARTY_PROVIDER";
      return new Response(
        JSON.stringify({
          error: validation.error,
          code: isProhibited ? "PROHIBITED_THIRD_PARTY_PROVIDER" : "UNTRUSTED_INFERENCE_HOST",
        }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    try {
      console.log(
        `[SAM-AI Chat] Invoking Gradio Space='${configuredBaseUrl}', messagesCount=${messages.length}, clientHash=${clientIp.slice(0, 6)}...`
      );
      const temperature = getGenerationTemperature();
      const apiKey = process.env.SAM_AI_API_KEY?.trim();
      const gradioStream = await streamGradioChat({
        baseUrl: configuredBaseUrl,
        messages,
        temperature,
        apiKey,
        systemPrompt: SAM_AI_SYSTEM_PROMPT,
      });

      return new Response(gradioStream, {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "x-vercel-ai-data-stream": "v1",
        },
      });
    } catch (gradioErr: any) {
      console.error("[SAM-AI Chat] Gradio streaming error:", gradioErr?.message || gradioErr);
      return new Response(
        JSON.stringify({
          error: gradioErr?.message || "Failed to generate stream response from Gradio Space",
          code: "PROVIDER_STREAM_ERROR",
        }),
        { status: 502, headers: { "Content-Type": "application/json" } }
      );
    }
  }

  // 5. Instantiate OpenAI-Compatible Model Adapter
  let modelInstance;
  try {
    modelInstance = getModelInstance();
  } catch (err: any) {
    const isProhibited = err?.code === "PROHIBITED_THIRD_PARTY_PROVIDER";
    const isBackendNotConfigured = err?.code === "MODEL_BACKEND_NOT_CONFIGURED";
    const isUntrustedHost = err?.code === "UNTRUSTED_INFERENCE_HOST";

    let statusCode = 500;
    if (isProhibited || isUntrustedHost) statusCode = 400;
    else if (isBackendNotConfigured) statusCode = 503;

    return new Response(
      JSON.stringify({
        error: err.message || "Model provider configuration error",
        code: isProhibited
          ? "PROHIBITED_THIRD_PARTY_PROVIDER"
          : isUntrustedHost
          ? "UNTRUSTED_INFERENCE_HOST"
          : isBackendNotConfigured
          ? "MODEL_BACKEND_NOT_CONFIGURED"
          : "PROVIDER_CONFIG_ERROR",
      }),
      {
        status: statusCode,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // 5. Stream Output via Vercel AI SDK
  try {
    // Audit log: record request metadata only (NEVER log private conversation text)
    console.log(
      `[SAM-AI Chat] Invoking model='${modelInstance.modelId}', messagesCount=${messages.length}, clientHash=${clientIp.slice(0, 6)}...`
    );

    const temperature = getGenerationTemperature();
    const result = streamText({
      model: modelInstance.model,
      system: SAM_AI_SYSTEM_PROMPT,
      messages,
      maxTokens: modelInstance.maxTokens,
      temperature,
    });

    return result.toDataStreamResponse({
      getErrorMessage: (err: any) => err?.message || "Stream transmission interrupted by model provider",
    });
  } catch (providerError: any) {
    console.error("[SAM-AI Chat] Provider streaming error:", providerError?.message || providerError);
    return new Response(
      JSON.stringify({
        error: providerError?.message || "Failed to generate stream response from model provider",
        code: "PROVIDER_STREAM_ERROR",
      }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }
}
