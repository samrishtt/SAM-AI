import { generateText } from "ai";
import { getModelInstance, getGenerationTemperature, validateInferenceEndpoint } from "@/lib/model-adapter";
import { isGradioEndpoint, generateGradioText } from "@/lib/gradio-connector";
import { validateSitePassword, getClientIp } from "@/lib/security";
import { checkRateLimit } from "@/lib/rate-limiter";
import { filterReasoningFromText } from "@/lib/reasoning-filter";

export const maxDuration = 30; // Max execution duration in seconds

const EXPLANATION_SYSTEM_PROMPT =
  "You are SAM-AI. Provide a concise, user-facing summary of the method, logic, and key steps " +
  "behind the provided answer in 2-4 clear bullet points. Do not reproduce the full answer or invent synthetic thoughts. " +
  "Keep it high-level, directly addressing how the solution was constructed.";

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

  const { prompt, answer } = body;
  if (!prompt || typeof prompt !== "string" || !prompt.trim()) {
    return new Response(
      JSON.stringify({ error: "Request must include a non-empty 'prompt' string", code: "BAD_REQUEST" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }
  if (!answer || typeof answer !== "string" || !answer.trim()) {
    return new Response(
      JSON.stringify({ error: "Request must include a non-empty 'answer' string", code: "BAD_REQUEST" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const trimmedPrompt = prompt.trim();
  const trimmedAnswer = answer.trim();

  // 4. Check for Trivial / Greeting Responses
  const isGreeting = /^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)[\s!.]*$/i.test(trimmedPrompt);
  if (isGreeting || trimmedAnswer.length < 30) {
    return new Response(
      JSON.stringify({
        explanation: "This response is a direct greeting or concise acknowledgement. No detailed derivation was necessary.",
        isTrivial: true,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }

  // 5. Construct Isolated Explanation Request
  const explainMessages = [
    {
      role: "user" as const,
      content:
        `Explain the core method, principles, or reasoning steps used to arrive at the following answer.\n` +
        `Keep the explanation high-level, clear, and structured in 2-4 brief bullet points. Do not repeat the entire answer.\n\n` +
        `User Prompt:\n${trimmedPrompt}\n\n` +
        `Final Answer:\n${trimmedAnswer}`,
    },
  ];

  // 6. Execute via Gradio Space or OpenAI-compatible backend
  const configuredBaseUrl = process.env.SAM_AI_BASE_URL?.trim();
  if (configuredBaseUrl && isGradioEndpoint(configuredBaseUrl)) {
    const validation = validateInferenceEndpoint(configuredBaseUrl);
    if (!validation.isValid) {
      return new Response(
        JSON.stringify({ error: validation.error, code: "UNTRUSTED_INFERENCE_HOST" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    try {
      const apiKey = process.env.SAM_AI_API_KEY?.trim();
      const rawText = await generateGradioText({
        baseUrl: configuredBaseUrl,
        messages: explainMessages,
        temperature: 0.3,
        maxTokens: 512,
        apiKey,
        systemPrompt: EXPLANATION_SYSTEM_PROMPT,
      });

      const cleanExplanation = filterReasoningFromText(rawText);
      return new Response(
        JSON.stringify({
          explanation: cleanExplanation || "Method explanation completed directly.",
          isTrivial: false,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    } catch (err: any) {
      console.error("[SAM-AI Explain] Gradio generation error:", err?.message || err);
      return new Response(
        JSON.stringify({
          error: err?.message || "Failed to generate explanation from Gradio Space",
          code: "PROVIDER_ERROR",
        }),
        { status: 502, headers: { "Content-Type": "application/json" } }
      );
    }
  }

  // Standard OpenAI-Compatible Backend
  try {
    const modelInstance = getModelInstance();
    const result = await generateText({
      model: modelInstance.model,
      system: EXPLANATION_SYSTEM_PROMPT,
      messages: explainMessages,
      maxTokens: 512,
      temperature: 0.3,
    });

    const cleanExplanation = filterReasoningFromText(result.text);
    return new Response(
      JSON.stringify({
        explanation: cleanExplanation || "Method explanation completed directly.",
        isTrivial: false,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  } catch (err: any) {
    console.error("[SAM-AI Explain] Model generation error:", err?.message || err);
    return new Response(
      JSON.stringify({
        error: err?.message || "Failed to generate explanation from model provider",
        code: "PROVIDER_ERROR",
      }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }
}
