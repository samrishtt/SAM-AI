import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";

export type StreamModel = Parameters<typeof streamText>[0]["model"];

/**
 * SAM-AI Canonical Model Identifiers and Types
 */
export const PRIMARY_RESEARCH_MODEL_ID = "Qwen/Qwen3.8-27B";
export const LEGACY_BASELINE_MODEL_ID =
  "gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1";

export type ModelProviderType = "qwen_openai_compatible" | "google";

export class ModelBackendNotConfiguredError extends Error {
  code = "MODEL_BACKEND_NOT_CONFIGURED";
  constructor(message?: string) {
    super(
      message ||
        "Qwen model backend endpoint is not configured. " +
          "Please set SAM_AI_BASE_URL in your server environment (.env.local) to point to your " +
          "vLLM, SGLang, or OpenAI-compatible inference server running Qwen/Qwen3.8-27B."
    );
    this.name = "ModelBackendNotConfiguredError";
  }
}

/**
 * Returns the currently active provider type.
 * Defaults to "qwen_openai_compatible" as the sovereign primary backend.
 * "google" is supported strictly as an explicit opt-in provider.
 */
export function getActiveProvider(): ModelProviderType {
  const provider = process.env.SAM_AI_PROVIDER?.trim().toLowerCase();
  if (provider === "google") {
    return "google";
  }
  return "qwen_openai_compatible";
}

/**
 * Returns the active model identifier based on provider and environment.
 */
export function getActiveModelId(): string {
  const envModel = process.env.SAM_AI_MODEL?.trim();
  if (envModel && envModel.length > 0) {
    return envModel;
  }
  const provider = getActiveProvider();
  if (provider === "google") {
    return "gemini-2.0-flash";
  }
  return PRIMARY_RESEARCH_MODEL_ID;
}

/**
 * Returns the maximum output tokens configured for the session.
 */
export function getMaxOutputTokens(): number {
  const envTokens = process.env.SAM_AI_MAX_OUTPUT_TOKENS?.trim();
  if (envTokens) {
    const parsed = parseInt(envTokens, 10);
    if (!isNaN(parsed) && parsed > 0 && parsed <= 8192) {
      return parsed;
    }
  }
  return 2048;
}

/**
 * System prompt establishing SAM-AI identity and boundaries.
 */
export const SAM_AI_SYSTEM_PROMPT = `You are SAM-AI, a helpful, honest, and sovereign artificial intelligence assistant.
Guidelines:
1. Provide accurate, clear, and well-structured answers with readable markdown formatting.
2. Be technically precise and transparent about your capabilities and limitations.
3. If code is requested, provide clean, idiomatic, well-commented code blocks.
4. Transparency: Maintain strict scientific integrity and do not claim to possess abilities or training you do not have.`;

export interface ModelInstanceResult {
  model: StreamModel;
  modelId: string;
  provider: ModelProviderType;
  providerName: string;
  maxTokens: number;
}

/**
/**
 * Returns the generation temperature configured for the session.
 */
export function getGenerationTemperature(): number {
  const envTemp = process.env.SAM_AI_TEMPERATURE?.trim();
  if (envTemp) {
    const parsed = parseFloat(envTemp);
    if (!isNaN(parsed) && parsed >= 0.0 && parsed <= 2.0) {
      return parsed;
    }
  }
  return 0.7;
}

/**
 * Instantiates and returns the configured language model.
 *
 * Rules:
 * - If SAM_AI_PROVIDER is 'qwen_openai_compatible' (default), connects via OpenAI-compatible endpoint.
 *   Throws ModelBackendNotConfiguredError if SAM_AI_BASE_URL is not set.
 *   Throws an error if a cloud router (e.g. Hugging Face) is targeted without an API token.
 *   NEVER silently falls back to Google!
 * - If SAM_AI_PROVIDER is 'google', connects via Google Generative AI.
 *   Throws an error if GOOGLE_GENERATIVE_AI_API_KEY is not configured.
 */
export function getModelInstance(): ModelInstanceResult {
  const provider = getActiveProvider();
  const modelId = getActiveModelId();
  const maxTokens = getMaxOutputTokens();

  if (provider === "qwen_openai_compatible") {
    const baseUrl = process.env.SAM_AI_BASE_URL?.trim();
    if (!baseUrl) {
      throw new ModelBackendNotConfiguredError();
    }

    // Support SAM_AI_API_KEY or standard HF_TOKEN / HUGGINGFACE_TOKEN
    const apiKey =
      process.env.SAM_AI_API_KEY?.trim() ||
      process.env.HF_TOKEN?.trim() ||
      process.env.HUGGINGFACE_TOKEN?.trim() ||
      "EMPTY";

    // If pointing to Hugging Face router, check for required authentication token
    const isHfRouter = baseUrl.toLowerCase().includes("huggingface.co");
    if (isHfRouter && (apiKey === "EMPTY" || !apiKey)) {
      throw new Error(
        "Hugging Face Inference Providers router requires a user access token. " +
          "Please configure SAM_AI_API_KEY or HF_TOKEN in your server environment (.env.local)."
      );
    }

    const openaiProvider = createOpenAI({
      baseURL: baseUrl,
      apiKey: apiKey,
    });

    const model = openaiProvider(modelId);

    return {
      model,
      modelId,
      provider: "qwen_openai_compatible",
      providerName: isHfRouter
        ? "Hugging Face Inference Router (Qwen/Qwen3.8-27B)"
        : "Qwen OpenAI-Compatible Inference Server",
      maxTokens,
    };
  }

  // Google Provider (explicit opt-in)
  const apiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY?.trim();
  if (!apiKey) {
    throw new Error(
      "GOOGLE_GENERATIVE_AI_API_KEY is not configured on the server. " +
        "When SAM_AI_PROVIDER=google is selected, a Google AI Studio API key must be set in .env.local or your deployment settings."
    );
  }

  const googleProvider = createGoogleGenerativeAI({
    apiKey,
  });

  const model = googleProvider(modelId);

  return {
    model,
    modelId,
    provider: "google",
    providerName: "Google Gemini (Optional Pretrained Provider)",
    maxTokens,
  };
}

/**
 * Returns safe status information for public /api/status route.
 * Never exposes API keys or sensitive credentials.
 */
export function getProviderStatus(): {
  provider: ModelProviderType;
  providerName: string;
  model: string;
  configured: boolean;
  status: "ready" | "not_configured" | "missing_api_key";
  endpoint: string;
  notice: string;
  setupGuide: string;
} {
  const provider = getActiveProvider();
  const model = getActiveModelId();

  if (provider === "qwen_openai_compatible") {
    const baseUrl = process.env.SAM_AI_BASE_URL?.trim();
    const apiKey =
      process.env.SAM_AI_API_KEY?.trim() ||
      process.env.HF_TOKEN?.trim() ||
      process.env.HUGGINGFACE_TOKEN?.trim();

    let sanitizedEndpoint = "NOT_CONFIGURED";
    let isHf = false;

    if (baseUrl) {
      try {
        const parsed = new URL(baseUrl);
        sanitizedEndpoint = `${parsed.protocol}//${parsed.host}${parsed.pathname}`;
        isHf = parsed.host.includes("huggingface.co");
      } catch {
        sanitizedEndpoint = baseUrl.replace(/\/\/.*@/, "//***@");
        isHf = baseUrl.includes("huggingface.co");
      }
    }

    if (!baseUrl) {
      return {
        provider: "qwen_openai_compatible",
        providerName: "Qwen Sovereign Inference Endpoint",
        model,
        configured: false,
        status: "not_configured",
        endpoint: "NOT_CONFIGURED",
        notice: `Backend endpoint for ${model} is not configured. Set SAM_AI_BASE_URL in server environment.`,
        setupGuide:
          "To enable Qwen3.8-27B, add SAM_AI_BASE_URL (e.g. https://router.huggingface.co/v1 or http://localhost:8000/v1) and SAM_AI_API_KEY to web/.env.local.",
      };
    }

    if (isHf && !apiKey) {
      return {
        provider: "qwen_openai_compatible",
        providerName: "Hugging Face Inference Providers",
        model,
        configured: false,
        status: "missing_api_key",
        endpoint: sanitizedEndpoint,
        notice: "Hugging Face router endpoint is configured, but SAM_AI_API_KEY or HF_TOKEN is missing.",
        setupGuide: "Set SAM_AI_API_KEY=hf_... in web/.env.local to authenticate with the Hugging Face router.",
      };
    }

    return {
      provider: "qwen_openai_compatible",
      providerName: isHf ? "Hugging Face Inference Router" : "Qwen Inference Server",
      model,
      configured: true,
      status: "ready",
      endpoint: sanitizedEndpoint,
      notice: `Connected to ${model} via ${sanitizedEndpoint}.`,
      setupGuide: "Inference endpoint is ready to receive requests.",
    };
  }

  const hasApiKey = Boolean(process.env.GOOGLE_GENERATIVE_AI_API_KEY?.trim());
  return {
    provider: "google",
    providerName: "Google Gemini (Optional Provider)",
    model,
    configured: hasApiKey,
    status: hasApiKey ? "ready" : "missing_api_key",
    endpoint: "https://generativelanguage.googleapis.com",
    notice: hasApiKey
      ? `Operating with ${model} via Google API.`
      : "Google API key is not configured.",
    setupGuide: hasApiKey
      ? "Google provider is ready."
      : "Set GOOGLE_GENERATIVE_AI_API_KEY in web/.env.local to activate Gemini.",
  };
}

