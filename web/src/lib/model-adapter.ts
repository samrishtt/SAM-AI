import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";
import { BackendStatus, BackendStatusCode } from "@/types/chat";
import { isGradioEndpoint, checkGradioHealth } from "./gradio-connector";

export type StreamModel = Parameters<typeof streamText>[0]["model"];

/**
 * SAM-AI Authoritative Sovereign Model & Adapter Identifiers
 */
export const PRIMARY_RESEARCH_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B";
export const PRIMARY_ADAPTER_PATH =
  "evidence/remote_r1_7b_run/extracted/sam_ai_r1_7b_gsm8k_adapter";

export type ModelProviderType = "sam_ai_sovereign";

/**
 * Known third-party managed inference services strictly prohibited by SAM-AI architecture.
 * SAM-AI models must run exclusively on owner-controlled self-hosted endpoints.
 */
export const PROHIBITED_MANAGED_ROUTER_HOSTS: readonly string[] = [
  "router.huggingface.co",
  "huggingface.co",
  "openrouter.ai",
  "api.openai.com",
  "api.together.xyz",
  "api.groq.com",
  "api.anthropic.com",
  "generativelanguage.googleapis.com",
  "api.deepseek.com",
];

export class ModelBackendNotConfiguredError extends Error {
  code = "MODEL_BACKEND_NOT_CONFIGURED";
  constructor(message?: string) {
    super(
      message ||
        "SAM-AI sovereign model backend endpoint is not configured. " +
          "Please set SAM_AI_BASE_URL in your server environment (.env.local) to point to your " +
          "self-hosted SAM-AI, vLLM, or SGLang inference server (e.g., http://127.0.0.1:8000/v1)."
    );
    this.name = "ModelBackendNotConfiguredError";
  }
}

export class ProhibitedProviderError extends Error {
  code = "PROHIBITED_THIRD_PARTY_PROVIDER";
  constructor(domain: string) {
    super(
      `Third-party managed inference router '${domain}' is rejected. ` +
        `SAM-AI architecture strictly forbids routing requests to commercial AI APIs or third-party hosted routers. ` +
        `Please configure SAM_AI_BASE_URL to point exclusively to your self-hosted or project-controlled server.`
    );
    this.name = "ProhibitedProviderError";
  }
}

export class UntrustedInferenceHostError extends Error {
  code = "UNTRUSTED_INFERENCE_HOST";
  constructor(message: string) {
    super(message);
    this.name = "UntrustedInferenceHostError";
  }
}

export interface EndpointValidationResult {
  isValid: boolean;
  isLoopback: boolean;
  hostname: string;
  normalizedBaseUrl?: string;
  healthCheckUrl?: string;
  error?: string;
  errorCode?: string;
}

/**
 * Checks whether a given hostname refers to a safe local loopback address.
 */
export function isLoopbackHostname(hostname: string): boolean {
  const lower = hostname.toLowerCase();
  if (
    lower === "localhost" ||
    lower === "127.0.0.1" ||
    lower === "0.0.0.0" ||
    lower === "::1" ||
    lower === "[::1]" ||
    lower.endsWith(".localhost")
  ) {
    return true;
  }
  // Standard IPv4 loopback subnet 127.0.0.0/8
  if (/^127(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d|\d)){3}$/.test(lower)) {
    return true;
  }
  return false;
}

/**
 * Retrieves explicitly trusted remote self-hosted hostnames from environment.
 */
export function getTrustedRemoteHosts(): Set<string> {
  const envHosts = process.env.SAM_AI_TRUSTED_HOSTS?.trim();
  if (!envHosts) return new Set();
  return new Set(
    envHosts
      .split(",")
      .map((h) => h.trim().toLowerCase())
      .filter(Boolean)
  );
}

/**
 * Validates whether an endpoint URL is permitted under SAM-AI security architecture:
 * 1. Permits loopback/local endpoints for local development.
 * 2. Permits remote HTTPS endpoints only when hostname is in SAM_AI_TRUSTED_HOSTS.
 * 3. Rejects third-party managed routers regardless of allowlist configuration.
 * 4. Rejects unexpected URL schemes and embedded credentials.
 * 5. Constructs normalized OpenAI-compatible base URL and health-check URL.
 */
export function validateInferenceEndpoint(urlStr: string): EndpointValidationResult {
  if (!urlStr || typeof urlStr !== "string" || urlStr.trim().length === 0) {
    return {
      isValid: false,
      isLoopback: false,
      hostname: "",
      error: "Inference endpoint URL is not configured.",
      errorCode: "EMPTY_ENDPOINT_URL",
    };
  }

  let parsed: URL;
  try {
    parsed = new URL(urlStr.trim());
  } catch {
    return {
      isValid: false,
      isLoopback: false,
      hostname: "",
      error: `Malformed endpoint URL: '${urlStr}'.`,
      errorCode: "MALFORMED_URL",
    };
  }

  // Reject embedded credentials in URL
  if (parsed.username || parsed.password) {
    return {
      isValid: false,
      isLoopback: false,
      hostname: parsed.hostname,
      error: "Credentials must not be embedded in SAM_AI_BASE_URL. Use server-side SAM_AI_API_KEY instead.",
      errorCode: "CREDENTIALS_IN_URL_FORBIDDEN",
    };
  }

  // Reject unexpected URL schemes
  const scheme = parsed.protocol.toLowerCase();
  if (scheme !== "http:" && scheme !== "https:") {
    return {
      isValid: false,
      isLoopback: false,
      hostname: parsed.hostname,
      error: `Invalid URL scheme '${scheme}'. Only HTTP and HTTPS are permitted.`,
      errorCode: "INVALID_URL_SCHEME",
    };
  }

  const hostname = parsed.hostname.toLowerCase();
  const loopback = isLoopbackHostname(hostname);

  // Prohibited third-party router check (OVERRIDE: always blocked regardless of allowlist)
  for (const prohibited of PROHIBITED_MANAGED_ROUTER_HOSTS) {
    if (hostname === prohibited || hostname.endsWith("." + prohibited)) {
      return {
        isValid: false,
        isLoopback: loopback,
        hostname,
        error: `Third-party managed inference router '${hostname}' is prohibited by SAM-AI sovereignty policy.`,
        errorCode: "PROHIBITED_THIRD_PARTY_PROVIDER",
      };
    }
  }

  // Enforce remote allowlist
  if (!loopback) {
    // Insecure HTTP is forbidden for remote endpoints
    if (scheme !== "https:") {
      return {
        isValid: false,
        isLoopback: false,
        hostname,
        error: `Insecure HTTP scheme is forbidden for remote endpoint '${hostname}'. Remote endpoints must use HTTPS.`,
        errorCode: "INSECURE_REMOTE_SCHEME",
      };
    }

    // Remote endpoints must be explicitly in SAM_AI_TRUSTED_HOSTS
    const trustedHosts = getTrustedRemoteHosts();
    if (!trustedHosts.has(hostname)) {
      return {
        isValid: false,
        isLoopback: false,
        hostname,
        error: `Remote inference host '${hostname}' is not in SAM_AI_TRUSTED_HOSTS allowlist.`,
        errorCode: "UNTRUSTED_REMOTE_HOST",
      };
    }
  }

  // Construct normalizedBaseUrl and healthCheckUrl
  const origin = parsed.origin;
  const rawPath = parsed.pathname.replace(/\/+$/, "");

  let normalizedBaseUrl: string;
  let healthCheckUrl: string;

  if (rawPath === "" || rawPath === "/") {
    normalizedBaseUrl = `${origin}/v1`;
    healthCheckUrl = `${origin}/health`;
  } else if (rawPath.endsWith("/v1")) {
    normalizedBaseUrl = `${origin}${rawPath}`;
    const basePathWithoutV1 = rawPath.slice(0, -3).replace(/\/+$/, "");
    healthCheckUrl = `${origin}${basePathWithoutV1}/health`;
  } else {
    normalizedBaseUrl = `${origin}${rawPath}/v1`;
    healthCheckUrl = `${origin}${rawPath}/health`;
  }

  return {
    isValid: true,
    isLoopback: loopback,
    hostname,
    normalizedBaseUrl,
    healthCheckUrl,
  };
}

/**
 * Validates whether an endpoint URL targets a prohibited external third-party router.
 */
export function checkProhibitedRouter(urlStr: string): { isProhibited: boolean; domain?: string } {
  const result = validateInferenceEndpoint(urlStr);
  if (result.errorCode === "PROHIBITED_THIRD_PARTY_PROVIDER") {
    return { isProhibited: true, domain: result.hostname };
  }
  return { isProhibited: false, domain: result.hostname };
}

/**
 * Returns the active provider type. Strictly sovereign self-hosted.
 */
export function getActiveProvider(): ModelProviderType {
  return "sam_ai_sovereign";
}

/**
 * Returns the active model identifier based on environment or canonical 7B default.
 */
export function getActiveModelId(): string {
  const envModel = process.env.SAM_AI_MODEL?.trim();
  if (envModel && envModel.length > 0) {
    return envModel;
  }
  return PRIMARY_RESEARCH_MODEL_ID;
}

/**
 * Returns the active adapter path if configured.
 */
export function getActiveAdapterPath(): string | undefined {
  const envAdapter = process.env.SAM_AI_ADAPTER_PATH?.trim();
  if (envAdapter && envAdapter.length > 0) {
    return envAdapter;
  }
  // The Experiment 001 adapter was rejected due to pilot regression.
  // Do not load or promote any adapter by default.
  return undefined;
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
 * System prompt establishing SAM-AI sovereign identity and boundaries.
 */
export const SAM_AI_SYSTEM_PROMPT = `You are SAM-AI, a self-hosted, sovereign artificial intelligence developed by Parallax Intelligence Lab.
Guidelines:
1. Provide accurate, clear, and well-structured answers with readable markdown formatting.
2. Be technically precise and transparent about your capabilities and limitations.
3. If code is requested, provide clean, idiomatic, well-commented code blocks.
4. Transparency: Maintain strict scientific integrity and do not claim to possess abilities or training you do not have.
5. Sovereign Operation: You run on a dedicated, owner-controlled inference backend.`;

export interface ModelInstanceResult {
  model: StreamModel;
  modelId: string;
  provider: ModelProviderType;
  providerName: string;
  maxTokens: number;
}

/**
 * Instantiates the sovereign language model via standard OpenAI-compatible protocol.
 * Rejects third-party commercial routers and fails closed if unconfigured or untrusted.
 */
export function getModelInstance(): ModelInstanceResult {
  const baseUrl = process.env.SAM_AI_BASE_URL?.trim();
  if (!baseUrl) {
    throw new ModelBackendNotConfiguredError();
  }

  // Security allowlist and blocklist validation
  const validation = validateInferenceEndpoint(baseUrl);
  if (!validation.isValid) {
    if (validation.errorCode === "PROHIBITED_THIRD_PARTY_PROVIDER") {
      throw new ProhibitedProviderError(validation.hostname || baseUrl);
    }
    throw new UntrustedInferenceHostError(validation.error || "Untrusted inference endpoint");
  }

  const modelId = getActiveModelId();
  const maxTokens = getMaxOutputTokens();

  // Server-side credential for authenticating only to the owner's controlled endpoint
  const apiKey = process.env.SAM_AI_API_KEY?.trim() || "EMPTY";

  const openaiProvider = createOpenAI({
    baseURL: validation.normalizedBaseUrl,
    apiKey,
  });

  const model = openaiProvider(modelId);

  return {
    model,
    modelId,
    provider: "sam_ai_sovereign",
    providerName: "SAM-AI Sovereign Inference Engine",
    maxTokens,
  };
}

/**
 * Performs a bounded-time HTTP health check against the controlled backend server.
 * Distinguishes: not_configured, prohibited_provider, unreachable, loading,
 * weights_missing, model_mismatch, and ready.
 */
export async function inspectBackendReadiness(): Promise<BackendStatus> {
  const model = getActiveModelId();
  const adapter = getActiveAdapterPath();
  const maxOutputTokens = getMaxOutputTokens();
  const rateLimit = parseInt(process.env.SAM_AI_RATE_LIMIT_PER_MINUTE || "20", 10);
  const requiresPassword = Boolean(process.env.SAM_AI_SITE_PASSWORD?.trim());

  const baseUrl = process.env.SAM_AI_BASE_URL?.trim();

  // 1. Not configured
  if (!baseUrl) {
    return {
      configured: false,
      status: "not_configured",
      model,
      adapter,
      provider: "sam_ai_sovereign",
      providerName: "SAM-AI Sovereign Inference Server",
      requiresPassword,
      rateLimitPerMinute: rateLimit,
      maxOutputTokens,
      notice: "SAM_AI_BASE_URL is not configured in .env.local.",
      setupGuide: "Set SAM_AI_BASE_URL=http://127.0.0.1:8000/v1 to connect to your local SAM-AI server.",
      privacyNotice: "Dedicated sovereign endpoint. Inputs processed strictly locally.",
    };
  }

  // 2. Validate endpoint against security allowlist and prohibited router blocklist
  const validation = validateInferenceEndpoint(baseUrl);
  if (!validation.isValid) {
    const isProhibited = validation.errorCode === "PROHIBITED_THIRD_PARTY_PROVIDER";
    return {
      configured: false,
      status: isProhibited ? "prohibited_provider" : "not_configured",
      model,
      adapter,
      provider: "sam_ai_sovereign",
      providerName: isProhibited ? "Prohibited External Service" : "Untrusted Endpoint",
      endpoint: validation.hostname || baseUrl,
      requiresPassword,
      rateLimitPerMinute: rateLimit,
      maxOutputTokens,
      notice: validation.error || "Inference endpoint failed security policy validation.",
      setupGuide: isProhibited
        ? "Switch SAM_AI_BASE_URL to an owner-controlled self-hosted server (e.g. http://127.0.0.1:8000/v1)."
        : "Permit loopback (127.0.0.1) or add remote hostname to SAM_AI_TRUSTED_HOSTS.",
      privacyNotice: "Unverified and commercial endpoints are rejected by policy.",
      backendError: validation.error,
    };
  }

  // 3. Gradio Space Health Inspection
  if (isGradioEndpoint(baseUrl)) {
    const gradioHealth = await checkGradioHealth(baseUrl, process.env.SAM_AI_API_KEY?.trim());
    return {
      configured: true,
      status: gradioHealth.isReady ? "ready" : "unreachable",
      model: gradioHealth.modelName || model,
      adapter: undefined, // Pure foundation model, no rejected adapter
      provider: "sam_ai_sovereign",
      providerName: "SAM-AI Sovereign ZeroGPU Space",
      endpoint: validation.hostname || baseUrl,
      requiresPassword,
      rateLimitPerMinute: rateLimit,
      maxOutputTokens,
      notice: gradioHealth.detail,
      setupGuide: gradioHealth.isReady
        ? "Connected to sovereign Gradio ZeroGPU Space."
        : "Check that the Hugging Face ZeroGPU Space is running and healthy.",
      privacyNotice: "Dedicated sovereign endpoint.",
    };
  }

  // 4. Bounded-time Health Inspection (2.0 second timeout)
  const healthUrl = validation.healthCheckUrl!;

  try {
    const res = await fetch(healthUrl, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(2000),
      cache: "no-store",
    });

    if (!res.ok) {
      return {
        configured: true,
        status: "unreachable",
        model,
        adapter,
        provider: "sam_ai_sovereign",
        providerName: "SAM-AI Sovereign Inference Server",
        requiresPassword,
        rateLimitPerMinute: rateLimit,
        maxOutputTokens,
        notice: `Backend health endpoint returned HTTP ${res.status}.`,
        setupGuide: "Verify the SAM-AI inference server process is running cleanly.",
        privacyNotice: "Dedicated sovereign endpoint.",
        backendError: `Server responded with HTTP ${res.status}`,
      };
    }

    const data = await res.json();

    // Verify model identity
    const reportedModel = data.model;
    if (reportedModel && reportedModel !== model) {
      return {
        configured: true,
        status: "model_mismatch",
        model,
        adapter,
        provider: "sam_ai_sovereign",
        providerName: "SAM-AI Sovereign Inference Server",
        requiresPassword,
        rateLimitPerMinute: rateLimit,
        maxOutputTokens,
        notice: `Model mismatch: Web configured for '${model}', server serving '${reportedModel}'.`,
        setupGuide: "Align SAM_AI_MODEL in web/.env.local with the server's --model argument.",
        privacyNotice: "Dedicated sovereign endpoint.",
        isLoaded: false,
      };
    }

    // Verify adapter identity if server reports it and we have an adapter configured
    const reportedAdapter = data.adapter;
    if (adapter && reportedAdapter && reportedAdapter !== adapter) {
      return {
        configured: true,
        status: "model_mismatch",
        model,
        adapter,
        provider: "sam_ai_sovereign",
        providerName: "SAM-AI Sovereign Inference Server",
        requiresPassword,
        rateLimitPerMinute: rateLimit,
        maxOutputTokens,
        notice: `Adapter mismatch: Web configured for '${adapter}', server loaded '${reportedAdapter}'.`,
        setupGuide: "Align SAM_AI_ADAPTER_PATH in web/.env.local with server's --adapter path.",
        privacyNotice: "Dedicated sovereign endpoint.",
        isLoaded: false,
      };
    }

    // Inspect server inference status
    const infStatus = data.inference_status;
    if (infStatus === "BLOCKED_MISSING_WEIGHTS") {
      return {
        configured: true,
        status: "weights_missing",
        model,
        adapter,
        provider: "sam_ai_sovereign",
        providerName: "SAM-AI Sovereign Inference Server",
        requiresPassword,
        rateLimitPerMinute: rateLimit,
        maxOutputTokens,
        notice: `Model weights absent on server: ${data.issues?.[0] || "Weights not found"}.`,
        setupGuide: "Run the server on a host with model weights or run preflight diagnostics.",
        privacyNotice: "Dedicated sovereign endpoint. Fails closed safely.",
        isLoaded: false,
      };
    }

    if (infStatus === "LOADING") {
      return {
        configured: true,
        status: "loading",
        model,
        adapter,
        provider: "sam_ai_sovereign",
        providerName: "SAM-AI Sovereign Inference Server",
        requiresPassword,
        rateLimitPerMinute: rateLimit,
        maxOutputTokens,
        notice: "Model weights are currently being loaded into memory...",
        setupGuide: "Wait for the server to finish weight loading.",
        privacyNotice: "Dedicated sovereign endpoint.",
        isLoaded: false,
      };
    }

    if ((data.is_loaded === true || data.is_ready === true) && (data.status === "ok" || data.status === "ready")) {
      return {
        configured: true,
        status: "ready",
        model,
        adapter,
        provider: "sam_ai_sovereign",
        providerName: "SAM-AI Sovereign Inference Server",
        requiresPassword,
        rateLimitPerMinute: rateLimit,
        maxOutputTokens,
        notice: "SAM-AI sovereign model is online and ready.",
        setupGuide: "Connected to local inference server.",
        privacyNotice: "Dedicated sovereign endpoint. Operating locally.",
        isLoaded: true,
      };
    }

    // Default fallback if not fully ready
    return {
      configured: true,
      status: "weights_missing",
      model,
      adapter,
      provider: "sam_ai_sovereign",
      providerName: "SAM-AI Sovereign Inference Server",
      requiresPassword,
      rateLimitPerMinute: rateLimit,
      maxOutputTokens,
      notice: `Server status: ${data.inference_status || data.status || "degraded"}.`,
      setupGuide: "Inspect server logs for hardware or dependency diagnostics.",
      privacyNotice: "Dedicated sovereign endpoint.",
      isLoaded: Boolean(data.is_loaded),
    };
  } catch (err: any) {
    const isTimeout = err?.name === "TimeoutError";
    return {
      configured: true,
      status: "unreachable",
      model,
      adapter,
      provider: "sam_ai_sovereign",
      providerName: "SAM-AI Sovereign Inference Server",
      requiresPassword,
      rateLimitPerMinute: rateLimit,
      maxOutputTokens,
      notice: isTimeout
        ? "Inference server health check timed out (2.0s)."
        : "Cannot connect to inference server at configured URL.",
      setupGuide: "Ensure your inference server is running on http://127.0.0.1:8000.",
      privacyNotice: "Dedicated sovereign endpoint.",
      backendError: err?.message || "Connection refused",
    };
  }
}
