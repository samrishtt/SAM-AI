/**
 * SAM-AI Web Search Provider Abstraction
 * =====================================
 * Pluggable search engine drivers conforming to WebSearchProvider interface:
 * 1. MockSearchProvider: Deterministic fixture driver for testing.
 * 2. SearXNGSearchProvider: Configurable, self-hosted meta-search engine JSON API client.
 * 3. SearchNotConfiguredError: Clear structured failure when search is unconfigured.
 *
 * Security Invariants:
 * - Never silently query an unauthenticated public SearXNG instance.
 * - Endpoint must be deliberately configured via SAM_AI_SEARXNG_BASE_URL.
 * - Server-side only: API keys and internal search configurations are never sent to client.
 */

import type {
  WebSearchProvider,
  SearchOptions,
  SearchResult,
  SearchProviderStatus,
} from "./types.ts";
import {
  SearchProviderError,
  SearchNotConfiguredError,
} from "./types.ts";

export { SearchProviderError, SearchNotConfiguredError };
import { validateUrlSyntax } from "./ssrf-guard.ts";

// ---------------------------------------------------------------------------
// 1. Mock Search Provider (Deterministic testing)
// ---------------------------------------------------------------------------

export interface MockSearchFixture {
  queryMatch?: RegExp | string;
  results: SearchResult[];
}

export class MockSearchProvider implements WebSearchProvider {
  public readonly name = "mock";
  private fixtures: MockSearchFixture[] = [];
  private defaultResults: SearchResult[];
  private shouldFailWith?: SearchProviderError;
  private delayMs: number = 0;

  constructor(options: {
    fixtures?: MockSearchFixture[];
    defaultResults?: SearchResult[];
    shouldFailWith?: SearchProviderError;
    delayMs?: number;
  } = {}) {
    this.fixtures = options.fixtures || [];
    this.shouldFailWith = options.shouldFailWith;
    this.delayMs = options.delayMs || 0;

    this.defaultResults = options.defaultResults || [
      {
        url: "https://docs.python.org/3/library/math.html",
        title: "math — Mathematical functions — Python documentation",
        snippet: "This module provides access to the mathematical functions defined by the C standard.",
        provider: "mock",
        retrievedAt: new Date().toISOString(),
        score: 0.95,
        engine: "mock_engine",
      },
      {
        url: "https://en.wikipedia.org/wiki/Prime_number",
        title: "Prime number - Wikipedia",
        snippet: "A prime number is a natural number greater than 1 that is not a product of two smaller natural numbers.",
        provider: "mock",
        retrievedAt: new Date().toISOString(),
        score: 0.91,
        engine: "mock_engine",
      },
      {
        url: "https://arxiv.org/abs/1706.03762",
        title: "Attention Is All You Need",
        snippet: "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        provider: "mock",
        retrievedAt: new Date().toISOString(),
        score: 0.88,
        engine: "mock_engine",
      },
    ];
  }

  public setFixtures(fixtures: MockSearchFixture[]): void {
    this.fixtures = fixtures;
  }

  public setFailure(err?: SearchProviderError): void {
    this.shouldFailWith = err;
  }

  public async search(options: SearchOptions): Promise<SearchResult[]> {
    if (!options.query || options.query.trim().length === 0) {
      throw new SearchProviderError("INVALID_SEARCH_QUERY", "Search query cannot be empty", this.name);
    }

    if (this.delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.delayMs));
    }

    if (this.shouldFailWith) {
      throw this.shouldFailWith;
    }

    const trimmedQuery = options.query.trim();
    const limit = Math.min(Math.max(1, options.limit ?? 5), 10);

    for (const fixture of this.fixtures) {
      if (fixture.queryMatch) {
        const matches =
          fixture.queryMatch instanceof RegExp
            ? fixture.queryMatch.test(trimmedQuery)
            : trimmedQuery.toLowerCase().includes(fixture.queryMatch.toLowerCase());
        if (matches) {
          return fixture.results.slice(0, limit);
        }
      }
    }

    return this.defaultResults.slice(0, limit);
  }

  public async checkHealth(): Promise<SearchProviderStatus> {
    return {
      name: this.name,
      configured: true,
      available: true,
      message: "Mock search provider operational",
    };
  }
}

// ---------------------------------------------------------------------------
// 2. SearXNG Search Provider (Self-hosted Meta-Search JSON Client)
// ---------------------------------------------------------------------------

export interface SearXNGConfig {
  baseUrl: string;
  apiKey?: string;
  timeoutMs?: number;
  customFetch?: typeof fetch;
}

export class SearXNGSearchProvider implements WebSearchProvider {
  public readonly name = "searxng";
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly fetchFn: typeof fetch;

  constructor(config: SearXNGConfig) {
    if (!config.baseUrl || config.baseUrl.trim().length === 0) {
      throw new SearchNotConfiguredError(
        "SearXNG base URL is required. Please set SAM_AI_SEARXNG_BASE_URL in your server environment."
      );
    }

    const trimmedUrl = config.baseUrl.trim().replace(/\/+$/, "");
    // Validate scheme and port: SearXNG must be HTTPS (or loopback for local development)
    const syntax = validateUrlSyntax(trimmedUrl, ["https:", "http:"], [443, 80, 8080, 8443, 8888]);
    if (!syntax.valid || !syntax.parsedUrl) {
      throw new SearchProviderError(
        "SEARCH_PROVIDER_ERROR",
        `Invalid SearXNG base URL: ${syntax.error}`,
        this.name
      );
    }

    this.baseUrl = trimmedUrl;
    this.apiKey = config.apiKey?.trim();
    this.timeoutMs = config.timeoutMs ?? 6000;
    this.fetchFn = config.customFetch ?? globalThis.fetch;
  }

  public async search(options: SearchOptions): Promise<SearchResult[]> {
    if (!options.query || options.query.trim().length === 0) {
      throw new SearchProviderError("INVALID_SEARCH_QUERY", "Search query cannot be empty", this.name);
    }

    const limit = Math.min(Math.max(1, options.limit ?? 5), 10);
    const searchUrl = new URL(`${this.baseUrl}/search`);
    searchUrl.searchParams.set("q", options.query.trim());
    searchUrl.searchParams.set("format", "json");

    if (options.language) {
      searchUrl.searchParams.set("language", options.language);
    }
    if (options.timeRange && options.timeRange !== "all") {
      searchUrl.searchParams.set("time_range", options.timeRange);
    }
    if (options.safesearch !== undefined) {
      searchUrl.searchParams.set("safesearch", String(options.safesearch));
    }
    if (options.categories && options.categories.length > 0) {
      searchUrl.searchParams.set("categories", options.categories.join(","));
    }

    const headers: Record<string, string> = {
      Accept: "application/json",
      "User-Agent": "SAM-AI-ResearchEngine/1.0",
    };

    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeoutMs ?? this.timeoutMs);

    let resp: Response;
    try {
      resp = await this.fetchFn(searchUrl.toString(), {
        method: "GET",
        headers,
        signal: controller.signal,
      });
    } catch (err: any) {
      if (err.name === "AbortError") {
        throw new SearchProviderError(
          "SEARCH_TIMEOUT",
          `SearXNG search request timed out after ${options.timeoutMs ?? this.timeoutMs}ms`,
          this.name
        );
      }
      throw new SearchProviderError(
        "SEARCH_PROVIDER_ERROR",
        `Network error communicating with SearXNG: ${err.message}`,
        this.name
      );
    } finally {
      clearTimeout(timeoutId);
    }

    if (resp.status === 429) {
      throw new SearchProviderError(
        "SEARCH_RATE_LIMITED",
        "SearXNG search rate limit exceeded. Please retry later.",
        this.name
      );
    }

    if (!resp.ok) {
      throw new SearchProviderError(
        "SEARCH_PROVIDER_ERROR",
        `SearXNG returned HTTP ${resp.status}`,
        this.name
      );
    }

    let data: any;
    try {
      data = await resp.json();
    } catch {
      throw new SearchProviderError(
        "SEARCH_MALFORMED_RESPONSE",
        "Failed to parse JSON response from SearXNG backend",
        this.name
      );
    }

    if (!data || typeof data !== "object" || !Array.isArray(data.results)) {
      throw new SearchProviderError(
        "SEARCH_MALFORMED_RESPONSE",
        "SearXNG response missing expected 'results' array",
        this.name
      );
    }

    const nowIso = new Date().toISOString();
    const results: SearchResult[] = [];

    for (const item of data.results) {
      if (!item || typeof item !== "object") continue;
      const rawUrl = item.url || item.link;
      const title = item.title;
      const snippet = item.content || item.snippet || "";

      if (rawUrl && typeof rawUrl === "string" && title && typeof title === "string") {
        results.push({
          url: rawUrl.trim(),
          title: title.trim(),
          snippet: typeof snippet === "string" ? snippet.trim() : "",
          provider: this.name,
          retrievedAt: nowIso,
          score: typeof item.score === "number" ? item.score : undefined,
          engine: typeof item.engine === "string" ? item.engine : undefined,
        });
      }

      if (results.length >= limit) break;
    }

    return results;
  }

  public async checkHealth(): Promise<SearchProviderStatus> {
    try {
      const searchUrl = new URL(`${this.baseUrl}/search`);
      searchUrl.searchParams.set("q", "test");
      searchUrl.searchParams.set("format", "json");

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);

      const resp = await this.fetchFn(searchUrl.toString(), {
        headers: { Accept: "application/json" },
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      return {
        name: this.name,
        configured: true,
        available: resp.ok,
        endpoint: this.baseUrl,
        message: resp.ok ? "SearXNG backend operational" : `HTTP ${resp.status}`,
      };
    } catch (err: any) {
      return {
        name: this.name,
        configured: true,
        available: false,
        endpoint: this.baseUrl,
        message: err.message,
      };
    }
  }
}

// ---------------------------------------------------------------------------
// 3. Provider Factory
// ---------------------------------------------------------------------------

/**
 * Instantiates the configured search provider from server environment variables.
 * Returns null or raises SearchNotConfiguredError if no search provider is configured.
 */
export function getSearchProvider(env: Record<string, string | undefined> = process.env): WebSearchProvider {
  const providerType = (env.SAM_AI_SEARCH_PROVIDER || "").trim().toLowerCase();

  if (!providerType || providerType === "none") {
    throw new SearchNotConfiguredError();
  }

  if (providerType === "mock") {
    return new MockSearchProvider();
  }

  if (providerType === "searxng") {
    const baseUrl = env.SAM_AI_SEARXNG_BASE_URL?.trim();
    if (!baseUrl) {
      throw new SearchNotConfiguredError(
        "SAM_AI_SEARCH_PROVIDER is set to 'searxng', but SAM_AI_SEARXNG_BASE_URL is not configured."
      );
    }
    const apiKey = env.SAM_AI_SEARXNG_API_KEY?.trim();
    return new SearXNGSearchProvider({ baseUrl, apiKey });
  }

  throw new SearchProviderError(
    "SEARCH_PROVIDER_ERROR",
    `Unsupported search provider type: '${providerType}'. Supported: 'searxng', 'mock'.`,
    providerType
  );
}
