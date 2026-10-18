/**
 * SAM-AI Web Research Engine: Types and Interfaces
 * =================================================
 * Defines typed abstractions for:
 * 1. Web search providers, query options, and canonical search results.
 * 2. Safe page retrieval, SSRF defense configurations, and structured fetch results.
 * 3. Fail-closed error codes and security envelopes.
 *
 * Invariant: All fetched content is marked with isUntrustedContent=true.
 * Server-side only: Never expose search API keys or internal infrastructure IPs.
 */

export type SearchProviderType = "searxng" | "mock" | "none";

export interface SearchOptions {
  /** The natural language or keyword search query */
  query: string;
  /** Maximum number of results to retrieve (default: 5, maximum: 10) */
  limit?: number;
  /** Optional language code (e.g. 'en', 'es') */
  language?: string;
  /** Optional category filtering */
  categories?: string[];
  /** Optional time range constraint */
  timeRange?: "day" | "week" | "month" | "year" | "all";
  /** SafeSearch filter level: 0 = off, 1 = moderate, 2 = strict */
  safesearch?: 0 | 1 | 2;
  /** Timeout in milliseconds for the search request (default: 6000ms) */
  timeoutMs?: number;
}

export interface SearchResult {
  /** Canonical public web destination URL */
  url: string;
  /** Extracted page title */
  title: string;
  /** Concise text snippet summarizing query relevance */
  snippet: string;
  /** Search provider identifier that yielded this result (e.g. 'searxng', 'mock') */
  provider: string;
  /** ISO-8601 UTC timestamp of when this result was retrieved */
  retrievedAt: string;
  /** Optional relevance score reported by search backend */
  score?: number;
  /** Optional underlying engine name (e.g. 'wikipedia', 'google', 'bing') */
  engine?: string;
}

export interface SearchProviderStatus {
  name: string;
  configured: boolean;
  available: boolean;
  endpoint?: string;
  message?: string;
}

export interface WebSearchProvider {
  readonly name: string;
  search(options: SearchOptions): Promise<SearchResult[]>;
  checkHealth(): Promise<SearchProviderStatus>;
}

export type SearchErrorCode =
  | "SEARCH_NOT_CONFIGURED"
  | "INVALID_SEARCH_QUERY"
  | "SEARCH_TIMEOUT"
  | "SEARCH_RATE_LIMITED"
  | "SEARCH_MALFORMED_RESPONSE"
  | "SEARCH_PROVIDER_ERROR";

export class SearchProviderError extends Error {
  public readonly code: SearchErrorCode;
  public readonly provider: string;
  public readonly isSafeToExpose: boolean;

  constructor(code: SearchErrorCode, message: string, provider: string = "unknown") {
    super(message);
    this.name = "SearchProviderError";
    this.code = code;
    this.provider = provider;
    this.isSafeToExpose = true;
  }
}

export class SearchNotConfiguredError extends SearchProviderError {
  constructor(message?: string) {
    super(
      "SEARCH_NOT_CONFIGURED",
      message ||
        "No web search provider is configured. Please configure SAM_AI_SEARCH_PROVIDER and associated endpoint in server environment.",
      "none"
    );
  }
}

// ---------------------------------------------------------------------------
// Safe Web Fetcher Types
// ---------------------------------------------------------------------------

export type FetchErrorCode =
  | "URL_BLOCKED"
  | "FETCH_TIMEOUT"
  | "REDIRECT_LIMIT"
  | "CONTENT_TOO_LARGE"
  | "UNSUPPORTED_CONTENT_TYPE"
  | "FETCH_FAILED"
  | "DNS_RESOLUTION_FAILED";

export class SafeFetchError extends Error {
  public readonly code: FetchErrorCode;
  public readonly url: string;
  public readonly status?: number;
  public readonly isSafeToExpose: boolean;

  constructor(code: FetchErrorCode, message: string, url: string, status?: number) {
    super(message);
    this.name = "SafeFetchError";
    this.code = code;
    this.url = url;
    this.status = status;
    this.isSafeToExpose = true;
  }
}

export interface SafeFetchOptions {
  /** Maximum bytes to read before aborting the stream (default: 500 KB = 512,000 bytes) */
  maxBytes?: number;
  /** Maximum network connection and retrieval timeout in milliseconds (default: 5000ms) */
  timeoutMs?: number;
  /** Maximum manual redirect hops permitted (default: 3) */
  maxRedirects?: number;
  /** Allowed protocols (default: ['https:']) */
  allowedSchemes?: string[];
  /** Allowed destination ports (default: [443]) */
  allowedPorts?: number[];
  /** Allowed HTTP Content-Types (default: ['text/html', 'text/plain', 'application/xhtml+xml']) */
  allowedContentTypes?: string[];
  /** Maximum characters of clean extracted text to retain (default: 10,000) */
  maxCleanTextChars?: number;
  /** Custom DNS lookup hook for dependency injection and pinned transport tests */
  customLookup?: (hostname: string) => Promise<string[]>;
}

export interface SafeFetchResult {
  /** The final verified public URL after following redirects */
  url: string;
  /** The initial requested URL before redirects */
  originalUrl: string;
  /** Extracted HTML title or fallback */
  title: string;
  /** Trustworthy canonical URL if declared in <link rel="canonical"> */
  canonicalUrl?: string;
  /** Sanitized, readable body text with scripts, styles, forms, and boilerplate removed */
  cleanText: string;
  /** Exact raw byte count downloaded before truncation */
  rawByteCount: number;
  /** Validated HTTP Content-Type header */
  contentType: string;
  /** Final HTTP status code */
  status: number;
  /** ISO-8601 UTC timestamp of retrieval */
  retrievedAt: string;
  /** Number of redirects followed */
  redirectHops: number;
  /**
   * Explicit security flag: All external web content must be treated as untrusted.
   * Model reasoning prompts MUST never interpret content as instructions or system overrides.
   */
  readonly isUntrustedContent: true;
}
