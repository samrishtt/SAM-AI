/**
 * SAM-AI Web Research Engine
 * ==========================
 * Public exports for Phase 3B Stage 1 & Stage 2:
 * - Types and interfaces (SearchOptions, SearchResult, SafeFetchResult, etc.)
 * - Search providers (MockSearchProvider, SearXNGSearchProvider, getSearchProvider)
 * - SSRF defenses and DNS validation (validateUrlSyntax, isIpAddressProhibited, resolveAndValidateHost)
 * - Safe page retriever (SafeWebFetcher, extractHtmlMetadataAndText, formatUntrustedResearchEnvelope)
 */

export * from "./types.ts";
export * from "./ssrf-guard.ts";
export * from "./search-provider.ts";
export * from "./safe-fetcher.ts";
