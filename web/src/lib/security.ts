/**
 * Security and Server-Side Password Gating.
 *
 * Protects the application and API from casual unauthorized quota consumption.
 * If SAM_AI_SITE_PASSWORD is configured, both UI and API requests must provide
 * a matching password in the 'x-sam-ai-password' request header.
 */

export function isPasswordGateConfigured(): boolean {
  const configuredPassword = process.env.SAM_AI_SITE_PASSWORD?.trim();
  return Boolean(configuredPassword && configuredPassword.length > 0);
}

export function validateSitePassword(providedPassword?: string | null): {
  authorized: boolean;
  isPreview: boolean;
  error?: string;
} {
  const configuredPassword = process.env.SAM_AI_SITE_PASSWORD?.trim();

  // If no password gate is configured, operate in open preview mode
  if (!configuredPassword) {
    return { authorized: true, isPreview: true };
  }

  if (!providedPassword || providedPassword.trim().length === 0) {
    return {
      authorized: false,
      isPreview: false,
      error: "Access password required. Please enter the site password to proceed.",
    };
  }

  // Exact comparison
  if (providedPassword.trim() === configuredPassword) {
    return { authorized: true, isPreview: false };
  }

  return {
    authorized: false,
    isPreview: false,
    error: "Invalid access password. Access to SAM-AI chat is restricted.",
  };
}

/**
 * Extracts client IP address for rate limiting.
 * Only trusts forwarded headers (x-forwarded-for, x-real-ip) if SAM_AI_TRUST_PROXY is explicitly enabled.
 * Otherwise defaults to loopback to prevent spoofed header rate-limiting bypass.
 */
export function getClientIp(headers: Headers): string {
  const trustProxy =
    process.env.SAM_AI_TRUST_PROXY === "true" || process.env.SAM_AI_TRUST_PROXY === "1";

  if (trustProxy) {
    const forwardedFor = headers.get("x-forwarded-for");
    if (forwardedFor) {
      const firstIp = forwardedFor.split(",")[0]?.trim();
      if (firstIp && /^[0-9a-fA-F:.]+$/.test(firstIp)) return firstIp;
    }
    const realIp = headers.get("x-real-ip");
    if (realIp && /^[0-9a-fA-F:.]+$/.test(realIp.trim())) return realIp.trim();
  }

  return "127.0.0.1";
}
