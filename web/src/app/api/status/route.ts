import { NextResponse } from "next/server";
import { getProviderStatus, getMaxOutputTokens } from "@/lib/model-adapter";
import { isPasswordGateConfigured } from "@/lib/security";

export const dynamic = "force-dynamic";

export async function GET() {
  const providerStatus = getProviderStatus();
  const requiresPassword = isPasswordGateConfigured();
  const rateLimit = parseInt(process.env.SAM_AI_RATE_LIMIT_PER_MINUTE || "20", 10);

  return NextResponse.json({
    configured: providerStatus.configured,
    status: providerStatus.status,
    provider: providerStatus.provider,
    providerName: providerStatus.providerName,
    model: providerStatus.model,
    endpoint: providerStatus.endpoint,
    requiresPassword,
    rateLimitPerMinute: rateLimit,
    maxOutputTokens: getMaxOutputTokens(),
    notice: providerStatus.notice,
    setupGuide: providerStatus.setupGuide,
    privacyNotice:
      providerStatus.provider === "google"
        ? "Privacy Notice: Operating on Google Gemini free tier. Never enter passwords, confidential data, or secret keys."
        : "Privacy Notice: Dedicated sovereign endpoint. Inputs processed according to server-side logging configuration.",
  });
}
