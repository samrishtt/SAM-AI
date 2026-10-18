"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Navbar } from "@/components/Navbar";
import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  RefreshCw,
  Shield,
  Cpu,
  Lock,
  Layers,
  Terminal,
  ExternalLink,
  ArrowRight,
} from "lucide-react";
import { BackendStatus, BackendStatusCode } from "@/types/chat";

export default function StatusPage() {
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastChecked, setLastChecked] = useState<string>("");

  const checkStatus = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/status");
      const data: BackendStatus = await res.json();
      setStatus(data);
    } catch {
      setStatus({
        configured: false,
        status: "unreachable",
        model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        provider: "sam_ai_sovereign",
        providerName: "SAM-AI Sovereign Inference Engine",
        requiresPassword: false,
        rateLimitPerMinute: 20,
        notice: "Failed to communicate with /api/status route.",
        privacyNotice: "",
      });
    } finally {
      setLoading(false);
      setLastChecked(new Date().toLocaleTimeString());
    }
  };

  useEffect(() => {
    checkStatus();
  }, []);

  const getStatusBadge = (code?: BackendStatusCode) => {
    switch (code) {
      case "ready":
        return { label: "Ready (Online)", color: "bg-[#EBF5F0] text-[#2D6A4F] border-[#B7E4C7]" };
      case "loading":
        return { label: "Loading Weights", color: "bg-[#FEF3C7] text-[#B45309] border-[#FDE68A]" };
      case "weights_missing":
        return { label: "Weights Missing", color: "bg-[#FEF3C7] text-[#B45309] border-[#FDE68A]" };
      case "model_mismatch":
        return { label: "Model Mismatch", color: "bg-[#FEF3C7] text-[#B45309] border-[#FDE68A]" };
      case "prohibited_provider":
        return { label: "Prohibited Router Blocked", color: "bg-[#FEE2E2] text-[#C53030] border-[#FECACA]" };
      case "unreachable":
        return { label: "Backend Offline", color: "bg-[#F5F0EA] text-[#766B62] border-[#E8E1D9]" };
      default:
        return { label: "Not Configured", color: "bg-[#FEF3C7] text-[#B45309] border-[#FDE68A]" };
    }
  };

  const badge = getStatusBadge(status?.status);

  return (
    <div className="min-h-screen bg-[#FAF8F5] text-[#28231F] flex flex-col font-sans">
      <Navbar />

      <main className="flex-1 max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-14 space-y-12">
        {/* Header */}
        <section className="space-y-3 border-b border-[#E8E1D9] pb-8 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#FBEBDD] border border-[#D97745]/20 text-xs font-semibold text-[#D97745]">
              <Activity className="w-3.5 h-3.5" />
              Live Diagnostics &bull; Public Status Matrix
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-[#28231F]">
              System &amp; Capability Status
            </h1>
            <p className="text-sm sm:text-base text-[#766B62] max-w-2xl">
              Transparent, real-time reporting distinguishing implemented, tested, experimental, unavailable, and planned features.
            </p>
          </div>

          <button
            onClick={checkStatus}
            disabled={loading}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-[#FFFFFF] hover:bg-[#F5F0EA] border border-[#E8E1D9] text-xs font-semibold text-[#28231F] self-start sm:self-auto transition disabled:opacity-50 shadow-xs"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-[#D97745]" : "text-[#766B62]"}`} />
            <span>Refresh Diagnostics</span>
          </button>
        </section>

        {/* Live Inference Probe Card */}
        <section className="space-y-4">
          <h2 className="text-lg font-bold text-[#28231F] tracking-tight flex items-center gap-2">
            <Cpu className="w-4 h-4 text-[#D97745]" />
            <span>Active Inference Endpoint Diagnostics</span>
            {lastChecked && (
              <span className="text-[11px] font-normal text-[#766B62] font-mono ml-auto">
                Checked at {lastChecked}
              </span>
            )}
          </h2>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-5 shadow-xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#E8E1D9]">
              <div className="space-y-1">
                <span className="text-[#766B62] text-xs block">Active Model Identity</span>
                <span className="font-mono text-sm sm:text-base text-[#28231F] font-semibold">
                  {status?.model || "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
                </span>
              </div>
              <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-semibold self-start sm:self-center ${badge.color}`}>
                <span className="w-2 h-2 rounded-full bg-current" />
                <span>{badge.label}</span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] block text-[11px]">Provider Routing</span>
                <span className="text-[#28231F] font-mono font-semibold mt-0.5 block truncate">
                  {status?.providerName || "Sovereign Inference Server"}
                </span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] block text-[11px]">Configured Endpoint</span>
                <span className="text-[#28231F] font-mono font-semibold mt-0.5 block truncate">
                  {status?.endpoint || "Unset (Local / Remote)"}
                </span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] block text-[11px]">Security Gate</span>
                <span className="text-[#28231F] font-semibold mt-0.5 block">
                  {status?.requiresPassword ? "Password Protected" : "Open Preview Mode"}
                </span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] text-xs text-[#28231F] space-y-1">
              <span className="font-bold text-[#28231F] block">Status Detail &bull; Guidance:</span>
              <p className="text-[#766B62] leading-relaxed">
                {status?.notice || "Awaiting status probe..."}
              </p>
              {status?.setupGuide && (
                <p className="text-[#D97745] font-mono text-[11px] pt-1">
                  &rarr; {status.setupGuide}
                </p>
              )}
            </div>
          </div>
        </section>

        {/* Comprehensive Capability Matrix */}
        <section className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-[#28231F] tracking-tight flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#2D6A4F]" />
              <span>Full Capability &amp; Research Status Matrix</span>
            </h2>
            <p className="text-xs text-[#766B62] mt-0.5">
              Authoritative categorization of capabilities to ensure transparent user expectations.
            </p>
          </div>

          <div className="border border-[#E8E1D9] rounded-2xl overflow-hidden text-xs shadow-xs">
            <div className="grid grid-cols-12 bg-[#F5F0EA] px-4 py-2.5 font-bold text-[#766B62] border-b border-[#E8E1D9] text-[11px] uppercase tracking-wider">
              <div className="col-span-5 sm:col-span-4">Capability / Component</div>
              <div className="col-span-4 sm:col-span-3">Status Category</div>
              <div className="col-span-3 sm:col-span-5 hidden sm:block">Verification Evidence / Notes</div>
            </div>

            <div className="divide-y divide-[#E8E1D9] bg-[#FFFFFF]">
              {/* Row 1 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-semibold text-[#28231F]">
                  Next.js Web Chat Interface
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded-full bg-[#EBF5F0] text-[#2D6A4F] border border-[#B7E4C7] font-semibold text-[11px]">
                    Deployed and verified
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-[#766B62]">
                  Live on Vercel; SSE stream keepalive heartbeats (3.5s), conversation isolation, greeting fast-path.
                </div>
              </div>

              {/* Row 2 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-semibold text-[#28231F]">
                  Hugging Face Space Backend
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded-full bg-[#EBF5F0] text-[#2D6A4F] border border-[#B7E4C7] font-semibold text-[11px]">
                    Deployed and verified
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-[#766B62]">
                  RUNNING on ZeroGPU (A10G); base DeepSeek-R1-7B serving with adapters detached.
                </div>
              </div>

              {/* Row 3 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-semibold text-[#28231F]">
                  Phase 3A Multi-Domain Baseline
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded-full bg-[#ECFDF5] text-[#047857] border border-[#A7F3D0] font-semibold text-[11px]">
                    Verified
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-[#766B62]">
                  84/95 (88.42%) across 7 domains audited; math parser corrected (+2); HumanEval not sandbox-certified.
                </div>
              </div>

              {/* Row 4 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-semibold text-[#28231F]">
                  Phase 3B Web Research Engine
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded-full bg-[#F5F3FF] text-[#5B21B6] border border-[#DDD6FE] font-semibold text-[11px]">
                    Implemented locally
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-[#766B62]">
                  Stage 1 &amp; 2 pass 14/14 tests; SSRF firewall and IP pinning; not yet integrated into user chat.
                </div>
              </div>

              {/* Row 5 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-semibold text-[#28231F]">
                  Experiment 001 Adapter
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded-full bg-[#FEF2F2] text-[#B91C1C] border border-[#FECACA] font-semibold text-[11px]">
                    Rejected &bull; Quarantined
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-[#766B62]">
                  Regressed accuracy from 19/20 to 16/20 due to missing closing tags; permanently detached.
                </div>
              </div>

              {/* Row 6 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-semibold text-[#28231F]">
                  Phase 4B Curated Dataset
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded-full bg-[#ECFDF5] text-[#047857] border border-[#A7F3D0] font-semibold text-[11px]">
                    Verified
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-[#766B62]">
                  250 examples from OpenR1-Math-220k with complete traces; 0% pilot leakage; SHA-256 authenticated.
                </div>
              </div>

              {/* Row 7 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-semibold text-[#28231F]">
                  Phase 3B Stage 3 (Citations)
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded-full bg-[#F1F5F9] text-[#475569] border border-[#CBD5E1] font-semibold text-[11px]">
                    Planned
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-[#766B62]">
                  Citation formatting prompt rules and clickable frontend reference cards in the chat interface.
                </div>
              </div>

              {/* Row 8 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-semibold text-[#28231F]">
                  Experiment 002 (CoT SFT)
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded-full bg-[#F1F5F9] text-[#475569] border border-[#CBD5E1] font-semibold text-[11px]">
                    Planned
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-[#766B62]">
                  QLoRA recipe with loss computed on reasoning tokens; gated against the frozen 84/95 baseline.
                </div>
              </div>

              {/* Row 9 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-semibold text-[#28231F]">
                  Read-Only Code Workspace
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded-full bg-[#EBF5F0] text-[#2D6A4F] border border-[#B7E4C7] font-semibold text-[11px]">
                    Deployed and verified
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-[#766B62]">
                  Isolated tree browsing; path traversal, symlink escapes, and secrets blocked.
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Security Invariants Checklist */}
        <section className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-[#28231F] tracking-tight flex items-center gap-2">
              <Shield className="w-4 h-4 text-[#D97745]" />
              <span>Architectural Security Invariants</span>
            </h2>
            <p className="text-xs text-[#766B62] mt-0.5">Guarantees enforced by the codebase at build and runtime.</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-[#28231F]">
            <div className="p-4 rounded-xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-1.5 shadow-xs">
              <div className="flex items-center gap-2 text-[#2D6A4F] font-bold">
                <CheckCircle2 className="w-4 h-4" />
                <span>Zero Third-Party Inference Leaks</span>
              </div>
              <p className="text-[#766B62] text-[11px] leading-relaxed">
                Blocklist enforces that no tokens, prompts, or conversation history can be forwarded to OpenAI, Anthropic, Google Gemini, Groq, or OpenRouter.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-1.5 shadow-xs">
              <div className="flex items-center gap-2 text-[#2D6A4F] font-bold">
                <CheckCircle2 className="w-4 h-4" />
                <span>Server-Side Secret Isolation</span>
              </div>
              <p className="text-[#766B62] text-[11px] leading-relaxed">
                Inference API keys, site passwords, and filesystem root paths are never serialized or transmitted to browser client JavaScript bundles.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-1.5 shadow-xs">
              <div className="flex items-center gap-2 text-[#2D6A4F] font-bold">
                <CheckCircle2 className="w-4 h-4" />
                <span>Local Browser Privacy</span>
              </div>
              <p className="text-[#766B62] text-[11px] leading-relaxed">
                Chat conversations and local history reside strictly in client <code className="font-mono text-[10px]">localStorage</code>. No persistent server database logs user conversation text.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-1.5 shadow-xs">
              <div className="flex items-center gap-2 text-[#2D6A4F] font-bold">
                <CheckCircle2 className="w-4 h-4" />
                <span>Fail-Closed Error Handling</span>
              </div>
              <p className="text-[#766B62] text-[11px] leading-relaxed">
                Unconfigured endpoints produce a truthful 503 setup error. The application never synthesizes fake simulated responses under the guise of model intelligence.
              </p>
            </div>
          </div>
        </section>

        {/* Footer Navigation */}
        <section className="pt-4 flex items-center justify-between border-t border-[#E8E1D9]">
          <Link
            href="/research"
            className="text-xs text-[#766B62] hover:text-[#28231F] flex items-center gap-1 transition font-medium"
          >
            <span>&larr; View Research &amp; Audits</span>
          </Link>

          <Link
            href="/chat"
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#D97745] hover:bg-[#B95F34] text-white text-xs font-semibold shadow-xs transition"
          >
            <span>Launch Chat &amp; Code Demo</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-[#E8E1D9] bg-[#FFFFFF] py-8 px-4 sm:px-6 mt-12 text-xs text-[#766B62]">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-bold text-[#28231F]">SAM-AI</span>
            <span>&bull;</span>
            <span>Parallax Intelligence Lab</span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/chat" className="hover:text-[#28231F] transition">Chat Demo</Link>
            <Link href="/research" className="hover:text-[#28231F] transition">Research</Link>
            <Link href="/status" className="hover:text-[#28231F] transition">System Status</Link>
            <a
              href="https://github.com/samrishtt/SAM-AI"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-[#28231F] transition"
            >
              GitHub
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
