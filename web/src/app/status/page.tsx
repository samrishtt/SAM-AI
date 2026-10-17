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
        return { label: "Ready (Online)", color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" };
      case "loading":
        return { label: "Loading Weights", color: "bg-blue-500/10 text-blue-400 border-blue-500/20" };
      case "weights_missing":
        return { label: "Weights Missing", color: "bg-amber-500/10 text-amber-400 border-amber-500/20" };
      case "model_mismatch":
        return { label: "Model Mismatch", color: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20" };
      case "prohibited_provider":
        return { label: "Prohibited Router Blocked", color: "bg-red-500/10 text-red-400 border-red-500/20" };
      case "unreachable":
        return { label: "Backend Offline", color: "bg-slate-800 text-slate-400 border-slate-700" };
      default:
        return { label: "Not Configured", color: "bg-amber-500/10 text-amber-400 border-amber-500/20" };
    }
  };

  const badge = getStatusBadge(status?.status);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-5xl mx-auto px-4 sm:px-6 py-10 sm:py-14 space-y-12">
        {/* Header */}
        <section className="space-y-3 border-b border-slate-800 pb-8 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-xs font-mono text-blue-400">
              <Activity className="w-3.5 h-3.5" />
              Live Diagnostics &bull; Public Status Matrix
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
              System &amp; Capability Status
            </h1>
            <p className="text-sm sm:text-base text-slate-400 max-w-2xl">
              Transparent, real-time reporting distinguishing implemented, tested, experimental, unavailable, and planned features.
            </p>
          </div>

          <button
            onClick={checkStatus}
            disabled={loading}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-medium text-slate-300 self-start sm:self-auto transition disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-blue-400" : ""}`} />
            <span>Refresh Diagnostics</span>
          </button>
        </section>

        {/* Live Inference Probe Card */}
        <section className="space-y-4">
          <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
            <Cpu className="w-4 h-4 text-blue-400" />
            <span>Active Inference Endpoint Diagnostics</span>
            {lastChecked && (
              <span className="text-[11px] font-normal text-slate-400 font-mono ml-auto">
                Checked at {lastChecked}
              </span>
            )}
          </h2>

          <div className="p-5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
              <div className="space-y-1">
                <span className="text-slate-400 text-xs block">Active Model Identity</span>
                <span className="font-mono text-sm sm:text-base text-slate-100 font-semibold">
                  {status?.model || "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
                </span>
              </div>
              <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-semibold self-start sm:self-center ${badge.color}`}>
                <span className="w-2 h-2 rounded-full bg-current" />
                <span>{badge.label}</span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Provider Routing</span>
                <span className="text-slate-200 font-mono font-medium mt-0.5 block truncate">
                  {status?.providerName || "Sovereign Inference Server"}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Configured Endpoint</span>
                <span className="text-slate-200 font-mono font-medium mt-0.5 block truncate">
                  {status?.endpoint || "Unset (Local / Remote)"}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Security Gate</span>
                <span className="text-slate-200 font-medium mt-0.5 block">
                  {status?.requiresPassword ? "Password Protected" : "Open Preview Mode"}
                </span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80 text-xs text-slate-300 space-y-1">
              <span className="font-semibold text-slate-200 block">Status Detail &bull; Guidance:</span>
              <p className="text-slate-400 leading-relaxed">
                {status?.notice || "Awaiting status probe..."}
              </p>
              {status?.setupGuide && (
                <p className="text-blue-400/90 font-mono text-[11px] pt-1">
                  &rarr; {status.setupGuide}
                </p>
              )}
            </div>
          </div>
        </section>

        {/* Comprehensive Capability Matrix */}
        <section className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" />
              <span>Full Capability &amp; Research Status Matrix</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Authoritative categorization of capabilities to ensure transparent user expectations.
            </p>
          </div>

          <div className="border border-slate-800 rounded-xl overflow-hidden text-xs">
            <div className="grid grid-cols-12 bg-slate-900 px-4 py-2.5 font-semibold text-slate-300 border-b border-slate-800 text-[11px] uppercase tracking-wider">
              <div className="col-span-5 sm:col-span-4">Capability / Component</div>
              <div className="col-span-4 sm:col-span-3">Status Category</div>
              <div className="col-span-3 sm:col-span-5 hidden sm:block">Verification Evidence / Notes</div>
            </div>

            <div className="divide-y divide-slate-800/80 bg-slate-950">
              {/* Row 1 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-medium text-slate-200">
                  Next.js Web Chat Interface
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold text-[11px]">
                    Implemented &bull; Tested
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-slate-400">
                  Build passes; streaming via Vercel AI SDK; 10/10 acceptance tests pass.
                </div>
              </div>

              {/* Row 2 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-medium text-slate-200">
                  Read-Only Code Workspace
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold text-[11px]">
                    Implemented &bull; Tested
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-slate-400">
                  Isolated tree browsing; path traversal, symlink escapes, and secrets blocked.
                </div>
              </div>

              {/* Row 3 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-medium text-slate-200">
                  Sovereign Endpoint Security
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold text-[11px]">
                    Implemented &bull; Tested
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-slate-400">
                  Third-party routers blocked; loopback and trusted HTTPS allowlist verified.
                </div>
              </div>

              {/* Row 4 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-medium text-slate-200">
                  Phase 4B Curated Dataset
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold text-[11px]">
                    Verified on Disk
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-slate-400">
                  250 examples with complete traces; 0% pilot leakage; SHA-256 authenticated.
                </div>
              </div>

              {/* Row 5 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-medium text-slate-200">
                  Experiment 001 Adapter
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20 font-semibold text-[11px]">
                    Rejected &bull; Quarantined
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-slate-400">
                  Regressed accuracy from 19/20 to 16/20 due to reasoning collapse.
                </div>
              </div>

              {/* Row 6 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-medium text-slate-200">
                  Experiment 002 (CoT-Preserving)
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-semibold text-[11px]">
                    Proposed &bull; Ready
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-slate-400">
                  Configuration complete; awaiting execution on free Kaggle GPU quota.
                </div>
              </div>

              {/* Row 7 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-medium text-slate-200">
                  Hugging Face Space Live URL
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 font-semibold text-[11px]">
                    Pending User Push
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-slate-400">
                  Package packaged in <code className="font-mono text-[10px]">spaces/sam_ai_zerogpu/</code>; requires user git push to Space repo.
                </div>
              </div>

              {/* Row 8 */}
              <div className="grid grid-cols-12 px-4 py-3 items-center">
                <div className="col-span-5 sm:col-span-4 font-medium text-slate-200">
                  RLVR / GRPO Training
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700 font-semibold text-[11px]">
                    Planned Future Phase
                  </span>
                </div>
                <div className="col-span-3 sm:col-span-5 hidden sm:block text-slate-400">
                  Reinforcement learning with verifiable mathematical rewards slated for Stage 5.
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Security Invariants Checklist */}
        <section className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Shield className="w-4 h-4 text-blue-400" />
              <span>Architectural Security Invariants</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">Guarantees enforced by the codebase at build and runtime.</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-300">
            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-1.5">
              <div className="flex items-center gap-2 text-emerald-400 font-semibold">
                <CheckCircle2 className="w-4 h-4" />
                <span>Zero Third-Party Inference Leaks</span>
              </div>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Blocklist enforces that no tokens, prompts, or conversation history can be forwarded to OpenAI, Anthropic, Google Gemini, Groq, or OpenRouter.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-1.5">
              <div className="flex items-center gap-2 text-emerald-400 font-semibold">
                <CheckCircle2 className="w-4 h-4" />
                <span>Server-Side Secret Isolation</span>
              </div>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Inference API keys, site passwords, and filesystem root paths are never serialized or transmitted to browser client JavaScript bundles.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-1.5">
              <div className="flex items-center gap-2 text-emerald-400 font-semibold">
                <CheckCircle2 className="w-4 h-4" />
                <span>Local Browser Privacy</span>
              </div>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Chat conversations and local history reside strictly in client <code className="font-mono text-[10px]">localStorage</code>. No persistent server database logs user conversation text.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-1.5">
              <div className="flex items-center gap-2 text-emerald-400 font-semibold">
                <CheckCircle2 className="w-4 h-4" />
                <span>Fail-Closed Error Handling</span>
              </div>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Unconfigured endpoints produce a truthful 503 setup error. The application never synthesizes fake simulated responses under the guise of model intelligence.
              </p>
            </div>
          </div>
        </section>

        {/* Footer Navigation */}
        <section className="pt-4 flex items-center justify-between border-t border-slate-800">
          <Link
            href="/research"
            className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1 transition"
          >
            <span>&larr; View Research &amp; Audits</span>
          </Link>

          <Link
            href="/chat"
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-md shadow-blue-500/20 transition"
          >
            <span>Launch Chat &amp; Code Demo</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </section>
      </main>
    </div>
  );
}
