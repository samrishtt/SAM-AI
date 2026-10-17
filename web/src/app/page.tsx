import React from "react";
import Link from "next/link";
import { Navbar } from "@/components/Navbar";
import {
  ShieldCheck,
  Cpu,
  Brain,
  Terminal,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Activity,
  Layers,
  Lock,
  Compass,
  Code2,
} from "lucide-react";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-6xl mx-auto px-4 sm:px-6 py-10 sm:py-16 space-y-16">
        {/* Hero Section */}
        <section className="space-y-6 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-xs font-mono text-blue-400">
            <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
            Parallax Intelligence Lab &bull; Sovereign AI Initiative
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Sovereign Intelligence &amp; Verifiable Reasoning
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed">
            SAM-AI investigates post-training dynamics, reasoning-trace integrity, and sovereign open-weight execution. We build verifiable models that preserve internal chain-of-thought without commercial API dependencies.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Link
              href="/chat"
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold shadow-lg shadow-blue-500/20 transition"
            >
              <span>Try SAM-AI Chat &amp; Code</span>
              <ArrowRight className="w-4 h-4" />
            </Link>

            <Link
              href="/research"
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-800 text-sm font-semibold transition"
            >
              <FileText className="w-4 h-4 text-slate-400" />
              <span>Research Log &amp; Audits</span>
            </Link>

            <Link
              href="/status"
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-800 text-sm font-semibold transition"
            >
              <Activity className="w-4 h-4 text-slate-400" />
              <span>System Status</span>
            </Link>
          </div>

          {/* Model Status Snapshot */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-4 border-t border-slate-900 text-xs">
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <span className="text-slate-400 block text-[11px]">Primary Research Base</span>
              <span className="font-mono text-slate-200 font-semibold truncate block mt-0.5">
                DeepSeek-R1-Distill-Qwen-7B
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <span className="text-slate-400 block text-[11px]">Active Experiment</span>
              <span className="font-mono text-amber-300 font-semibold block mt-0.5">
                EXP-002: CoT-Preserving SFT
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <span className="text-slate-400 block text-[11px]">Inference Mode</span>
              <span className="text-slate-200 font-semibold block mt-0.5">
                Self-Hosted / ZeroGPU Gradio
              </span>
            </div>
          </div>
        </section>

        {/* Core Principles */}
        <section className="space-y-6">
          <div>
            <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight">Core Research Principles</h2>
            <p className="text-sm text-slate-400 mt-1">
              Methodological foundations governing our model development, evaluation, and serving architecture.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-3">
              <div className="w-9 h-9 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                <Brain className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-white text-base">Verifiable Chain-of-Thought</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Preserving step-by-step reasoning tokens inside <code className="text-blue-300 font-mono text-[11px]">&lt;think&gt;</code> tags. We reject superficial answer-only distillation that collapses mathematical reasoning capacity.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-3">
              <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-white text-base">Sovereign Cloud Serving</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Zero dependence on commercial closed AI APIs (OpenRouter, Gemini, OpenAI). All inference executes on owner-controlled self-hosted servers or isolated Hugging Face ZeroGPU containers.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-3">
              <div className="w-9 h-9 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                <Activity className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-white text-base">Paired-Outcome Auditing</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Every candidate adapter is audited against a frozen, model-neutral evaluation harness (20 fixed GSM8K pilot questions) with zero leakage and paired regression tracking before promotion.
              </p>
            </div>
          </div>
        </section>

        {/* Empirical Research & Audits Snapshot */}
        <section className="space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight">Empirical Research &amp; Milestones</h2>
              <p className="text-sm text-slate-400 mt-1">Honest, evidence-supported experimental outcomes.</p>
            </div>
            <Link
              href="/research"
              className="text-xs font-semibold text-blue-400 hover:text-blue-300 flex items-center gap-1 self-start sm:self-center"
            >
              <span>View Full Research Archive</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="space-y-3">
            {/* Phase 4B */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider font-mono">Phase 4B &bull; Verified</span>
                  <span className="text-slate-400 text-xs">&bull; October 2026</span>
                </div>
                <h3 className="font-semibold text-slate-100 text-sm">Verified Reasoning Dataset Preparation</h3>
                <p className="text-xs text-slate-400 max-w-2xl leading-relaxed">
                  Curated 250 verified mathematical reasoning traces from <code className="text-slate-300 font-mono text-[11px]">open-r1/OpenR1-Math-220k</code> with 0% pilot leakage, Math-Verify gold validation, and strict token length bounds for T4 GPU training.
                </p>
              </div>
              <Link
                href="/research#phase-4b"
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium whitespace-nowrap self-start md:self-center transition"
              >
                Inspect Dataset
              </Link>
            </div>

            {/* Phase 4A */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-amber-400 uppercase tracking-wider font-mono">Phase 4A &bull; Failure Audit</span>
                  <span className="text-slate-400 text-xs">&bull; October 2026</span>
                </div>
                <h3 className="font-semibold text-slate-100 text-sm">Experiment 001 Reasoning Collapse Diagnosis</h3>
                <p className="text-xs text-slate-400 max-w-2xl leading-relaxed">
                  Audited why Experiment 001 reduced accuracy from 95% (19/20) to 80% (16/20). Diagnosed that training without closing <code className="text-slate-300 font-mono text-[11px]">&lt;/think&gt;</code> tags suppressed generation length from 560.9 down to 111.2 tokens.
                </p>
              </div>
              <Link
                href="/research#phase-4a"
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium whitespace-nowrap self-start md:self-center transition"
              >
                Read Audit
              </Link>
            </div>

            {/* Experiment 002 */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-blue-400 uppercase tracking-wider font-mono">Experiment 002 &bull; Design Ready</span>
                  <span className="text-slate-400 text-xs">&bull; Proposed</span>
                </div>
                <h3 className="font-semibold text-slate-100 text-sm">CoT-Preserving Supervised Fine-Tuning</h3>
                <p className="text-xs text-slate-400 max-w-2xl leading-relaxed">
                  Corrected target formatting contract: loss computed strictly on reasoning tokens, closing delimiter, solution synthesis, and final answer marker. Sequence length expanded to 1,536 tokens with cosine LR decay.
                </p>
              </div>
              <Link
                href="/research#exp-002"
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium whitespace-nowrap self-start md:self-center transition"
              >
                View Protocol
              </Link>
            </div>
          </div>
        </section>

        {/* Features & Architecture Grid */}
        <section className="space-y-6">
          <div>
            <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight">Platform Architecture</h2>
            <p className="text-sm text-slate-400 mt-1">Modular components supporting the research and demonstration workflow.</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2.5">
              <div className="flex items-center gap-2 font-semibold text-sm text-slate-200">
                <Terminal className="w-4 h-4 text-blue-400" />
                <span>Next.js Web Chat Interface</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Vercel AI SDK streaming integration with local browser session persistence, Markdown/LaTeX rendering, code copy blocks, and bounded conversational history.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2.5">
              <div className="flex items-center gap-2 font-semibold text-sm text-slate-200">
                <Code2 className="w-4 h-4 text-emerald-400" />
                <span>Read-Only Code Workspace</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Built-in repository inspector with strict path traversal prevention, symlink escape rejection, sensitive file exclusion, and 512KB file read limits.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2.5">
              <div className="flex items-center gap-2 font-semibold text-sm text-slate-200">
                <Lock className="w-4 h-4 text-amber-400" />
                <span>Sovereign Security Gateway</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Strict domain allowlist rejecting commercial third-party inference routers. Server-side secrets remain hidden; optional site password protection and IP rate-limiting.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2.5">
              <div className="flex items-center gap-2 font-semibold text-sm text-slate-200">
                <Cpu className="w-4 h-4 text-purple-400" />
                <span>Hugging Face ZeroGPU Space Package</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Self-contained Gradio package under <code className="text-slate-300 font-mono text-[11px]">spaces/sam_ai_zerogpu/</code> configured for free-tier dynamic GPU allocation on Hugging Face Spaces.
              </p>
            </div>
          </div>
        </section>

        {/* Truthful Deployment Disclosure Callout */}
        <section className="p-5 rounded-xl bg-slate-900/70 border border-slate-800 text-xs text-slate-300 space-y-2">
          <div className="flex items-center gap-2 text-slate-200 font-semibold">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span>Honest Model Inference &amp; Deployment Status</span>
          </div>
          <p className="text-slate-400 leading-relaxed">
            This Next.js web application is ready for production deployment to Vercel Hobby. However, model inference requires an active, owner-controlled self-hosted backend (<code className="text-slate-300 font-mono">SAM_AI_BASE_URL</code>) or a deployed Hugging Face ZeroGPU Space. When no backend is configured, the chat demo produces an informative setup status rather than fake mock responses.
          </p>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-8 px-4 sm:px-6 mt-12 text-xs text-slate-400">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-bold text-slate-200">SAM-AI</span>
            <span>&bull;</span>
            <span>Parallax Intelligence Lab</span>
            <span>&bull;</span>
            <span className="font-mono text-[11px]">integrate/claude-2026-10-12</span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/chat" className="hover:text-slate-200 transition">Chat Demo</Link>
            <Link href="/research" className="hover:text-slate-200 transition">Research</Link>
            <Link href="/status" className="hover:text-slate-200 transition">System Status</Link>
            <a
              href="https://github.com/samrishtt/SAM-AI"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-slate-200 transition"
            >
              GitHub
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
