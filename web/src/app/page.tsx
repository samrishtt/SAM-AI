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
  Code2,
  Sparkles,
  Search,
  ExternalLink,
  Clock,
  Database,
  XCircle,
  HelpCircle,
} from "lucide-react";

type StatusType =
  | "deployed_verified"
  | "published"
  | "implemented_locally"
  | "verified"
  | "in_progress"
  | "planned"
  | "blocked_quarantined";

function StatusBadge({ type, label }: { type: StatusType; label?: string }) {
  const configs: Record<StatusType, { defaultLabel: string; bg: string; text: string; border: string }> = {
    deployed_verified: {
      defaultLabel: "Deployed and verified",
      bg: "bg-[#EBF5F0]",
      text: "text-[#2D6A4F]",
      border: "border-[#B7E4C7]",
    },
    published: {
      defaultLabel: "Published to repository",
      bg: "bg-[#EFF6FF]",
      text: "text-[#1E40AF]",
      border: "border-[#BFDBFE]",
    },
    implemented_locally: {
      defaultLabel: "Implemented locally",
      bg: "bg-[#F5F3FF]",
      text: "text-[#5B21B6]",
      border: "border-[#DDD6FE]",
    },
    verified: {
      defaultLabel: "Verified",
      bg: "bg-[#ECFDF5]",
      text: "text-[#047857]",
      border: "border-[#A7F3D0]",
    },
    in_progress: {
      defaultLabel: "In progress",
      bg: "bg-[#FEF3C7]",
      text: "text-[#B45309]",
      border: "border-[#FDE68A]",
    },
    planned: {
      defaultLabel: "Planned",
      bg: "bg-[#F1F5F9]",
      text: "text-[#475569]",
      border: "border-[#CBD5E1]",
    },
    blocked_quarantined: {
      defaultLabel: "Rejected & Quarantined",
      bg: "bg-[#FEF2F2]",
      text: "text-[#B91C1C]",
      border: "border-[#FECACA]",
    },
  };

  const c = configs[type];
  return (
    <span
      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-mono font-semibold border ${c.bg} ${c.text} ${c.border}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current opacity-70" />
      {label || c.defaultLabel}
    </span>
  );
}

export default function HomePage() {
  const samplePrompts = [
    {
      title: "Algebraic Reasoning",
      tag: "Math",
      prompt: "Solve for x: 3^(2x) - 10 * 3^x + 9 = 0. Show the substitution and each step.",
    },
    {
      title: "Algorithmic Implementation",
      tag: "Code",
      prompt: "Write a Python function to find the longest palindromic substring with O(n^2) time and O(1) space.",
    },
    {
      title: "Thermodynamics Deduction",
      tag: "Science",
      prompt: "Explain how entropy change dictates the spontaneity of an endothermic reaction at high temperatures.",
    },
    {
      title: "Constraint Logic",
      tag: "Logic",
      prompt: "Five people sit in a row. A cannot sit next to B. C sits in the exact middle. Deduce valid seating orders.",
    },
  ];

  return (
    <div className="min-h-screen bg-[#FAF8F5] text-[#28231F] flex flex-col font-sans">
      <Navbar />

      <main className="flex-1 max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-14 space-y-12">
        {/* Hero Section */}
        <section className="space-y-6 text-center max-w-3xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#FBEBDD] border border-[#D97745]/20 text-xs font-semibold text-[#D97745]">
            <span className="w-2 h-2 rounded-full bg-[#D97745] animate-pulse" />
            Parallax Intelligence Lab &bull; Experimental AI Research
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-[#28231F] leading-[1.15]">
            Sovereign Open-Weight Reasoning &amp; Empirical Research
          </h1>

          <p className="text-base sm:text-lg text-[#766B62] leading-relaxed max-w-2xl mx-auto">
            SAM-AI is an experimental open-weight AI research project developing verified reasoning models, safe web-retrieval architectures, and sovereign cloud inference without closed third-party API dependencies.
          </p>

          {/* Honest Status Banner */}
          <div className="flex flex-wrap items-center justify-center gap-2 p-3 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] text-xs text-[#766B62] shadow-xs">
            <span className="font-semibold text-[#28231F]">Current Runtime:</span>
            <span className="font-mono text-[#D97745]">DeepSeek-R1-Distill-Qwen-7B</span>
            <span className="text-[#E8E1D9] hidden sm:inline">|</span>
            <StatusBadge type="deployed_verified" label="Live Assistant (Base Model Only)" />
            <span className="text-[#E8E1D9] hidden sm:inline">|</span>
            <StatusBadge type="blocked_quarantined" label="Adapters Detached" />
            <Link href="/research" className="text-[#D97745] font-semibold hover:underline ml-1">
              Read Research Log &rarr;
            </Link>
          </div>
        </section>

        {/* Current Architecture Card */}
        <section className="bg-[#FFFFFF] border border-[#E8E1D9] rounded-2xl p-5 sm:p-7 shadow-sm space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#E8E1D9]">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[#FBEBDD] border border-[#D97745]/20 text-[#D97745] flex items-center justify-center font-bold font-mono text-sm">
                S
              </div>
              <div>
                <h2 className="text-sm sm:text-base font-bold text-[#28231F] flex items-center gap-2">
                  <span>SAM-AI Assistant Architecture</span>
                  <StatusBadge type="deployed_verified" label="Deployed &amp; Verified" />
                </h2>
                <p className="text-xs text-[#766B62]">
                  Production Web Client &bull; Hugging Face ZeroGPU Backend &bull; Untouched Foundation Weights
                </p>
              </div>
            </div>

            <Link
              href="/chat"
              className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-[#D97745] hover:bg-[#B95F34] text-white text-xs font-semibold shadow-xs transition self-start sm:self-auto"
            >
              <span>Launch Chat Interface</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          {/* Three-Tier Architecture Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <Terminal className="w-4 h-4 text-[#D97745]" />
                  Next.js Web Gateway
                </span>
                <StatusBadge type="deployed_verified" label="Live on Vercel" />
              </div>
              <p className="text-[#766B62] leading-relaxed">
                Server-side streaming gateway handling SSE stream keepalive heartbeats (3.5s interval), conversation isolation, query prefill cleanup, and client IP rate limiting.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <Cpu className="w-4 h-4 text-[#2D6A4F]" />
                  Hugging Face Space
                </span>
                <StatusBadge type="deployed_verified" label="ZeroGPU (A10G)" />
              </div>
              <p className="text-[#766B62] leading-relaxed">
                Hosted at <code className="text-[#28231F] font-mono">Samrish2009/sam-ai-assistant</code>. Dynamic ZeroGPU allocation running pure foundation model weights with all experimental adapters detached.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <Brain className="w-4 h-4 text-[#5B21B6]" />
                  Foundation Model
                </span>
                <StatusBadge type="verified" label="DeepSeek-R1-7B" />
              </div>
              <p className="text-[#766B62] leading-relaxed">
                Untouched <code className="text-[#28231F] font-mono">DeepSeek-R1-Distill-Qwen-7B</code>. Internal thinking tokens are filtered server-side so answers remain concise by default.
              </p>
            </div>
          </div>

          {/* Honest Operational Notice */}
          <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-2 text-xs">
            <div className="flex items-center gap-2 font-bold text-[#28231F]">
              <AlertTriangle className="w-4 h-4 text-[#B45309]" />
              <span>Operational Clarity: No Degraded Adapters, No Commercial Routers</span>
            </div>
            <p className="text-[#766B62] leading-relaxed">
              The live assistant does <strong>not</strong> use the rejected Experiment 001 adapter (which suffered reasoning collapse). It runs exclusively on the base foundation model. Furthermore, the website does <strong>not</strong> route queries through proprietary APIs (OpenAI, Anthropic, Gemini, or OpenRouter). All inference is self-hosted.
            </p>
          </div>

          {/* Sample Prompts */}
          <div className="space-y-2.5">
            <span className="text-xs font-semibold text-[#766B62] tracking-wider uppercase block">
              Test Real Questions in the Chat Interface
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {samplePrompts.map((card, idx) => (
                <Link
                  key={idx}
                  href={`/chat?q=${encodeURIComponent(card.prompt)}`}
                  className="p-3.5 rounded-xl bg-[#FFFFFF] border border-[#E8E1D9] hover:border-[#D97745]/40 hover:bg-[#FAF8F5] transition group flex flex-col justify-between text-left shadow-xs"
                >
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-[#28231F] group-hover:text-[#D97745] transition">
                        {card.title}
                      </span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#F5F0EA] text-[#766B62]">
                        {card.tag}
                      </span>
                    </div>
                    <p className="text-[11px] text-[#766B62] line-clamp-2 leading-relaxed">
                      {card.prompt}
                    </p>
                  </div>
                  <span className="text-[11px] text-[#D97745] font-medium mt-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition">
                    Ask in chat &rarr;
                  </span>
                </Link>
              ))}
            </div>
          </div>
        </section>

        {/* What Works Now vs What Remains Experimental */}
        <section className="space-y-5">
          <div className="text-center max-w-xl mx-auto space-y-1">
            <h2 className="text-2xl font-bold text-[#28231F] tracking-tight">System Status: Verified vs. Experimental</h2>
            <p className="text-xs sm:text-sm text-[#766B62]">
              Strictly distinguishing production capabilities from local prototypes and planned milestones.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            {/* Column 1: Live & Verified */}
            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-4 shadow-xs">
              <div className="flex items-center justify-between pb-3 border-b border-[#E8E1D9]">
                <h3 className="font-bold text-[#28231F] text-sm flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-[#2D6A4F]" />
                  What Works Now (Deployed &amp; Verified)
                </h3>
                <StatusBadge type="deployed_verified" label="Active" />
              </div>

              <ul className="space-y-2.5 text-[#766B62] leading-relaxed">
                <li className="flex items-start gap-2">
                  <span className="text-[#2D6A4F] font-bold">&bull;</span>
                  <div>
                    <strong className="text-[#28231F]">Pure Foundation Model Inference:</strong> Responsive 7B reasoning via Hugging Face ZeroGPU with 60s GPU allocation window.
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#2D6A4F] font-bold">&bull;</span>
                  <div>
                    <strong className="text-[#28231F]">Chat Reliability &amp; SSE Keepalives:</strong> Gradio SSE parser properly processes completed event payloads; 3.5s heartbeats eliminate Vercel gateway dropouts.
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#2D6A4F] font-bold">&bull;</span>
                  <div>
                    <strong className="text-[#28231F]">Conversation Isolation:</strong> Fresh conversation state is hydrated cleanly; in-flight responses are strictly bound to their originating conversation.
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#2D6A4F] font-bold">&bull;</span>
                  <div>
                    <strong className="text-[#28231F]">Concise Responses &amp; Greeting Fast-Path:</strong> Simple greetings receive direct replies rather than triggering mathematical deliberation loops.
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#2D6A4F] font-bold">&bull;</span>
                  <div>
                    <strong className="text-[#28231F]">Read-Only Codebase Explorer:</strong> Built-in repository inspector with strict path traversal prevention, symlink protection, and 512 KB read bounds.
                  </div>
                </li>
              </ul>
            </div>

            {/* Column 2: Experimental & Under Development */}
            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-4 shadow-xs">
              <div className="flex items-center justify-between pb-3 border-b border-[#E8E1D9]">
                <h3 className="font-bold text-[#28231F] text-sm flex items-center gap-2">
                  <Clock className="w-4 h-4 text-[#B45309]" />
                  Experimental &amp; Local Engineering
                </h3>
                <StatusBadge type="implemented_locally" label="In Progress" />
              </div>

              <ul className="space-y-2.5 text-[#766B62] leading-relaxed">
                <li className="flex items-start gap-2">
                  <span className="text-[#5B21B6] font-bold">&bull;</span>
                  <div>
                    <strong className="text-[#28231F]">Web Research Engine (Phase 3B Stage 1 &amp; 2):</strong> Implemented locally in backend modules with 14/14 acceptance tests passing. <em>Not yet wired into user chat</em>.
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#B45309] font-bold">&bull;</span>
                  <div>
                    <strong className="text-[#28231F]">Frozen 95-Item Multi-Domain Baseline (Phase 3A):</strong> Audited score of 84/95 (88.42%) on internal challenge set. Serves as ground-truth evaluation, not proof of frontier ability.
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#B91C1C] font-bold">&bull;</span>
                  <div>
                    <strong className="text-[#28231F]">Fine-Tuning Adapters (Exp 001 Rejected):</strong> The first QLoRA adapter collapsed reasoning (16/20 vs 19/20 base). Quarantined; Experiment 002 recipe designed.
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#475569] font-bold">&bull;</span>
                  <div>
                    <strong className="text-[#28231F]">Grounded Citation UI (Phase 3B Stage 3):</strong> Clickable source badges and citations in chat responses are planned for the next engineering sprint.
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#475569] font-bold">&bull;</span>
                  <div>
                    <strong className="text-[#28231F]">Coding Sandbox Security:</strong> Offline HumanEval benchmark (17/20) was evaluated without a hardened, strongly isolated execution container.
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Empirical Milestones Snapshot */}
        <section className="space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-xl sm:text-2xl font-bold text-[#28231F] tracking-tight">Key Empirical Research Milestones</h2>
              <p className="text-xs text-[#766B62] mt-0.5">Audited numbers, reproducible harnesses, and verified lessons.</p>
            </div>
            <Link
              href="/research"
              className="text-xs font-semibold text-[#D97745] hover:text-[#B95F34] flex items-center gap-1 self-start sm:self-center transition"
            >
              <span>View Full Research Archive</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Milestone 1: Multi-Domain Baseline */}
            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-3 shadow-xs flex flex-col justify-between">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-[#2D6A4F] font-bold uppercase">Phase 3A.1</span>
                  <StatusBadge type="verified" label="84/95 (88.4%)" />
                </div>
                <h3 className="font-bold text-[#28231F] text-sm">Multi-Domain Baseline Benchmark</h3>
                <p className="text-xs text-[#766B62] leading-relaxed">
                  Evaluated untouched DeepSeek-R1-7B on a frozen 95-item internal test suite across 7 domains. Following a LaTeX fraction parser audit, the benchmark stands certified at 84/95 (88.42%).
                </p>
              </div>
              <Link
                href="/research#phase-3a"
                className="text-xs font-semibold text-[#D97745] hover:underline pt-2 block"
              >
                Inspect Domain Breakdown &rarr;
              </Link>
            </div>

            {/* Milestone 2: Experiment 001 Lesson */}
            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-3 shadow-xs flex flex-col justify-between">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-[#B91C1C] font-bold uppercase">Phase 4A</span>
                  <StatusBadge type="blocked_quarantined" label="16/20 (Rejected)" />
                </div>
                <h3 className="font-bold text-[#28231F] text-sm">Experiment 001 Adapter Quarantined</h3>
                <p className="text-xs text-[#766B62] leading-relaxed">
                  The first QLoRA adapter reduced math accuracy from 95% (19/20) down to 80% (16/20) by omitting closing <code className="text-[#28231F] font-mono">&lt;/think&gt;</code> tags. The adapter was permanently rejected and detached.
                </p>
              </div>
              <Link
                href="/research#phase-4a"
                className="text-xs font-semibold text-[#D97745] hover:underline pt-2 block"
              >
                Read Failure Audit &rarr;
              </Link>
            </div>

            {/* Milestone 3: Web Research Engine */}
            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-3 shadow-xs flex flex-col justify-between">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-[#5B21B6] font-bold uppercase">Phase 3B</span>
                  <StatusBadge type="implemented_locally" label="14/14 Tests Pass" />
                </div>
                <h3 className="font-bold text-[#28231F] text-sm">Web Research Engine (Stage 1 &amp; 2)</h3>
                <p className="text-xs text-[#766B62] leading-relaxed">
                  Server-side search provider abstraction and safe page retriever with full SSRF firewall, multi-record DNS checks, transport IP-pinning, 500 KB limit, and untrusted envelope tagging.
                </p>
              </div>
              <Link
                href="/research#phase-3b"
                className="text-xs font-semibold text-[#D97745] hover:underline pt-2 block"
              >
                View Engine Specifications &rarr;
              </Link>
            </div>
          </div>
        </section>

        {/* Next Engineering Milestones */}
        <section className="space-y-5">
          <div>
            <h2 className="text-xl sm:text-2xl font-bold text-[#28231F] tracking-tight">Next Engineering Milestones</h2>
            <p className="text-xs text-[#766B62] mt-0.5">Strictly phased progression separating web retrieval from model training.</p>
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-4 shadow-xs text-xs">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[#D97745] font-bold">Step 1</span>
                  <StatusBadge type="in_progress" label="Current" />
                </div>
                <p className="font-bold text-[#28231F]">Website &amp; Research Sync</p>
                <p className="text-[#766B62] text-[11px] leading-relaxed">
                  Publish updated documentation, transparent status matrix, and local research engine verification.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[#766B62] font-bold">Step 2</span>
                  <StatusBadge type="planned" label="Planned" />
                </div>
                <p className="font-bold text-[#28231F]">Phase 3B Stage 3</p>
                <p className="text-[#766B62] text-[11px] leading-relaxed">
                  Integrate retrieved sources into answer generation with grounded, clickable frontend citations.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[#766B62] font-bold">Step 3</span>
                  <StatusBadge type="planned" label="Planned" />
                </div>
                <p className="font-bold text-[#28231F]">Phase 3B Stage 4</p>
                <p className="text-[#766B62] text-[11px] leading-relaxed">
                  Paired retrieval benchmark: evaluate ~100 factual queries with search versus without search.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[#766B62] font-bold">Step 4</span>
                  <StatusBadge type="planned" label="Planned" />
                </div>
                <p className="font-bold text-[#28231F]">Experiment 002 (CoT SFT)</p>
                <p className="text-[#766B62] text-[11px] leading-relaxed">
                  Controlled QLoRA candidate evaluated strictly against the frozen 84/95 baseline.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[#766B62] font-bold">Step 5</span>
                  <StatusBadge type="planned" label="Planned" />
                </div>
                <p className="font-bold text-[#28231F]">Generalization Gate</p>
                <p className="text-[#766B62] text-[11px] leading-relaxed">
                  Test passing candidates on an untouched, held-out evaluation set before considering promotion.
                </p>
              </div>
            </div>
          </div>
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
            <Link href="/chat" className="hover:text-[#28231F] transition">Chat Interface</Link>
            <Link href="/research" className="hover:text-[#28231F] transition">Research Archive</Link>
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
