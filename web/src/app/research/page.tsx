import React from "react";
import Link from "next/link";
import { Navbar } from "@/components/Navbar";
import {
  BookOpen,
  ArrowRight,
  Brain,
  AlertTriangle,
  CheckCircle2,
  Database,
  Cpu,
  Layers,
  FileCode,
  FileText,
  Shield,
  Activity,
  Compass,
  Search,
  ExternalLink,
  Lock,
  Terminal,
  Clock,
  XCircle,
  HelpCircle,
  ArrowUpRight,
} from "lucide-react";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "SAM-AI — Research Log & Empirical Audits",
  description: "Evidence-based documentation of architecture, training runs, failure audits, evaluation metrics, and roadmap.",
};

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

export default function ResearchPage() {
  return (
    <div className="min-h-screen bg-[#FAF8F5] text-[#28231F] flex flex-col font-sans">
      <Navbar />

      <main className="flex-1 max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-14 space-y-12">
        {/* Header */}
        <section className="space-y-4 border-b border-[#E8E1D9] pb-8">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#FBEBDD] border border-[#D97745]/20 text-xs font-semibold text-[#D97745]">
            <BookOpen className="w-3.5 h-3.5" />
            Parallax Intelligence Lab &bull; Research Archive
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-[#28231F]">
            Research Log, Benchmark Baselines &amp; Verified Progress
          </h1>
          <p className="text-sm sm:text-base text-[#766B62] leading-relaxed max-w-3xl">
            Transparent, evidence-backed documentation of SAM-AI&apos;s architecture, chat reliability engineering, frozen baseline benchmarks, adapter failure audits, web-research prototypes, and forward roadmap.
          </p>

          {/* Quick Jump Navigation */}
          <div className="flex flex-wrap gap-2 pt-2">
            <a
              href="#chat-reliability"
              className="px-3 py-1.5 rounded-lg bg-[#FFFFFF] border border-[#E8E1D9] text-xs font-medium text-[#28231F] hover:border-[#D97745] hover:text-[#D97745] transition"
            >
              A. Chat Reliability
            </a>
            <a
              href="#phase-3a"
              className="px-3 py-1.5 rounded-lg bg-[#FFFFFF] border border-[#E8E1D9] text-xs font-medium text-[#28231F] hover:border-[#D97745] hover:text-[#D97745] transition"
            >
              B. Phase 3A Baseline (84/95)
            </a>
            <a
              href="#phase-4a"
              className="px-3 py-1.5 rounded-lg bg-[#FFFFFF] border border-[#E8E1D9] text-xs font-medium text-[#28231F] hover:border-[#D97745] hover:text-[#D97745] transition"
            >
              C. Experiment 001 Audit
            </a>
            <a
              href="#phase-3b"
              className="px-3 py-1.5 rounded-lg bg-[#FFFFFF] border border-[#E8E1D9] text-xs font-medium text-[#28231F] hover:border-[#D97745] hover:text-[#D97745] transition"
            >
              D. Web Research Engine
            </a>
            <a
              href="#roadmap"
              className="px-3 py-1.5 rounded-lg bg-[#FFFFFF] border border-[#E8E1D9] text-xs font-medium text-[#28231F] hover:border-[#D97745] hover:text-[#D97745] transition"
            >
              E. Research Roadmap
            </a>
            <a
              href="#phase-4b"
              className="px-3 py-1.5 rounded-lg bg-[#FFFFFF] border border-[#E8E1D9] text-xs font-medium text-[#28231F] hover:border-[#D97745] hover:text-[#D97745] transition"
            >
              F. Curated Dataset
            </a>
          </div>
        </section>

        {/* Section A: Chat Reliability & UX Hardening */}
        <section id="chat-reliability" className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-[#EBF5F0] border border-[#B7E4C7] flex items-center justify-center text-[#2D6A4F]">
                <Shield className="w-4 h-4" />
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#2D6A4F] font-bold uppercase tracking-wider">Milestone A</span>
                <h2 className="text-xl font-bold text-[#28231F] tracking-tight">
                  Chat Reliability &amp; Production Gateway Hardening
                </h2>
              </div>
            </div>
            <StatusBadge type="deployed_verified" label="Deployed &amp; Verified on Vercel" />
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-5 text-xs text-[#28231F] shadow-xs">
            <p className="leading-relaxed text-[#766B62]">
              In October 2026, extensive chat reliability fixes were deployed to resolve conversation cross-talk, stream dropouts, and greeting deliberations on the production site (<code className="text-[#28231F] font-mono">https://sam-ai-lilac-six.vercel.app</code>):
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#2D6A4F]" />
                  Conversation Hydration &amp; Isolation
                </span>
                <p className="text-[#766B62] leading-relaxed">
                  Local storage hydration cleanly resets state when starting a new session. Sending <code className="text-[#28231F] font-mono">hi</code> sends strictly zero previous messages, eliminating contamination from prior sessions.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#2D6A4F]" />
                  In-Flight Conversation Binding
                </span>
                <p className="text-[#766B62] leading-relaxed">
                  Incoming streaming tokens are bound directly to the active conversation ID that initiated the request. Switching conversations while generation is in flight cannot corrupt other conversations.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#2D6A4F]" />
                  Gradio SSE Parser &amp; 3.5s Stream Keepalives
                </span>
                <p className="text-[#766B62] leading-relaxed">
                  State-tracked parsing processes Gradio&apos;s <code className="text-[#28231F] font-mono">data:</code> payload following <code className="text-[#28231F] font-mono">event: complete</code>. A 3.5s interval emits lightweight keepalive frames, preventing Vercel edge timeouts during model deliberation.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#2D6A4F]" />
                  Greeting Fast-Path &amp; Concise Policy
                </span>
                <p className="text-[#766B62] leading-relaxed">
                  Conversational greetings receive immediate, polite replies via a greeting fast-path, preventing DeepSeek-R1 from embarking on unnecessary mathematical deliberation loops.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#2D6A4F]" />
                  Query Prefill Cleanup
                </span>
                <p className="text-[#766B62] leading-relaxed">
                  URL query parameters (<code className="text-[#28231F] font-mono">?q=</code>, <code className="text-[#28231F] font-mono">?prompt=</code>) are consumed exactly once upon load and purged from browser history to prevent repeated auto-submissions on refresh.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#2D6A4F]" />
                  Fail-Closed Reasoning Guard
                </span>
                <p className="text-[#766B62] leading-relaxed">
                  If token generation terminates before closing <code className="text-[#28231F] font-mono">&lt;/think&gt;</code>, the reasoning filter emits an honest interruption notice rather than exposing internal thought drafts.
                </p>
              </div>
            </div>

            {/* Space Sync Status Notice */}
            <div className="p-4 rounded-xl bg-[#FEF3C7] border border-[#FDE68A] space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-[#B45309] flex items-center gap-1.5">
                  <Clock className="w-4 h-4" />
                  Hugging Face Space File Parity Status
                </span>
                <StatusBadge type="in_progress" label="Manual Space Sync Pending" />
              </div>
              <p className="text-[#B45309] leading-relaxed">
                The live Space (<code className="text-[#28231F] font-mono">Samrish2009/sam-ai-assistant</code>) is running on ZeroGPU with all adapters detached. Local repository files (<code className="text-[#28231F] font-mono">spaces/sam_ai_zerogpu/app.py</code> and <code className="text-[#28231F] font-mono">model_loader.py</code>) contain the latest 60s duration and concise prompt rules. Manual browser synchronization to the Hugging Face Space repository is pending user upload per credential safety protocols.
              </p>
            </div>
          </div>
        </section>

        {/* Section B: Phase 3A Frozen Multi-Domain Baseline */}
        <section id="phase-3a" className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-[#EBF5F0] border border-[#B7E4C7] flex items-center justify-center text-[#2D6A4F]">
                <Activity className="w-4 h-4" />
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#2D6A4F] font-bold uppercase tracking-wider">Milestone B</span>
                <h2 className="text-xl font-bold text-[#28231F] tracking-tight">
                  Phase 3A: Frozen Multi-Domain Baseline Benchmark
                </h2>
              </div>
            </div>
            <StatusBadge type="verified" label="84 / 95 (88.42%) Audited" />
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-5 text-xs text-[#28231F] shadow-xs">
            <p className="leading-relaxed text-[#766B62]">
              Before undertaking any new model training, SAM-AI established a frozen, reproducible baseline benchmark evaluating the untouched foundation model (<code className="text-[#28231F] font-mono">deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</code>) with zero adapters and zero retrieval tools on Kaggle GPU hardware (dual Tesla T4, 7,682s runtime).
            </p>

            {/* Score Table */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-xs text-[#28231F]">
                  Certified Multi-Domain Baseline Matrix (95 Held-Out Questions)
                </span>
                <span className="text-[11px] font-mono text-[#766B62]">
                  Kernel: samrishb/sam-ai-multi-domain-baseline
                </span>
              </div>
              <div className="border border-[#E8E1D9] rounded-xl overflow-hidden text-[11px]">
                <div className="grid grid-cols-12 bg-[#F5F0EA] px-3.5 py-2.5 font-bold text-[#766B62] border-b border-[#E8E1D9]">
                  <div className="col-span-5">Evaluation Domain</div>
                  <div className="col-span-2 text-center">Sample Count</div>
                  <div className="col-span-2 text-center">Initial Reported</div>
                  <div className="col-span-3 text-right">Audited Ground-Truth Score</div>
                </div>
                <div className="divide-y divide-[#E8E1D9] bg-[#FFFFFF]">
                  <div className="grid grid-cols-12 px-3.5 py-2 items-center">
                    <div className="col-span-5 font-mono text-[#28231F]">mathematics_gsm8k</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">20</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">19 / 20</div>
                    <div className="col-span-3 text-right font-mono font-bold text-[#2D6A4F]">19 / 20 (95.0%)</div>
                  </div>
                  <div className="grid grid-cols-12 px-3.5 py-2 items-center bg-[#FAF8F5]">
                    <div className="col-span-5 font-mono text-[#28231F]">mathematics_math500</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">20</div>
                    <div className="col-span-2 text-center font-mono text-[#B45309]">18 / 20 *</div>
                    <div className="col-span-3 text-right font-mono font-bold text-[#2D6A4F]">20 / 20 (100.0%)</div>
                  </div>
                  <div className="grid grid-cols-12 px-3.5 py-2 items-center">
                    <div className="col-span-5 font-mono text-[#28231F]">coding_humaneval **</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">20</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">17 / 20</div>
                    <div className="col-span-3 text-right font-mono font-bold text-[#2D6A4F]">17 / 20 (85.0%)</div>
                  </div>
                  <div className="grid grid-cols-12 px-3.5 py-2 items-center bg-[#FAF8F5]">
                    <div className="col-span-5 font-mono text-[#28231F]">science_reasoning</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">10</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">9 / 10</div>
                    <div className="col-span-3 text-right font-mono font-bold text-[#2D6A4F]">9 / 10 (90.0%)</div>
                  </div>
                  <div className="grid grid-cols-12 px-3.5 py-2 items-center">
                    <div className="col-span-5 font-mono text-[#28231F]">logic_constraints</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">10</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">7 / 10</div>
                    <div className="col-span-3 text-right font-mono font-bold text-[#2D6A4F]">7 / 10 (70.0%)</div>
                  </div>
                  <div className="grid grid-cols-12 px-3.5 py-2 items-center bg-[#FAF8F5]">
                    <div className="col-span-5 font-mono text-[#28231F]">instruction_following</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">10</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">7 / 10</div>
                    <div className="col-span-3 text-right font-mono font-bold text-[#2D6A4F]">7 / 10 (70.0%)</div>
                  </div>
                  <div className="grid grid-cols-12 px-3.5 py-2 items-center">
                    <div className="col-span-5 font-mono text-[#28231F]">multiturn_context</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">5</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">5 / 5</div>
                    <div className="col-span-3 text-right font-mono font-bold text-[#2D6A4F]">5 / 5 (100.0%)</div>
                  </div>
                  <div className="grid grid-cols-12 px-3.5 py-2.5 items-center bg-[#F5F0EA] font-bold border-t border-[#E8E1D9]">
                    <div className="col-span-5 text-[#28231F]">AGGREGATE TOTAL</div>
                    <div className="col-span-2 text-center font-mono">95</div>
                    <div className="col-span-2 text-center font-mono text-[#766B62]">82 / 95 (86.32%)</div>
                    <div className="col-span-3 text-right font-mono text-[#2D6A4F] text-xs">84 / 95 (88.42%)</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Disclosures & Audits */}
            <div className="space-y-3 pt-2 border-t border-[#E8E1D9]">
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1">
                <span className="font-bold text-[#28231F] block">
                  * LaTeX Fraction Parser Correction (+2 items, 82 &rarr; 84)
                </span>
                <p className="text-[#766B62] leading-relaxed">
                  During Phase 3A.1 auditing (<code className="text-[#28231F] font-mono">docs/phase_3a1_baseline_audit.md</code>), two competition-math items (<code className="text-[#28231F] font-mono">math500_05_precalc</code> answering <code className="text-[#28231F] font-mono">\boxed&#123;-\dfrac&#123;3&#125;&#123;4&#125;&#125;</code> and <code className="text-[#28231F] font-mono">math500_10_prob</code> answering <code className="text-[#28231F] font-mono">\boxed&#123;\dfrac&#123;1&#125;&#123;9&#125;&#125;</code>) were found to be 100% mathematically sound. The original regex extractor did not support the LaTeX display macro <code className="text-[#28231F] font-mono">\dfrac</code>. The parser was patched in <code className="text-[#28231F] font-mono">sam_ai/evaluation/multidomain_harness.py</code> and verified via 5 passing tests in <code className="text-[#28231F] font-mono">tests/test_multidomain_latex_rescoring.py</code>.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-[#FEF3C7] border border-[#FDE68A] space-y-1">
                <span className="font-bold text-[#B45309] block">
                  ** Scope &amp; Sandbox Security Disclosures
                </span>
                <p className="text-[#B45309] leading-relaxed">
                  <strong>Internal Challenge Set:</strong> This 95-question test suite is an internal benchmark used strictly as a paired gate for post-training comparisons. It is not a full standardized benchmark or proof of frontier-level capability.
                </p>
                <p className="text-[#B45309] leading-relaxed">
                  <strong>Coding Security Validation:</strong> The HumanEval subset score (17/20, 85.0%) was evaluated offline in the Kaggle environment without a hardened, security-isolated Docker or gVisor execution sandbox. It is an internal research result and must not be treated as a security-certified metric.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Section C: Experiment 001 Failed Adapter */}
        <section id="phase-4a" className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-[#FEF2F2] border border-[#FECACA] flex items-center justify-center text-[#B91C1C]">
                <XCircle className="w-4 h-4" />
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#B91C1C] font-bold uppercase tracking-wider">Milestone C</span>
                <h2 className="text-xl font-bold text-[#28231F] tracking-tight">
                  Experiment 001: Failed Adapter Audit &amp; Quarantined Weights
                </h2>
              </div>
            </div>
            <StatusBadge type="blocked_quarantined" label="Adapter Rejected &amp; Quarantined" />
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-5 text-xs text-[#28231F] shadow-xs">
            <p className="leading-relaxed text-[#766B62]">
              In Experiment 001, a candidate QLoRA adapter was trained on standard GSM8K for 100 optimizer steps. The model was evaluated against an immutable 20-question mathematical pilot benchmark with zero leakage:
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-center">
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[11px] block font-sans">Base Model (No Adapter)</span>
                <span className="text-[#2D6A4F] text-base font-bold">19 / 20 (95.0%)</span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[11px] block font-sans">Experiment 001 Adapter</span>
                <span className="text-[#B91C1C] text-base font-bold">16 / 20 (80.0%)</span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[11px] block font-sans">Paired Regressions</span>
                <span className="text-[#B91C1C] text-base font-bold">3 Failures, 0 Gains</span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[11px] block font-sans">Mean Reasoning Tokens</span>
                <span className="text-[#B91C1C] text-base font-bold">560.9 &rarr; 111.2</span>
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-[#E8E1D9]">
              <h3 className="font-bold text-[#28231F] text-sm">Key Research Lesson</h3>
              <p className="text-[#766B62] leading-relaxed">
                Training prompts ended with <code className="text-[#28231F] font-mono">&lt;｜Assistant｜&gt;&lt;think&gt;\n</code>, but the training completions lacked a closing <code className="text-[#28231F] font-mono">&lt;/think&gt;</code> delimiter. As a result, the model learned to aggressively suppress its internal chain-of-thought, collapsing reasoning length by 80% and introducing arithmetic errors.
              </p>
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] text-[#28231F] space-y-1">
                <strong>Empirical Law:</strong> A shorter response is not automatically a smarter response. Any candidate adapter must preserve useful reasoning and demonstrate genuine improvement against the same frozen evaluation before consideration for promotion.
              </div>
            </div>
          </div>
        </section>

        {/* Section D: Phase 3B Web Research Engine */}
        <section id="phase-3b" className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-[#F5F3FF] border border-[#DDD6FE] flex items-center justify-center text-[#5B21B6]">
                <Search className="w-4 h-4" />
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#5B21B6] font-bold uppercase tracking-wider">Milestone D</span>
                <h2 className="text-xl font-bold text-[#28231F] tracking-tight">
                  Phase 3B: Web Research Engine (Stage 1 &amp; Stage 2)
                </h2>
              </div>
            </div>
            <StatusBadge type="implemented_locally" label="Implemented Locally &bull; 14/14 Tests" />
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-5 text-xs text-[#28231F] shadow-xs">
            <p className="leading-relaxed text-[#766B62]">
              Phase 3B establishes a sovereign, server-side web research foundation allowing SAM-AI to search the public internet and safely retrieve external documentation without third-party LLM routing or client exposure:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#5B21B6]" />
                  Stage 1: Pluggable Search Abstraction
                </span>
                <p className="text-[#766B62] leading-relaxed">
                  Defined typed <code className="text-[#28231F] font-mono">WebSearchProvider</code> interface with <code className="text-[#28231F] font-mono">MockSearchProvider</code> for deterministic testing and <code className="text-[#28231F] font-mono">SearXNGSearchProvider</code> for self-hosted instances. Fail-closed: unconfigured environments throw structured <code className="text-[#28231F] font-mono">SearchNotConfiguredError</code>.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#5B21B6]" />
                  Stage 2: Hardened SSRF Firewall
                </span>
                <p className="text-[#766B62] leading-relaxed">
                  Full IPv4/IPv6 normalization. Blocks loopback (<code className="text-[#28231F] font-mono">127.0.0.0/8</code>, <code className="text-[#28231F] font-mono">::1</code>), RFC 1918 private subnets (<code className="text-[#28231F] font-mono">10/8</code>, <code className="text-[#28231F] font-mono">172.16/12</code>, <code className="text-[#28231F] font-mono">192.168/16</code>), link-local, cloud metadata (<code className="text-[#28231F] font-mono">169.254.169.254</code>), and non-HTTPS schemes.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#5B21B6]" />
                  Transport IP-Pinning (TOCTOU Defense)
                </span>
                <p className="text-[#766B62] leading-relaxed">
                  Multi-record DNS resolution verifies every returned IP record. A custom transport hook binds the socket directly to the validated IP, eliminating DNS rebinding and validation-to-connection race conditions.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="font-bold text-[#28231F] flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#5B21B6]" />
                  Streaming Limits &amp; Untrusted Envelope
                </span>
                <p className="text-[#766B62] leading-relaxed">
                  Streaming byte count terminates at 500 KB. Redirects are manually validated at each hop (max 3). Active content (<code className="text-[#28231F] font-mono">&lt;script&gt;</code>, <code className="text-[#28231F] font-mono">&lt;form&gt;</code>) is stripped. Results are tagged <code className="text-[#28231F] font-mono">isUntrustedContent: true</code> with explicit prompt-injection fences.
                </p>
              </div>
            </div>

            {/* Test Verification Grid */}
            <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-2">
              <span className="font-bold text-[#28231F] block">
                Verification Suites (100% Passing)
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 font-mono text-center">
                <div className="p-2.5 rounded-lg bg-[#FFFFFF] border border-[#E8E1D9]">
                  <span className="text-[#766B62] text-[10px] block font-sans">Node Acceptance Suite</span>
                  <span className="text-[#2D6A4F] text-xs font-bold">14 / 14 Passed</span>
                </div>
                <div className="p-2.5 rounded-lg bg-[#FFFFFF] border border-[#E8E1D9]">
                  <span className="text-[#766B62] text-[10px] block font-sans">Pytest Integration Suite</span>
                  <span className="text-[#2D6A4F] text-xs font-bold">14 / 14 Passed</span>
                </div>
                <div className="p-2.5 rounded-lg bg-[#FFFFFF] border border-[#E8E1D9]">
                  <span className="text-[#766B62] text-[10px] block font-sans">Next.js Production Build</span>
                  <span className="text-[#2D6A4F] text-xs font-bold">0 Errors / Success</span>
                </div>
              </div>
            </div>

            {/* Crucial Disclosure: Not Live in Chat */}
            <div className="p-4 rounded-xl bg-[#FEF3C7] border border-[#FDE68A] space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-bold text-[#B45309] flex items-center gap-1.5">
                  <AlertTriangle className="w-4 h-4" />
                  Integration Status: Local Modules Only &bull; Not Live in User Chat
                </span>
                <StatusBadge type="implemented_locally" label="Backend Only" />
              </div>
              <p className="text-[#B45309] leading-relaxed">
                The presence of backend search and retrieval modules does <strong>not</strong> mean live browsing is active in user chat. The engine is currently <strong>not</strong> integrated into <code className="text-[#28231F] font-mono">/api/chat</code>, no live search provider API is configured in the production environment, and end-to-end web grounding has not yet been rolled out. Live citation synthesis is scheduled for Phase 3B Stage 3.
              </p>
            </div>
          </div>
        </section>

        {/* Section E: Roadmap */}
        <section id="roadmap" className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] flex items-center justify-center text-[#D97745]">
                <Compass className="w-4 h-4" />
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#D97745] font-bold uppercase tracking-wider">Milestone E</span>
                <h2 className="text-xl font-bold text-[#28231F] tracking-tight">
                  Research Roadmap &amp; Execution Milestones
                </h2>
              </div>
            </div>
            <StatusBadge type="in_progress" label="Sequential Milestones" />
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-4 text-xs shadow-xs">
            <p className="text-[#766B62] leading-relaxed">
              SAM-AI follows a disciplined, evidence-based roadmap. Better web retrieval and better base-model reasoning are evaluated on separate, independent evaluation tracks:
            </p>

            <div className="space-y-3">
              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[#D97745] font-bold">1. Website &amp; Engine Publication</span>
                    <StatusBadge type="in_progress" label="Current Task" />
                  </div>
                  <p className="text-[#766B62]">
                    Publish verified documentation, baseline metrics, and Phase 3B Stage 1/2 backend modules to the repository and verify production deployments.
                  </p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[#766B62] font-bold">2. Phase 3B Stage 3: Grounded Citation Synthesis</span>
                    <StatusBadge type="planned" label="Next Up" />
                  </div>
                  <p className="text-[#766B62]">
                    Synthesize retrieved external sources into answer generation with clickable frontend source reference badges and citation cards in the chat UI.
                  </p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[#766B62] font-bold">3. Phase 3B Stage 4: Comparative Retrieval Evaluation</span>
                    <StatusBadge type="planned" label="Planned" />
                  </div>
                  <p className="text-[#766B62]">
                    Evaluate ~100 contemporary factual questions comparing Model-Only versus Model + Web Retrieval under identical decoding parameters.
                  </p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[#766B62] font-bold">4. Controlled Fine-Tuning Candidate Experiment</span>
                    <StatusBadge type="planned" label="Planned" />
                  </div>
                  <p className="text-[#766B62]">
                    Execute Experiment 002 (CoT-preserving SFT on curated 250 Math-Verify rows) and evaluate strictly against the frozen 84/95 baseline.
                  </p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[#766B62] font-bold">5. Independent Generalization Gate</span>
                    <StatusBadge type="planned" label="Planned" />
                  </div>
                  <p className="text-[#766B62]">
                    Test successful candidates on an untouched, held-out evaluation set before considering promotion to production serving.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section F: Curated Dataset */}
        <section id="phase-4b" className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-[#EBF5F0] border border-[#B7E4C7] flex items-center justify-center text-[#2D6A4F]">
                <Database className="w-4 h-4" />
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#2D6A4F] font-bold uppercase tracking-wider">Milestone F</span>
                <h2 className="text-xl font-bold text-[#28231F] tracking-tight">
                  Dataset Curation: Verified Reasoning Traces (OpenR1-Math-220k)
                </h2>
              </div>
            </div>
            <StatusBadge type="verified" label="250 Verified Records" />
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-4 text-xs text-[#28231F] shadow-xs">
            <p className="text-[#766B62] leading-relaxed">
              To address Experiment 001 reasoning collapse, we curated 250 verified mathematical reasoning traces from <code className="text-[#D97745] font-mono">open-r1/OpenR1-Math-220k</code> (commit <code className="text-[#766B62] font-mono">e4e141ec</code>) with 0.0% benchmark pilot contamination:
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-center">
              <div className="p-3 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[10px] block font-sans">Inspected Candidate Rows</span>
                <span className="text-[#28231F] text-sm font-bold">2,281</span>
              </div>
              <div className="p-3 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[10px] block font-sans">Curated Gold Traces</span>
                <span className="text-[#2D6A4F] text-sm font-bold">250 Verified</span>
              </div>
              <div className="p-3 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[10px] block font-sans">Benchmark Pilot Leakage</span>
                <span className="text-[#2D6A4F] text-sm font-bold">0.0% (Zero)</span>
              </div>
            </div>

            <ul className="list-disc list-inside space-y-1 text-[#766B62] leading-relaxed pt-2 border-t border-[#E8E1D9]">
              <li><strong>Math-Verify Gold:</strong> Traces verified mathematically correct against official gold answers.</li>
              <li><strong>Complete Delimiters:</strong> Naturally completed traces with balanced <code className="text-[#28231F] font-mono">&lt;think&gt;</code> and <code className="text-[#28231F] font-mono">&lt;/think&gt;</code> tags.</li>
              <li><strong>Token Length Budget:</strong> Capped under 4,500 characters (~958 tokens) for reliable training on 16GB GPUs without truncation.</li>
            </ul>
          </div>
        </section>

        {/* Call to Action */}
        <section className="pt-4 flex items-center justify-between border-t border-[#E8E1D9]">
          <Link
            href="/status"
            className="text-xs text-[#766B62] hover:text-[#28231F] flex items-center gap-1 transition font-medium"
          >
            <span>&larr; View Live System Status</span>
          </Link>

          <Link
            href="/chat"
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#D97745] hover:bg-[#B95F34] text-white text-xs font-semibold shadow-xs transition"
          >
            <span>Launch Chat Interface</span>
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
