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
  Sparkles,
  MessageSquare,
  Search,
} from "lucide-react";

export default function HomePage() {
  const suggestedPrompts = [
    {
      title: "Mathematical Logic",
      tag: "Proof",
      prompt: "Prove why the square root of 2 is irrational using proof by contradiction, showing all intermediate deductions.",
    },
    {
      title: "Chain-of-Thought Dynamics",
      tag: "Research",
      prompt: "Explain how CoT-preserving SFT prevents reasoning collapse in distilled DeepSeek-R1 models.",
    },
    {
      title: "Noise-Aware Benchmarking",
      tag: "Evaluation",
      prompt: "What are best practices for noise-aware paired evaluation on GSM8K without benchmark contamination?",
    },
    {
      title: "Sovereign Architecture",
      tag: "Security",
      prompt: "Explain how self-hosted ZeroGPU inference ensures data privacy and eliminates closed third-party dependencies.",
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
            Parallax Intelligence Lab &bull; Sovereign AI Initiative
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-[#28231F] leading-[1.15]">
            Open-Weight Reasoning &amp; Sovereign Intelligence
          </h1>

          <p className="text-base sm:text-lg text-[#766B62] leading-relaxed max-w-2xl mx-auto">
            SAM-AI investigates post-training dynamics, verifiable reasoning traces, and sovereign open-weight execution — preserving deep internal chain-of-thought without commercial API dependencies.
          </p>

          {/* Quiet Honest Notice */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-[#FFFFFF] border border-[#E8E1D9] text-xs text-[#766B62] shadow-xs">
            <span className="w-1.5 h-1.5 rounded-full bg-[#D97745]" />
            <span>Model connection is being set up. The research preview is available while live inference is prepared.</span>
            <Link href="/research" className="text-[#D97745] font-medium hover:underline ml-1">
              Read log &rarr;
            </Link>
          </div>
        </section>

        {/* Interactive Conversation Preview Card */}
        <section className="bg-[#FFFFFF] border border-[#E8E1D9] rounded-2xl p-5 sm:p-7 shadow-sm space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#E8E1D9]">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[#FBEBDD] border border-[#D97745]/20 text-[#D97745] flex items-center justify-center font-bold font-mono text-sm">
                S
              </div>
              <div>
                <h2 className="text-sm sm:text-base font-bold text-[#28231F] flex items-center gap-2">
                  <span>Start a Conversation</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#F5F0EA] text-[#766B62] font-medium border border-[#E8E1D9]">
                    DeepSeek-R1-Distill-7B
                  </span>
                </h2>
                <p className="text-xs text-[#766B62]">
                  Select a research prompt below or launch the full workspace
                </p>
              </div>
            </div>

            <Link
              href="/chat"
              className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-[#D97745] hover:bg-[#B95F34] text-white text-xs font-semibold shadow-xs transition self-start sm:self-auto"
            >
              <span>Open Chat &amp; Workspace</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          {/* Simulated Dialogue Preview */}
          <div className="space-y-3.5 bg-[#FAF8F5] p-4 sm:p-5 rounded-xl border border-[#E8E1D9]">
            {/* User message */}
            <div className="flex justify-end">
              <div className="max-w-xl bg-[#FFFFFF] border border-[#E8E1D9] rounded-2xl rounded-tr-sm px-4 py-2.5 text-xs text-[#28231F] shadow-xs">
                How does SAM-AI preserve internal chain-of-thought during supervised fine-tuning?
              </div>
            </div>

            {/* Assistant message */}
            <div className="flex justify-start gap-3 items-start">
              <div className="w-7 h-7 rounded-lg bg-[#FBEBDD] text-[#D97745] font-bold text-xs flex items-center justify-center flex-shrink-0 font-mono mt-0.5 border border-[#D97745]/20">
                S
              </div>
              <div className="space-y-2 max-w-2xl text-xs text-[#28231F]">
                {/* Collapsed thinking badge */}
                <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#FFFFFF] border border-[#E8E1D9] text-[11px] text-[#766B62] font-mono">
                  <Brain className="w-3 h-3 text-[#D97745]" />
                  <span>Thought trace: verified 4-stage deduction sequence &bull; 412 tokens</span>
                </div>
                <p className="leading-relaxed text-[#766B62]">
                  Unlike naive distillation that computes loss only on final answers, SAM-AI trains with strict boundary tokens <code className="text-[#D97745] bg-[#FFFFFF] px-1 py-0.5 rounded font-mono text-[11px] border border-[#E8E1D9]">&lt;think&gt; ... &lt;/think&gt;</code>. This prevents reasoning collapse and guarantees mathematical verification before answer emission.
                </p>
              </div>
            </div>
          </div>

          {/* Quick-Launch Prompt Cards */}
          <div className="space-y-2.5">
            <span className="text-xs font-semibold text-[#766B62] tracking-wider uppercase block">
              Suggested Research Inquiries
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {suggestedPrompts.map((card, idx) => (
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
                    Ask this in chat &rarr;
                  </span>
                </Link>
              ))}
            </div>
          </div>
        </section>

        {/* Core Principles */}
        <section className="space-y-6">
          <div className="text-center max-w-xl mx-auto space-y-1">
            <h2 className="text-2xl font-bold text-[#28231F] tracking-tight">Core Research Principles</h2>
            <p className="text-xs sm:text-sm text-[#766B62]">
              Methodological standards governing our model training, verification, and sovereign serving.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-3 shadow-xs">
              <div className="w-9 h-9 rounded-xl bg-[#FBEBDD] border border-[#D97745]/20 flex items-center justify-center text-[#D97745]">
                <Brain className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-[#28231F] text-sm">Verifiable Chain-of-Thought</h3>
              <p className="text-xs text-[#766B62] leading-relaxed">
                Preserving step-by-step reasoning tokens inside <code className="text-[#D97745] font-mono text-[11px]">&lt;think&gt;</code> tags. We reject superficial answer-only distillation that collapses mathematical reasoning capacity.
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-3 shadow-xs">
              <div className="w-9 h-9 rounded-xl bg-[#EBF5F0] border border-[#B7E4C7] flex items-center justify-center text-[#2D6A4F]">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-[#28231F] text-sm">Sovereign Cloud Serving</h3>
              <p className="text-xs text-[#766B62] leading-relaxed">
                Zero dependence on commercial closed AI APIs (OpenRouter, Gemini, OpenAI). All inference executes on owner-controlled self-hosted servers or isolated Hugging Face ZeroGPU containers.
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-3 shadow-xs">
              <div className="w-9 h-9 rounded-xl bg-[#FEF3C7] border border-[#FDE68A] flex items-center justify-center text-[#B45309]">
                <Activity className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-[#28231F] text-sm">Paired-Outcome Auditing</h3>
              <p className="text-xs text-[#766B62] leading-relaxed">
                Every candidate adapter is audited against a frozen, model-neutral evaluation harness (20 fixed GSM8K pilot questions) with zero leakage and paired regression tracking before promotion.
              </p>
            </div>
          </div>
        </section>

        {/* Empirical Research & Audits Snapshot */}
        <section className="space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-xl sm:text-2xl font-bold text-[#28231F] tracking-tight">Empirical Research &amp; Milestones</h2>
              <p className="text-xs text-[#766B62] mt-0.5">Honest, evidence-supported experimental outcomes.</p>
            </div>
            <Link
              href="/research"
              className="text-xs font-semibold text-[#D97745] hover:text-[#B95F34] flex items-center gap-1 self-start sm:self-center transition"
            >
              <span>View Full Research Archive</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="space-y-3">
            {/* Phase 4B */}
            <div className="p-4 rounded-xl bg-[#FFFFFF] border border-[#E8E1D9] flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-xs">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold text-[#2D6A4F] uppercase tracking-wider font-mono">Phase 4B &bull; Verified</span>
                  <span className="text-[#766B62] text-xs">&bull; October 2026</span>
                </div>
                <h3 className="font-bold text-[#28231F] text-sm">Verified Reasoning Dataset Preparation</h3>
                <p className="text-xs text-[#766B62] max-w-2xl leading-relaxed">
                  Curated 250 verified mathematical reasoning traces from <code className="text-[#28231F] font-mono text-[11px]">open-r1/OpenR1-Math-220k</code> with 0% pilot leakage, Math-Verify gold validation, and strict token length bounds for T4 GPU training.
                </p>
              </div>
              <Link
                href="/research#phase-4b"
                className="px-3 py-1.5 rounded-lg bg-[#FAF8F5] hover:bg-[#F5F0EA] text-[#28231F] border border-[#E8E1D9] text-xs font-medium whitespace-nowrap self-start md:self-center transition"
              >
                Inspect Dataset
              </Link>
            </div>

            {/* Phase 4A */}
            <div className="p-4 rounded-xl bg-[#FFFFFF] border border-[#E8E1D9] flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-xs">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold text-[#B45309] uppercase tracking-wider font-mono">Phase 4A &bull; Failure Audit</span>
                  <span className="text-[#766B62] text-xs">&bull; October 2026</span>
                </div>
                <h3 className="font-bold text-[#28231F] text-sm">Experiment 001 Reasoning Collapse Diagnosis</h3>
                <p className="text-xs text-[#766B62] max-w-2xl leading-relaxed">
                  Audited why Experiment 001 reduced accuracy from 95% (19/20) to 80% (16/20). Diagnosed that training without closing <code className="text-[#28231F] font-mono text-[11px]">&lt;/think&gt;</code> tags suppressed generation length from 560.9 down to 111.2 tokens.
                </p>
              </div>
              <Link
                href="/research#phase-4a"
                className="px-3 py-1.5 rounded-lg bg-[#FAF8F5] hover:bg-[#F5F0EA] text-[#28231F] border border-[#E8E1D9] text-xs font-medium whitespace-nowrap self-start md:self-center transition"
              >
                Read Audit
              </Link>
            </div>

            {/* Experiment 002 */}
            <div className="p-4 rounded-xl bg-[#FFFFFF] border border-[#E8E1D9] flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-xs">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold text-[#D97745] uppercase tracking-wider font-mono">Experiment 002 &bull; Design Ready</span>
                  <span className="text-[#766B62] text-xs">&bull; Proposed</span>
                </div>
                <h3 className="font-bold text-[#28231F] text-sm">CoT-Preserving Supervised Fine-Tuning</h3>
                <p className="text-xs text-[#766B62] max-w-2xl leading-relaxed">
                  Corrected target formatting contract: loss computed strictly on reasoning tokens, closing delimiter, solution synthesis, and final answer marker. Sequence length expanded to 1,536 tokens with cosine LR decay.
                </p>
              </div>
              <Link
                href="/research#exp-002"
                className="px-3 py-1.5 rounded-lg bg-[#FAF8F5] hover:bg-[#F5F0EA] text-[#28231F] border border-[#E8E1D9] text-xs font-medium whitespace-nowrap self-start md:self-center transition"
              >
                View Protocol
              </Link>
            </div>
          </div>
        </section>

        {/* Features & Architecture Grid */}
        <section className="space-y-5">
          <div>
            <h2 className="text-xl sm:text-2xl font-bold text-[#28231F] tracking-tight">Platform Architecture</h2>
            <p className="text-xs text-[#766B62] mt-0.5">Modular components supporting the research and demonstration workflow.</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-2 shadow-xs">
              <div className="flex items-center gap-2 font-bold text-sm text-[#28231F]">
                <Terminal className="w-4 h-4 text-[#D97745]" />
                <span>Next.js Web Chat Interface</span>
              </div>
              <p className="text-xs text-[#766B62] leading-relaxed">
                Vercel AI SDK streaming integration with local browser session persistence, Markdown/LaTeX rendering, code copy blocks, and bounded conversational history.
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-2 shadow-xs">
              <div className="flex items-center gap-2 font-bold text-sm text-[#28231F]">
                <Code2 className="w-4 h-4 text-[#2D6A4F]" />
                <span>Read-Only Code Workspace</span>
              </div>
              <p className="text-xs text-[#766B62] leading-relaxed">
                Built-in repository inspector with strict path traversal prevention, symlink escape rejection, sensitive file exclusion, and 512KB file read limits.
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-2 shadow-xs">
              <div className="flex items-center gap-2 font-bold text-sm text-[#28231F]">
                <Lock className="w-4 h-4 text-[#B45309]" />
                <span>Sovereign Security Gateway</span>
              </div>
              <p className="text-xs text-[#766B62] leading-relaxed">
                Strict domain allowlist rejecting commercial third-party inference routers. Server-side secrets remain hidden; optional site password protection and IP rate-limiting.
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-2 shadow-xs">
              <div className="flex items-center gap-2 font-bold text-sm text-[#28231F]">
                <Cpu className="w-4 h-4 text-[#766B62]" />
                <span>Hugging Face ZeroGPU Space Package</span>
              </div>
              <p className="text-xs text-[#766B62] leading-relaxed">
                Self-contained Gradio package under <code className="text-[#28231F] font-mono text-[11px]">spaces/sam_ai_zerogpu/</code> configured for free-tier dynamic GPU allocation on Hugging Face Spaces.
              </p>
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
            <span>&bull;</span>
            <span className="font-mono text-[11px]">integrate/claude-2026-10-12</span>
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
