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
} from "lucide-react";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "SAM-AI — Research Log & Audits",
  description: "Evidence-based documentation of architecture, training runs, failure audits, and evaluation metrics.",
};

export default function ResearchPage() {
  return (
    <div className="min-h-screen bg-[#FAF8F5] text-[#28231F] flex flex-col font-sans">
      <Navbar />

      <main className="flex-1 max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-14 space-y-12">
        {/* Header */}
        <section className="space-y-3 border-b border-[#E8E1D9] pb-8">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#FBEBDD] border border-[#D97745]/20 text-xs font-semibold text-[#D97745]">
            <BookOpen className="w-3.5 h-3.5" />
            Parallax Intelligence Lab &bull; Research Archive
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-[#28231F]">
            Research Log &amp; Empirical Audits
          </h1>
          <p className="text-sm sm:text-base text-[#766B62] leading-relaxed max-w-3xl">
            Complete, evidence-backed documentation of training experiments, failure audits, dataset curation pipelines, and evaluation harnesses in the SAM-AI project.
          </p>
        </section>

        {/* Experiment 001 Audit */}
        <section id="phase-4a" className="space-y-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-[#FEF3C7] border border-[#FDE68A] flex items-center justify-center text-[#B45309]">
              <AlertTriangle className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[11px] font-mono text-[#B45309] font-bold uppercase tracking-wider">Phase 4A Audit</span>
              <h2 className="text-xl font-bold text-[#28231F] tracking-tight">
                Experiment 001: GSM8K QLoRA Failure Audit &amp; Reasoning Collapse
              </h2>
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-5 text-xs text-[#28231F] shadow-xs">
            <p className="leading-relaxed text-[#766B62]">
              Experiment 001 fine-tuned <code className="text-[#D97745] font-mono font-semibold">deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</code> on standard GSM8K using QLoRA for 100 optimizer steps. The Phase 3C paired pilot evaluation revealed severe performance degradation across identical evaluation fixtures:
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-center">
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[11px] block font-sans">Base Model</span>
                <span className="text-[#2D6A4F] text-base font-bold">19 / 20 (95%)</span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[11px] block font-sans">Exp 001 Adapter</span>
                <span className="text-[#C53030] text-base font-bold">16 / 20 (80%)</span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[11px] block font-sans">Paired Outcomes</span>
                <span className="text-[#B45309] text-base font-bold">0 Gains, 3 Regr.</span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[11px] block font-sans">Mean Tokens</span>
                <span className="text-[#C53030] text-base font-bold">560.9 &rarr; 111.2</span>
              </div>
            </div>

            {/* Paired Failure Table */}
            <div className="space-y-2">
              <span className="font-bold text-xs text-[#28231F] block">
                Paired Regression Case Breakdown
              </span>
              <div className="border border-[#E8E1D9] rounded-xl overflow-hidden text-[11px]">
                <div className="grid grid-cols-12 bg-[#F5F0EA] px-3.5 py-2 font-bold text-[#766B62] border-b border-[#E8E1D9]">
                  <div className="col-span-2">Pilot Item</div>
                  <div className="col-span-4">Base Model (DeepSeek-R1-7B)</div>
                  <div className="col-span-4">Candidate Adapter (Exp 001)</div>
                  <div className="col-span-2">Outcome</div>
                </div>
                <div className="divide-y divide-[#E8E1D9] bg-[#FFFFFF]">
                  <div className="grid grid-cols-12 px-3.5 py-2.5 items-center">
                    <div className="col-span-2 font-mono text-[#766B62]">gsm8k_004</div>
                    <div className="col-span-4 text-[#2D6A4F] font-mono">Correct (542 tok, CoT)</div>
                    <div className="col-span-4 text-[#C53030] font-mono">Incorrect (84 tok, collapsed)</div>
                    <div className="col-span-2 text-[#C53030] font-semibold">Regression</div>
                  </div>
                  <div className="grid grid-cols-12 px-3.5 py-2.5 items-center">
                    <div className="col-span-2 font-mono text-[#766B62]">gsm8k_012</div>
                    <div className="col-span-4 text-[#2D6A4F] font-mono">Correct (610 tok, CoT)</div>
                    <div className="col-span-4 text-[#C53030] font-mono">Incorrect (102 tok, collapsed)</div>
                    <div className="col-span-2 text-[#C53030] font-semibold">Regression</div>
                  </div>
                  <div className="grid grid-cols-12 px-3.5 py-2.5 items-center">
                    <div className="col-span-2 font-mono text-[#766B62]">gsm8k_017</div>
                    <div className="col-span-4 text-[#2D6A4F] font-mono">Correct (578 tok, CoT)</div>
                    <div className="col-span-4 text-[#C53030] font-mono">Incorrect (95 tok, collapsed)</div>
                    <div className="col-span-2 text-[#C53030] font-semibold">Regression</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-[#E8E1D9]">
              <h3 className="font-bold text-[#28231F] text-sm">Root Cause Identification</h3>
              <p className="leading-relaxed text-[#766B62]">
                In Experiment 001, prompts ended with <code className="text-[#28231F] font-mono">&lt;｜Assistant｜&gt;&lt;think&gt;\n</code>, but the training target was formatted with standard answer-only text lacking a closing <code className="text-[#28231F] font-mono">&lt;/think&gt;</code> delimiter. Consequently, the model penalized generating internal chain-of-thought tokens, collapsing sequence length from 560.9 down to 111.2 tokens and inducing severe mathematical reasoning failure.
              </p>
              <div className="p-3.5 rounded-xl bg-[#FEE2E2] border border-[#FECACA] text-[#C53030] text-xs font-medium">
                <strong>Audit Decision:</strong> The Experiment 001 adapter is permanently rejected and quarantined. No degraded weights are promoted to production.
              </div>
            </div>
          </div>
        </section>

        {/* Phase 4B Dataset Preparation */}
        <section id="phase-4b" className="space-y-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-[#EBF5F0] border border-[#B7E4C7] flex items-center justify-center text-[#2D6A4F]">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[11px] font-mono text-[#2D6A4F] font-bold uppercase tracking-wider">Phase 4B Curation</span>
              <h2 className="text-xl font-bold text-[#28231F] tracking-tight">
                Verified Reasoning Dataset Preparation (OpenR1-Math-220k)
              </h2>
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-5 text-xs text-[#28231F] shadow-xs">
            <p className="leading-relaxed text-[#766B62]">
              To systematically resolve reasoning collapse, we developed a deterministic curation pipeline (<code className="text-[#D97745] font-mono">scripts/prepare_experiment_002_dataset.py</code>) filtering <code className="text-[#D97745] font-mono">open-r1/OpenR1-Math-220k</code> (commit <code className="text-[#766B62] font-mono">e4e141ec...</code>, Apache 2.0).
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-center">
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[11px] block font-sans">Inspected Rows</span>
                <span className="text-[#28231F] text-base font-bold">2,281</span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[11px] block font-sans">Curated Records</span>
                <span className="text-[#2D6A4F] text-base font-bold">250 Verified</span>
              </div>
              <div className="p-3.5 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9]">
                <span className="text-[#766B62] text-[11px] block font-sans">Pilot Leakage</span>
                <span className="text-[#2D6A4F] text-base font-bold">0.0% (Zero)</span>
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-[#E8E1D9]">
              <h3 className="font-bold text-[#28231F] text-sm">Quality Filters Enforced</h3>
              <ul className="list-disc list-inside space-y-1.5 text-[#766B62] leading-relaxed">
                <li><strong>Math-Verify Validation:</strong> Only candidate traces where <code className="text-[#28231F] font-mono">correctness_math_verify == True</code> against gold ground-truth answers.</li>
                <li><strong>Completeness:</strong> Traces must have terminated naturally (<code className="text-[#28231F] font-mono">is_reasoning_complete == True</code>) with balanced <code className="text-[#28231F] font-mono">&lt;think&gt;</code> and <code className="text-[#28231F] font-mono">&lt;/think&gt;</code> tags.</li>
                <li><strong>Sequence Budgeting:</strong> Total length strictly capped under 4,500 characters (mean ~3,833 chars / ~958 tokens) to prevent out-of-memory or truncation on 16GB T4 GPUs (<code className="text-[#28231F] font-mono">max_seq_length = 1536</code>).</li>
                <li><strong>Leakage Prevention:</strong> Dual-stage exact substring and token Jaccard similarity purging against the 20 fixed Phase 3C pilot fixtures (89 borderline rows rejected).</li>
                <li><strong>Non-Trivial Answers:</strong> Single-letter multiple choice options (<code className="text-[#28231F] font-mono">A-E</code>) excluded (668 rows rejected).</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Experiment 002 Protocol */}
        <section id="exp-002" className="space-y-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-[#FBEBDD] border border-[#D97745]/20 flex items-center justify-center text-[#D97745]">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[11px] font-mono text-[#D97745] font-bold uppercase tracking-wider">Proposed Protocol</span>
              <h2 className="text-xl font-bold text-[#28231F] tracking-tight">
                Experiment 002: CoT-Preserving SFT Protocol &amp; Acceptance Gates
              </h2>
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-5 text-xs text-[#28231F] shadow-xs">
            <p className="leading-relaxed text-[#766B62]">
              Experiment 002 defines a corrected QLoRA recipe designed to preserve the base model&apos;s deep chain-of-thought while aligning output formatting:
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="text-[#766B62] text-[11px] font-bold block uppercase tracking-wider">Hyperparameters</span>
                <p><strong>LoRA Rank &amp; Alpha:</strong> <code className="text-[#D97745] font-mono">r=16, alpha=32, dropout=0.05</code></p>
                <p><strong>Target Modules:</strong> <code className="text-[#28231F] font-mono">q, k, v, o, gate, up, down</code></p>
                <p><strong>Learning Rate:</strong> <code className="text-[#D97745] font-mono">1e-4</code> (cosine decay, halved from 2e-4)</p>
                <p><strong>Steps:</strong> 75 optimizer steps (effective batch size 16 = 1,200 examples)</p>
                <p><strong>Max Sequence Length:</strong> 1,536 tokens (expanded from 512)</p>
              </div>

              <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E8E1D9] space-y-1.5">
                <span className="text-[#766B62] text-[11px] font-bold block uppercase tracking-wider">Target Contract</span>
                <p><strong>Prompt:</strong> <code className="text-[#28231F] font-mono">&lt;｜User｜&gt;...&lt;｜Assistant｜&gt;&lt;think&gt;\n</code></p>
                <p><strong>Supervised Completion:</strong> <code className="text-[#2D6A4F] font-mono">&#123;reasoning&#125;\n&lt;/think&gt;\n\n&#123;solution&#125;\n\n#### &#123;answer&#125;&lt;｜end of sentence｜&gt;</code></p>
                <p><strong>Loss Masking:</strong> Prompt masked (-100); loss computed strictly on reasoning trace, closing tag, solution, and answer marker.</p>
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-[#E8E1D9]">
              <h3 className="font-bold text-[#28231F] text-sm">Promotion &amp; Rejection Gates</h3>
              <p className="text-[#766B62] leading-relaxed">
                Candidate adapter promotion requires meeting all four conditions:
              </p>
              <ul className="list-disc list-inside space-y-1 text-[#766B62]">
                <li>Minimum 95.0% accuracy on the fixed 20-sample pilot benchmark (&ge; 19/20 correct).</li>
                <li>Zero paired regressions (candidate must not fail any question correctly solved by the base model).</li>
                <li>Average generated tokens &ge; 300.0 (preventing reasoning collapse).</li>
                <li>Valid delimiter compliance &ge; 95.0%.</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Evaluation Harness & Test Isolation */}
        <section className="space-y-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-[#EBF5F0] border border-[#B7E4C7] flex items-center justify-center text-[#2D6A4F]">
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[11px] font-mono text-[#2D6A4F] font-bold uppercase tracking-wider">Harness Protocol</span>
              <h2 className="text-xl font-bold text-[#28231F] tracking-tight">
                Evaluation Harness &amp; Leakage Isolation
              </h2>
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-[#FFFFFF] border border-[#E8E1D9] space-y-3 text-xs text-[#28231F] shadow-xs">
            <p className="leading-relaxed text-[#766B62]">
              The evaluation harness (<code className="text-[#D97745] font-mono">sam_ai/evaluation/reproducible_eval_harness.py</code>) operates as an immutable benchmark fixture. It provides:
            </p>
            <ul className="list-disc list-inside space-y-1.5 text-[#766B62] leading-relaxed">
              <li><strong>Model Neutrality:</strong> Evaluates base open-weight model and candidate adapter under identical prompt templates, decoding parameters (<code className="text-[#28231F] font-mono">temperature=0.0, max_new_tokens=2048</code>), and numeric parsers.</li>
              <li><strong>Dual Extraction:</strong> Extracts numerical answers using official <code className="text-[#28231F] font-mono">####</code> markers, LaTeX <code className="text-[#28231F] font-mono">\boxed&#123;&#125;</code> expressions, natural phrasing, and trailing numeric fallbacks.</li>
              <li><strong>Zero Contamination:</strong> The 20 pilot questions are permanently excluded from all training sets and verified via automated CI checks.</li>
            </ul>
          </div>
        </section>

        {/* Call to Action */}
        <section className="pt-4 flex items-center justify-between border-t border-[#E8E1D9]">
          <Link
            href="/status"
            className="text-xs text-[#766B62] hover:text-[#28231F] flex items-center gap-1 transition font-medium"
          >
            <span>&larr; View System Status</span>
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
