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
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-5xl mx-auto px-4 sm:px-6 py-10 sm:py-14 space-y-12">
        {/* Header */}
        <section className="space-y-3 border-b border-slate-800 pb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-xs font-mono text-blue-400">
            <BookOpen className="w-3.5 h-3.5" />
            Parallax Intelligence Lab &bull; Research Archive
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
            Research Log &amp; Empirical Audits
          </h1>
          <p className="text-sm sm:text-base text-slate-400 leading-relaxed max-w-3xl">
            Complete, evidence-backed documentation of training experiments, failure audits, dataset curation pipelines, and evaluation harnesses in the SAM-AI project.
          </p>
        </section>

        {/* Experiment 001 Audit */}
        <section id="phase-4a" className="space-y-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
              <AlertTriangle className="w-4 h-4" />
            </div>
            <div>
              <span className="text-xs font-mono text-amber-400 font-semibold uppercase">Phase 4A Audit</span>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Experiment 001: GSM8K QLoRA Failure Audit &amp; Reasoning Collapse
              </h2>
            </div>
          </div>

          <div className="p-5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-4 text-xs text-slate-300">
            <p className="leading-relaxed">
              Experiment 001 fine-tuned <code className="text-blue-300 font-mono">deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</code> on standard GSM8K using QLoRA for 100 optimizer steps. The Phase 3C paired pilot evaluation revealed a significant performance degradation:
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-center">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 text-[11px] block">Base Model</span>
                <span className="text-emerald-400 text-base font-bold">19 / 20 (95%)</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 text-[11px] block">Exp 001 Adapter</span>
                <span className="text-rose-400 text-base font-bold">16 / 20 (80%)</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 text-[11px] block">Paired Outcomes</span>
                <span className="text-amber-400 text-base font-bold">0 Gains, 3 Regr.</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 text-[11px] block">Mean Tokens</span>
                <span className="text-rose-400 text-base font-bold">560.9 &rarr; 111.2</span>
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <h3 className="font-semibold text-slate-100 text-sm">Root Cause Identification</h3>
              <p className="leading-relaxed text-slate-400">
                In Experiment 001, prompts were constructed ending with <code className="text-slate-300 font-mono">&lt;｜Assistant｜&gt;&lt;think&gt;\n</code>, but the supervised target completion contained the raw GSM8K solution without any internal thinking tokens and without a closing <code className="text-slate-300 font-mono">&lt;/think&gt;</code> delimiter. As a result, the model learned that immediately following <code className="text-slate-300 font-mono">&lt;think&gt;\n</code>, it should emit short final summaries and stop, effectively suppressing its pre-trained Chain-of-Thought mechanism.
              </p>
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
                <strong>Audit Decision:</strong> The Experiment 001 adapter is permanently rejected and quarantined.
              </div>
            </div>
          </div>
        </section>

        {/* Phase 4B Dataset Preparation */}
        <section id="phase-4b" className="space-y-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <span className="text-xs font-mono text-emerald-400 font-semibold uppercase">Phase 4B Curation</span>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Verified Reasoning Dataset Preparation (OpenR1-Math-220k)
              </h2>
            </div>
          </div>

          <div className="p-5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-4 text-xs text-slate-300">
            <p className="leading-relaxed">
              To address reasoning collapse, we developed a deterministic curation pipeline (<code className="text-blue-300 font-mono">scripts/prepare_experiment_002_dataset.py</code>) querying <code className="text-blue-300 font-mono">open-r1/OpenR1-Math-220k</code> (commit <code className="text-slate-400 font-mono">e4e141ec...</code>, Apache 2.0).
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-center">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 text-[11px] block">Inspected Rows</span>
                <span className="text-slate-200 text-base font-bold">2,281</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 text-[11px] block">Curated Records</span>
                <span className="text-emerald-400 text-base font-bold">250 Verified</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 text-[11px] block">Pilot Leakage</span>
                <span className="text-emerald-400 text-base font-bold">0.0% (Zero)</span>
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <h3 className="font-semibold text-slate-100 text-sm">Quality Filters Enforced</h3>
              <ul className="list-disc list-inside space-y-1 text-slate-400 leading-relaxed">
                <li><strong>Math-Verify Validation:</strong> Only candidate traces where <code className="text-slate-300 font-mono">correctness_math_verify == True</code> against gold answers.</li>
                <li><strong>Completeness:</strong> Traces must have finished naturally (<code className="text-slate-300 font-mono">is_reasoning_complete == True</code>) with balanced <code className="text-slate-300 font-mono">&lt;think&gt;</code> and <code className="text-slate-300 font-mono">&lt;/think&gt;</code> tags.</li>
                <li><strong>Sequence Budgeting:</strong> Total length strictly capped under 4,500 characters (mean ~3,833 chars / ~958 tokens) to prevent out-of-memory or truncation on 16GB T4 GPUs (<code className="text-slate-300 font-mono">max_seq_length = 1536</code>).</li>
                <li><strong>Leakage Prevention:</strong> Dual-stage exact substring and token Jaccard similarity purging against the 20 fixed Phase 3C pilot fixtures (89 borderline rows rejected).</li>
                <li><strong>Non-Trivial Answers:</strong> Single-letter multiple choice options (<code className="text-slate-300 font-mono">A-E</code>) excluded (668 rows rejected).</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Experiment 002 Protocol */}
        <section id="exp-002" className="space-y-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <span className="text-xs font-mono text-blue-400 font-semibold uppercase">Proposed Protocol</span>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Experiment 002: CoT-Preserving SFT Protocol &amp; Acceptance Gates
              </h2>
            </div>
          </div>

          <div className="p-5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-4 text-xs text-slate-300">
            <p className="leading-relaxed">
              Experiment 002 defines a corrected QLoRA recipe designed to preserve the base model&apos;s deep chain-of-thought while aligning output formatting:
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-slate-400 text-[11px] font-semibold block uppercase">Hyperparameters</span>
                <p><strong>LoRA Rank &amp; Alpha:</strong> <code className="text-blue-300 font-mono">r=16, alpha=32, dropout=0.05</code></p>
                <p><strong>Target Modules:</strong> <code className="text-slate-300 font-mono">q, k, v, o, gate, up, down</code></p>
                <p><strong>Learning Rate:</strong> <code className="text-blue-300 font-mono">1e-4</code> (cosine decay, halved from 2e-4)</p>
                <p><strong>Steps:</strong> 75 optimizer steps (effective batch size 16 = 1,200 examples)</p>
                <p><strong>Max Sequence Length:</strong> 1,536 tokens (expanded from 512)</p>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-slate-400 text-[11px] font-semibold block uppercase">Target Contract</span>
                <p><strong>Prompt:</strong> <code className="text-slate-300 font-mono">&lt;｜User｜&gt;...&lt;｜Assistant｜&gt;&lt;think&gt;\n</code></p>
                <p><strong>Supervised Completion:</strong> <code className="text-emerald-300 font-mono">&#123;reasoning&#125;\n&lt;/think&gt;\n\n&#123;solution&#125;\n\n#### &#123;answer&#125;&lt;｜end of sentence｜&gt;</code></p>
                <p><strong>Loss Masking:</strong> Prompt masked (-100); loss computed strictly on reasoning trace, closing tag, solution, and answer marker.</p>
              </div>
            </div>

            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <h3 className="font-semibold text-slate-100 text-sm">Promotion &amp; Rejection Gates</h3>
              <p className="text-slate-400 leading-relaxed">
                Candidate adapter promotion requires:
              </p>
              <ul className="list-disc list-inside space-y-1 text-slate-400">
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
            <div className="w-7 h-7 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <span className="text-xs font-mono text-indigo-400 font-semibold uppercase">Harness Protocol</span>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Evaluation Harness &amp; Leakage Isolation
              </h2>
            </div>
          </div>

          <div className="p-5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-3 text-xs text-slate-300">
            <p className="leading-relaxed">
              The evaluation harness (<code className="text-blue-300 font-mono">sam_ai/evaluation/reproducible_eval_harness.py</code>) operates as an immutable benchmark fixture. It provides:
            </p>
            <ul className="list-disc list-inside space-y-1 text-slate-400 leading-relaxed">
              <li><strong>Model Neutrality:</strong> Evaluates base open-weight model and candidate adapter under identical prompt templates, decoding parameters (<code className="text-slate-300 font-mono">temperature=0.0, max_new_tokens=2048</code>), and numeric parsers.</li>
              <li><strong>Dual Extraction:</strong> Extracts numerical answers using official <code className="text-slate-300 font-mono">####</code> markers, LaTeX <code className="text-slate-300 font-mono">\boxed&#123;&#125;</code> expressions, natural phrasing, and trailing numeric fallbacks.</li>
              <li><strong>Zero Contamination:</strong> The 20 pilot questions are permanently excluded from all training sets and verified via automated CI checks.</li>
            </ul>
          </div>
        </section>

        {/* Call to Action */}
        <section className="pt-4 flex items-center justify-between border-t border-slate-800">
          <Link
            href="/status"
            className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1 transition"
          >
            <span>&larr; View System Status</span>
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
