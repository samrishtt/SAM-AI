"use client";

import React, { useState } from "react";
import { Info, AlertTriangle, CheckCircle2, ShieldAlert, Loader2, X, AlertOctagon } from "lucide-react";
import { BackendStatus } from "@/types/chat";

interface NoticeBannersProps {
  status: BackendStatus;
}

export const NoticeBanners: React.FC<NoticeBannersProps> = ({ status }) => {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) {
    return (
      <div className="px-4 py-1 text-center bg-slate-900/40 border-b border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-center gap-2">
        <span className="font-mono text-[10px] text-slate-300">
          [{status.status.toUpperCase()}] {status.model} ({status.providerName})
        </span>
        <button
          onClick={() => setDismissed(false)}
          className="text-blue-400 hover:underline text-[11px]"
        >
          Show Status
        </button>
      </div>
    );
  }

  // 1. Prohibited third-party inference provider detected
  if (status.status === "prohibited_provider") {
    return (
      <div className="bg-red-950/40 border-b border-red-500/40 text-xs text-red-200 px-4 py-2.5 transition-all">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
          <div className="flex items-start gap-2.5">
            <ShieldAlert className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <p className="font-semibold text-red-300">
                Security Policy: Prohibited Third-Party Managed Provider Rejected
              </p>
              <p className="text-red-200/80 text-[11px] leading-relaxed">
                {status.notice} SAM-AI requires an authentic, owner-controlled sovereign endpoint.
              </p>
            </div>
          </div>
          <button
            onClick={() => setDismissed(true)}
            className="text-red-400/70 hover:text-red-200 p-1 rounded hover:bg-red-900/40 self-end sm:self-center transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  // 2. Weights missing locally on inference server
  if (status.status === "weights_missing") {
    return (
      <div className="bg-amber-950/40 border-b border-amber-500/30 text-xs text-amber-200 px-4 py-2.5 transition-all">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
          <div className="flex items-start gap-2.5">
            <AlertOctagon className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <p className="font-semibold text-amber-300">
                Model Weights Missing — {status.model}
              </p>
              <p className="text-amber-200/80 text-[11px] leading-relaxed">
                {status.notice} {status.backendError ? `(${status.backendError})` : ""}
              </p>
            </div>
          </div>
          <button
            onClick={() => setDismissed(true)}
            className="text-amber-400/70 hover:text-amber-200 p-1 rounded hover:bg-amber-900/40 self-end sm:self-center transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  // 3. Model ID mismatch
  if (status.status === "model_mismatch") {
    return (
      <div className="bg-yellow-950/40 border-b border-yellow-500/30 text-xs text-yellow-200 px-4 py-2.5 transition-all">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <p className="font-semibold text-yellow-300">
                Model Identity Mismatch
              </p>
              <p className="text-yellow-200/80 text-[11px] leading-relaxed">
                {status.notice}
              </p>
            </div>
          </div>
          <button
            onClick={() => setDismissed(true)}
            className="text-yellow-400/70 hover:text-yellow-200 p-1 rounded hover:bg-yellow-900/40 self-end sm:self-center transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  // 4. Backend unreachable
  if (status.status === "unreachable") {
    return (
      <div className="bg-slate-900 border-b border-slate-700/60 text-xs text-slate-300 px-4 py-2.5 transition-all">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <p className="font-semibold text-slate-200">
                Inference Server Unreachable — {status.endpoint || "127.0.0.1:8000"}
              </p>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                {status.notice}
              </p>
            </div>
          </div>
          <button
            onClick={() => setDismissed(true)}
            className="text-slate-400 hover:text-slate-200 p-1 rounded hover:bg-slate-800 self-end sm:self-center transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  // 5. Model Loading
  if (status.status === "loading") {
    return (
      <div className="bg-blue-950/40 border-b border-blue-500/30 text-xs text-blue-200 px-4 py-2.5 transition-all">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
          <div className="flex items-start gap-2.5">
            <Loader2 className="w-4 h-4 text-blue-400 animate-spin flex-shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <p className="font-semibold text-blue-300">
                Loading Sovereign Model Weights — {status.model}
              </p>
              <p className="text-blue-200/80 text-[11px] leading-relaxed">
                {status.notice}
              </p>
            </div>
          </div>
          <button
            onClick={() => setDismissed(true)}
            className="text-blue-400/70 hover:text-blue-200 p-1 rounded hover:bg-blue-900/40 self-end sm:self-center transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  // 6. Backend Not Configured
  if (status.status === "not_configured" || !status.configured) {
    return (
      <div className="bg-amber-950/30 border-b border-amber-500/20 text-xs text-amber-200 px-4 py-2.5 transition-all">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <p className="font-semibold text-amber-300">
                Backend Endpoint Not Configured — {status.model}
              </p>
              <p className="text-amber-200/80 text-[11px] leading-relaxed">
                {status.notice} <span className="text-amber-100 font-medium">{status.setupGuide}</span>
              </p>
            </div>
          </div>
          <button
            onClick={() => setDismissed(true)}
            className="text-amber-400/70 hover:text-amber-200 p-1 rounded hover:bg-amber-900/40 self-end sm:self-center transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  // 7. Ready
  return (
    <div className="bg-emerald-950/30 border-b border-emerald-500/20 text-xs text-emerald-200 px-4 py-2.5 transition-all">
      <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
        <div className="flex items-start gap-2.5">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <p className="font-semibold text-emerald-300">
              Sovereign Model Online: {status.model}
            </p>
            <p className="text-emerald-200/80 text-[11px] leading-relaxed">
              Serving via owner-controlled backend {status.endpoint ? `(${status.endpoint})` : ""}. Streaming ready.
            </p>
          </div>
        </div>
        <button
          onClick={() => setDismissed(true)}
          className="text-emerald-400/70 hover:text-emerald-200 p-1 rounded hover:bg-emerald-900/40 self-end sm:self-center transition"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
