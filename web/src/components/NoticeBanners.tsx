"use client";

import React, { useState } from "react";
import { Info, AlertTriangle, CheckCircle2, X } from "lucide-react";
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
          [{status.configured ? "ONLINE" : "OFFLINE"}] {status.model} ({status.providerName})
        </span>
        <button
          onClick={() => setDismissed(false)}
          className="text-blue-400 hover:underline text-[11px]"
        >
          Show Banner
        </button>
      </div>
    );
  }

  // Case 1: Backend not configured
  if (!status.configured) {
    return (
      <div className="bg-amber-950/30 border-b border-amber-500/20 text-xs text-amber-200 px-4 py-2.5 transition-all">
        <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <p className="font-semibold text-amber-300">
                Backend Endpoint Not Configured — Sovereign Model ({status.model})
              </p>
              <p className="text-amber-200/80 text-[11px] leading-relaxed">
                {status.notice}{" "}
                <span className="text-amber-100 font-medium">
                  {status.setupGuide}
                </span>
              </p>
            </div>
          </div>

          <button
            onClick={() => setDismissed(true)}
            title="Dismiss banner"
            className="text-amber-400/70 hover:text-amber-200 p-1 rounded hover:bg-amber-900/40 self-end sm:self-center transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  // Case 2: Google Gemini explicit opt-in
  if (status.provider === "google") {
    return (
      <div className="bg-indigo-950/30 border-b border-indigo-500/20 text-xs text-indigo-200 px-4 py-2.5 transition-all">
        <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
          <div className="flex items-start gap-2.5">
            <Info className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <p className="font-semibold text-indigo-300">
                Optional Fallback Active: Google Gemini ({status.model})
              </p>
              <p className="text-indigo-200/80 text-[11px] leading-relaxed">
                {status.notice}{" "}
                <span className="text-amber-300 ml-1 font-medium">
                  {status.privacyNotice}
                </span>
              </p>
            </div>
          </div>

          <button
            onClick={() => setDismissed(true)}
            title="Dismiss banner"
            className="text-indigo-400/70 hover:text-indigo-200 p-1 rounded hover:bg-indigo-900/40 self-end sm:self-center transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  // Case 3: Sovereign Qwen OpenAI-compatible inference endpoint ready
  return (
    <div className="bg-blue-950/30 border-b border-blue-500/20 text-xs text-blue-200 px-4 py-2.5 transition-all">
      <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
        <div className="flex items-start gap-2.5">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <p className="font-semibold text-blue-300">
              Sovereign Model Active: {status.model}
            </p>
            <p className="text-blue-200/80 text-[11px] leading-relaxed">
              Connected to {status.providerName} via {status.endpoint}. Streaming ready.
            </p>
          </div>
        </div>

        <button
          onClick={() => setDismissed(true)}
          title="Dismiss banner"
          className="text-blue-400/70 hover:text-blue-200 p-1 rounded hover:bg-blue-900/40 self-end sm:self-center transition"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

