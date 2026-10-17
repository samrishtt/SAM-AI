"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Info, AlertCircle, CheckCircle2, ShieldAlert, Loader2, X, ArrowRight } from "lucide-react";
import { BackendStatus } from "@/types/chat";

interface NoticeBannersProps {
  status: BackendStatus;
}

export const NoticeBanners: React.FC<NoticeBannersProps> = ({ status }) => {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) {
    return null;
  }

  // 1. Prohibited third-party inference provider detected
  if (status.status === "prohibited_provider") {
    return (
      <div className="bg-red-50 border-b border-red-200 text-xs text-red-800 px-4 py-2 transition-all">
        <div className="max-w-5xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-red-600 flex-shrink-0" />
            <p className="font-medium">
              Sovereign Policy: Third-party commercial routers are rejected. SAM-AI uses self-hosted open weights.
            </p>
          </div>
          <button
            onClick={() => setDismissed(true)}
            className="text-red-500 hover:text-red-800 p-0.5 rounded transition"
            aria-label="Dismiss notice"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    );
  }

  // 2. Model Loading
  if (status.status === "loading") {
    return (
      <div className="bg-[#FBEBDD] border-b border-[#E8E1D9] text-xs text-[#B95F34] px-4 py-2 transition-all">
        <div className="max-w-5xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Loader2 className="w-4 h-4 text-[#D97745] animate-spin flex-shrink-0" />
            <p className="font-medium">
              Loading sovereign foundation model ({status.model.split("/").pop()})...
            </p>
          </div>
          <button
            onClick={() => setDismissed(true)}
            className="text-[#D97745] hover:text-[#B95F34] p-0.5 rounded transition"
            aria-label="Dismiss notice"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    );
  }

  // 3. Backend Not Configured or Unreachable (Truthful, compact notice)
  if (status.status === "not_configured" || !status.configured || status.status === "unreachable") {
    return (
      <div className="bg-[#F5F0EA] border-b border-[#E8E1D9] text-xs text-[#766B62] px-4 py-2 transition-all">
        <div className="max-w-5xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <Info className="w-3.5 h-3.5 text-[#D97745] flex-shrink-0" />
            <span className="truncate">
              Model connection is being set up. The SAM-AI research preview is available while live inference is prepared.
            </span>
          </div>
          <div className="flex items-center gap-3 flex-shrink-0">
            <Link
              href="/research"
              className="text-[#D97745] hover:text-[#B95F34] font-medium text-xs flex items-center gap-1 hover:underline"
            >
              <span>Research Log</span>
              <ArrowRight className="w-3 h-3" />
            </Link>
            <button
              onClick={() => setDismissed(true)}
              className="text-[#766B62] hover:text-[#28231F] p-0.5 rounded transition"
              aria-label="Dismiss notice"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  // 4. Online & Ready
  return null;
};
