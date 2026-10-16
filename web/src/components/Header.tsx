"use client";

import React from "react";
import { Sparkles, ShieldCheck, Lock, Unlock, Menu, Info, AlertTriangle } from "lucide-react";

interface HeaderProps {
  modelId: string;
  providerName?: string;
  isConfigured?: boolean;
  hasPasswordGate: boolean;
  isUnlocked: boolean;
  onOpenPasswordModal: () => void;
  onToggleSidebar: () => void;
  isSidebarOpen: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  modelId,
  providerName = "Qwen Sovereign Endpoint",
  isConfigured = false,
  hasPasswordGate,
  isUnlocked,
  onOpenPasswordModal,
  onToggleSidebar,
  isSidebarOpen,
}) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur px-4 flex items-center justify-between z-10 sticky top-0">
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          aria-label="Toggle Sidebar"
          className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition md:hidden"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20 font-bold text-white text-base">
            S
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-slate-100 tracking-wide text-sm md:text-base">SAM-AI</span>
              <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                Web MVP
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">Conversational Intelligence</p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-3">
        {/* Model Status Badge */}
        <div
          title={`Active model: ${modelId} (${isConfigured ? "Connected" : "Offline / Unconfigured"})`}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-800/80 border border-slate-700/60 text-xs text-slate-300"
        >
          <span
            className={`w-2 h-2 rounded-full ${
              isConfigured ? "bg-emerald-400 shadow-sm shadow-emerald-400/50" : "bg-amber-400"
            }`}
          />
          <span className="font-mono text-[11px] max-w-[120px] sm:max-w-none truncate">{modelId}</span>
          <span className="hidden md:inline text-[10px] text-slate-400 font-sans border-l border-slate-700 pl-1.5">
            {isConfigured ? "Online" : "Setup Needed"}
          </span>
        </div>

        {/* Security / Password Gate Status */}
        {hasPasswordGate ? (
          <button
            onClick={onOpenPasswordModal}
            title={isUnlocked ? "Site unlocked via access password" : "Site locked. Click to enter password"}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition ${
              isUnlocked
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20"
                : "bg-amber-500/10 text-amber-400 border border-amber-500/20 hover:bg-amber-500/20"
            }`}
          >
            {isUnlocked ? <Unlock className="w-3.5 h-3.5" /> : <Lock className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline">{isUnlocked ? "Access Granted" : "Password Gate"}</span>
          </button>
        ) : (
          <div
            title="Preview Mode: No site password configured"
            className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700/60 text-xs text-slate-400"
          >
            <ShieldCheck className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-[11px]">Preview Mode</span>
          </div>
        )}
      </div>
    </header>
  );
};
