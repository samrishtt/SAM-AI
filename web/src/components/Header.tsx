"use client";

import React from "react";
import Link from "next/link";
import { MessageSquare, Code2, ShieldCheck, Lock, Unlock, Menu } from "lucide-react";
import { BackendStatusCode } from "@/types/chat";

interface HeaderProps {
  modelId: string;
  providerName?: string;
  isConfigured?: boolean;
  backendStatus?: BackendStatusCode;
  hasPasswordGate: boolean;
  isUnlocked: boolean;
  activeTab?: "chat" | "code";
  onTabChange?: (tab: "chat" | "code") => void;
  onOpenPasswordModal: () => void;
  onToggleSidebar: () => void;
  isSidebarOpen: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  modelId,
  providerName = "Sovereign Inference Engine",
  isConfigured = false,
  backendStatus = "not_configured",
  hasPasswordGate,
  isUnlocked,
  activeTab = "chat",
  onTabChange,
  onOpenPasswordModal,
  onToggleSidebar,
  isSidebarOpen,
}) => {
  const getStatusColor = () => {
    switch (backendStatus) {
      case "ready":
        return "bg-emerald-400 shadow-sm shadow-emerald-400/50";
      case "loading":
        return "bg-blue-400 animate-pulse";
      case "weights_missing":
      case "model_mismatch":
        return "bg-yellow-400";
      case "prohibited_provider":
        return "bg-red-500";
      default:
        return "bg-amber-400";
    }
  };

  const getStatusLabel = () => {
    switch (backendStatus) {
      case "ready":
        return "Online";
      case "loading":
        return "Loading";
      case "weights_missing":
        return "Weights Missing";
      case "model_mismatch":
        return "Mismatch";
      case "prohibited_provider":
        return "Prohibited";
      case "unreachable":
        return "Offline";
      default:
        return "Setup Needed";
    }
  };

  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur px-4 flex items-center justify-between z-10 sticky top-0">
      <div className="flex items-center gap-3 sm:gap-4">
        {activeTab === "chat" && (
          <button
            onClick={onToggleSidebar}
            aria-label="Toggle Sidebar"
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition md:hidden"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}

        <Link href="/" className="flex items-center gap-2.5 group hover:opacity-95 transition" title="Return to SAM-AI Homepage">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20 font-bold text-white text-base group-hover:bg-blue-500 transition">
            S
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-slate-100 tracking-wide text-sm md:text-base group-hover:text-blue-400 transition">SAM-AI</span>
              <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                MVP
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">Sovereign Intelligence</p>
          </div>
        </Link>

        {/* Workspace Mode Tabs */}
        {onTabChange && (
          <div className="flex items-center bg-slate-950/70 p-1 rounded-lg border border-slate-800 ml-2">
            <button
              onClick={() => onTabChange("chat")}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition ${
                activeTab === "chat"
                  ? "bg-blue-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>Chat</span>
            </button>
            <button
              onClick={() => onTabChange("code")}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition ${
                activeTab === "code"
                  ? "bg-blue-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Code2 className="w-3.5 h-3.5" />
              <span>Code</span>
              <span className="text-[9px] uppercase px-1 py-0.2 rounded bg-slate-800 text-slate-300 font-mono border border-slate-700">
                Read-Only
              </span>
            </button>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 sm:gap-3">
        {/* Model Status Badge */}
        <div
          title={`Active model: ${modelId} (${getStatusLabel()})`}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-800/80 border border-slate-700/60 text-xs text-slate-300"
        >
          <span className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
          <span className="font-mono text-[11px] max-w-[120px] sm:max-w-[200px] truncate">{modelId}</span>
          <span className="hidden md:inline text-[10px] text-slate-400 font-sans border-l border-slate-700 pl-1.5">
            {getStatusLabel()}
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
