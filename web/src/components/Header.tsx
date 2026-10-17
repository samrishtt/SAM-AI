"use client";

import React from "react";
import Link from "next/link";
import { MessageSquare, Code2, ShieldCheck, Lock, Unlock, Menu, ArrowLeft } from "lucide-react";
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
        return "bg-emerald-600";
      case "loading":
        return "bg-amber-500 animate-pulse";
      case "weights_missing":
      case "model_mismatch":
        return "bg-amber-600";
      case "prohibited_provider":
        return "bg-red-500";
      default:
        return "bg-amber-600";
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
    <header className="h-14 border-b border-[#E8E1D9] bg-[#FFFFFF]/90 backdrop-blur px-4 flex items-center justify-between z-10 sticky top-0">
      <div className="flex items-center gap-3">
        {activeTab === "chat" && (
          <button
            onClick={onToggleSidebar}
            aria-label="Toggle Conversations"
            className="p-1.5 rounded-lg text-[#766B62] hover:text-[#28231F] hover:bg-[#F5F0EA] transition"
          >
            <Menu className="w-4 h-4" />
          </button>
        )}

        <Link
          href="/"
          className="flex items-center gap-2 group text-[#766B62] hover:text-[#28231F] transition text-xs font-medium mr-1"
          title="Return to Overview"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Overview</span>
        </Link>

        <div className="h-4 w-[1px] bg-[#E8E1D9] hidden sm:block" />

        {/* Workspace Mode Tabs */}
        {onTabChange && (
          <div className="flex items-center bg-[#F5F0EA] p-0.5 rounded-lg border border-[#E8E1D9]">
            <button
              onClick={() => onTabChange("chat")}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition ${
                activeTab === "chat"
                  ? "bg-[#FFFFFF] text-[#28231F] shadow-xs"
                  : "text-[#766B62] hover:text-[#28231F]"
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>Chat</span>
            </button>
            <button
              onClick={() => onTabChange("code")}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition ${
                activeTab === "code"
                  ? "bg-[#FFFFFF] text-[#28231F] shadow-xs"
                  : "text-[#766B62] hover:text-[#28231F]"
              }`}
            >
              <Code2 className="w-3.5 h-3.5" />
              <span>Code</span>
              <span className="text-[9px] uppercase px-1 py-0.2 rounded bg-[#FAF8F5] text-[#766B62] font-mono border border-[#E8E1D9]">
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
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#F5F0EA] border border-[#E8E1D9] text-xs text-[#766B62]"
        >
          <span className={`w-1.5 h-1.5 rounded-full ${getStatusColor()}`} />
          <span className="font-mono text-[11px] text-[#28231F] max-w-[120px] sm:max-w-[180px] truncate">
            {modelId.split("/").pop() || modelId}
          </span>
          <span className="hidden md:inline text-[10px] text-[#766B62] border-l border-[#E8E1D9] pl-1.5">
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
                ? "bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100"
                : "bg-[#FBEBDD] text-[#D97745] border border-[#E8E1D9] hover:bg-[#F5F0EA]"
            }`}
          >
            {isUnlocked ? <Unlock className="w-3.5 h-3.5" /> : <Lock className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline">{isUnlocked ? "Unlocked" : "Password"}</span>
          </button>
        ) : (
          <div
            title="Preview Mode: Open research access"
            className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#FAF8F5] border border-[#E8E1D9] text-xs text-[#766B62]"
          >
            <ShieldCheck className="w-3.5 h-3.5 text-[#766B62]" />
            <span className="text-[11px]">Preview</span>
          </div>
        )}
      </div>
    </header>
  );
};
