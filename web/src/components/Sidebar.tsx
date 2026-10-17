"use client";

import React from "react";
import { Plus, MessageSquare, Trash2, X, Sparkles, Database } from "lucide-react";
import { Conversation } from "@/types/chat";

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
  isOpen: boolean;
  onClose: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  conversations,
  activeId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  isOpen,
  onClose,
}) => {
  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 bg-black/60 z-20 md:hidden backdrop-blur-sm transition-opacity"
        />
      )}

      <aside
        className={`fixed md:static inset-y-0 left-0 z-30 w-72 bg-[#FFFFFF] border-r border-[#E8E1D9] flex flex-col transition-transform duration-200 ease-in-out ${
          isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        {/* New Chat Button */}
        <div className="p-3.5 border-b border-[#E8E1D9] flex items-center gap-2">
          <button
            onClick={() => {
              onNewConversation();
              onClose();
            }}
            className="flex-1 flex items-center justify-center gap-2 px-3.5 py-2.5 rounded-lg bg-[#D97745] hover:bg-[#B95F34] text-white text-xs font-semibold shadow-sm transition"
          >
            <Plus className="w-4 h-4" />
            <span>New Conversation</span>
          </button>
          <button
            onClick={onClose}
            aria-label="Close sidebar"
            className="p-2 rounded-lg text-[#766B62] hover:text-[#28231F] hover:bg-[#F5F0EA] md:hidden transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <div className="px-2 py-1.5 text-[11px] font-semibold tracking-wider text-[#766B62] uppercase">
            Local History
          </div>

          {conversations.length === 0 ? (
            <div className="px-3 py-6 text-center text-xs text-[#766B62] space-y-1">
              <MessageSquare className="w-6 h-6 mx-auto text-[#766B62]/50 mb-2" />
              <p className="font-medium text-[#28231F]">No conversations yet</p>
              <p className="text-[11px]">Chat messages are stored locally in your browser.</p>
            </div>
          ) : (
            conversations.map((conv) => {
              const isActive = conv.id === activeId;
              return (
                <div
                  key={conv.id}
                  onClick={() => {
                    onSelectConversation(conv.id);
                    onClose();
                  }}
                  className={`group flex items-center justify-between px-3 py-2 rounded-lg text-xs cursor-pointer transition ${
                    isActive
                      ? "bg-[#FBEBDD] text-[#D97745] font-semibold border border-[#D97745]/20"
                      : "text-[#28231F] hover:bg-[#F5F0EA]"
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0 flex-1">
                    <MessageSquare className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? "text-[#D97745]" : "text-[#766B62]"}`} />
                    <span className="truncate">{conv.title}</span>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConversation(conv.id);
                    }}
                    title="Delete conversation"
                    className="opacity-0 group-hover:opacity-100 p-1 text-[#766B62] hover:text-[#C53030] rounded transition"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })
          )}
        </div>

        {/* Local Storage Privacy Footer */}
        <div className="p-3 border-t border-[#E8E1D9] text-[11px] text-[#766B62] flex items-center justify-between bg-[#FAF8F5]">
          <div className="flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-[#766B62]" />
            <span>Local Browser Storage</span>
          </div>
          <span className="text-[10px] text-[#766B62] font-mono">MVP v0.1</span>
        </div>
      </aside>
    </>
  );
};
