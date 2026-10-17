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
        className={`fixed md:static inset-y-0 left-0 z-30 w-72 bg-slate-900 border-r border-slate-800 flex flex-col transition-transform duration-200 ease-in-out ${
          isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        {/* New Chat Button */}
        <div className="p-3.5 border-b border-slate-800 flex items-center gap-2">
          <button
            onClick={() => {
              onNewConversation();
              onClose();
            }}
            className="flex-1 flex items-center justify-center gap-2 px-3.5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-md shadow-blue-500/10 transition"
          >
            <Plus className="w-4 h-4" />
            <span>New Conversation</span>
          </button>
          <button
            onClick={onClose}
            aria-label="Close sidebar"
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 md:hidden transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <div className="px-2 py-1.5 text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
            Local History
          </div>

          {conversations.length === 0 ? (
            <div className="px-3 py-6 text-center text-xs text-slate-400 space-y-1">
              <MessageSquare className="w-6 h-6 mx-auto text-slate-600 mb-2 opacity-60" />
              <p>No conversations yet</p>
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
                      ? "bg-slate-800 text-slate-100 font-medium"
                      : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200"
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0 flex-1">
                    <MessageSquare className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? "text-blue-400" : "text-slate-500"}`} />
                    <span className="truncate">{conv.title}</span>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConversation(conv.id);
                    }}
                    title="Delete conversation"
                    className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-rose-400 rounded transition"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })
          )}
        </div>

        {/* Local Storage Privacy Footer */}
        <div className="p-3 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between bg-slate-950/40">
          <div className="flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-slate-500" />
            <span>Local Browser Storage</span>
          </div>
          <span className="text-[10px] text-slate-400 font-mono">MVP v0.1</span>
        </div>
      </aside>
    </>
  );
};
