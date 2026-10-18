"use client";

import React, { useState, useEffect, useRef } from "react";
import { useChat } from "ai/react";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { NoticeBanners } from "./NoticeBanners";
import { PasswordModal } from "./PasswordModal";
import {
  loadConversations,
  saveConversations,
  getActiveConversationId,
  setActiveConversationId,
  createNewConversation,
  deleteConversation as deleteStorageConversation,
  getStoredPassword,
} from "@/lib/storage";
import { Conversation, Message, BackendStatus } from "@/types/chat";
import { Sparkles, MessageSquare, Terminal, Lightbulb, Compass, Loader2, AlertTriangle } from "lucide-react";
import { CodeWorkspace } from "./CodeWorkspace";

export const ChatInterface: React.FC = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isPasswordModalOpen, setIsPasswordModalOpen] = useState(false);
  const [storedPassword, setStoredPasswordState] = useState("");
  const [activeTab, setActiveTab] = useState<"chat" | "code">("chat");
  const [status, setStatus] = useState<BackendStatus>({
    configured: false,
    status: "not_configured",
    model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    provider: "sam_ai_sovereign",
    providerName: "SAM-AI Sovereign Inference Engine",
    requiresPassword: false,
    rateLimitPerMinute: 20,
    notice: "Initializing SAM-AI backend...",
    privacyNotice: "",
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inFlightConvIdRef = useRef<string | null>(null);

  // Initialize and load conversations + backend status
  useEffect(() => {
    const loaded = loadConversations();
    setConversations(loaded);

    const activeId = getActiveConversationId();
    let currentActive: Conversation | undefined;
    if (activeId && loaded.some((c) => c.id === activeId)) {
      setActiveConvId(activeId);
      currentActive = loaded.find((c) => c.id === activeId);
    } else if (loaded.length > 0) {
      setActiveConvId(loaded[0].id);
      setActiveConversationId(loaded[0].id);
      currentActive = loaded[0];
    }

    // Hydrate useChat internal message state from active conversation
    if (currentActive && Array.isArray(currentActive.messages) && currentActive.messages.length > 0) {
      setMessages(currentActive.messages as any);
    } else {
      setMessages([]);
    }

    setStoredPasswordState(getStoredPassword());

    // Fetch public status
    fetch("/api/status")
      .then((res) => res.json())
      .then((data: BackendStatus) => {
        setStatus(data);
        if (data.requiresPassword && !getStoredPassword()) {
          setIsPasswordModalOpen(true);
        }
      })
      .catch((err) => console.error("Failed to load status:", err));

    // Support prefilling prompt from URL query params (?q=... or ?prompt=...)
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const queryPrompt = params.get("q") || params.get("prompt");
      if (queryPrompt) {
        handleInputChange({ target: { value: queryPrompt } } as any);
        // Clean URL search parameters once consumed so refreshing never re-injects stale input
        try {
          const cleanUrl = new URL(window.location.href);
          cleanUrl.searchParams.delete("q");
          cleanUrl.searchParams.delete("prompt");
          window.history.replaceState({}, document.title, cleanUrl.pathname + (cleanUrl.search ? cleanUrl.search : ""));
        } catch {}
      }
    }
  }, []);

  const activeConversation = conversations.find((c) => c.id === activeConvId);

  // Initialize useChat with custom headers for password gate
  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    stop,
    reload,
    error,
    setMessages,
  } = useChat({
    api: "/api/chat",
    headers: {
      "x-sam-ai-password": storedPassword,
    },
    initialMessages: activeConversation ? (activeConversation.messages as any) : [],
    onFinish: (message) => {
      // Save into the conversation where the in-flight request originated to prevent misfiling
      const targetId = inFlightConvIdRef.current || activeConvId || createNewConversation(message.content).id;
      inFlightConvIdRef.current = null;

      setConversations((prev) => {
        const existing = prev.find((c) => c.id === targetId);

        // Strictly deduplicate messages by ID to prevent duplicate assistant or user turns
        const allCandidates = messages.some((m) => m.id === message.id) ? messages : [...messages, message];
        const seenIds = new Set<string>();
        const mergedMessages: Message[] = [];
        for (const m of allCandidates) {
          if (m.id) {
            if (!seenIds.has(m.id)) {
              seenIds.add(m.id);
              mergedMessages.push(m as Message);
            }
          } else {
            mergedMessages.push(m as Message);
          }
        }

        let updatedList: Conversation[];
        if (existing) {
          updatedList = prev.map((c) =>
            c.id === targetId
              ? {
                  ...c,
                  updatedAt: Date.now(),
                  messages: mergedMessages as Message[],
                }
              : c
          );
        } else {
          const newConv: Conversation = {
            id: targetId,
            title: message.content.slice(0, 30) || "Conversation",
            createdAt: Date.now(),
            updatedAt: Date.now(),
            messages: mergedMessages as Message[],
          };
          updatedList = [newConv, ...prev];
        }
        saveConversations(updatedList);
        return updatedList;
      });
    },
  });

  // Switch conversation
  const handleSelectConversation = (id: string) => {
    setActiveConvId(id);
    setActiveConversationId(id);
    const conv = conversations.find((c) => c.id === id);
    if (conv && Array.isArray(conv.messages)) {
      setMessages((conv.messages as any) || []);
    } else {
      setMessages([]);
    }
  };

  // Create new conversation
  const handleNewConversation = () => {
    inFlightConvIdRef.current = null;
    const newConv = createNewConversation();
    const updated = [newConv, ...conversations];
    setConversations(updated);
    saveConversations(updated);
    setActiveConvId(newConv.id);
    setActiveConversationId(newConv.id);
    setMessages([]);
  };

  // Delete conversation
  const handleDeleteConversation = (id: string) => {
    const updated = deleteStorageConversation(id);
    setConversations(updated);
    if (activeConvId === id) {
      const nextId = updated.length > 0 ? updated[0].id : null;
      setActiveConvId(nextId);
      if (nextId) {
        const nextConv = updated.find((c) => c.id === nextId);
        setMessages((nextConv?.messages as any) || []);
      } else {
        setMessages([]);
      }
    }
  };

  // Sync title when first user message is sent
  const onFormSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    let targetId = activeConvId;
    if (!targetId) {
      const newConv = createNewConversation(input);
      const updated = [newConv, ...conversations];
      setConversations(updated);
      saveConversations(updated);
      targetId = newConv.id;
      setActiveConvId(targetId);
      setActiveConversationId(targetId);
    } else if (messages.length === 0 && input.trim()) {
      setConversations((prev) => {
        const updated = prev.map((c) =>
          c.id === targetId
            ? { ...c, title: input.slice(0, 32).trim() + (input.length > 32 ? "..." : "") }
            : c
        );
        saveConversations(updated);
        return updated;
      });
    }
    inFlightConvIdRef.current = targetId;
    handleSubmit(e);
  };

  // Auto-scroll as tokens stream
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const starterPrompts = [
    {
      title: "Architecture Walkthrough",
      desc: "Explain the difference between System 1 fast heuristics and System 2 MCTS deliberation.",
      prompt: "Can you explain the difference between System 1 fast heuristics and System 2 MCTS deliberation in autonomous AI systems?",
    },
    {
      title: "Write Clean Code",
      desc: "Generate a TypeScript LRU Cache implementation with complete unit tests.",
      prompt: "Implement an efficient, fully-typed TypeScript LRU Cache class with a maximum capacity and thorough unit tests.",
    },
    {
      title: "Research Methodology",
      desc: "How should an AI lab conduct rigorous, noise-aware evaluation benchmarks?",
      prompt: "What are best practices for conducting noise-aware, contamination-free evaluation benchmarks on cognitive reasoning tasks?",
    },
  ];

  const lastAssistantIndex = messages
    .map((m, idx) => (m.role === "assistant" ? idx : -1))
    .filter((idx) => idx !== -1)
    .pop();

  return (
    <div className="flex h-[100dvh] w-full bg-[#FAF8F5] text-[#28231F] overflow-hidden">
      {/* Sidebar Navigation (Chat mode only) */}
      {activeTab === "chat" && (
        <Sidebar
          conversations={conversations}
          activeId={activeConvId}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
          isOpen={isSidebarOpen}
          onClose={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Main Stage */}
      <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#FAF8F5]">
        <Header
          modelId={status.model}
          providerName={status.providerName}
          isConfigured={status.configured}
          backendStatus={status.status}
          hasPasswordGate={status.requiresPassword}
          isUnlocked={Boolean(storedPassword)}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          onOpenPasswordModal={() => setIsPasswordModalOpen(true)}
          onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
          isSidebarOpen={isSidebarOpen}
        />

        <NoticeBanners status={status} />

        {activeTab === "code" ? (
          <CodeWorkspace backendStatus={status} />
        ) : (
          <>
            {/* Message Stream Scroll Area */}
            <div className="flex-1 overflow-y-auto">
              {messages.length === 0 ? (
                <div className="max-w-3xl mx-auto px-4 py-8 sm:py-12 text-center">
                  <div className="w-12 h-12 rounded-2xl bg-[#FBEBDD] border border-[#D97745]/20 text-[#D97745] flex items-center justify-center mx-auto mb-4 shadow-sm">
                    <Sparkles className="w-6 h-6 animate-pulse" />
                  </div>
                  <h2 className="text-xl font-bold text-[#28231F] mb-2 tracking-tight">Welcome to SAM-AI</h2>
                  <p className="text-xs text-[#766B62] max-w-md mx-auto mb-6 leading-relaxed">
                    A calm, sovereign conversational interface. Explore reasoning trajectories, inspect system architectures,
                    and review empirical research.
                  </p>

                  {/* Starter cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-left">
                    {starterPrompts.map((card, idx) => (
                      <button
                        key={idx}
                        onClick={() => {
                          handleInputChange({ target: { value: card.prompt } } as any);
                        }}
                        className="p-3.5 rounded-xl bg-[#FFFFFF] border border-[#E8E1D9] hover:border-[#D97745]/40 hover:bg-[#FAF8F5] hover:shadow-sm transition group text-left flex flex-col justify-between"
                      >
                        <div>
                          <div className="flex items-center gap-1.5 text-xs font-semibold text-[#28231F] group-hover:text-[#D97745] mb-1 transition">
                            {idx === 0 && <Compass className="w-3.5 h-3.5 text-[#D97745]" />}
                            {idx === 1 && <Terminal className="w-3.5 h-3.5 text-[#2D6A4F]" />}
                            {idx === 2 && <Lightbulb className="w-3.5 h-3.5 text-[#B45309]" />}
                            <span>{card.title}</span>
                          </div>
                          <p className="text-[11px] text-[#766B62] leading-relaxed">{card.desc}</p>
                        </div>
                        <span className="text-[10px] text-[#D97745] mt-2 block font-medium opacity-0 group-hover:opacity-100 transition">
                          Click to insert prompt &rarr;
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="divide-y divide-[#E8E1D9]">
                  {messages.map((m, idx) => (
                    <ChatMessage
                      key={m.id}
                      message={m as Message}
                      modelId={status.model}
                      isLastAssistant={m.role === "assistant" && idx === lastAssistantIndex}
                      userPrompt={
                        m.role === "assistant" && idx > 0 && messages[idx - 1]?.role === "user"
                          ? messages[idx - 1].content
                          : undefined
                      }
                      onRetry={reload}
                      isLoading={isLoading}
                    />
                  ))}

                  {isLoading && messages[messages.length - 1]?.role === "user" && (
                    <div className="py-4 px-4 sm:px-6 bg-[#FAF8F5]/80">
                      <div className="max-w-3xl mx-auto flex gap-3.5 items-center text-xs text-[#766B62]">
                        <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-[#FBEBDD] text-[#D97745] border border-[#D97745]/20 flex items-center justify-center">
                          <Loader2 className="w-4 h-4 animate-spin" />
                        </div>
                        <span>SAM-AI is generating stream...</span>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Input Dock */}
            <ChatInput
              input={input}
              handleInputChange={handleInputChange}
              handleSubmit={onFormSubmit}
              isLoading={isLoading}
              stop={stop}
              onRetry={reload}
              error={error}
              requiresPassword={status.requiresPassword}
              isUnlocked={Boolean(storedPassword)}
              onOpenPasswordModal={() => setIsPasswordModalOpen(true)}
            />
          </>
        )}
      </div>

      {/* Password Modal */}
      <PasswordModal
        isOpen={isPasswordModalOpen}
        onClose={() => setIsPasswordModalOpen(false)}
        onPasswordSaved={() => {
          setStoredPasswordState(getStoredPassword());
        }}
      />
    </div>
  );
};
