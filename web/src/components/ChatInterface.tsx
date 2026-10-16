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

export const ChatInterface: React.FC = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isPasswordModalOpen, setIsPasswordModalOpen] = useState(false);
  const [storedPassword, setStoredPasswordState] = useState("");
  const [status, setStatus] = useState<BackendStatus>({
    configured: false,
    status: "not_configured",
    model: "Qwen/Qwen3.8-27B",
    provider: "qwen_openai_compatible",
    providerName: "Qwen Sovereign Inference Endpoint",
    requiresPassword: false,
    rateLimitPerMinute: 20,
    notice: "Initializing SAM-AI backend...",
    privacyNotice: "",
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize and load conversations + backend status
  useEffect(() => {
    const loaded = loadConversations();
    setConversations(loaded);

    const activeId = getActiveConversationId();
    if (activeId && loaded.some((c) => c.id === activeId)) {
      setActiveConvId(activeId);
    } else if (loaded.length > 0) {
      setActiveConvId(loaded[0].id);
      setActiveConversationId(loaded[0].id);
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
      // Sync messages to local conversation
      setConversations((prev) => {
        const targetId = activeConvId || createNewConversation(message.content).id;
        const existing = prev.find((c) => c.id === targetId);

        let updatedList: Conversation[];
        if (existing) {
          updatedList = prev.map((c) =>
            c.id === targetId
              ? {
                  ...c,
                  updatedAt: Date.now(),
                  messages: [...messages, message] as Message[],
                }
              : c
          );
        } else {
          const newConv: Conversation = {
            id: targetId,
            title: message.content.slice(0, 30) || "Conversation",
            createdAt: Date.now(),
            updatedAt: Date.now(),
            messages: [...messages, message] as Message[],
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
    if (conv) {
      setMessages((conv.messages as any) || []);
    }
  };

  // Create new conversation
  const handleNewConversation = () => {
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
    if (!activeConvId) {
      const newConv = createNewConversation(input);
      const updated = [newConv, ...conversations];
      setConversations(updated);
      saveConversations(updated);
      setActiveConvId(newConv.id);
      setActiveConversationId(newConv.id);
    } else if (messages.length === 0 && input.trim()) {
      setConversations((prev) => {
        const updated = prev.map((c) =>
          c.id === activeConvId
            ? { ...c, title: input.slice(0, 32).trim() + (input.length > 32 ? "..." : "") }
            : c
        );
        saveConversations(updated);
        return updated;
      });
    }
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
    <div className="flex h-[100dvh] w-full bg-slate-950 overflow-hidden">
      {/* Sidebar Navigation */}
      <Sidebar
        conversations={conversations}
        activeId={activeConvId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
      />

      {/* Main Chat Stage */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <Header
          modelId={status.model}
          providerName={status.providerName}
          isConfigured={status.configured}
          hasPasswordGate={status.requiresPassword}
          isUnlocked={Boolean(storedPassword)}
          onOpenPasswordModal={() => setIsPasswordModalOpen(true)}
          onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
          isSidebarOpen={isSidebarOpen}
        />

        <NoticeBanners status={status} />

        {/* Message Stream Scroll Area */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="max-w-3xl mx-auto px-4 py-8 sm:py-12 text-center">
              <div className="w-12 h-12 rounded-2xl bg-blue-600/10 border border-blue-500/20 text-blue-400 flex items-center justify-center mx-auto mb-4 shadow-lg shadow-blue-500/10">
                <Sparkles className="w-6 h-6 animate-pulse" />
              </div>
              <h2 className="text-xl font-bold text-slate-100 mb-2">Welcome to SAM-AI</h2>
              <p className="text-xs text-slate-400 max-w-md mx-auto mb-6 leading-relaxed">
                A clean, streaming conversational AI interface. Ask questions, explore code architectures,
                and draft technical research.
              </p>

              {/* Setup Guidance Box when backend is unconfigured */}
              {!status.configured && (
                <div className="mb-6 p-4 rounded-xl bg-slate-900 border border-amber-500/30 text-left space-y-3 shadow-xl">
                  <div className="flex items-center gap-2 text-xs font-semibold text-amber-300">
                    <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
                    <span>Inference Backend Setup Required</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    SAM-AI is configured to serve <strong className="text-slate-100">{status.model}</strong> via an OpenAI-compatible inference endpoint. To activate responses, add your credentials to <code className="text-blue-300 bg-slate-950 px-1.5 py-0.5 rounded font-mono text-[11px]">web/.env.local</code>:
                  </p>
                  <div className="space-y-2">
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 font-mono text-[11px] text-slate-300">
                      <div className="text-slate-400 text-[10px] mb-1 font-sans"># Option A: Hugging Face Inference Providers Router (Recommended Hosted Endpoint)</div>
                      <div className="text-blue-300">SAM_AI_BASE_URL=https://router.huggingface.co/v1</div>
                      <div className="text-blue-300">SAM_AI_API_KEY=hf_your_token_here</div>
                      <div className="text-blue-300">SAM_AI_MODEL=Qwen/Qwen3.8-27B</div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 font-mono text-[11px] text-slate-300">
                      <div className="text-slate-400 text-[10px] mb-1 font-sans"># Option B: Local vLLM, SGLang, or Ollama Server</div>
                      <div className="text-emerald-300">SAM_AI_BASE_URL=http://localhost:8000/v1</div>
                      <div className="text-emerald-300">SAM_AI_MODEL=Qwen/Qwen3.8-27B</div>
                    </div>
                  </div>
                </div>
              )}

              {/* Starter cards */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-left">
                {starterPrompts.map((card, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      handleInputChange({ target: { value: card.prompt } } as any);
                    }}
                    className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 hover:border-slate-700 hover:bg-slate-800/60 transition group text-left flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-200 group-hover:text-blue-400 mb-1 transition">
                        {idx === 0 && <Compass className="w-3.5 h-3.5 text-blue-400" />}
                        {idx === 1 && <Terminal className="w-3.5 h-3.5 text-emerald-400" />}
                        {idx === 2 && <Lightbulb className="w-3.5 h-3.5 text-amber-400" />}
                        <span>{card.title}</span>
                      </div>
                      <p className="text-[11px] text-slate-400 leading-relaxed">{card.desc}</p>
                    </div>
                    <span className="text-[10px] text-blue-400 mt-2 block font-medium opacity-0 group-hover:opacity-100 transition">
                      Click to insert prompt →
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="divide-y divide-slate-900">
              {messages.map((m, idx) => (
                <ChatMessage
                  key={m.id}
                  message={m as Message}
                  modelId={status.model}
                  isLastAssistant={m.role === "assistant" && idx === lastAssistantIndex}
                  onRetry={reload}
                  isLoading={isLoading}
                />
              ))}

              {isLoading && messages[messages.length - 1]?.role === "user" && (
                <div className="py-4 px-4 sm:px-6 bg-slate-950/40">
                  <div className="max-w-3xl mx-auto flex gap-3.5 items-center text-xs text-slate-400">
                    <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white">
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
