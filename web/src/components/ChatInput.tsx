"use client";

import React, { useRef, useEffect } from "react";
import { Send, Square, AlertCircle, ArrowUp } from "lucide-react";

interface ChatInputProps {
  input: string;
  handleInputChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  isLoading: boolean;
  stop: () => void;
  onRetry?: () => void;
  error?: Error | null;
  requiresPassword?: boolean;
  isUnlocked?: boolean;
  onOpenPasswordModal?: () => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  input,
  handleInputChange,
  handleSubmit,
  isLoading,
  stop,
  onRetry,
  error,
  requiresPassword,
  isUnlocked,
  onOpenPasswordModal,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && input.trim()) {
        const form = e.currentTarget.form;
        if (form) form.requestSubmit();
      }
    }
  };

  const isLocked = requiresPassword && !isUnlocked;

  const formatErrorMessage = (err: Error): string => {
    try {
      const parsed = JSON.parse(err.message);
      if (parsed?.error) return parsed.error;
    } catch {}
    return err.message || "Failed to communicate with SAM-AI backend.";
  };

  return (
    <div className="p-3 sm:p-4 bg-[#FAF8F5]/90 border-t border-[#E8E1D9] backdrop-blur sticky bottom-0">
      <div className="max-w-3xl mx-auto space-y-2">
        {/* Error alert if any */}
        {error && (
          <div className="p-2.5 rounded-xl bg-red-50 border border-red-200 text-xs text-red-800 flex items-center justify-between gap-2 shadow-xs">
            <div className="flex items-start gap-2 flex-1">
              <AlertCircle className="w-4 h-4 flex-shrink-0 text-red-600 mt-0.5" />
              <div className="flex-1">
                <span className="font-semibold">Backend Notice: </span>
                {formatErrorMessage(error)}
              </div>
            </div>
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                className="px-2.5 py-1 rounded-lg bg-red-100 hover:bg-red-200 text-red-800 text-xs font-medium transition flex-shrink-0"
              >
                Retry
              </button>
            )}
          </div>
        )}

        {/* Locked alert if password required */}
        {isLocked && (
          <div className="p-2.5 rounded-xl bg-[#FBEBDD] border border-[#E8E1D9] text-xs text-[#B95F34] flex items-center justify-between shadow-xs">
            <span>Site is currently locked by access password.</span>
            <button
              type="button"
              onClick={onOpenPasswordModal}
              className="text-xs font-semibold text-[#D97745] underline hover:text-[#B95F34]"
            >
              Unlock Now
            </button>
          </div>
        )}

        {/* Input Form */}
        <form
          onSubmit={handleSubmit}
          className="relative flex items-end gap-2 bg-[#FFFFFF] border border-[#E8E1D9] rounded-2xl p-2 shadow-xs focus-within:border-[#D97745] focus-within:ring-1 focus-within:ring-[#D97745]/30 transition"
        >
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={
              isLocked
                ? "Please unlock the site with SAM_AI_SITE_PASSWORD..."
                : "Ask SAM-AI anything... (Shift+Enter for newline)"
            }
            disabled={isLoading || isLocked}
            className="flex-1 max-h-[180px] bg-transparent text-sm text-[#28231F] placeholder:text-[#766B62]/60 resize-none px-2 py-1.5 focus:outline-none disabled:opacity-50"
          />

          <div className="flex items-center gap-1.5 flex-shrink-0">
            {isLoading ? (
              <button
                type="button"
                onClick={stop}
                title="Stop generation"
                className="p-2 rounded-xl bg-[#F5F0EA] hover:bg-[#E8E1D9] text-red-600 transition"
              >
                <Square className="w-4 h-4 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim() || isLocked}
                title="Send message"
                className="p-2 rounded-xl bg-[#D97745] hover:bg-[#B95F34] disabled:opacity-30 disabled:hover:bg-[#D97745] text-white transition shadow-xs"
              >
                <ArrowUp className="w-4 h-4" />
              </button>
            )}
          </div>
        </form>

        <div className="flex items-center justify-between text-[11px] text-[#766B62] px-1">
          <span>SAM-AI Research Preview • Verifiable Reasoning</span>
          <span>{input.length} / 8,000</span>
        </div>
      </div>
    </div>
  );
};
