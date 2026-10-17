"use client";

import React, { useRef, useEffect } from "react";
import { Send, Square, AlertCircle } from "lucide-react";

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

  // Format error message cleanly if it contains JSON payload
  const formatErrorMessage = (err: Error): string => {
    try {
      const parsed = JSON.parse(err.message);
      if (parsed?.error) return parsed.error;
    } catch {}
    return err.message || "Failed to communicate with SAM-AI backend.";
  };

  return (
    <div className="p-3 sm:p-4 bg-slate-950/80 border-t border-slate-800 backdrop-blur">
      <div className="max-w-3xl mx-auto space-y-2">
        {/* Error alert if any */}
        {error && (
          <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 flex items-center justify-between gap-2">
            <div className="flex items-start gap-2 flex-1">
              <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-400 mt-0.5" />
              <div className="flex-1">
                <span className="font-semibold">Backend Error: </span>
                {formatErrorMessage(error)}
              </div>
            </div>
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                className="px-2.5 py-1 rounded bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 text-xs font-medium transition flex-shrink-0"
              >
                Retry
              </button>
            )}
          </div>
        )}

        {/* Locked alert if password required */}
        {isLocked && (
          <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300 flex items-center justify-between">
            <span>Site is currently locked by access password.</span>
            <button
              type="button"
              onClick={onOpenPasswordModal}
              className="text-xs font-semibold text-amber-400 underline hover:text-amber-300"
            >
              Unlock Now
            </button>
          </div>
        )}

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="relative flex items-end gap-2 bg-slate-900 border border-slate-800 rounded-xl p-2 shadow-inner focus-within:border-blue-500 transition">
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
            className="flex-1 max-h-[180px] bg-transparent text-sm text-slate-100 placeholder:text-slate-500 resize-none px-2 py-1.5 focus:outline-none disabled:opacity-50"
          />

          <div className="flex items-center gap-1.5 flex-shrink-0">
            {isLoading ? (
              <button
                type="button"
                onClick={stop}
                title="Stop response"
                className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-rose-400 transition"
              >
                <Square className="w-4 h-4 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim() || isLocked}
                title="Send message"
                className="p-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:hover:bg-blue-600 text-white transition shadow-sm"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </div>
        </form>

        <div className="flex items-center justify-between text-[11px] text-slate-400 px-1">
          <span>SAM-AI Web MVP • Streamed via Vercel AI SDK</span>
          <span>{input.length} / 8,000</span>
        </div>
      </div>
    </div>
  );
};
