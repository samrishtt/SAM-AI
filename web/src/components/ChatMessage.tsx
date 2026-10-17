"use client";

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot, Check, Copy, RotateCw } from "lucide-react";
import { Message } from "@/types/chat";

interface ChatMessageProps {
  message: Message;
  modelId?: string;
  isLastAssistant?: boolean;
  onRetry?: () => void;
  isLoading?: boolean;
}

const CodeBlock: React.FC<{ language?: string; children: React.ReactNode }> = ({
  language,
  children,
}) => {
  const [copied, setCopied] = useState(false);
  const codeString = String(children).replace(/\n$/, "");

  const handleCopyCode = () => {
    navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group my-3 rounded-lg border border-slate-800 bg-slate-900 overflow-hidden">
      <div className="flex items-center justify-between px-3.5 py-1.5 bg-slate-950/70 border-b border-slate-800 text-[11px] font-mono text-slate-400">
        <span>{language || "code"}</span>
        <button
          onClick={handleCopyCode}
          title="Copy code"
          className="flex items-center gap-1 text-slate-400 hover:text-slate-200 transition"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-[10px] text-emerald-400 font-sans">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span className="text-[10px] font-sans">Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-3.5 overflow-x-auto font-mono text-xs text-slate-200 leading-relaxed">
        <code>{children}</code>
      </pre>
    </div>
  );
};

export const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  modelId = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  isLastAssistant = false,
  onRetry,
  isLoading = false,
}) => {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!message.content) return;
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={`py-4 px-4 sm:px-6 transition-colors ${
        isUser ? "bg-slate-900/40" : "bg-slate-950/40 border-y border-slate-900"
      }`}
    >
      <div className="max-w-3xl mx-auto flex gap-3.5 sm:gap-4 items-start">
        {/* Avatar */}
        <div
          className={`w-7 h-7 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-xs font-semibold shadow-sm ${
            isUser
              ? "bg-slate-800 text-slate-200 border border-slate-700"
              : "bg-blue-600 text-white shadow-blue-500/20"
          }`}
        >
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </div>

        {/* Message Content */}
        <div className="flex-1 min-w-0 space-y-1.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-slate-300">
                {isUser ? "You" : "SAM-AI"}
              </span>
              {!isUser && (
                <span className="text-[10px] font-mono text-slate-400 px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800">
                  {modelId}
                </span>
              )}
            </div>

            <div className="flex items-center gap-1.5">
              {!isUser && isLastAssistant && onRetry && !isLoading && (
                <button
                  onClick={onRetry}
                  title="Regenerate response"
                  className="text-slate-500 hover:text-blue-400 p-1 rounded transition text-xs flex items-center gap-1"
                >
                  <RotateCw className="w-3.5 h-3.5" />
                  <span className="text-[11px] hidden sm:inline">Retry</span>
                </button>
              )}

              <button
                onClick={handleCopy}
                title="Copy message content"
                className="text-slate-500 hover:text-slate-300 p-1 rounded transition text-xs flex items-center gap-1"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-[11px] text-emerald-400">Copied</span>
                  </>
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
          </div>

          {/* Render Markdown or plain text */}
          <div className="prose text-sm text-slate-200 leading-relaxed break-words">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ node, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || "");
                  const isInline = !match && !String(children).includes("\n");
                  if (isInline) {
                    return (
                      <code
                        className="bg-slate-800/80 text-blue-300 px-1.5 py-0.5 rounded font-mono text-xs"
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  }
                  return (
                    <CodeBlock language={match ? match[1] : undefined}>
                      {children}
                    </CodeBlock>
                  );
                },
                table({ children }) {
                  return (
                    <div className="overflow-x-auto my-3 border border-slate-800 rounded-lg">
                      <table className="min-w-full divide-y divide-slate-800 text-xs text-left">
                        {children}
                      </table>
                    </div>
                  );
                },
                thead({ children }) {
                  return (
                    <thead className="bg-slate-900/90 text-slate-200 font-semibold">
                      {children}
                    </thead>
                  );
                },
                th({ children }) {
                  return (
                    <th className="px-3.5 py-2.5 text-xs font-semibold text-slate-200 border-b border-slate-800">
                      {children}
                    </th>
                  );
                },
                td({ children }) {
                  return (
                    <td className="px-3.5 py-2 text-xs text-slate-300 border-b border-slate-800/60">
                      {children}
                    </td>
                  );
                },
                ul({ children }) {
                  return (
                    <ul className="list-disc list-outside ml-5 my-2 space-y-1 text-slate-200">
                      {children}
                    </ul>
                  );
                },
                ol({ children }) {
                  return (
                    <ol className="list-decimal list-outside ml-5 my-2 space-y-1 text-slate-200">
                      {children}
                    </ol>
                  );
                },
                blockquote({ children }) {
                  return (
                    <blockquote className="border-l-2 border-blue-500/50 pl-3.5 my-2 text-slate-400 italic">
                      {children}
                    </blockquote>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
};

