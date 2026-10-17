"use client";

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Sparkles, Check, Copy, RotateCw } from "lucide-react";
import { Message } from "@/types/chat";
import { filterReasoningFromText } from "@/lib/reasoning-filter";

interface ChatMessageProps {
  message: Message;
  modelId?: string;
  isLastAssistant?: boolean;
  onRetry?: () => void;
  isLoading?: boolean;
}

export interface ParsedMessageContent {
  thought: null;
  answer: string;
  isThinking: boolean;
}

/**
 * Returns clean final answer, ensuring zero raw reasoning traces are displayed.
 */
export function parseMessageContent(content: string, isStreaming: boolean): ParsedMessageContent {
  if (!content) {
    return { thought: null, answer: "", isThinking: isStreaming };
  }
  const clean = filterReasoningFromText(content);
  return {
    thought: null,
    answer: clean,
    isThinking: isStreaming && !clean,
  };
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
    <div className="relative group my-3 rounded-lg border border-[#E8E1D9] bg-[#F5F0EA] overflow-hidden">
      <div className="flex items-center justify-between px-3.5 py-1.5 bg-[#EAE3D9] border-b border-[#E8E1D9] text-[11px] font-mono text-[#766B62]">
        <span>{language || "code"}</span>
        <button
          onClick={handleCopyCode}
          title="Copy code"
          className="flex items-center gap-1 text-[#766B62] hover:text-[#28231F] transition"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-600" />
              <span className="text-[10px] text-emerald-700 font-sans">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span className="text-[10px] font-sans">Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-3.5 overflow-x-auto font-mono text-xs text-[#28231F] leading-relaxed">
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

  // Client-side defense-in-depth sanitization: strips any legacy raw reasoning traces
  const answer = isUser ? message.content : filterReasoningFromText(message.content);
  const isStreamingThis = !isUser && isLastAssistant && isLoading && !answer;

  const handleCopy = () => {
    if (!answer) return;
    navigator.clipboard.writeText(answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const shortModel = modelId.split("/").pop() || modelId;

  return (
    <div
      className={`py-5 px-4 sm:px-6 transition-colors ${
        isUser ? "bg-[#F5F0EA]/70 border-b border-[#E8E1D9]/50" : "bg-transparent"
      }`}
    >
      <div className="max-w-3xl mx-auto flex gap-3.5 sm:gap-4 items-start">
        {/* Avatar */}
        <div
          className={`w-7 h-7 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-xs font-semibold shadow-xs ${
            isUser
              ? "bg-[#E8E1D9] text-[#766B62]"
              : "bg-[#D97745] text-white"
          }`}
        >
          {isUser ? <User className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
        </div>

        {/* Message Content */}
        <div className="flex-1 min-w-0 space-y-1.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-[#28231F]">
                {isUser ? "You" : "SAM-AI"}
              </span>
              {!isUser && (
                <span className="text-[10px] font-mono text-[#766B62] px-1.5 py-0.5 rounded bg-[#F5F0EA] border border-[#E8E1D9]">
                  {shortModel}
                </span>
              )}
            </div>

            <div className="flex items-center gap-1.5">
              {!isUser && isLastAssistant && onRetry && !isLoading && (
                <button
                  onClick={onRetry}
                  title="Regenerate response"
                  className="text-[#766B62] hover:text-[#D97745] p-1 rounded transition text-xs flex items-center gap-1"
                >
                  <RotateCw className="w-3.5 h-3.5" />
                  <span className="text-[11px] hidden sm:inline">Retry</span>
                </button>
              )}

              <button
                onClick={handleCopy}
                title="Copy message content"
                className="text-[#766B62] hover:text-[#28231F] p-1 rounded transition text-xs flex items-center gap-1"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                    <span className="text-[11px] text-emerald-700">Copied</span>
                  </>
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
          </div>

          {/* Render Markdown Answer */}
          {answer ? (
            <div className="prose text-sm text-[#28231F] leading-relaxed break-words">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ node, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "");
                    const isInline = !match && !String(children).includes("\n");
                    if (isInline) {
                      return (
                        <code
                          className="bg-[#F5F0EA] text-[#B95F34] px-1.5 py-0.5 rounded font-mono text-xs border border-[#E8E1D9]"
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
                      <div className="overflow-x-auto my-3 border border-[#E8E1D9] rounded-lg">
                        <table className="min-w-full divide-y divide-[#E8E1D9] text-xs text-left">
                          {children}
                        </table>
                      </div>
                    );
                  },
                  thead({ children }) {
                    return (
                      <thead className="bg-[#F5F0EA] text-[#28231F] font-semibold">
                        {children}
                      </thead>
                    );
                  },
                  th({ children }) {
                    return (
                      <th className="px-3.5 py-2.5 text-xs font-semibold text-[#28231F] border-b border-[#E8E1D9]">
                        {children}
                      </th>
                    );
                  },
                  td({ children }) {
                    return (
                      <td className="px-3.5 py-2 text-xs text-[#28231F] border-b border-[#E8E1D9]/60">
                        {children}
                      </td>
                    );
                  },
                  ul({ children }) {
                    return (
                      <ul className="list-disc list-outside ml-5 my-2 space-y-1 text-[#28231F]">
                        {children}
                      </ul>
                    );
                  },
                  ol({ children }) {
                    return (
                      <ol className="list-decimal list-outside ml-5 my-2 space-y-1 text-[#28231F]">
                        {children}
                      </ol>
                    );
                  },
                  blockquote({ children }) {
                    return (
                      <blockquote className="border-l-2 border-[#D97745] pl-3.5 my-2 text-[#766B62] italic bg-[#FBEBDD]/40 py-1 rounded-r">
                        {children}
                      </blockquote>
                    );
                  },
                }}
              >
                {answer}
              </ReactMarkdown>
            </div>
          ) : isStreamingThis ? (
            <div className="flex items-center gap-2 text-xs text-[#766B62] italic py-1">
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#D97745] opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#D97745]"></span>
              </span>
              <span>Generating response...</span>
            </div>
          ) : (
            <div className="text-sm text-[#766B62] italic">
              *[Empty response]*
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
