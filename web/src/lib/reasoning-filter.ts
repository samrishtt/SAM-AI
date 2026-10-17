/**
 * SAM-AI Reasoning Stream Filter
 * ==============================
 * Server-side stream processor that intercepts raw reasoning models (such as
 * DeepSeek-R1-Distill-Qwen-7B), buffers internal <think>...</think> traces,
 * handles split delimiter boundaries, and streams ONLY clean final answers.
 *
 * Invariants:
 * 1. Zero raw <think> or </think> tags exposed to the client.
 * 2. Zero partial or incomplete reasoning traces leaked if generation aborts early.
 * 3. Delimiters split across chunk boundaries (e.g. "</" and "think>") are safely buffered.
 * 4. Preserves all valid markdown, math LaTeX, code blocks, and formatting in the final answer.
 */

export interface ReasoningFilterOptions {
  /** If true, treats all text prior to </think> as internal reasoning. Default: true */
  stripThinking?: boolean;
  /** Custom fallback message if generation terminates before </think> is closed */
  interruptedMessage?: string;
}

export class ReasoningStreamFilter {
  private inThinking: boolean = true;
  private buffer: string = "";
  private hasSeenClosingTag: boolean = false;
  private answerStarted: boolean = false;
  private readonly interruptedMessage: string;

  constructor(options: ReasoningFilterOptions = {}) {
    this.interruptedMessage =
      options.interruptedMessage ||
      "\n\n*[SAM-AI Notice: The reasoning process was interrupted before a final answer could be generated. Please retry your request.]*";
  }

  /**
   * Processes an incoming incremental text chunk.
   * Returns text to emit to the client, or null if currently buffering reasoning.
   */
  public processChunk(chunk: string): string | null {
    if (!chunk) return null;

    if (this.hasSeenClosingTag) {
      // Already past </think>, pass through directly
      return chunk;
    }

    this.buffer += chunk;

    // Check for closing delimiter: </think>
    const closingTag = "</think>";
    const closingIndex = this.buffer.indexOf(closingTag);

    if (closingIndex !== -1) {
      this.hasSeenClosingTag = true;
      this.inThinking = false;

      // Extract content after </think>
      let answerPortion = this.buffer.slice(closingIndex + closingTag.length);

      // Strip leading newlines/whitespace immediately following </think>
      if (!this.answerStarted) {
        answerPortion = answerPortion.replace(/^[\r\n\s]+/, "");
        if (answerPortion.length > 0) {
          this.answerStarted = true;
        }
      }

      // Clear internal buffer to free memory
      this.buffer = "";
      return answerPortion.length > 0 ? answerPortion : null;
    }

    // Still buffering inside thinking
    return null;
  }

  /**
   * Finalizes the stream when the upstream provider finishes.
   * If generation finished before </think> was encountered, returns an honest
   * recoverable error instead of leaking the buffered internal thoughts.
   */
  public finalize(): string | null {
    if (!this.hasSeenClosingTag) {
      // Check if buffer contains no reasoning markers and no thinking characteristics
      // (e.g., if a non-reasoning endpoint was used and never emitted <think> or </think>)
      const trimmed = this.buffer.trim();
      const hasExplicitOpenTag = trimmed.includes("<think>");

      if (!hasExplicitOpenTag && trimmed.length > 0 && !this.buffer.startsWith("<｜Assistant｜>")) {
        // Safe check: If the output does NOT start with thinking and has no open tag,
        // and doesn't look like an unclosed thinking trace, check if it's a direct response
        // In DeepSeek-R1, reasoning always precedes the answer. If </think> was never emitted,
        // it means max_tokens cut off the reasoning trace before the answer was formed.
        return this.interruptedMessage;
      }

      // Truncated during reasoning
      return this.interruptedMessage;
    }

    return null;
  }

  /**
   * Returns whether a valid final answer was successfully started.
   */
  public hasCompletedReasoning(): boolean {
    return this.hasSeenClosingTag;
  }
}

/**
 * Pure function to filter complete text (non-streaming).
 */
export function filterReasoningFromText(rawText: string): string {
  if (!rawText) return "";

  const closingTag = "</think>";
  const closingIdx = rawText.indexOf(closingTag);

  if (closingIdx !== -1) {
    const after = rawText.slice(closingIdx + closingTag.length).trim();
    if (after.length > 0) {
      return after;
    }
    return "*[SAM-AI completed reasoning, but did not emit a final answer statement.]*";
  }

  // If starts with <think> but no closing tag, it was truncated
  if (rawText.trim().startsWith("<think>")) {
    return "*[SAM-AI Notice: The reasoning process was interrupted before a final answer could be generated. Please retry your request.]*";
  }

  // If no <think> or </think> tags exist, return as-is
  return rawText.trim();
}
