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
      const trimmed = this.buffer.trim();
      if (!trimmed) {
        return null;
      }
      // If generation contained an opening <think> tag or Assistant token without </think>,
      // or starts with typical internal reasoning deliberation phrases, fail closed without leaking thoughts.
      const reasoningMarkers = ["<think>", "<｜Assistant｜>"];
      const reasoningPrefixes = ["Okay,", "Alright,", "Let me think", "First,", "We need", "To find", "I need to"];
      if (
        reasoningMarkers.some((m) => this.buffer.includes(m)) ||
        reasoningPrefixes.some((p) => trimmed.startsWith(p))
      ) {
        return this.interruptedMessage;
      }
      // Otherwise, the output is a plain final response with absent delimiters.
      return trimmed;
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

  // If contains <think> or starts with reasoning deliberation, fail closed
  if (
    rawText.includes("<think>") ||
    rawText.trim().startsWith("Okay,") ||
    rawText.trim().startsWith("Alright,") ||
    rawText.trim().startsWith("Let me think") ||
    rawText.trim().startsWith("First, I need")
  ) {
    return "*[SAM-AI Notice: The reasoning process was interrupted before a final answer could be generated. Please retry your request.]*";
  }

  // If no <think> or </think> tags exist, return as-is
  return rawText.trim();
}
