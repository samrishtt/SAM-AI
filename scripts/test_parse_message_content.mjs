import assert from "node:assert";
import { filterReasoningFromText } from "../web/src/lib/reasoning-filter.ts";

export function parseMessageContent(content, isStreaming) {
  if (!content) {
    return { thought: null, answer: "", isThinking: Boolean(isStreaming) };
  }
  const clean = filterReasoningFromText(content);
  return {
    thought: null,
    answer: clean,
    isThinking: Boolean(isStreaming) && !clean,
  };
}

console.log("=== Testing Phase 2G Final-Answer-Only Message Parsing ===");

// Case 1: Image 1 scenario (no open tag, has closing tag + answer)
{
  const raw = `Alright, the user greeted me with "hi." I need to respond politely and set the stage for our conversation.
Since I'm SAM-AI, I should mention that. </think>
Hello! I'm here to help with any questions or topics you'd like to discuss. How can I assist you today?`;

  const parsed = parseMessageContent(raw, false);
  assert.strictEqual(parsed.thought, null, "Raw thoughts must NEVER be returned");
  assert.ok(parsed.answer.startsWith("Hello!"), "Answer must be extracted cleanly");
  assert.ok(!parsed.answer.includes("user greeted me"), "Raw reasoning must be completely absent from answer");
  assert.strictEqual(parsed.isThinking, false);
  console.log("[PASS] Case 1: Image 1 scenario cleanly extracts final answer without leaking thoughts");
}

// Case 2: Standard <think> ... </think> ... answer
{
  const raw = "<think>\nStep 1: Compute 2 + 2\nStep 2: Result is 4\n</think>\n\nThe sum is 4.";
  const parsed = parseMessageContent(raw, false);
  assert.strictEqual(parsed.thought, null, "Thought must be null");
  assert.strictEqual(parsed.answer, "The sum is 4.", "Answer must be exact");
  assert.strictEqual(parsed.isThinking, false);
  console.log("[PASS] Case 2: Standard <think>...</think> yields clean final answer only");
}

// Case 3: Streaming thought scenario before answer is ready
{
  const raw = "<think>\nOkay, so I'm trying to figure out how to solve this problem...";
  const parsed = parseMessageContent(raw, true);
  assert.strictEqual(parsed.thought, null, "Thought must never leak");
  assert.ok(parsed.answer.includes("interrupted"), "Unclosed think fails closed");
  console.log("[PASS] Case 3: Streaming thought fails closed safely without leaking");
}

// Case 4: Standard direct response without reasoning
{
  const raw = "I am ready to help you with your coding project.";
  const parsed = parseMessageContent(raw, false);
  assert.strictEqual(parsed.thought, null);
  assert.strictEqual(parsed.answer, raw);
  assert.strictEqual(parsed.isThinking, false);
  console.log("[PASS] Case 4: Direct response without thinking passes through cleanly");
}

console.log("ALL Phase 2G parseMessageContent TESTS PASSED!");
