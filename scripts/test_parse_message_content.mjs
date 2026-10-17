import assert from "node:assert";

export function parseMessageContent(content, isStreaming) {
  if (!content) {
    return { thought: null, answer: "", isThinking: Boolean(isStreaming) };
  }

  const closingTag = "</think>";
  const closingIdx = content.indexOf(closingTag);

  if (closingIdx !== -1) {
    let rawThought = content.slice(0, closingIdx);
    rawThought = rawThought.replace(/^\s*<think>\s*/i, "").trim();
    const answer = content.slice(closingIdx + closingTag.length).trim();
    return {
      thought: rawThought.length > 0 ? rawThought : null,
      answer,
      isThinking: false,
    };
  }

  // No </think> closing tag
  const openTag = "<think>";
  const openIdx = content.indexOf(openTag);
  if (openIdx !== -1) {
    const rawThought = content.slice(openIdx + openTag.length).trim();
    return {
      thought: rawThought.length > 0 ? rawThought : null,
      answer: "",
      isThinking: Boolean(isStreaming),
    };
  }

  if (isStreaming) {
    return {
      thought: content.trim(),
      answer: "",
      isThinking: true,
    };
  }

  // Not streaming, no think tags
  return {
    thought: null,
    answer: content.trim(),
    isThinking: false,
  };
}

console.log("=== Testing parseMessageContent ===");

// Case 1: Image 1 scenario (no open tag, has closing tag + answer)
{
  const raw = `Alright, the user greeted me with "hi." I need to respond politely and set the stage for our conversation.
Since I'm SAM-AI, I should mention that. </think>
Hello! I'm here to help with any questions or topics you'd like to discuss. How can I assist you today?`;

  const parsed = parseMessageContent(raw, false);
  assert.ok(parsed.thought !== null, "Thought must be extracted");
  assert.ok(parsed.thought.includes("the user greeted me"), "Thought content must match");
  assert.ok(parsed.answer.startsWith("Hello!"), "Answer must be separated cleanly");
  assert.strictEqual(parsed.isThinking, false);
  console.log("[PASS] Case 1: Image 1 scenario with closing tag only");
}

// Case 2: Standard <think> ... </think> ... answer
{
  const raw = "<think>\nStep 1: Compute 2 + 2\nStep 2: Result is 4\n</think>\n\nThe sum is 4.";
  const parsed = parseMessageContent(raw, false);
  assert.strictEqual(parsed.thought, "Step 1: Compute 2 + 2\nStep 2: Result is 4");
  assert.strictEqual(parsed.answer, "The sum is 4.");
  assert.strictEqual(parsed.isThinking, false);
  console.log("[PASS] Case 2: Standard <think>...</think>");
}

// Case 3: Image 2 scenario (currently streaming in middle of thought, isStreaming = true)
{
  const raw = "Okay, so I'm trying to figure out how to solve this problem. It seems like a math problem...";
  const parsed = parseMessageContent(raw, true);
  assert.strictEqual(parsed.thought, raw);
  assert.strictEqual(parsed.answer, "");
  assert.strictEqual(parsed.isThinking, true);
  console.log("[PASS] Case 3: Image 2 streaming thought scenario");
}

// Case 4: Standard direct response without reasoning (finished)
{
  const raw = "I am ready to help you with your coding project.";
  const parsed = parseMessageContent(raw, false);
  assert.strictEqual(parsed.thought, null);
  assert.strictEqual(parsed.answer, raw);
  assert.strictEqual(parsed.isThinking, false);
  console.log("[PASS] Case 4: Direct response without thinking");
}

console.log("ALL parseMessageContent TESTS PASSED!");
