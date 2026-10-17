import assert from "node:assert";
import { ReasoningStreamFilter, filterReasoningFromText } from "../web/src/lib/reasoning-filter.ts";

console.log("=== Testing ReasoningStreamFilter ===");

// Test 1: Standard reasoning + answer
{
  const filter = new ReasoningStreamFilter();
  const c1 = filter.processChunk("<think>\nCalculating 2 + 2\n");
  assert.strictEqual(c1, null, "Thinking chunk must be suppressed");
  const c2 = filter.processChunk("</think>\n\nThe sum is 4.");
  assert.strictEqual(c2, "The sum is 4.", "Answer after </think> must be emitted without leading whitespace");
  const c3 = filter.processChunk(" Very simple.");
  assert.strictEqual(c3, " Very simple.", "Subsequent answer chunks must be passed through");
  const final = filter.finalize();
  assert.strictEqual(final, null, "Finalize must be null when answer was completed");
  console.log("[PASS] Test 1: Standard reasoning + answer");
}

// Test 2: Delimiter split across chunks ("</" then "think>")
{
  const filter = new ReasoningStreamFilter();
  assert.strictEqual(filter.processChunk("Initial thinking steps... </"), null);
  assert.strictEqual(filter.processChunk("think>\nHere is the clean result."), "Here is the clean result.");
  assert.strictEqual(filter.processChunk(" Additional details."), " Additional details.");
  assert.strictEqual(filter.finalize(), null);
  console.log("[PASS] Test 2: Split delimiter across chunks");
}

// Test 3: Delimiter split into 3 chunks ("<", "/th", "ink>\nFinal")
{
  const filter = new ReasoningStreamFilter();
  assert.strictEqual(filter.processChunk("Thoughts... <"), null);
  assert.strictEqual(filter.processChunk("/th"), null);
  assert.strictEqual(filter.processChunk("ink>\nFinal statement."), "Final statement.");
  assert.strictEqual(filter.finalize(), null);
  console.log("[PASS] Test 3: Multi-split delimiter");
}

// Test 4: Truncated generation (never emitted </think>)
{
  const filter = new ReasoningStreamFilter();
  assert.strictEqual(filter.processChunk("<think>\nI am thinking deeply about a hard problem..."), null);
  assert.strictEqual(filter.processChunk(" still thinking..."), null);
  const final = filter.finalize();
  assert.ok(final !== null, "Must emit interrupted notice when truncated in thinking");
  assert.ok(final.includes("interrupted"), "Notice must explain reasoning was interrupted");
  assert.ok(!final.includes("I am thinking deeply"), "Must NOT leak internal thoughts");
  console.log("[PASS] Test 4: Truncated generation fails closed");
}

// Test 5: filterReasoningFromText pure function
{
  const raw = "<think>\nStep 1: x = 1\nStep 2: y = 2\n</think>\n\nResult is 3.";
  const clean = filterReasoningFromText(raw);
  assert.strictEqual(clean, "Result is 3.", "Must cleanly strip thinking block");

  const rawNoTag = "Direct answer without reasoning.";
  assert.strictEqual(filterReasoningFromText(rawNoTag), "Direct answer without reasoning.");
  console.log("[PASS] Test 5: filterReasoningFromText pure function");
}

// Test 6: Plain final response with absent delimiters
{
  const filter = new ReasoningStreamFilter();
  const c1 = filter.processChunk("42 is the answer to everything.");
  assert.strictEqual(c1, null, "Buffers until </think> or finalize");
  const final = filter.finalize();
  assert.strictEqual(final, "42 is the answer to everything.", "Finalize emits plain response when no think tags were present");
  console.log("[PASS] Test 6: Plain final response with absent delimiters");
}

// Test 7: Plain response split across multiple chunks
{
  const filter = new ReasoningStreamFilter();
  filter.processChunk("The prime numbers below 10 are: ");
  filter.processChunk("2, 3, 5, 7.");
  const final = filter.finalize();
  assert.strictEqual(final, "The prime numbers below 10 are: 2, 3, 5, 7.", "Finalize emits concatenated plain response");
  console.log("[PASS] Test 7: Multi-chunk plain response without delimiters");
}

// Test 8: Malformed unclosed <think> with arbitrary content
{
  const filter = new ReasoningStreamFilter();
  filter.processChunk("<think>malformed unclosed output");
  const final = filter.finalize();
  assert.ok(final !== null, "Must fail closed");
  assert.ok(final.includes("interrupted"), "Must emit interrupted notice");
  assert.ok(!final.includes("malformed"), "Must not leak internal text");
  console.log("[PASS] Test 8: Malformed unclosed <think> fails closed");
}

// Test 9: Mathematical LaTeX and Code preserved in final answer
{
  const raw = "<think>\nComputing integral of 2x\n</think>\n\n$$\\int 2x dx = x^2 + C$$\n\n```python\ndef f(x):\n    return x**2\n```";
  const clean = filterReasoningFromText(raw);
  assert.ok(!clean.includes("<think>"));
  assert.ok(!clean.includes("</think>"));
  assert.ok(!clean.includes("Computing integral"));
  assert.ok(clean.includes("$$\\int 2x dx = x^2 + C$$"));
  assert.ok(clean.includes("```python\ndef f(x):\n    return x**2\n```"));
  console.log("[PASS] Test 9: Mathematical LaTeX and code preserved");
}

// Test 10: Empty stream handling
{
  const filter = new ReasoningStreamFilter();
  assert.strictEqual(filter.processChunk(""), null);
  assert.strictEqual(filter.finalize(), null);
  console.log("[PASS] Test 10: Empty stream handling");
}

console.log("ALL REASONING FILTER TESTS PASSED (10/10)!");
