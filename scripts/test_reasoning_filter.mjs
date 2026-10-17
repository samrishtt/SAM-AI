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

console.log("ALL REASONING FILTER TESTS PASSED!");
