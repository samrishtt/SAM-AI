# Mutation test of think_commit.py's tests (Claude, 2026-10-13)

Method: inject ONE deliberate bug into think_commit.py, run the tests, expect a failure. A mutant that
survives means the tests cannot tell correct code from broken code at that spot. Tool: mutate.py
(copy it next to think_commit.py + the two test files; offline, ~8 s per mutant).

Round 1 (57 -> 47 tests then): 30 mutants, 20 killed, 10 survived.
  7 REAL gaps in my tests: M03 (prefix conflict, second insertion order), M05 (top_p ignored),
  M07 (temperature ignored), M09 (uniform instead of softmax), M27 (HF start() reads wrong position),
  M28 (HF seed not applied), M30 (trailing whitespace in think_text on budget path).
  3 equivalent mutants (cannot change behaviour): M20, M22, M23.
Fix: 10 new tests (prefix order x5, softmax distribution, temperature, exact top_p nucleus,
  budget-path whitespace, start() logits, HF seed reproducibility).
Round 2: 36 mutants (6 new, written without looking at the tests), 32 killed, 4 survived.
  All 4 are equivalent: M20 (index filter is redundant with the trie walk), M22/M23 (a defensive final check
  raises the same GrammarError class), M33 (swapping x/y in the template enumerates the same set).
Test count after fix: 57 (38 think_commit + 9 path discovery, +10 new). All pass.

What this does NOT show: that the decoder makes the real 14B model better. It shows the decoder code does what its
tests say, and that those tests fail when the code is broken.
