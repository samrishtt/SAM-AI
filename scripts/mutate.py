#!/usr/bin/env python3
"""Mutation test for think_commit.py: inject one deliberate bug at a time and see if the tests notice.

A mutant that SURVIVES means the tests cannot tell correct code from broken code at that spot.
Run: python3 mutate.py   (offline; ~8 s per mutant)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "think_commit.py"), encoding="utf-8").read()
TESTS = ["test_think_commit.py", "test_think_commit_hf_glue.py"]

# (id, description, old, new)
MUTANTS = [
    ("M01", "ARC grid 64 -> 65", "grid: int = 64,", "grid: int = 65,"),
    ("M02", "grammar: drop earlier-is-prefix check", "                if node.terminal:\n                    raise ValueError(f\"an earlier action string is a proper prefix of {s!r}\")\n", ""),
    ("M03", "grammar: drop later-is-prefix check", "            if node.children:\n                raise ValueError(f\"action string is a proper prefix of another: {s!r}\")\n", ""),
    ("M04", "accepts(): ignore terminal flag", "        return node.terminal\n\n    def find_dead_ends", "        return True\n\n    def find_dead_ends"),
    ("M05", "top_p ignored (keep everything)", "        if acc >= top_p:\n            break", "        if False:\n            break"),
    ("M06", "top_p keeps only the single best token", "        if acc >= top_p:\n            break", "        break"),
    ("M07", "temperature ignored in softmax", "math.exp((s - m) / temperature)", "math.exp((s - m))"),
    ("M08", "temperature<=0 returns index 0 not argmax", "    if temperature <= 0:\n        return best", "    if temperature <= 0:\n        return 0"),
    ("M09", "pick: sample uniformly among kept", "        acc += ws[i]\n        if r <= acc:", "        acc += total / len(keep)\n        if r <= acc:"),
    ("M10", "no close tag injected on budget exhaustion", 'suffix_ids = enc("\\n" + think_close + "\\n\\n")', "suffix_ids = []"),
    ("M11", "suffix not fed to logits source", "list(prompt_ids) + list(ids) + list(suffix_ids)", "list(prompt_ids) + list(ids)"),
    ("M12", "think thoughts dropped from commit context", "list(prompt_ids) + list(ids) + list(suffix_ids)", "list(prompt_ids) + list(suffix_ids)"),
    ("M13", "KV state not advanced in commit_decode", "        if node.children:\n            source.step(tid)", "        if node.children:\n            pass"),
    ("M14", "over-budget think output tolerated", "    if len(think_ids) > think_budget:\n        raise", "    if False:\n        raise"),
    ("M15", "ended_by: budget test off by one", 'ended_by = "budget" if len(think_ids) >= think_budget else', 'ended_by = "budget" if len(think_ids) > think_budget else'),
    ("M16", "text after close tag not cut off", "            ids = enc(text[: at + len(think_close)])\n            need = \"\\n\\n\"", "            ids = think_ids\n            need = \"\\n\\n\""),
    ("M17", "no separator added after close tag", "need = \"\" if after.endswith(\"\\n\\n\") else (\"\\n\" if after.endswith(\"\\n\") else \"\\n\\n\")", "need = \"\""),
    ("M18", "seed ignored", "rng = random.Random(seed)", "rng = random.Random()"),
    ("M19", "completion_text single newline", 'f"{self.think_text}\\n{THINK_CLOSE}\\n\\n{self.action_text}"', 'f"{self.think_text}\\n{THINK_CLOSE}\\n{self.action_text}"'),
    ("M20", "TokenIndex keeps tokens with ANY char outside alphabet", "if not s or any(c not in alphabet for c in s):", "if not s or all(c not in alphabet for c in s):"),
    ("M21", "find_dead_ends never reports", "            if not cands:\n                bad.append(prefix)\n                continue", "            if not cands:\n                continue"),
    ("M22", "commit_decode: no-candidate error swallowed to empty", "        if not cands:\n            raise GrammarError(f\"no token can continue after {''.join(parts)!r}\")", "        if not cands:\n            break"),
    ("M23", "commit_decode: step limit not enforced", "    else:\n        raise GrammarError(f\"grammar not finished after {max_steps} tokens: {''.join(parts)!r}\")", "    else:\n        pass"),
    ("M24", "HF think: max_new_tokens budget+1", "max_new_tokens=budget,", "max_new_tokens=budget + 1,"),
    ("M25", "HF think: stop-on-close disabled", "hit = close in tok.decode(tail, skip_special_tokens=False, clean_up_tokenization_spaces=False)", "hit = False"),
    ("M26", "HF glue: KV cache not passed (wrong logits)", "                past_key_values=self._past,\n", ""),
    ("M27", "HF glue: reads logits of position 0", "self._past, self._logits = out.past_key_values, out.logits[0, -1]\n\n    def step", "self._past, self._logits = out.past_key_values, out.logits[0, 0]\n\n    def step"),
    ("M28", "HF glue: seed not applied", "        if self.seed is not None:\n            torch.manual_seed(self.seed)\n", ""),
    ("M29", "pick: non-finite max not guarded", "    if not math.isfinite(m):\n        raise GrammarError(\"all candidate scores are non-finite\")\n", ""),
    ("M30", "think_text keeps trailing whitespace (budget path)", "think_text = text.rstrip()", "think_text = text"),
    ("M31", "pick: temperature multiplies instead of divides", "math.exp((s - m) / temperature)", "math.exp((s - m) * temperature)"),
    ("M32", "pick: argmax becomes argmin", "best = max(range(n), key=lambda i: scores[i])", "best = min(range(n), key=lambda i: scores[i])"),
    ("M33", "ARC click template swaps x and y", "% (click_action, x, y) for x in range(grid) for y in range(grid)]", "% (click_action, y, x) for x in range(grid) for y in range(grid)]"),
    ("M34", "HF think: sampling disabled", "kw = dict(do_sample=self.temperature > 0)", "kw = dict(do_sample=False)"),
    ("M35", "pick: nucleus built from lowest-weight first", "order = sorted(range(n), key=lambda i: -ws[i])", "order = sorted(range(n), key=lambda i: ws[i])"),
    ("M36", "commit_decode: first allowed token always chosen", "tid, s, node = cands[pick(scores, temperature, top_p, rng)]", "tid, s, node = cands[0]"),
]


def run(mutant):
    mid, desc, old, new = mutant
    if SRC.count(old) != 1:
        return mid, desc, "BAD_PATCH", f"pattern occurs {SRC.count(old)}x"
    d = tempfile.mkdtemp(prefix="mut_")
    try:
        for f in TESTS:
            shutil.copy(os.path.join(HERE, f), d)
        open(os.path.join(d, "think_commit.py"), "w", encoding="utf-8").write(SRC.replace(old, new))
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS],
                cwd=d, env=env, capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return mid, desc, "KILLED", "timeout/hang"
        out = p.stdout + p.stderr
        if p.returncode == 0:
            return mid, desc, "SURVIVED", ""
        failed = [l for l in out.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")]
        return mid, desc, "KILLED", (failed[0][:110] if failed else out.strip().splitlines()[-1][:110])
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    only = set(sys.argv[1:])
    results = []
    for m in MUTANTS:
        if only and m[0] not in only:
            continue
        r = run(m)
        results.append(r)
        print(f"{r[2]:9s} {r[0]} {r[1]}  {r[3]}", flush=True)
    killed = sum(r[2] == "KILLED" for r in results)
    surv = [r[0] for r in results if r[2] == "SURVIVED"]
    bad = [r[0] for r in results if r[2] == "BAD_PATCH"]
    print(f"\nkilled {killed}/{len(results)}; survived {surv}; bad patches {bad}")
    json.dump(results, open(os.path.join(HERE, "mutation_results.json"), "w"), indent=1)
