"""Think-then-commit decoding (upgrade U1).

Problem: an R1-style model thinks for thousands of tokens and then writes an
answer. If the thinking budget runs out, the answer is never written, and if
the answer is written it may not be valid JSON. Both failures cost a model call
and, in an interactive game, a wasted or missing action.

Method:
  1. THINK: let the model reason freely up to `think_budget` new tokens, stopping
     early when it emits `</think>`.
  2. CLOSE: if the model did not close its thinking, inject "\\n</think>\\n\\n".
  3. COMMIT: generate the action under a fixed grammar. At every step only tokens
     that keep the output a prefix of some allowed action string can be chosen,
     so the result is valid by construction.

Properties (tested without any model, see tests/test_think_commit.py):
  * tokenizer-agnostic: the allowed-token set is derived by decoding every
    vocabulary entry once, so it works with any tokenizer, including multi-
    character tokens;
  * dead ends are detected up front (`ActionGrammar.find_dead_ends`), not at run
    time in the middle of a game;
  * no fallback action is ever invented: if the grammar cannot be satisfied it
    raises `GrammarError`.

What this does NOT do: it does not make the chosen action *good*. It removes
format failures and truncation failures only. Whether the model's reasoning is
good is a separate, measured question.

The HF glue at the bottom (`HFThinkSource`, `HFLogitsSource`) is a thin wrapper
around `model.generate` / `model(...)`. It needs torch and transformers and is
covered by `tests/test_think_commit_hf_glue.py`, which is skipped when they are
not installed. The core logic needs neither.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

THINK_CLOSE = "</think>"
DEFAULT_SIMPLE_ACTIONS = ("RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7")


class GrammarError(RuntimeError):
    """The grammar cannot be satisfied with this tokenizer, or decoding stalled."""


# --------------------------------------------------------------------------- grammar
class _Node:
    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: Dict[str, "_Node"] = {}
        self.terminal = False


class ActionGrammar:
    """A finite set of allowed action strings, stored as a character trie."""

    def __init__(self, strings: Iterable[str]) -> None:
        self.root = _Node()
        self._strings: List[str] = []
        self.alphabet: set = set()
        for s in strings:
            if not s:
                raise ValueError("empty action string")
            node = self.root
            for ch in s:
                if node.terminal:
                    raise ValueError(f"an earlier action string is a proper prefix of {s!r}")
                node = node.children.setdefault(ch, _Node())
                self.alphabet.add(ch)
            if node.children:
                raise ValueError(f"action string is a proper prefix of another: {s!r}")
            node.terminal = True
            self._strings.append(s)
        if not self._strings:
            raise ValueError("grammar has no strings")

    @classmethod
    def arc3(
        cls,
        simple_template: str = '{"action": "%s"}',
        click_template: str = '{"action": "%s", "x": %d, "y": %d}',
        simple_actions: Sequence[str] = DEFAULT_SIMPLE_ACTIONS,
        click_action: str = "ACTION6",
        grid: int = 64,
    ) -> "ActionGrammar":
        """ARC-AGI-3 actions: RESET and ACTION1-5,7 take no coordinates;
        ACTION6 takes x,y in 0..grid-1. The templates MUST match the schema that
        the agent's parser accepts; a test must parse every string through it."""
        strings = [simple_template % a for a in simple_actions]
        strings += [click_template % (click_action, x, y) for x in range(grid) for y in range(grid)]
        return cls(strings)

    def strings(self) -> List[str]:
        return list(self._strings)

    def accepts(self, text: str) -> bool:
        node = self.root
        for ch in text:
            node = node.children.get(ch)
            if node is None:
                return False
        return node.terminal

    def find_dead_ends(self, index: "TokenIndex", limit: int = 20) -> List[str]:
        """Prefixes (at token boundaries) from which the tokenizer cannot continue.

        An empty list means every path the decoder can take can be completed.
        """
        bad: List[str] = []
        seen = {id(self.root)}
        stack: List[Tuple[_Node, str]] = [(self.root, "")]
        while stack and len(bad) < limit:
            node, prefix = stack.pop()
            if not node.children:
                continue
            cands = index.allowed(node)
            if not cands:
                bad.append(prefix)
                continue
            for _tid, s, nxt in cands:
                if id(nxt) not in seen:
                    seen.add(id(nxt))
                    stack.append((nxt, prefix + s))
        return bad


# --------------------------------------------------------------------------- tokens
class TokenIndex:
    """Vocabulary entries that could appear in the grammar, bucketed by first char."""

    def __init__(self, token_strings: Sequence[str], alphabet: set) -> None:
        self.by_first: Dict[str, List[Tuple[int, str]]] = {}
        self.n_usable = 0
        for tid, s in enumerate(token_strings):
            if not s or any(c not in alphabet for c in s):
                continue
            self.by_first.setdefault(s[0], []).append((tid, s))
            self.n_usable += 1

    @classmethod
    def from_tokenizer(cls, tokenizer, grammar: ActionGrammar, vocab_size: Optional[int] = None) -> "TokenIndex":
        n = vocab_size if vocab_size is not None else len(tokenizer)
        strs = [
            tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for i in range(n)
        ]
        return cls(strs, grammar.alphabet)

    def allowed(self, node: _Node) -> List[Tuple[int, str, _Node]]:
        out: List[Tuple[int, str, _Node]] = []
        for ch in node.children:
            for tid, s in self.by_first.get(ch, ()):
                n = node
                for c in s:
                    n = n.children.get(c)
                    if n is None:
                        break
                else:
                    out.append((tid, s, n))
        return out


# --------------------------------------------------------------------------- sources
class ThinkSource(Protocol):
    def think(self, prompt_ids: List[int], budget: int, close: str) -> List[int]:
        """Return new token ids (never more than `budget`), stopping after `close`."""


class LogitsSource(Protocol):
    def start(self, ids: List[int]) -> None: ...
    def step(self, token_id: int) -> None: ...
    def gather(self, token_ids: Sequence[int]) -> Sequence[float]:
        """Scores for the given ids at the current position."""


# --------------------------------------------------------------------------- sampling
def pick(scores: Sequence[float], temperature: float, top_p: float, rng: random.Random) -> int:
    """Choose an index among the allowed candidates. top_p applies to the allowed set."""
    n = len(scores)
    if n == 0:
        raise GrammarError("no candidates")
    if n == 1:
        return 0
    best = max(range(n), key=lambda i: scores[i])
    if temperature <= 0:
        return best
    m = scores[best]
    if not math.isfinite(m):
        raise GrammarError("all candidate scores are non-finite")
    ws = [math.exp((s - m) / temperature) if math.isfinite(s) else 0.0 for s in scores]
    z = sum(ws)
    order = sorted(range(n), key=lambda i: -ws[i])
    keep, acc = [], 0.0
    for i in order:
        keep.append(i)
        acc += ws[i] / z
        if acc >= top_p:
            break
    total = sum(ws[i] for i in keep)
    r = rng.random() * total
    acc = 0.0
    for i in keep:
        acc += ws[i]
        if r <= acc:
            return i
    return keep[-1]


def commit_decode(
    source: LogitsSource,
    index: TokenIndex,
    grammar: ActionGrammar,
    *,
    temperature: float,
    top_p: float,
    rng: random.Random,
    max_steps: int = 128,
) -> Tuple[List[int], str]:
    node, ids, parts = grammar.root, [], []
    for _ in range(max_steps):
        if not node.children:
            break
        cands = index.allowed(node)
        if not cands:
            raise GrammarError(f"no token can continue after {''.join(parts)!r}")
        scores = source.gather([t for t, _s, _n in cands])
        tid, s, node = cands[pick(scores, temperature, top_p, rng)]
        ids.append(tid)
        parts.append(s)
        if node.children:
            source.step(tid)
    else:
        raise GrammarError(f"grammar not finished after {max_steps} tokens: {''.join(parts)!r}")
    text = "".join(parts)
    if not node.terminal or not grammar.accepts(text):  # defensive; cannot happen by construction
        raise GrammarError(f"decoded text is not an allowed action: {text!r}")
    return ids, text


# --------------------------------------------------------------------------- result
@dataclass
class ThinkCommitResult:
    action_text: str
    think_text: str
    think_ended_by: str  # "close_tag" | "budget" | "stopped_early"
    close_injected: bool
    think_tokens: int
    injected_tokens: int
    commit_tokens: int
    think_seconds: float
    commit_seconds: float
    seed: Optional[int]
    temperature: float
    top_p: float
    think_budget: int
    parse_valid_by_construction: bool = True
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def completion_text(self) -> str:
        """Text shaped like a normal R1 completion, so existing parsers keep working."""
        return f"{self.think_text}\n{THINK_CLOSE}\n\n{self.action_text}"

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["completion_chars"] = len(self.completion_text)
        return d


def think_then_commit(
    prompt_ids: List[int],
    think_source: ThinkSource,
    logits_source: LogitsSource,
    tokenizer,
    grammar: ActionGrammar,
    index: TokenIndex,
    *,
    think_budget: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: Optional[int] = None,
    think_close: str = THINK_CLOSE,
) -> ThinkCommitResult:
    if think_budget < 1:
        raise ValueError("think_budget must be >= 1")
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
    dec = lambda ids: tokenizer.decode(  # noqa: E731
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    t0 = time.perf_counter()
    think_ids = list(think_source.think(prompt_ids, think_budget, think_close))
    if len(think_ids) > think_budget:
        raise GrammarError(f"think source returned {len(think_ids)} tokens for budget {think_budget}")
    t1 = time.perf_counter()

    text = dec(think_ids)
    at = text.find(think_close)
    injected = 0
    if at >= 0:
        before, after = text[:at], text[at + len(think_close):]
        ended_by, close_injected = "close_tag", False
        if after.strip() == "":
            ids = think_ids
            need = "" if after.endswith("\n\n") else ("\n" if after.endswith("\n") else "\n\n")
        else:
            # text after the tag in the same chunk: cut it off and re-tokenise the head
            ids = enc(text[: at + len(think_close)])
            need = "\n\n"
        suffix_ids = enc(need) if need else []
        injected = len(suffix_ids)
        think_text = before.rstrip()
    else:
        ended_by = "budget" if len(think_ids) >= think_budget else "stopped_early"
        close_injected = True
        suffix_ids = enc("\n" + think_close + "\n\n")
        injected = len(suffix_ids)
        ids = think_ids
        think_text = text.rstrip()

    logits_source.start(list(prompt_ids) + list(ids) + list(suffix_ids))
    rng = random.Random(seed)
    commit_ids, action_text = commit_decode(
        logits_source, index, grammar, temperature=temperature, top_p=top_p, rng=rng
    )
    t2 = time.perf_counter()
    return ThinkCommitResult(
        action_text=action_text,
        think_text=think_text,
        think_ended_by=ended_by,
        close_injected=close_injected,
        think_tokens=len(think_ids),
        injected_tokens=injected,
        commit_tokens=len(commit_ids),
        think_seconds=t1 - t0,
        commit_seconds=t2 - t1,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        think_budget=think_budget,
    )


# --------------------------------------------------------------------------- HF glue
class HFThinkSource:
    """Phase 1 with transformers `generate`. Needs torch + transformers."""

    def __init__(self, model, tokenizer, temperature: float = 0.6, top_p: float = 0.95, seed: Optional[int] = None):
        self.model, self.tokenizer = model, tokenizer
        self.temperature, self.top_p, self.seed = temperature, top_p, seed

    def think(self, prompt_ids: List[int], budget: int, close: str) -> List[int]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        tok, n_prompt = self.tokenizer, len(prompt_ids)

        class _StopOnClose(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
                tail = input_ids[0, max(n_prompt, input_ids.shape[1] - 8):].tolist()
                hit = close in tok.decode(tail, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                return torch.tensor([hit], dtype=torch.bool, device=input_ids.device)

        if self.seed is not None:
            torch.manual_seed(self.seed)
        device = next(self.model.parameters()).device
        x = torch.tensor([prompt_ids], device=device)
        kw = dict(do_sample=self.temperature > 0)
        if self.temperature > 0:
            kw.update(temperature=self.temperature, top_p=self.top_p)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        out = self.model.generate(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            max_new_tokens=budget,
            stopping_criteria=StoppingCriteriaList([_StopOnClose()]),
            pad_token_id=pad,
            **kw,
        )
        return out[0, n_prompt:].tolist()


class HFLogitsSource:
    """Phase 3 with a KV cache. Needs torch."""

    def __init__(self, model):
        self.model = model
        self._logits = None
        self._past = None

    def _device(self):
        return next(self.model.parameters()).device

    def start(self, ids: List[int]) -> None:
        import torch

        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids], device=self._device()), use_cache=True)
        self._past, self._logits = out.past_key_values, out.logits[0, -1]

    def step(self, token_id: int) -> None:
        import torch

        with torch.no_grad():
            out = self.model(
                input_ids=torch.tensor([[token_id]], device=self._device()),
                past_key_values=self._past,
                use_cache=True,
            )
        self._past, self._logits = out.past_key_values, out.logits[0, -1]

    def gather(self, token_ids: Sequence[int]) -> Sequence[float]:
        import torch

        idx = torch.tensor(list(token_ids), device=self._logits.device)
        return self._logits.index_select(0, idx).float().tolist()
