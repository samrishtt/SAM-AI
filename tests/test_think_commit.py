"""Tests for think_commit.py. Pure Python + the `tokenizers` library; no torch, no model, no network."""
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # in the repo the module lives in sam_ai/agent/
    from sam_ai.agent import think_commit as tc  # noqa: E402
except ImportError:  # standalone checkout: sibling file
    import think_commit as tc  # noqa: E402

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")


# ------------------------------------------------------------------ fixtures
def _build_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False)]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700,
        special_tokens=["<|eos|>", "<think>", "</think>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    g = tc.ActionGrammar.arc3()
    rng = random.Random(0)
    corpus = rng.sample(g.strings(), 600) + [
        "Let me think about this carefully. The grid has a red block near the top.",
        "I should try moving the block to the right and see what changes.",
        "</think>\n\nThe best next move is to click the highlighted cell.",
    ] * 20
    tok.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|eos|>", pad_token="<|eos|>")


@pytest.fixture(scope="module")
def tokenizer():
    return _build_tokenizer()


@pytest.fixture(scope="module")
def grammar():
    return tc.ActionGrammar.arc3()


@pytest.fixture(scope="module")
def index(tokenizer, grammar):
    return tc.TokenIndex.from_tokenizer(tokenizer, grammar)


class RandomLogits:
    """Scores are random; records the emitted text so tests can inspect it."""

    def __init__(self, index, seed=0, scale=1.0):
        self.rng = random.Random(seed)
        self.scale = scale
        self.started_with = None

    def start(self, ids):
        self.started_with = list(ids)

    def step(self, token_id):
        pass

    def gather(self, ids):
        return [self.rng.gauss(0, self.scale) for _ in ids]


class TargetLogits:
    """Prefers the longest token that keeps the output on the way to `target`."""

    def __init__(self, index, target):
        self.table = {tid: s for bucket in index.by_first.values() for tid, s in bucket}
        self.target, self.emitted, self.started_with = target, "", None

    def start(self, ids):
        self.started_with, self.emitted = list(ids), ""

    def step(self, token_id):
        self.emitted += self.table[token_id]

    def gather(self, ids):
        out = []
        for t in ids:
            s = self.table[t]
            out.append(float(len(s)) if self.target.startswith(self.emitted + s) else -5.0)
        return out


class ScriptedThink:
    def __init__(self, ids):
        self.ids, self.seen = list(ids), None

    def think(self, prompt_ids, budget, close):
        self.seen = (list(prompt_ids), budget, close)
        return self.ids[:budget]


# ------------------------------------------------------------------ grammar
def test_arc3_grammar_size_and_membership(grammar):
    s = grammar.strings()
    assert len(s) == 7 + 64 * 64 and len(set(s)) == len(s)
    assert grammar.accepts('{"action": "RESET"}')
    assert grammar.accepts('{"action": "ACTION6", "x": 63, "y": 0}')


def test_every_string_is_valid_json_with_in_range_coordinates(grammar):
    for s in grammar.strings():
        d = json.loads(s)
        assert d["action"].startswith(("ACTION", "RESET"))
        if d["action"] == "ACTION6":
            assert set(d) == {"action", "x", "y"} and 0 <= d["x"] < 64 and 0 <= d["y"] < 64
        else:
            assert set(d) == {"action"}


@pytest.mark.parametrize(
    "bad",
    [
        '{"action": "ACTION6", "x": 64, "y": 0}',
        '{"action": "ACTION6", "x": 5, "y": 100}',
        '{"action": "ACTION8"}',
        '{"action": "action1"}',
        '{"action": "ACTION1"',
        '{"action": "ACTION6", "x": 05, "y": 0}',
        "",
    ],
)
def test_grammar_rejects_out_of_grammar(grammar, bad):
    assert not grammar.accepts(bad)


@pytest.mark.parametrize(
    "strings",
    [["ab", "abc"], ["abc", "ab"], ["a", "ab", "abc"], ["abc", "ab", "a"], ["xy", "abc", "ab"]],
)
def test_prefix_strings_are_refused(strings):
    # both insertion orders matter: the conflict is detected in two different places
    with pytest.raises(ValueError):
        tc.ActionGrammar(strings)


# ------------------------------------------------------------------ tokens
def test_index_has_multichar_tokens(index):
    assert any(len(s) > 1 for b in index.by_first.values() for _t, s in b)


def test_no_dead_ends_with_bpe_tokenizer(grammar, index):
    assert grammar.find_dead_ends(index) == []


def test_dead_end_is_detected_and_decode_raises():
    g = tc.ActionGrammar(['{"a": 7}'])
    strings = ['{', '"', 'a', ':', ' ', '}', '{"', '":']  # no '7'
    idx = tc.TokenIndex(strings, g.alphabet)
    assert g.find_dead_ends(idx) == ['{"a": ']
    src = type("S", (), {"start": lambda s, i: None, "step": lambda s, t: None, "gather": lambda s, ids: [0.0] * len(ids)})()
    with pytest.raises(tc.GrammarError):
        tc.commit_decode(src, idx, g, temperature=0.0, top_p=1.0, rng=random.Random(0))


# ------------------------------------------------------------------ commit decoding
TARGETS = [
    '{"action": "RESET"}',
    '{"action": "ACTION1"}',
    '{"action": "ACTION7"}',
    '{"action": "ACTION6", "x": 63, "y": 0}',
    '{"action": "ACTION6", "x": 0, "y": 63}',
    '{"action": "ACTION6", "x": 7, "y": 7}',
    '{"action": "ACTION6", "x": 10, "y": 20}',
]


@pytest.mark.parametrize("target", TARGETS)
def test_greedy_reaches_each_target(grammar, index, target):
    src = TargetLogits(index, target)
    src.start([])
    ids, text = tc.commit_decode(src, index, grammar, temperature=0.0, top_p=1.0, rng=random.Random(0))
    assert text == target and grammar.accepts(text) and len(ids) >= 1


@pytest.mark.parametrize("temperature,top_p", [(1.0, 1.0), (0.6, 0.95), (2.0, 0.9)])
def test_fuzz_random_logits_always_valid(grammar, index, temperature, top_p):
    seen = set()
    for seed in range(600):
        src = RandomLogits(index, seed, scale=3.0)
        _ids, text = tc.commit_decode(src, index, grammar, temperature=temperature, top_p=top_p, rng=random.Random(seed))
        assert grammar.accepts(text)
        seen.add(text)
    assert len(seen) > 30


def test_pick_is_deterministic_and_respects_top_p():
    scores = [5.0, 4.9, -10.0, -10.0]
    picks = {tc.pick(scores, 0.6, 0.9, random.Random(s)) for s in range(200)}
    assert picks <= {0, 1}
    assert tc.pick(scores, 0.0, 1.0, random.Random(0)) == 0
    assert tc.pick([1.0, 2.0], 0.6, 0.95, random.Random(7)) == tc.pick([1.0, 2.0], 0.6, 0.95, random.Random(7))
    with pytest.raises(tc.GrammarError):
        tc.pick([float("-inf"), float("-inf")], 0.6, 0.95, random.Random(0))


def _freq(scores, temperature, top_p, n=6000, seed0=0):
    rng = random.Random(seed0)
    counts = [0] * len(scores)
    for _ in range(n):
        counts[tc.pick(scores, temperature, top_p, rng)] += 1
    return [c / n for c in counts]


def test_pick_follows_softmax_distribution():
    # scores [2,1,0] at T=1 -> p = 0.665, 0.245, 0.090 (exact softmax)
    f = _freq([2.0, 1.0, 0.0], 1.0, 1.0)
    assert abs(f[0] - 0.665) < 0.03 and abs(f[1] - 0.245) < 0.03 and abs(f[2] - 0.090) < 0.03


def test_pick_temperature_changes_the_distribution():
    # scores [1,0]: p(best) = 0.982 at T=0.25, 0.731 at T=1, 0.562 at T=4
    cold, mid, hot = (_freq([1.0, 0.0], t, 1.0)[0] for t in (0.25, 1.0, 4.0))
    assert abs(cold - 0.982) < 0.02 and abs(mid - 0.731) < 0.03 and abs(hot - 0.562) < 0.03


def test_pick_top_p_truncates_the_tail_exactly():
    scores = [2.0, 1.0, 0.0]  # p = 0.665, 0.245, 0.090
    assert set(_freq(scores, 1.0, 0.60, n=500)) == {1.0, 0.0}  # nucleus = {0} only
    f = _freq(scores, 1.0, 0.80)  # nucleus = {0, 1}: never index 2, renormalised 0.731 / 0.269
    assert f[2] == 0.0 and abs(f[0] - 0.731) < 0.03
    assert _freq(scores, 1.0, 1.0)[2] > 0.05  # top_p=1 keeps the whole tail


# ------------------------------------------------------------------ think phase
def _run(tokenizer, grammar, index, think_text, budget, target='{"action": "ACTION2"}', seed=1):
    prompt = tokenizer.encode("PROMPT ", add_special_tokens=False)
    think = ScriptedThink(tokenizer.encode(think_text, add_special_tokens=False))
    logits = TargetLogits(index, target)
    res = tc.think_then_commit(prompt, think, logits, tokenizer, grammar, index, think_budget=budget, seed=seed)
    return prompt, think, logits, res


def test_model_closes_its_own_thinking(tokenizer, grammar, index):
    prompt, think, logits, res = _run(tokenizer, grammar, index, "I will go up. </think>", 500)
    assert res.think_ended_by == "close_tag" and not res.close_injected
    assert res.action_text == '{"action": "ACTION2"}'
    assert res.think_text == "I will go up."
    sent = tokenizer.decode(logits.started_with, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert sent.startswith("PROMPT I will go up. </think>") and sent.endswith("\n\n")
    assert res.completion_text == 'I will go up.\n</think>\n\n{"action": "ACTION2"}'
    assert think.seen[1:] == (500, "</think>")


def test_budget_exhaustion_injects_close(tokenizer, grammar, index):
    long_text = "thinking " * 200
    prompt, think, logits, res = _run(tokenizer, grammar, index, long_text, 30)
    assert res.think_ended_by == "budget" and res.close_injected
    assert res.think_tokens == 30 and res.injected_tokens > 0
    sent = tokenizer.decode(logits.started_with, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert sent.endswith("\n</think>\n\n") and grammar.accepts(res.action_text)


def test_budget_path_think_text_has_no_trailing_whitespace(tokenizer, grammar, index):
    text = "alpha beta gamma \n\n  "
    n = len(tokenizer.encode(text, add_special_tokens=False))
    _p, _t, _l, res = _run(tokenizer, grammar, index, text, n)
    assert res.think_ended_by == "budget"
    assert res.think_text == res.think_text.rstrip() and res.think_text.startswith("alpha")


def test_model_stops_early_without_close(tokenizer, grammar, index):
    _p, _t, logits, res = _run(tokenizer, grammar, index, "short thought", 500)
    assert res.think_ended_by == "stopped_early" and res.close_injected
    assert grammar.accepts(res.action_text)


def test_text_after_close_tag_is_cut(tokenizer, grammar, index):
    _p, _t, logits, res = _run(tokenizer, grammar, index, "idea </think>\n\nJUNK ANSWER", 500)
    sent = tokenizer.decode(logits.started_with, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert "JUNK" not in sent and sent.endswith("</think>\n\n")
    assert res.think_ended_by == "close_tag"


def test_over_budget_source_is_rejected(tokenizer, grammar, index):
    class Bad:
        def think(self, p, b, c):
            return list(range(b + 5))

    with pytest.raises(tc.GrammarError):
        tc.think_then_commit([1], Bad(), TargetLogits(index, TARGETS[0]), tokenizer, grammar, index, think_budget=10)


def test_bad_budget_rejected(tokenizer, grammar, index):
    with pytest.raises(ValueError):
        tc.think_then_commit([1], ScriptedThink([]), TargetLogits(index, TARGETS[0]), tokenizer, grammar, index, think_budget=0)


def test_same_seed_same_action_and_seeds_vary(tokenizer, grammar, index):
    class Uniform(RandomLogits):
        def gather(self, ids):
            return [0.0] * len(ids)

    def go(seed):
        think = ScriptedThink(tokenizer.encode("x </think>", add_special_tokens=False))
        r = tc.think_then_commit([1], think, Uniform(index), tokenizer, grammar, index, think_budget=50, seed=seed)
        return r.action_text

    assert go(3) == go(3)
    assert len({go(s) for s in range(200)}) > 20


def test_result_to_dict_has_telemetry_fields(tokenizer, grammar, index):
    _p, _t, _l, res = _run(tokenizer, grammar, index, "a </think>", 50)
    d = res.to_dict()
    for k in ("think_tokens", "commit_tokens", "injected_tokens", "close_injected", "think_ended_by",
              "think_seconds", "commit_seconds", "seed", "temperature", "top_p", "think_budget",
              "parse_valid_by_construction"):
        assert k in d
