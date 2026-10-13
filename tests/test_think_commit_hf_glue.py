"""End-to-end check of the HF glue with a tiny RANDOM Qwen2 model. No weights are downloaded.

This proves the plumbing (generate -> stop on </think> -> KV-cache stepping -> grammar)
and that the cached logits equal full-recompute logits. It says nothing about model quality.
Skipped automatically when torch or transformers is missing.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # in the repo the module lives in sam_ai/agent/
    from sam_ai.agent import think_commit as tc  # noqa: E402
except ImportError:  # standalone checkout: sibling file
    import think_commit as tc  # noqa: E402
from test_think_commit import _build_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def stack():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    tok = _build_tokenizer()
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(tok) + 8, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    g = tc.ActionGrammar.arc3()
    return tok, model, g, tc.TokenIndex.from_tokenizer(tok, g)


def test_kv_cache_logits_match_full_recompute(stack):
    tok, model, g, idx = stack
    ids = tok.encode('PROMPT {"action": "ACT', add_special_tokens=False)
    src = tc.HFLogitsSource(model)
    src.start(ids[:-3])
    for t in ids[-3:]:
        src.step(t)
    cached = torch.tensor(src.gather(list(range(50))))
    with torch.no_grad():
        full = model(input_ids=torch.tensor([ids])).logits[0, -1][:50].float()
    assert torch.allclose(cached, full, atol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_end_to_end_budget_path_is_valid(stack, seed):
    tok, model, g, idx = stack
    prompt = tok.encode("Choose an action.", add_special_tokens=False)
    res = tc.think_then_commit(
        prompt, tc.HFThinkSource(model, tok, 0.6, 0.95, seed), tc.HFLogitsSource(model), tok, g, idx,
        think_budget=20, seed=seed,
    )
    assert g.accepts(res.action_text)
    assert res.think_tokens <= 20
    assert res.close_injected or res.think_ended_by == "close_tag"


def test_stop_on_close_stops_generation(stack):
    tok, model, g, idx = stack
    close_id = tok.convert_tokens_to_ids("</think>")
    # bias the random model so that it emits </think> immediately
    with torch.no_grad():
        bias = model.lm_head.weight.clone()
        model.lm_head.weight[close_id] += 50.0 * model.lm_head.weight.mean(0).sign()
    try:
        out = tc.HFThinkSource(model, tok, 0.0, 1.0, 0).think(tok.encode("go", add_special_tokens=False), 40, "</think>")
        assert len(out) < 40 and "</think>" in tok.decode(out, skip_special_tokens=False)
    finally:
        with torch.no_grad():
            model.lm_head.weight.copy_(bias)
