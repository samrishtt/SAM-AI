"""Unit tests for the Sovereign Neural Transformer Model."""

import pytest
import numpy as np
from core.neural.transformer import SovereignNeuralTransformer, TransformerConfig


def test_neural_transformer_initialization():
    config = TransformerConfig(
        vocab_size=128,
        seq_len=32,
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=1,
    )
    model = SovereignNeuralTransformer(config)
    param_count = model.count_parameters()
    assert param_count > 1000
    print(f"Initialized neural model with {param_count:,} parameters.")


def test_neural_transformer_forward_pass():
    config = TransformerConfig(vocab_size=128, seq_len=16, d_model=32, n_heads=2, d_ff=64, n_layers=1)
    model = SovereignNeuralTransformer(config)

    tokens = model.tokenizer.encode("Hello World")
    logits, cache = model.forward(tokens)

    assert logits.shape == (len(tokens), config.vocab_size)
    assert not np.isnan(logits).any()


def test_neural_transformer_loss_and_training():
    config = TransformerConfig(vocab_size=128, seq_len=32, d_model=32, n_heads=2, d_ff=64, n_layers=1)
    model = SovereignNeuralTransformer(config)

    text = "Artificial intelligence requires continuous learning."
    initial_loss = model.train_step(text)
    assert initial_loss > 0.0

    # Train for a few steps
    for _ in range(5):
        loss = model.train_step(text)

    assert loss is not None


def test_neural_transformer_autoregressive_generation():
    config = TransformerConfig(vocab_size=128, seq_len=32, d_model=32, n_heads=2, d_ff=64, n_layers=1)
    model = SovereignNeuralTransformer(config)

    prompt = "AI"
    output = model.generate(prompt, max_new_tokens=10, temperature=0.7)
    assert isinstance(output, str)
    assert output.startswith("AI")
    assert len(output) > len(prompt)


def test_moe_transformer_routing_and_forward():
    config = TransformerConfig(
        vocab_size=128,
        seq_len=32,
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=1,
        use_moe=True,
        n_experts=4,
        top_k=2,
        use_shared_expert=True,
    )
    model = SovereignNeuralTransformer(config)
    total_params = model.count_parameters()
    active_params = model.count_active_parameters()

    # Total params should be strictly greater than active params due to sparsity
    assert total_params > active_params
    assert active_params > 1000

    tokens = model.tokenizer.encode("Autonomous MoE Test")
    logits, cache = model.forward(tokens)
    assert logits.shape == (len(tokens), config.vocab_size)
    assert not np.isnan(logits).any()

    # Check that MoE router cache exists
    block_cache = cache["block_caches"][0]
    ffn_cache = block_cache["c_ffn"]
    assert "router_logits" in ffn_cache
    assert "top_indices" in ffn_cache
    assert ffn_cache["top_indices"].shape == (len(tokens), 2)
    assert ffn_cache["shared_out"] is not None


def test_recurrent_depth_execution():
    config = TransformerConfig(
        vocab_size=128,
        seq_len=32,
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=2,
        recurrent_depth=3,
    )
    model = SovereignNeuralTransformer(config)
    tokens = model.tokenizer.encode("Recurrent Depth Looped Test")
    logits, cache = model.forward(tokens)

    assert logits.shape == (len(tokens), config.vocab_size)
    assert cache["recurrent_depth_applied"] == 3
    # 2 layers * 3 recurrent iterations = 6 block executions
    assert len(cache["block_caches"]) == 6


def test_moe_full_parameter_training():
    from core.neural.full_trainer import FullParameterTrainer

    config = TransformerConfig(
        vocab_size=128,
        seq_len=16,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1,
        use_moe=True,
        n_experts=3,
        top_k=2,
    )
    model = SovereignNeuralTransformer(config)
    trainer = FullParameterTrainer(model, lr=0.01)

    # Verify MoE router and expert weights are registered
    param_names = [p["name"] for p in trainer.params]
    assert any("moe_router" in n for n in param_names)
    assert any("moe_exp0" in n for n in param_names)

    tokens = model.tokenizer.encode("Training MoE")
    loss_before = trainer.compute_loss(tokens)
    assert loss_before > 0.0

    # Perturb/train step
    loss_after = trainer.train_step(tokens)
    assert loss_after > 0.0
