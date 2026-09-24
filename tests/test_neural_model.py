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
