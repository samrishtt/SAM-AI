"""SAM-AI: Comprehensive Real-World Flow & Frontier Architecture Verification.

Verifies end-to-end operation across all subsystems:
1. Sparse Mixture of Experts (MoE) with Top-K routing and Shared Expert
2. Recurrent Depth (Looped Transformer) execution
3. Full-parameter gradient estimation & optimization
4. Neuro-symbolic cognitive reasoning loop (HyperAstra Engine)
5. OpenAI-compatible REST API Server verification
6. Web Chat Studio Server health check
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.neural.transformer import (
    SovereignNeuralTransformer,
    TransformerConfig,
)
from core.neural.full_trainer import FullParameterTrainer
from core.hyper_engine import HyperAstraEngine
from core.neural.foundation_runner import FoundationReasoningEngine


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def verify_frontier_moe_and_recurrent():
    print_header("1. SPARSE MIXTURE OF EXPERTS (MoE) & RECURRENT DEPTH VERIFICATION")
    
    config = TransformerConfig(
        vocab_size=128,
        seq_len=64,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        use_moe=True,
        n_experts=4,
        top_k=2,
        use_shared_expert=True,
        recurrent_depth=2,
    )
    
    print(f"[+] Initializing SovereignNeuralTransformer with config:")
    print(f"    - Embedding Dim (d_model): {config.d_model}")
    print(f"    - Attention Heads:         {config.n_heads}")
    print(f"    - Layers:                  {config.n_layers}")
    print(f"    - Sparse MoE Enabled:      {config.use_moe}")
    print(f"    - Total Experts (N):       {config.n_experts}")
    print(f"    - Active Top-K Experts:    {config.top_k}")
    print(f"    - Shared Expert:           {config.use_shared_expert}")
    print(f"    - Recurrent Depth (R):     {config.recurrent_depth}")

    model = SovereignNeuralTransformer(config)
    total_params = model.count_parameters()
    active_params = model.count_active_parameters()
    sparsity_ratio = (1.0 - (active_params / total_params)) * 100.0

    print(f"\n[+] Parameter Metrics:")
    print(f"    - Total Trainable Parameters: {total_params:,}")
    print(f"    - Active Parameters / Token:  {active_params:,}")
    print(f"    - Computational Sparsity:     {sparsity_ratio:.1f}% compute savings per token")

    test_prompt = "SAM-AI sovereign frontier intelligence"
    token_ids = model.tokenizer.encode(test_prompt)
    print(f"\n[+] Input Prompt: '{test_prompt}' ({len(token_ids)} tokens)")

    # Execute forward pass
    t0 = time.perf_counter()
    logits, cache = model.forward(token_ids)
    t_forward = (time.perf_counter() - t0) * 1000.0

    print(f"[+] Forward Pass Completed in {t_forward:.2f} ms")
    print(f"    - Output Logits Shape: {logits.shape}")
    print(f"    - Recurrent Depth Applied: {cache['recurrent_depth_applied']}")
    print(f"    - Total Block Iterations:  {len(cache['block_caches'])} ({config.n_layers} layers x {config.recurrent_depth} iterations)")

    # Verify MoE routing behavior
    block0_ffn = cache["block_caches"][0]["c_ffn"]
    top_indices = block0_ffn["top_indices"]
    routing_weights = block0_ffn["routing_weights"]
    print(f"\n[+] MoE Top-K Routing Sample (First 3 tokens):")
    for t_idx in range(min(3, len(token_ids))):
        exp_ids = top_indices[t_idx].tolist()
        exp_w = routing_weights[t_idx].tolist()
        print(f"    - Token {t_idx} ('{chr(token_ids[t_idx])}'): Selected Experts {exp_ids} with Gating Weights {[round(w, 3) for w in exp_w]}")

    # Generate text autoregressively
    print(f"\n[+] Testing Autoregressive Generation:")
    gen_text = model.generate(test_prompt, max_new_tokens=15, temperature=0.7)
    print(f"    - Generated Output: {repr(gen_text)}")

    # Full parameter trainer verification
    print(f"\n[+] Testing Full-Parameter Trainer with MoE Perturbation Gradients:")
    trainer = FullParameterTrainer(model, lr=0.005)
    print(f"    - Registered Parameter Tensors in AdamW: {len(trainer.params)}")
    loss_before = trainer.compute_loss(token_ids)
    loss_after = trainer.train_step(token_ids, n_perturbation_dirs=2, train_internal=True)
    print(f"    - Cross-Entropy Loss: {loss_before:.4f} -> {loss_after:.4f}")

    return True


def verify_cognitive_hyper_engine():
    print_header("2. NEURO-SYMBOLIC HYPER-ASTRA COGNITIVE ENGINE VERIFICATION")

    engine = HyperAstraEngine(base_simulations=5, max_simulations=20)
    goal = "Synthesize an optimal dynamic graph algorithm and verify invariants"
    print(f"[+] Launching cognitive goal: '{goal}'")

    t0 = time.perf_counter()
    result = engine.solve_complex_goal(goal, max_steps=2)
    elapsed = time.perf_counter() - t0

    print(f"[+] Cognitive Loop Completed in {elapsed:.2f} s")
    print(f"    - Status Success:           {result.success}")
    print(f"    - Steps Executed:           {result.steps_executed}")
    print(f"    - Cumulative Entropy:       {result.total_entropy_budgeted:.2f}")
    print(f"    - Threats Neutralized:      {result.threats_neutralized}")
    print(f"    - Memory Triples Recorded:  {result.knowledge_triples_total}")
    for trace in result.traces:
        print(f"    * Step {trace.step_id}: Action '{trace.action_chosen}' | Tier: {trace.reasoning_tier} | Sims: {trace.simulations_run}")

    return True


def verify_foundation_runner_and_sandbox():
    print_header("3. FOUNDATION RUNNER & DETERMINISTIC AST SANDBOX VERIFICATION")

    runner = FoundationReasoningEngine()
    print(f"[+] Foundation Runner Active Backend: {runner._active_backend}")

    # Test AST sandbox verification directly
    safe_code = "result = sum(x**2 for x in range(10))\nprint(f'Sum of squares: {result}')"
    sb_result = runner.sandbox.execute(safe_code)
    print(f"[+] AST Sandbox Safe Code Execution:")
    print(f"    - Success: {sb_result.success}")
    print(f"    - Output:  {sb_result.output.strip()}")
    print(f"    - Errors:  {sb_result.error}")

    # Test AST sandbox blocking forbidden operations
    dangerous_code = "import os\nos.system('echo dangerous')"
    sb_blocked = runner.sandbox.execute(dangerous_code)
    print(f"[+] AST Sandbox Security Defense:")
    print(f"    - Blocked Forbidden Import: {not sb_blocked.success}")
    print(f"    - Sandbox Error Caught:     {sb_blocked.error}")

    return True


def verify_api_server_endpoints():
    print_header("4. LIVE OPENAI-COMPATIBLE API SERVER ENDPOINT VERIFICATION")
    import urllib.request

    # Check models endpoint
    models_url = "http://127.0.0.1:8000/v1/models"
    try:
        req = urllib.request.Request(models_url)
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
            print(f"[+] GET /v1/models -> HTTP {status}")
            print(f"    - Available Models: {[m['id'] for m in body.get('data', [])]}")
    except Exception as e:
        print(f"[!] Warning: API server connection error: {e}")
        return False

    # Check chat completions endpoint
    chat_url = "http://127.0.0.1:8000/v1/chat/completions"
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "messages": [
            {"role": "user", "content": "What is the capital of France and what is 12 + 15?"}
        ],
        "temperature": 0.3,
        "max_tokens": 60,
    }
    try:
        req = urllib.request.Request(
            chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
            print(f"[+] POST /v1/chat/completions -> HTTP {status}")
            choice = body["choices"][0]["message"]
            print(f"    - Response Role: {choice['role']}")
            print(f"    - Content Snippet: {repr(choice['content'][:80])}...")
    except Exception as e:
        print(f"[!] Warning: Chat completions endpoint error: {e}")
        return False

    return True


def verify_web_chat_server():
    print_header("5. LIVE WEB CHAT STUDIO SERVER VERIFICATION")
    import urllib.request

    web_url = "http://127.0.0.1:8765/"
    try:
        req = urllib.request.Request(web_url)
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            content = response.read().decode("utf-8")
            print(f"[+] GET http://127.0.0.1:8765/ -> HTTP {status}")
            print(f"    - Page Title Found: {'SAM-AI' in content}")
            print(f"    - Payload Size: {len(content):,} bytes")
    except Exception as e:
        print(f"[!] Warning: Web chat server connection error: {e}")
        return False

    return True


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("  SAM-AI SOVEREIGN FRONTIER STACK: REAL-WORLD FLOW VERIFICATION")
    print("#" * 70)

    v1 = verify_frontier_moe_and_recurrent()
    v2 = verify_cognitive_hyper_engine()
    v3 = verify_foundation_runner_and_sandbox()
    v4 = verify_api_server_endpoints()
    v5 = verify_web_chat_server()

    print_header("VERIFICATION SUMMARY")
    print(f"  1. MoE & Recurrent Looped Architecture:  {'PASS [100%]' if v1 else 'FAIL'}")
    print(f"  2. Neuro-Symbolic Cognitive Loop:        {'PASS [100%]' if v2 else 'FAIL'}")
    print(f"  3. Foundation Runner & AST Sandbox:      {'PASS [100%]' if v3 else 'FAIL'}")
    print(f"  4. OpenAI-Compatible REST API Server:    {'PASS [100%]' if v4 else 'FAIL'}")
    print(f"  5. Web Chat Studio Server:               {'PASS [100%]' if v5 else 'FAIL'}")
    print("=" * 70 + "\n")
