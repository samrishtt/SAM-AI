"""SAM-AI v2: Autonomous Self-Improvement with Curriculum Learning.

This is the REAL self-training pipeline that actually makes the model smarter.

What's different from v1:
1. FULL-PARAMETER TRAINING: Updates ALL 119K params (attention, FFN, norms, embeddings)
   not just the lm_head output projection.
2. PRE-TRAINING PHASE: First teaches the model basic byte patterns via next-token
   prediction on a math corpus, so it can produce coherent output.
3. CURRICULUM LEARNING: 4 stages from trivial -> hard:
   Stage 1: Pattern copying (echo "aaa" -> "aaa")
   Stage 2: Sequence completion ("abc" -> "d")  
   Stage 3: Single-digit arithmetic ("2+3=" -> "5")
   Stage 4: Multi-digit arithmetic ("15+27=" -> "42")
4. HUNDREDS of procedurally generated tasks (not just 5 hardcoded ones)
5. MEASURABLE REWARD IMPROVEMENT over training epochs
"""

from __future__ import annotations
import os
import sys
import time
import random
from typing import Callable, Dict, List, Tuple
import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.neural.transformer import SovereignNeuralTransformer, TransformerConfig
from core.neural.grpo_trainer import GRPOTrainer, GRPOSummary
from core.neural.full_trainer import FullParameterTrainer


# =============================================================================
# 1. Math Pre-Training Corpus Generator
# =============================================================================
def generate_pretrain_corpus(n_equations: int = 2000) -> str:
    """Generate a large corpus of simple math equations for pre-training.
    
    The model needs to see patterns like "2+3=5" thousands of times before
    it can learn to produce them. This is the equivalent of GPT's pre-training
    on internet text, but focused on math patterns.
    """
    random.seed(42)
    lines = []
    
    # Single digit addition (most important - need LOTS of these)
    for _ in range(n_equations // 3):
        a, b = random.randint(0, 9), random.randint(0, 9)
        lines.append(f"{a}+{b}={a+b}")
    
    # Single digit subtraction
    for _ in range(n_equations // 6):
        a = random.randint(0, 9)
        b = random.randint(0, a)
        lines.append(f"{a}-{b}={a-b}")
    
    # Single digit multiplication
    for _ in range(n_equations // 6):
        a, b = random.randint(0, 9), random.randint(0, 9)
        lines.append(f"{a}*{b}={a*b}")
    
    # Two digit addition
    for _ in range(n_equations // 6):
        a, b = random.randint(10, 50), random.randint(10, 50)
        lines.append(f"{a}+{b}={a+b}")
    
    # Sequential patterns
    for _ in range(n_equations // 6):
        start = random.randint(0, 20)
        seq = ",".join(str(start + i) for i in range(5))
        lines.append(seq)
    
    # Repeat patterns
    for ch in "0123456789abcdef":
        lines.append(ch * 8)
    
    random.shuffle(lines)
    corpus = "\n".join(lines) + "\n"
    return corpus


# =============================================================================
# 2. Curriculum Task Generators (Procedural, Unlimited Tasks)
# =============================================================================
class VerifiableTask:
    def __init__(self, prompt: str, verifier: Callable[[str], float], description: str):
        self.prompt = prompt
        self.verifier = verifier
        self.description = description


def generate_echo_tasks(n: int = 30) -> List[VerifiableTask]:
    """Stage 1: Echo/Copy tasks. Simplest possible - just reproduce the pattern."""
    tasks = []
    patterns = ["000", "111", "222", "aaa", "bbb", "abc", "123", "012"]
    
    for _ in range(n):
        p = random.choice(patterns)
        
        def make_verifier(expected):
            def v(text):
                # Check what fraction of expected chars appear in output
                if expected in text:
                    return 1.0
                matches = sum(1 for c in expected if c in text)
                return max(0.05, matches / len(expected) * 0.5)
            return v
        
        tasks.append(VerifiableTask(
            prompt=p,
            verifier=make_verifier(p),
            description=f"Echo '{p}'"
        ))
    
    return tasks


def generate_single_digit_math_tasks(n: int = 100) -> List[VerifiableTask]:
    """Stage 3: Single digit addition. 0+0=0 through 9+9=18."""
    tasks = []
    
    for _ in range(n):
        a, b = random.randint(0, 9), random.randint(0, 9)
        answer = str(a + b)
        prompt = f"{a}+{b}="
        
        def make_verifier(expected):
            def v(text):
                # Check if the answer appears in the output
                if expected in text:
                    return 1.0
                # Partial credit: any digit from answer found
                for ch in expected:
                    if ch in text:
                        return 0.3
                return 0.05
            return v
        
        tasks.append(VerifiableTask(
            prompt=prompt,
            verifier=make_verifier(answer),
            description=f"{a}+{b}={a+b}"
        ))
    
    return tasks


def generate_multi_digit_math_tasks(n: int = 50) -> List[VerifiableTask]:
    """Stage 4: Multi-digit addition. 10+10=20 through 50+50=100."""
    tasks = []
    
    for _ in range(n):
        a, b = random.randint(10, 50), random.randint(10, 50)
        answer = str(a + b)
        prompt = f"{a}+{b}="
        
        def make_verifier(expected):
            def v(text):
                if expected in text:
                    return 1.0
                # Partial: first digit correct
                if len(expected) > 0 and len(text) > 0 and expected[0] in text[:3]:
                    return 0.3
                return 0.05
            return v
        
        tasks.append(VerifiableTask(
            prompt=prompt,
            verifier=make_verifier(answer),
            description=f"{a}+{b}={a+b}"
        ))
    
    return tasks


def generate_sequence_tasks(n: int = 30) -> List[VerifiableTask]:
    """Stage 2: Sequence completion. Given '1,2,3,' predict '4'."""
    tasks = []
    
    for _ in range(n):
        start = random.randint(0, 6)
        seq = ",".join(str(start + i) for i in range(3))
        next_val = str(start + 3)
        prompt = seq + ","
        
        def make_verifier(expected):
            def v(text):
                if expected in text:
                    return 1.0
                return 0.05
            return v
        
        tasks.append(VerifiableTask(
            prompt=prompt,
            verifier=make_verifier(next_val),
            description=f"{seq},? -> {next_val}"
        ))
    
    return tasks


# =============================================================================
# 3. Main Self-Training Pipeline
# =============================================================================
def run_enhanced_self_improvement(
    pretrain_epochs: int = 8,
    grpo_epochs_per_stage: int = 10,
    group_size: int = 6,
):
    print("=" * 72)
    print("  SAM-AI v2: AUTONOMOUS SELF-IMPROVEMENT ENGINE")
    print("  Full-Parameter Training + Curriculum Learning + GRPO")
    print("=" * 72)
    
    # -------------------------------------------------------------------------
    # Model Setup
    # -------------------------------------------------------------------------
    config = TransformerConfig(
        vocab_size=128,
        seq_len=64,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        learning_rate=0.003,  # Slightly higher LR for small model
    )
    model = SovereignNeuralTransformer(config=config)
    full_trainer = FullParameterTrainer(model, lr=0.003, weight_decay=0.005)
    grpo_trainer = GRPOTrainer(model=model, group_size=group_size, clip_epsilon=0.2)
    
    print(f"\n  Model Parameters: {model.count_parameters():,}")
    print(f"  Architecture: {config.n_layers}L-{config.d_model}D-{config.n_heads}H")
    print(f"  Vocabulary: {config.vocab_size} (byte-level)")
    
    all_history = []
    
    # =========================================================================
    # PHASE 1: PRE-TRAINING (Next-Token Prediction on Math Corpus)
    # =========================================================================
    print("\n" + "=" * 72)
    print("  PHASE 1: PRE-TRAINING (Learning byte patterns from math corpus)")
    print("=" * 72)
    
    # Sample from model BEFORE training
    sample_before = model.generate("2+3=", max_new_tokens=5, temperature=0.8)
    print(f"\n  Sample BEFORE training: '2+3=' -> '{sample_before}'")
    
    corpus = generate_pretrain_corpus(2000)
    print(f"  Corpus size: {len(corpus):,} characters ({corpus.count(chr(10)):,} equations)")
    print(f"  Pre-training for {pretrain_epochs} epochs...\n")
    
    t0 = time.time()
    pretrain_losses = full_trainer.pretrain_on_corpus(
        corpus=corpus,
        epochs=pretrain_epochs,
        chunk_size=40,
        log_every=100,
        train_internal_every=3,
    )
    pretrain_time = time.time() - t0
    
    # Sample AFTER pre-training
    sample_after = model.generate("2+3=", max_new_tokens=5, temperature=0.5)
    print(f"\n  Sample AFTER pre-training: '2+3=' -> '{sample_after}'")
    print(f"  Pre-training time: {pretrain_time:.1f}s")
    print(f"  Loss: {pretrain_losses[0]:.4f} (start) -> {pretrain_losses[-1]:.4f} (end)")
    
    if len(pretrain_losses) > 20:
        improvement = pretrain_losses[0] - np.mean(pretrain_losses[-20:])
        print(f"  Loss improvement: {improvement:.4f}")
    
    # =========================================================================
    # PHASE 2: GRPO SELF-IMPROVEMENT (Curriculum Stages)
    # =========================================================================
    print("\n" + "=" * 72)
    print("  PHASE 2: GRPO SELF-IMPROVEMENT (Curriculum Learning)")
    print("=" * 72)
    
    stages = [
        ("Stage 1: Echo/Copy", generate_echo_tasks(20)),
        ("Stage 2: Sequences", generate_sequence_tasks(20)),
        ("Stage 3: Single-Digit Math", generate_single_digit_math_tasks(60)),
        ("Stage 4: Multi-Digit Math", generate_multi_digit_math_tasks(30)),
    ]
    
    for stage_idx, (stage_name, tasks) in enumerate(stages, 1):
        print(f"\n  --- {stage_name} ({len(tasks)} tasks, {grpo_epochs_per_stage} epochs) ---")
        
        stage_rewards = []
        
        for epoch in range(1, grpo_epochs_per_stage + 1):
            epoch_rewards = []
            random.shuffle(tasks)
            
            # Sample a batch of tasks each epoch (not all, to keep it fast)
            batch = tasks[:min(15, len(tasks))]
            
            for task in batch:
                summary: GRPOSummary = grpo_trainer.train_grpo_step(
                    prompt=task.prompt,
                    reward_verifier=task.verifier,
                    max_tokens=10,
                )
                epoch_rewards.append(summary.mean_reward)
                
                # Also do a full-param train step on the best output
                if summary.best_reward > 0.2:
                    full_trainer.train_step_full(
                        task.prompt + summary.best_output,
                        train_internal=(epoch % 3 == 0),
                    )
            
            avg_reward = np.mean(epoch_rewards)
            stage_rewards.append(avg_reward)
            
            solved = sum(1 for r in epoch_rewards if r >= 0.9)
            total = len(epoch_rewards)
            
            print(f"    Epoch {epoch:2d}/{grpo_epochs_per_stage} | "
                  f"Reward: {avg_reward:.3f} | "
                  f"Solved: {solved}/{total} | "
                  f"Best: {max(epoch_rewards):.2f}")
        
        # Stage summary
        if len(stage_rewards) >= 2:
            delta = stage_rewards[-1] - stage_rewards[0]
            direction = "+" if delta > 0 else ""
            print(f"  {stage_name} complete: {stage_rewards[0]:.3f} -> {stage_rewards[-1]:.3f} ({direction}{delta:.3f})")
        
        all_history.append({
            "stage": stage_name,
            "rewards": stage_rewards,
        })
    
    # =========================================================================
    # PHASE 3: SAVE & REPORT
    # =========================================================================
    print("\n" + "=" * 72)
    print("  PHASE 3: SAVING CHECKPOINT")
    print("=" * 72)
    
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "sam_ai_v2_trained.npz")
    
    # Save ALL model weights
    save_dict = {
        "tok_embeddings": model.tok_embeddings,
        "pos_embeddings": model.pos_embeddings,
        "lm_head": model.lm_head,
        "final_norm_gamma": model.final_norm.gamma,
    }
    for i, block in enumerate(model.blocks):
        save_dict[f"block{i}_W_q"] = block.attn.W_q
        save_dict[f"block{i}_W_k"] = block.attn.W_k
        save_dict[f"block{i}_W_v"] = block.attn.W_v
        save_dict[f"block{i}_W_o"] = block.attn.W_o
        save_dict[f"block{i}_W1"] = block.ffn.W1
        save_dict[f"block{i}_b1"] = block.ffn.b1
        save_dict[f"block{i}_W2"] = block.ffn.W2
        save_dict[f"block{i}_b2"] = block.ffn.b2
        save_dict[f"block{i}_norm1_gamma"] = block.norm1.gamma
        save_dict[f"block{i}_norm2_gamma"] = block.norm2.gamma
    
    np.savez_compressed(ckpt_path, **save_dict)
    
    print(f"  Checkpoint saved: {ckpt_path}")
    print(f"  File size: {os.path.getsize(ckpt_path) / 1024:.1f} KB")
    
    # Final demo
    print("\n  --- FINAL MODEL SAMPLES ---")
    test_prompts = ["2+3=", "5+4=", "1+1=", "7+8=", "0+0=", "3+3="]
    for p in test_prompts:
        output = model.generate(p, max_new_tokens=5, temperature=0.3)
        # Extract just the generated part
        generated = output[len(p):]
        print(f"    '{p}' -> '{generated.strip()}'")
    
    total_time = time.time() - t0
    print(f"\n  Total training time: {total_time:.1f}s")
    print("=" * 72)
    print("  SAM-AI v2 SELF-IMPROVEMENT COMPLETE!")
    print("=" * 72)
    
    return all_history


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    pretrain_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    grpo_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    run_enhanced_self_improvement(
        pretrain_epochs=pretrain_ep,
        grpo_epochs_per_stage=grpo_ep,
        group_size=6,
    )
