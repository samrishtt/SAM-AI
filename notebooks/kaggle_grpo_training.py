"""
SAM-AI: GRPO Self-Training on Kaggle Free T4 GPU
=================================================
Upload this script to Kaggle as a notebook (or paste into cells).
Uses Qwen2.5-0.5B-Instruct with Unsloth 4-bit QLoRA + TRL GRPOTrainer.

HOW TO USE:
1. Go to https://www.kaggle.com/ and sign in (free account)
2. Create a new Notebook
3. Settings -> Accelerator -> GPU T4 x2 (free 30hrs/week)
4. Paste this entire script into a code cell
5. Run it - training takes ~2-4 hours
6. Download the trained model from the output

The model trains itself to be smarter at math reasoning using GRPO
(the same algorithm DeepSeek used to create R1).
"""

# ==============================================================================
# CELL 1: Install Dependencies
# ==============================================================================
# !pip install -q unsloth "trl>=0.15.0" vllm datasets transformers accelerate

# Uncomment the line above and run it first, then run the rest.

import os
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================================================================
# CELL 2: Configuration
# ==============================================================================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Small enough for free T4
OUTPUT_DIR = "/kaggle/working/sam-ai-grpo-trained"
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 4
GROUP_SIZE = 8  # Number of rollouts per prompt (GRPO G parameter)
MAX_NEW_TOKENS = 256
LEARNING_RATE = 5e-6

# ==============================================================================
# CELL 3: Generate Training Dataset (Self-Generated Math Problems)
# ==============================================================================
def generate_math_dataset(n_samples=2000):
    """Generate verifiable math problems with ground truth answers."""
    import random
    random.seed(42)
    
    problems = []
    
    # Single digit addition
    for _ in range(n_samples // 4):
        a, b = random.randint(0, 99), random.randint(0, 99)
        problems.append({
            "prompt": f"What is {a} + {b}? Think step by step, then give the final answer.",
            "answer": str(a + b),
            "type": "addition"
        })
    
    # Multiplication
    for _ in range(n_samples // 4):
        a, b = random.randint(2, 20), random.randint(2, 20)
        problems.append({
            "prompt": f"What is {a} * {b}? Think step by step, then give the final answer.",
            "answer": str(a * b),
            "type": "multiplication"
        })
    
    # Subtraction
    for _ in range(n_samples // 4):
        a = random.randint(10, 200)
        b = random.randint(1, a)
        problems.append({
            "prompt": f"What is {a} - {b}? Think step by step, then give the final answer.",
            "answer": str(a - b),
            "type": "subtraction"
        })
    
    # Word problems
    for _ in range(n_samples // 4):
        a, b = random.randint(1, 50), random.randint(1, 50)
        problems.append({
            "prompt": f"If you have {a} apples and buy {b} more, how many apples do you have? Think step by step.",
            "answer": str(a + b),
            "type": "word_problem"
        })
    
    return problems


def generate_code_dataset(n_samples=500):
    """Generate verifiable coding problems."""
    import random
    random.seed(123)
    
    templates = [
        {
            "prompt": "Write a Python function called `factorial` that computes n!. Include a docstring.",
            "test": "assert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(1) == 1",
        },
        {
            "prompt": "Write a Python function called `fibonacci` that returns the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1).",
            "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55",
        },
        {
            "prompt": "Write a Python function called `is_palindrome` that checks if a string is a palindrome. Return True or False.",
            "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('abba') == True",
        },
        {
            "prompt": "Write a Python function called `reverse_string` that reverses a string.",
            "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''",
        },
        {
            "prompt": "Write a Python function called `sum_list` that returns the sum of all numbers in a list.",
            "test": "assert sum_list([1,2,3]) == 6\nassert sum_list([]) == 0\nassert sum_list([10]) == 10",
        },
        {
            "prompt": "Write a Python function called `max_element` that returns the maximum element in a list.",
            "test": "assert max_element([1,5,3,9,2]) == 9\nassert max_element([42]) == 42",
        },
        {
            "prompt": "Write a Python function called `count_vowels` that counts the number of vowels in a string.",
            "test": "assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0\nassert count_vowels('aeiou') == 5",
        },
        {
            "prompt": "Write a Python function called `is_prime` that checks if a number is prime. Return True or False.",
            "test": "assert is_prime(2) == True\nassert is_prime(4) == False\nassert is_prime(17) == True\nassert is_prime(1) == False",
        },
    ]
    
    problems = []
    for _ in range(n_samples):
        t = random.choice(templates)
        problems.append({
            "prompt": t["prompt"],
            "test_code": t["test"],
            "type": "code"
        })
    
    return problems


# ==============================================================================
# CELL 4: Reward Functions (Deterministic Verifiers)
# ==============================================================================
def math_reward_fn(completions, answer, **kwargs):
    """Reward function that checks if the model's answer matches the expected answer."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        # Extract numbers from the response
        numbers = re.findall(r'\b\d+\b', text)
        if answer in numbers:
            # Give higher reward if the answer appears near the end (final answer)
            last_pos = text.rfind(answer)
            if last_pos > len(text) * 0.5:
                rewards.append(1.0)  # Answer at the end = good reasoning
            else:
                rewards.append(0.5)  # Answer found but maybe not as final answer
        else:
            rewards.append(0.0)
    return rewards


def code_reward_fn(completions, test_code, **kwargs):
    """Reward function that executes generated code and runs test assertions."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        
        # Extract code block
        code_match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = text  # Try raw text as code
        
        try:
            exec_globals = {}
            exec(code + "\n" + test_code, exec_globals)
            rewards.append(1.0)  # All assertions passed
        except AssertionError:
            rewards.append(0.2)  # Code ran but tests failed
        except Exception:
            rewards.append(0.0)  # Code didn't run
    
    return rewards


# ==============================================================================
# CELL 5: Setup Model & Trainer
# ==============================================================================
def setup_training():
    """Initialize model and GRPO trainer."""
    
    try:
        from unsloth import FastLanguageModel
        print("[*] Using Unsloth for 4-bit QLoRA (2x faster, 60% less memory)")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=512,
            load_in_4bit=True,
            dtype=None,  # Auto-detect
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
        )
        
    except ImportError:
        print("[*] Unsloth not available, using standard HF + PEFT")
        from peft import LoraConfig, get_peft_model
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# ==============================================================================
# CELL 6: Training Loop
# ==============================================================================
def train():
    """Main GRPO training loop."""
    
    print("=" * 72)
    print("  SAM-AI: GRPO SELF-TRAINING ON KAGGLE GPU")
    print("  Algorithm: DeepSeek-R1 Group Relative Policy Optimization")
    print("  Model: Qwen2.5-0.5B-Instruct + 4-bit QLoRA")
    print("=" * 72)
    
    # Generate datasets
    print("\n[1/4] Generating self-training dataset...")
    math_problems = generate_math_dataset(2000)
    code_problems = generate_code_dataset(500)
    
    print(f"  Math problems: {len(math_problems)}")
    print(f"  Code problems: {len(code_problems)}")
    
    # Format for TRL GRPOTrainer
    math_data = Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": "You are SAM-AI, a sovereign reasoning engine. Think step by step and provide clear answers."},
                {"role": "user", "content": p["prompt"]}
            ],
            "answer": p["answer"]
        }
        for p in math_problems
    ])
    
    print("\n[2/4] Loading model...")
    model, tokenizer = setup_training()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Setup GRPO Trainer
    print("\n[3/4] Configuring GRPO trainer...")
    
    try:
        from trl import GRPOConfig, GRPOTrainer
        
        training_args = GRPOConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            num_generations=GROUP_SIZE,
            max_completion_length=MAX_NEW_TOKENS,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            report_to="none",
        )
        
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=math_data,
            reward_funcs=math_reward_fn,
            tokenizer=tokenizer,
        )
        
        print("\n[4/4] Starting GRPO self-training...")
        print("  This will take 2-4 hours on a T4 GPU.")
        print("  The model is training itself to reason better at math.\n")
        
        trainer.train()
        
        # Save final model
        print("\n[*] Saving trained model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"\n{'='*72}")
        print(f"  TRAINING COMPLETE!")
        print(f"  Model saved to: {OUTPUT_DIR}")
        print(f"  Download the model files and use with SAM-AI locally.")
        print(f"{'='*72}")
        
    except ImportError as e:
        print(f"\n[!] TRL not installed: {e}")
        print("    Run: pip install trl>=0.15.0")
        
        # Fallback: manual GRPO loop
        print("\n[*] Running manual GRPO training loop instead...")
        manual_grpo_training(model, tokenizer, math_problems)


def manual_grpo_training(model, tokenizer, problems):
    """Fallback manual GRPO training without TRL library."""
    import random
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    
    model.train()
    step = 0
    total_reward = 0
    
    for epoch in range(NUM_TRAIN_EPOCHS):
        random.shuffle(problems)
        epoch_rewards = []
        
        for i, problem in enumerate(problems):
            prompt = problem["prompt"]
            expected = problem["answer"]
            
            # Generate group of rollouts
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
            
            group_texts = []
            group_rewards = []
            
            for _ in range(GROUP_SIZE):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        temperature=0.9,
                        do_sample=True,
                        top_k=50,
                    )
                generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Reward
                numbers = re.findall(r'\b\d+\b', generated)
                reward = 1.0 if expected in numbers else 0.0
                
                group_texts.append(generated)
                group_rewards.append(reward)
            
            # GRPO: compute advantages
            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = max((sum((r-mean_r)**2 for r in group_rewards)/len(group_rewards))**0.5, 1e-6)
            advantages = [(r - mean_r) / std_r for r in group_rewards]
            
            # Train on best trajectory
            best_idx = group_rewards.index(max(group_rewards))
            if advantages[best_idx] > 0:
                best_text = prompt + " " + group_texts[best_idx]
                train_inputs = tokenizer(best_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
                outputs = model(**train_inputs, labels=train_inputs.input_ids)
                loss = outputs.loss * advantages[best_idx]
                loss.backward()
                
                if (step + 1) % 4 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            epoch_rewards.append(mean_r)
            step += 1
            
            if step % 50 == 0:
                avg = sum(epoch_rewards[-50:]) / min(50, len(epoch_rewards))
                print(f"  Step {step} | Epoch {epoch+1} | Avg Reward: {avg:.3f}")
        
        epoch_avg = sum(epoch_rewards) / len(epoch_rewards)
        print(f"\n  === Epoch {epoch+1} Complete | Mean Reward: {epoch_avg:.3f} ===\n")
    
    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n  Model saved to: {OUTPUT_DIR}")


# ==============================================================================
# CELL 7: Run Training
# ==============================================================================
if __name__ == "__main__":
    train()
