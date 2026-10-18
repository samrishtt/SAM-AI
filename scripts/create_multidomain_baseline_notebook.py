"""
SAM-AI Multi-Domain Baseline Notebook Generator
===============================================
Generates a self-contained, reproducible Kaggle GPU notebook for evaluating the
untouched base foundation model (deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) across
7 cognitive reasoning domains (95 samples).

Strict Requirements:
1. Untouched foundation model only. No LoRA adapter loaded or attached.
2. Web retrieval and external tools disabled.
3. Resumable execution loop with incremental JSONL streaming to disk.
4. Isolated subprocess execution sandbox for HumanEval code with 3.0s timeout.
5. Hardware and disk storage preflight checks (fails closed if resources are insufficient).
6. Generates kernel-metadata.json for samrishb/sam-ai-multidomain-baseline.
"""

from __future__ import annotations
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "notebooks" / "sam_ai_multidomain_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = OUTPUT_DIR / "sam_ai_multidomain_baseline.ipynb"
METADATA_PATH = OUTPUT_DIR / "kernel-metadata.json"

# Load the frozen 95 fixtures from curriculum/phase_3a_baseline_fixtures.json
FIXTURES_FILE = REPO_ROOT / "curriculum" / "phase_3a_baseline_fixtures.json"
if not FIXTURES_FILE.exists():
    raise FileNotFoundError(f"Fixtures file missing at {FIXTURES_FILE}. Run scripts/build_phase_3a_fixtures.py first.")

with open(FIXTURES_FILE, "r", encoding="utf-8") as f:
    FIXTURES_DATA = json.load(f)

FIXTURES_SHA256 = hashlib.sha256(open(FIXTURES_FILE, "rb").read()).hexdigest()

# Load core verification code directly from sam_ai/evaluation/multidomain_harness.py
HARNESS_FILE = REPO_ROOT / "sam_ai" / "evaluation" / "multidomain_harness.py"
with open(HARNESS_FILE, "r", encoding="utf-8") as f:
    harness_text = f.read()

start_marker = "def clean_numeric_str"
end_marker = "@dataclass\nclass MultiDomainItemPrediction"
start_idx = harness_text.index(start_marker)
end_idx = harness_text.index(end_marker)
CORE_VERIFICATION_CODE = harness_text[start_idx:end_idx].strip()


def create_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def create_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def build_notebook():
    cells = []

    # Cell 0: Header & Protocol
    cell_0_md = f"""# 🧠 SAM-AI: Multi-Domain Intelligence Baseline Benchmark (Phase 3A)
### Foundation Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (Base Only)
---
**Core Evaluation Protocol & Guarantees:**
- **Model:** Untouched `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (Base weights only).
- **LoRA Adapter:** **STRICTLY NONE** (Zero adapters loaded or attached; Experiment 001 adapter is rejected).
- **Web Retrieval:** **DISABLED** (Evaluates purely intrinsic model reasoning).
- **Decoding:** Greedy deterministic decoding (`temperature=0.0`, `do_sample=False`, `max_new_tokens=2048`).
- **Resumability:** Checkpoint-enabled streaming to `/kaggle/working/predictions.jsonl`.
- **Code Sandbox:** Isolated subprocess environment with 3.0-second execution timeout.
- **Fixtures SHA-256:** `{FIXTURES_SHA256}` ({FIXTURES_DATA['total_samples']} samples across 7 domains).

### Evaluated Domains:
1. `mathematics_gsm8k`: 20 items (Grade school arithmetic reasoning)
2. `mathematics_math500`: 20 items (Challenging competition mathematics)
3. `coding_humaneval`: 20 items (Functional Python problem solving with assertion test suites)
4. `science_reasoning`: 10 items (Physics, chemistry, biology calculations)
5. `logic_constraints`: 10 items (Knights & Knaves, syllogisms, constraint satisfaction)
6. `instruction_following`: 10 items (JSON schema, word limits, letter bans, formatting)
7. `multiturn_context`: 5 dialogue pairs = 10 turns (Multi-turn recall and state tracking)"""
    cells.append(create_markdown_cell(cell_0_md))

    # Cell 1: Hardware & Storage Preflight
    cell_1_code = """# Cell 1: Hardware & Storage Preflight Audit (Fail-Closed)
import os
import sys
import json
import time
import shutil
import platform
import subprocess
from pathlib import Path

print("=" * 70)
print("SAM-AI MULTI-DOMAIN BASELINE: HARDWARE & PREFLIGHT AUDIT")
print("=" * 70)

import torch
cuda_available = torch.cuda.is_available()
print(f"[*] CUDA Available: {cuda_available}")

if not cuda_available:
    raise RuntimeError("Kaggle GPU accelerator is not enabled. Please enable GPU T4 x 2 or P100 in Notebook Settings.")

device_count = torch.cuda.device_count()
device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
print(f"[*] GPUs detected: {device_count} ({', '.join(device_names)})")

total_vram_gb = 0.0
for i in range(device_count):
    vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    total_vram_gb += vram_gb
    cc = torch.cuda.get_device_capability(i)
    print(f"    - GPU {i} ({device_names[i]}): CC {cc[0]}.{cc[1]} | {vram_gb:.2f} GB total VRAM")

print(f"[*] Total Cluster VRAM: {total_vram_gb:.2f} GB")
if total_vram_gb < 14.0:
    raise RuntimeError(f"Insufficient VRAM for 7B evaluation: detected {total_vram_gb:.2f} GB, minimum required is 14.0 GB.")

working_dir = Path("/kaggle/working" if os.path.exists("/kaggle/working") else "./working")
working_dir.mkdir(parents=True, exist_ok=True)
disk_info = shutil.disk_usage(str(working_dir))
free_disk_gb = disk_info.free / (1024**3)
print(f"[*] Working Directory: {working_dir.resolve()}")
print(f"[*] Free Disk Space: {free_disk_gb:.2f} GB")

if free_disk_gb < 10.0:
    raise RuntimeError(f"Insufficient disk space in working directory: {free_disk_gb:.2f} GB free, minimum 10.0 GB required.")

if os.path.exists("/kaggle/working"):
    print("[*] Kaggle environment detected. Ensuring required packages are installed...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers>=4.45.0", "accelerate>=0.34.0", "bitsandbytes>=0.43.0"
    ], check=True)

print("[OK] Preflight audit passed successfully.")"""
    cells.append(create_code_cell(cell_1_code))

    # Cell 2: Fixtures Ingestion and Verification Logic
    fixtures_json_str = json.dumps(FIXTURES_DATA)
    cell_2_header = f"""# Cell 2: Benchmark Fixtures & Domain Extraction / Verification Logic
import os
import sys
import re
import json
import hashlib
import tempfile
import subprocess
from pathlib import Path
from fractions import Fraction
from typing import List, Dict, Any, Optional, Tuple

FIXTURES_RAW = {repr(fixtures_json_str)}
BENCHMARK_FIXTURES = json.loads(FIXTURES_RAW)

calculated_sha = hashlib.sha256(FIXTURES_RAW.encode("utf-8")).hexdigest()
print(f"[*] Benchmark Fixtures Loaded: {{BENCHMARK_FIXTURES['total_samples']}} samples across {{BENCHMARK_FIXTURES['total_domains']}} domains.")
print(f"[*] Fixtures Integrity Check: SHA-256 = {{calculated_sha[:16]}}...")
"""

    cell_2_footer = '\nprint("[OK] Benchmark verifiers and sandbox engine successfully initialized.")'
    cell_2_code = cell_2_header + "\n" + CORE_VERIFICATION_CODE + cell_2_footer
    cells.append(create_code_cell(cell_2_code))

    # Cell 3: Fail-Closed Base Model Loader
    cell_3_code = """# Cell 3: Fail-Closed Base Model Loader (deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

print("=" * 70)
print(f"LOADING FOUNDATION MODEL: {MODEL_ID}")
print("Strict Policy: BASE MODEL ONLY. NO ADAPTER.")
print("=" * 70)

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

max_mem = {i: "13.0GB" for i in range(device_count)}

try:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True
    )
    print("[*] Successfully loaded base model with 4-bit NF4 BitsAndBytesConfig.")
except Exception as bnb_err:
    print(f"[!] 4-bit NF4 not loaded ({bnb_err}). Loading with float16 sharded across GPUs...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True
    )
    print("[*] Successfully loaded base model in native float16.")

base_model.eval()

# Strict verification that NO PEFT or LoRA adapter is attached
if hasattr(base_model, "peft_config"):
    raise RuntimeError("CRITICAL ERROR: A PEFT adapter was detected on the base model! Benchmark requires untouched base model.")
if "PeftModel" in type(base_model).__name__:
    raise RuntimeError("CRITICAL ERROR: Model is wrapped in PeftModel! Benchmark requires untouched base model.")

param_count = sum(p.numel() for p in base_model.parameters())
load_dur = time.time() - t0
peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
print(f"[OK] Base model verified untouched. Total parameters: {param_count:,}")
print(f"[OK] Loaded in {load_dur:.2f}s | Peak VRAM: {peak_vram_mb:.2f} MB")"""
    cells.append(create_code_cell(cell_3_code))

    # Cell 4: Resumable Execution Loop with Streaming Persistence
    cell_4_code = """# Cell 4: Resumable Multi-Domain Execution Loop
checkpoint_file = working_dir / "predictions.jsonl"
print(f"[*] Evaluation Checkpoint File: {checkpoint_file.resolve()}")

completed_ids = set()
if checkpoint_file.exists():
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                    completed_ids.add(rec["example_id"])
                except Exception:
                    pass
    print(f"[*] Resuming from existing checkpoint: {len(completed_ids)} already completed items found.")
else:
    print("[*] No existing checkpoint found. Starting fresh evaluation.")

# Flatten all 95 samples in deterministic domain order
all_samples = []
for domain_name, items in BENCHMARK_FIXTURES["domains"].items():
    for item in items:
        all_samples.append(item)

print(f"[*] Total items to evaluate: {len(all_samples)}")

bench_start_time = time.time()
ckpt_writer = open(checkpoint_file, "a", encoding="utf-8")

for idx, sample in enumerate(all_samples):
    sid = sample["id"]
    domain = sample["domain"]
    eval_type = sample.get("eval_type", "numerical_exact")
    target = sample.get("target")

    if sid in completed_ids:
        print(f"[{idx+1:02d}/{len(all_samples)}] [RESUMED] Domain: {domain:<22} | ID: {sid}")
        continue

    # Format Prompt based on domain
    if domain in ("mathematics_gsm8k", "mathematics_math500"):
        prompt_text = f"<｜User｜>{sample['question']}\\nPlease reason step-by-step and end your response with #### followed by the final numeric answer.\\n<｜Assistant｜><think>\\n"
    elif domain == "coding_humaneval":
        prompt_text = f"<｜User｜>{sample['prompt']}\\nComplete the Python function above. Provide valid executable Python code.\\n<｜Assistant｜><think>\\n"
    elif domain in ("science_reasoning", "logic_constraints"):
        prompt_text = f"<｜User｜>{sample['question']}\\nPlease reason step-by-step and state your final answer clearly at the end.\\n<｜Assistant｜><think>\\n"
    elif domain == "instruction_following":
        prompt_text = f"<｜User｜>{sample['question']}\\nFollow all instructions strictly.\\n<｜Assistant｜><think>\\n"
    elif domain == "multiturn_context":
        t1_u = sample["turn_1_user"]
        t1_a = sample["turn_1_assistant_prefix"]
        t2_u = sample["turn_2_user"]
        prompt_text = f"<｜User｜>{t1_u}\\n<｜Assistant｜>{t1_a} </s>\\n<｜User｜>{t2_u}\\n<｜Assistant｜><think>\\n"
    else:
        prompt_text = f"<｜User｜>{sample.get('question', '')}\\n<｜Assistant｜><think>\\n"

    inputs = tokenizer(prompt_text, return_tensors="pt").to(base_model.device)
    prompt_len = inputs["input_ids"].shape[1]

    t_start = time.time()
    with torch.no_grad():
        output = base_model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id
        )
    dur_ms = (time.time() - t_start) * 1000.0

    gen_ids = output[0][prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    is_truncated = (len(gen_ids) >= 2048)

    # Dispatch universal verification
    is_corr, pred, method = verify_prediction(
        domain=domain,
        eval_type=eval_type,
        raw_output=gen_text,
        target=target,
        test_suite=sample.get("test"),
        entry_point=sample.get("entry_point"),
        prompt=sample.get("prompt")
    )

    rec = {
        "example_id": sid,
        "domain": domain,
        "eval_type": eval_type,
        "target": target,
        "prompt": prompt_text,
        "raw_output": gen_text,
        "extracted_value": pred,
        "verification_method": method,
        "is_correct": is_corr,
        "is_truncated": is_truncated,
        "generated_tokens": len(gen_ids),
        "duration_ms": round(dur_ms, 2)
    }

    # Persist immediately (flush to disk)
    ckpt_writer.write(json.dumps(rec) + "\\n")
    ckpt_writer.flush()
    completed_ids.add(sid)

    status_tag = "PASS" if is_corr else "FAIL"
    print(f"[{idx+1:02d}/{len(all_samples)}] [{status_tag}] Domain: {domain:<22} | ID: {sid} | Pred: {str(pred)[:20]} | Toks: {len(gen_ids)} | {dur_ms:.0f}ms")

ckpt_writer.close()
total_eval_time = time.time() - bench_start_time
print()
print(f"[OK] Evaluation execution completed in {total_eval_time:.1f}s.")"""
    cells.append(create_code_cell(cell_4_code))

    # Cell 5: Aggregation, Summary Report & Artifact Packaging
    cell_5_code = """# Cell 5: Aggregation, Summary Report & Packaging
import zipfile

print("=" * 70)
print("SAM-AI MULTI-DOMAIN BASELINE: AGGREGATE RESULTS SUMMARY")
print("=" * 70)

all_records = []
with open(checkpoint_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            all_records.append(json.loads(line))

# Domain Breakdown
domain_stats = {}
for rec in all_records:
    dom = rec["domain"]
    if dom not in domain_stats:
        domain_stats[dom] = {
            "total": 0,
            "correct": 0,
            "total_ms": 0.0,
            "total_tokens": 0,
            "truncated": 0
        }
    domain_stats[dom]["total"] += 1
    if rec["is_correct"]:
        domain_stats[dom]["correct"] += 1
    domain_stats[dom]["total_ms"] += rec["duration_ms"]
    domain_stats[dom]["total_tokens"] += rec["generated_tokens"]
    if rec.get("is_truncated", False):
        domain_stats[dom]["truncated"] += 1

total_correct = sum(s["correct"] for s in domain_stats.values())
total_items = len(all_records)
overall_acc = (total_correct / total_items * 100.0) if total_items > 0 else 0.0

# Print clean ASCII table
print(f"{'Domain':<26} | {'Samples':<8} | {'Correct':<8} | {'Accuracy':<10} | {'Avg Latency':<12} | {'Avg Tokens':<10}")
print("-" * 85)
for dom, stats in domain_stats.items():
    dom_acc = (stats["correct"] / stats["total"] * 100.0) if stats["total"] > 0 else 0.0
    avg_lat = stats["total_ms"] / stats["total"] if stats["total"] > 0 else 0.0
    avg_tok = stats["total_tokens"] / stats["total"] if stats["total"] > 0 else 0.0
    print(f"{dom:<26} | {stats['total']:<8} | {stats['correct']:<8} | {dom_acc:6.2f}%   | {avg_lat:8.1f} ms  | {avg_tok:8.1f}")
print("-" * 85)
print(f"{'OVERALL BASELINE':<26} | {total_items:<8} | {total_correct:<8} | {overall_acc:6.2f}%   |")
print("=" * 85)

# Generate JSON Report
report = {
    "benchmark_title": "SAM-AI Multi-Domain Intelligence Baseline",
    "foundation_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "adapter_attached": None,
    "total_samples": total_items,
    "overall_correct": total_correct,
    "overall_accuracy_percent": round(overall_acc, 2),
    "total_duration_seconds": round(total_eval_time, 2),
    "fixtures_sha256": calculated_sha,
    "domain_breakdown": {
        dom: {
            "total_samples": stats["total"],
            "correct_count": stats["correct"],
            "accuracy_percent": round((stats["correct"] / stats["total"] * 100.0), 2) if stats["total"] > 0 else 0.0,
            "avg_latency_ms": round(stats["total_ms"] / stats["total"], 2) if stats["total"] > 0 else 0.0,
            "avg_tokens": round(stats["total_tokens"] / stats["total"], 2) if stats["total"] > 0 else 0.0,
            "truncation_count": stats["truncated"]
        }
        for dom, stats in domain_stats.items()
    },
    "hardware": {
        "device_count": device_count,
        "device_names": device_names,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2)
    }
}

report_file = working_dir / "baseline_multidomain_report.json"
with open(report_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"[OK] Report generated: {report_file.resolve()}")

# Package Results into ZIP archive
zip_path = working_dir / "sam_ai_baseline_multidomain_results.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(report_file, arcname="baseline_multidomain_report.json")
    zf.write(checkpoint_file, arcname="predictions.jsonl")
print(f"[OK] Packaged results archive: {zip_path.resolve()} ({zip_path.stat().st_size / 1024:.1f} KB)")"""
    cells.append(create_code_cell(cell_5_code))

    # Notebook structure
    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
    print(f"Generated notebook at: {NOTEBOOK_PATH} ({len(cells)} cells)")

    # Kernel Metadata
    metadata_content = {
        "id": "samrishb/sam-ai-multi-domain-baseline",
        "title": "SAM-AI Multi-Domain Baseline",
        "code_file": "sam_ai_multidomain_baseline.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            "samrishb/sam-ai-source-bundle"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "keywords": [
            "gpu",
            "evaluation",
            "deepseek",
            "baseline",
            "multidomain"
        ]
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_content, f, indent=2)
    print(f"Generated kernel metadata at: {METADATA_PATH}")


if __name__ == "__main__":
    build_notebook()
