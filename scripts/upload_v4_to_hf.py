import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "Samrish2009/SAM-AI-Reasoning-v4"
CHECKPOINT_DIR = Path(r"C:\Users\Sam Pavi\.gemini\antigravity\scratch\micro_agi\kaggle_flywheel_check\sam_ai_v4_trained\checkpoint-25")

MODEL_CARD = """---
language:
- en
license: apache-2.0
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
tags:
- reasoning
- grpo
- r1-distill
- arc-agi
- code-generation
- math
- synthetic-reasoning
pipeline_tag: text-generation
library_name: peft
---

# SAM-AI Reasoning v4 (Parallax)

**SAM-AI Reasoning v4** is a 14B parameter reasoning adapter developed by **Parallax (Samrish)**. It is fine-tuned using Group Relative Policy Optimization (GRPO) with rule-based verifiable rewards on top of `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.

## Model Overview
- **Base Architecture**: DeepSeek-R1-Distill-Qwen-14B (Qwen2.5 14B transformer backbone)
- **Adapter Type**: LoRA (Rank = 16, Alpha = 32, Dropout = 0.05)
- **Target Modules**: All linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Training Method**: GRPO (Group Relative Policy Optimization) with format verification, multi-step math/code reward checks, and reasoning traces (`<think>...</think>`).
- **Trained Parameters**: 137.7 MB adapter safetensors.

## Usage with PEFT & Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
adapter_name = "Samrish2009/SAM-AI-Reasoning-v4"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_name)

prompt = "<｜User｜>Solve step by step: Prove that for any positive integer n, n^3 + 2n is divisible by 3.<｜Assistant｜><think>\\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.6, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Curriculum & Training Objectives
SAM-AI v4 incorporates:
1. **Verifiable Reasoning**: Strict formatting enforcement and multi-step deduction traces.
2. **Abstract Spatial Logic**: Curriculum drawn from ARC inductive reasoning patterns.
3. **Mathematical Derivations**: Step-by-step rigorous proof generation.
4. **Code Execution & Verification**: Synthesizing verifiable Python programs.

## Developed by
- **Team**: Parallax
- **Lead Developer**: Samrish
"""

def main():
    print(f"[*] Initializing Hugging Face API for {REPO_ID}...")
    api = HfApi(token=HF_TOKEN)
    
    try:
        repo_url = create_repo(
            repo_id=REPO_ID,
            token=HF_TOKEN,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"[+] Repository verified / ready: {repo_url}")
    except Exception as e:
        print(f"[-] Note on create_repo: {e}")

    # Files to upload
    files_to_upload = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    # Write model card locally in checkpoint dir
    readme_path = CHECKPOINT_DIR / "README.md"
    readme_path.write_text(MODEL_CARD, encoding="utf-8")
    files_to_upload.append("README.md")

    print("[*] Uploading model artifacts...")
    for filename in files_to_upload:
        file_path = CHECKPOINT_DIR / filename
        if file_path.exists():
            print(f" -> Uploading {filename} ({file_path.stat().st_size / (1024*1024):.2f} MB)...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model",
                token=HF_TOKEN
            )
            print(f"    [+] {filename} uploaded successfully.")
        else:
            print(f"    [!] Skipping {filename} (not found).")

    print(f"\n[🎉] SAM-AI Reasoning v4 successfully uploaded to https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
