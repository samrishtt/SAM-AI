import json

nb_path = 'notebooks/kaggle_sam_ai_v2_training/sam-ai-v2-training.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 0 (Title)
nb['cells'][0]['source'] = [
    "# 🧠 NEXIS v2: Universal 8-Domain Autonomous Self-Training Flywheel\n",
    "\n",
    "**Model:** Nexis v2 (14B Sovereign Core)  \n",
    "**Creator / Organization:** Parallax (Founder: Samrish)  \n",
    "**Optimization:** Group Relative Policy Optimization (GRPO) + Multi-Domain Deterministic RLVR  \n",
    "**Curriculum:** 8 Verifiable Frontiers (Olympiad Math, Software Engineering, ARC Spatial Grids, Microsoft Z3 SMT, SymPy Physics, Memory Safety ASan, BABILong Multi-Hop State)  \n",
    "**Target Hub:** [Samrish2009/SAM-AI-Reasoning-v2](https://huggingface.co/Samrish2009/SAM-AI-Reasoning-v2)  \n",
    "\n",
    "---\n"
]

# Update Cell 1 (Environment)
nb['cells'][1]['source'] = [
    "# ==============================================================================\n",
    "# 1. Environment & Dependency Conflict Resolution\n",
    "# ==============================================================================\n",
    "import os\n",
    "import sys\n",
    "import subprocess\n",
    "\n",
    "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
    "print('[*] Checking and resolving Kaggle pre-installed environment conflicts...')\n",
    "subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)\n",
    "\n",
    "pkgs = ['trl>=0.15.0', 'peft>=0.12.0', 'transformers>=4.48.0', 'accelerate>=1.2.0', 'datasets', 'bitsandbytes', 'huggingface_hub', 'z3-solver', 'sympy']\n",
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs)\n",
    "print('[OK] Environment sanitized: Modern TRL, PEFT, Z3, SymPy, and Transformers ready.')\n"
]

# Update Cell 3 (Target Model to pre-quantized unsloth bnb-4bit)
c3 = ''.join(nb['cells'][3]['source'])
c3_new = c3.replace(
    'MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"',
    'MODEL_ID = "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit"  # Pre-quantized 4-bit 14B fits inside 8.5 GB VRAM'
)
lines3 = [l + '\n' for l in c3_new.split('\n')]
if lines3 and lines3[-1] == '\n':
    lines3.pop()
nb['cells'][3]['source'] = lines3

# Update Cell 6 (Model loading with pre-quantized 4-bit)
nb['cells'][6]['source'] = [
    "# ==============================================================================\n",
    "# 6. Model Loading & PEFT LoRA Injection (Pre-Quantized 4-Bit Core)\n",
    "# ==============================================================================\n",
    "import os\n",
    "import torch\n",
    "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
    "from peft import LoraConfig, get_peft_model\n",
    "\n",
    "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
    "torch.cuda.empty_cache()\n",
    "\n",
    "print(f'[*] Loading tokenizer: {MODEL_ID}...')\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)\n",
    "if tokenizer.pad_token is None:\n",
    "    tokenizer.pad_token = tokenizer.eos_token\n",
    "\n",
    "print(f'[*] Loading pre-quantized 4-bit base weights for {MODEL_ID}...')\n",
    "compute_dtype = torch.bfloat16 if bf16_supported else torch.float16\n",
    "\n",
    "base_model = AutoModelForCausalLM.from_pretrained(\n",
    "    MODEL_ID,\n",
    "    device_map='auto',\n",
    "    torch_dtype=compute_dtype,\n",
    "    trust_remote_code=True,\n",
    ")\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=32,\n",
    "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],\n",
    "    lora_dropout=0.05,\n",
    "    bias='none',\n",
    "    task_type='CAUSAL_LM',\n",
    ")\n",
    "\n",
    "model = get_peft_model(base_model, lora_config)\n",
    "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
    "total = sum(p.numel() for p in model.parameters())\n",
    "print(f'[OK] LoRA Policy injected: {trainable:,} / {total:,} ({100 * trainable / total:.2f}% trainable)')\n"
]

# Update Cell 7 (Checkpoints & Commit messages)
c7 = ''.join(nb['cells'][7]['source'])
c7_new = c7.replace('SAM-AI v2', 'Nexis v2').replace('SAM-AI', 'Nexis')
lines7 = [l + '\n' for l in c7_new.split('\n')]
if lines7 and lines7[-1] == '\n':
    lines7.pop()
nb['cells'][7]['source'] = lines7

# Update Cell 8 (Identity in prompt)
c8 = ''.join(nb['cells'][8]['source'])
c8_new = c8.replace('You are SAM-AI, a sovereign reasoning intelligence', 'You are Nexis, a sovereign reasoning intelligence created by Parallax (Founder: Samrish)')
c8_new = c8_new.replace('SAM-AI v2', 'NEXIS v2')
lines8 = [l + '\n' for l in c8_new.split('\n')]
if lines8 and lines8[-1] == '\n':
    lines8.pop()
nb['cells'][8]['source'] = lines8

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('[OK] sam-ai-v2-training.ipynb successfully updated with device_map=auto, expandable_segments, and Nexis by Parallax branding!')
