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
c1 = ''.join(nb['cells'][1]['source'])
if 'PYTORCH_CUDA_ALLOC_CONF' not in c1:
    nb['cells'][1]['source'].insert(5, "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n")

# Update Cell 6 (Model loading)
c6 = ''.join(nb['cells'][6]['source'])
c6_new = c6.replace('device_map={"": 0}', 'device_map="auto"')
c6_new = c6_new.replace(
    'print(f"[*] Loading base weights for {MODEL_ID}...")',
    'import os, torch\nos.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"\ntorch.cuda.empty_cache()\nprint(f"[*] Loading base weights for {MODEL_ID}...")'
)
lines = [l + '\n' for l in c6_new.split('\n')]
if lines and lines[-1] == '\n':
    lines.pop()
nb['cells'][6]['source'] = lines

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
