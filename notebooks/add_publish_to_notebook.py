import json

path = 'notebooks/kaggle_sam_ai_arc_r1/sam-ai-arc-grpo-training.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][7]
source = cell['source']

push_code = [
    "\n",
    "# ==============================================================================\n",
    "# 8. Publish Live to Hugging Face Hub\n",
    "# ==============================================================================\n",
    "HF_REPO = 'Samrish2009/SAM-AI-Reasoning-14B'\n",
    "HF_TOKEN = os.environ.get('HF_TOKEN', '')\n",
    "try:\n",
    "    print(f'[*] Publishing live checkpoint directly to Hugging Face Hub: {HF_REPO}...')\n",
    "    trainer.model.push_to_hub(HF_REPO, token=HF_TOKEN)\n",
    "    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)\n",
    "    print(f'[✓] MODEL IS OFFICIALLY LIVE ON THE WEB: https://huggingface.co/{HF_REPO}')\n",
    "except Exception as e:\n",
    "    print(f'[Notice] Hub push error: {e}')\n"
]

src_str = ''.join(source)
if 'Publishing live checkpoint directly to Hugging Face Hub' not in src_str:
    source.extend(push_code)
    cell['source'] = source
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('Updated notebook with live HF Hub publishing logic!')
else:
    print('Already present!')
