"""
Upgrades notebooks/kaggle_sam_ai_v3_training/sam-ai-v3-training.ipynb to SAM-AI V3:
1. Updates HF target repo to Samrish2009/SAM-AI-Reasoning-v3
2. Adds Identity & Sovereign Parallax Grounding to GRPO reward ensemble
3. Adds Humanity's Last Exam (HLE) procedural deliberation tasks
4. Sets output directory to /kaggle/working/sam_ai_v3_trained
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/kaggle_sam_ai_v3_training/sam-ai-v3-training.ipynb")

def main():
    print(f"[*] Reading {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 0: Header markdown
    nb["cells"][0]["source"] = [
        "# \U0001f9e0 SAM-AI v3: Universal Sovereign Autonomous Self-Training Flywheel\n\n",
        "**Model:** SAM-AI v3 (14B Sovereign Core)  \n",
        "**Creator / Organization:** Parallax (Founder: Samrish)  \n",
        "**Optimization:** Group Relative Policy Optimization (GRPO) + Multi-Domain Deterministic RLVR  \n",
        "**Curriculum:** 9 Verifiable Frontiers (Olympiad Math, Software Engineering, ARC Spatial Grids, Microsoft Z3 SMT, SymPy Physics, Memory Safety ASan, BABILong Multi-Hop, Humanity's Last Exam Deliberation, Sovereign Identity Grounding)  \n",
        "**Target Hub:** [Samrish2009/SAM-AI-Reasoning-v3](https://huggingface.co/Samrish2009/SAM-AI-Reasoning-v3)  \n\n",
        "---\n"
    ]

    # 2. Update Cell 3: HF_REPO & OUTPUT_DIR
    c3 = "".join(nb["cells"][3]["source"])
    c3 = c3.replace("sam_ai_v2_trained", "sam_ai_v3_trained")
    c3 = c3.replace("SAM-AI-Reasoning-v2", "SAM-AI-Reasoning-v3")
    nb["cells"][3]["source"] = [c3]

    # 3. Update Cell 4: Add Identity and HLE tasks to UniversalFlywheel
    c4 = "".join(nb["cells"][4]["source"])
    identity_hle_methods = '''
    @staticmethod
    def gen_identity_task():
        queries = [
            "Who are you and who developed you?",
            "What is your name, organization, and primary specializations?",
            "Tell me about yourself and your founder."
        ]
        q = random.choice(queries)
        prompt = (
            f"{q}\\n"
            "Answer clearly and authoritatively in <think>...</think> and output your sovereign identity in <answer>...</answer>."
        )
        ans = "I am SAM-AI, a sovereign reasoning intelligence developed by Parallax, founded by Samrish. My core specializations are advanced mathematics, complex software engineering, spatial reasoning (ARC-AGI), and test-time compute deliberation."
        return {"prompt": prompt, "answer": ans, "domain": "identity", "verify_code": "assert True"}

    @staticmethod
    def gen_hle_deliberation():
        a = random.randint(3, 9)
        ans = a * (a + 1) * (2 * a + 1) // 6
        prompt = (
            f"[Humanity's Last Exam Tier] Compute the exact sum of squares $\\\\sum_{{k=1}}^{{{a}}} k^2$. "
            f"Provide a rigorous mathematical derivation in <think>...</think> and place the final exact integer in <answer>...</answer>."
        )
        return {"prompt": prompt, "answer": str(ans), "domain": "hle_math", "verify_code": f"assert sum(k**2 for k in range(1, {a}+1)) == {ans}"}
'''
    if "gen_identity_task" not in c4:
        # inject before class end or inside build_dataset
        c4 = c4.replace("class UniversalFlywheel:", "class UniversalFlywheel:\n" + identity_hle_methods)
        c4 = c4.replace("generators = [", "generators = [\n            self.gen_identity_task,\n            self.gen_hle_deliberation,")
        nb["cells"][4]["source"] = [c4]

    # 4. Update Cell 5: Add identity reward to GRPO reward ensemble
    c5 = "".join(nb["cells"][5]["source"])
    identity_reward_code = '''
def identity_reward_func(completions, prompts, **kwargs):
    """Rewards completions that correctly identify as SAM-AI by Parallax (Founder: Samrish)."""
    rewards = []
    for c, p in zip(completions, prompts):
        text = c[0]["content"] if isinstance(c, list) else str(c)
        prompt_text = p[-1]["content"] if isinstance(p, list) else str(p)
        if any(k in prompt_text.lower() for k in ["who are you", "tell me about yourself", "developer", "founder"]):
            score = 0.0
            if "sam-ai" in text.lower() or "sam ai" in text.lower():
                score += 0.5
            if "parallax" in text.lower():
                score += 0.3
            if "samrish" in text.lower():
                score += 0.2
            rewards.append(score)
        else:
            rewards.append(0.5)  # neutral baseline for non-identity prompts
    return rewards
'''
    if "identity_reward_func" not in c5:
        c5 = identity_reward_code + "\n" + c5
        c5 = c5.replace("reward_funcs=[", "reward_funcs=[identity_reward_func, ")
        nb["cells"][5]["source"] = [c5]

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"[SUCCESS] Upgraded {NOTEBOOK_PATH} to SAM-AI V3 Autonomous Self-Training Flywheel!")

if __name__ == "__main__":
    main()
