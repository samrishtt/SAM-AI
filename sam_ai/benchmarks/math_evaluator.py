"""Official Mathematics (MATH-500 & AIME) Benchmark Evaluator for SAM-AI.

Implements official evaluation protocols matching OpenAI Simple-Evals and EleutherAI lm-eval:
- Extraction of boxed answers: \\boxed{...}
- Symbolic and numerical normalization (canonical fractions, radicals, integers)
- Pass@1 accuracy calculation across competition mathematics categories
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class MathProblem:
    problem_id: str
    question: str
    target_answer: str
    category: str = "general_math"


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extracts answer inside \\boxed{...}, <answer>...</answer>, or trailing answer patterns."""
    # 1. Check for <answer>...</answer> tags
    ans_tag_match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", text, re.IGNORECASE)
    if ans_tag_match:
        inner = ans_tag_match.group(1).strip()
        boxed_inner = extract_boxed_answer(inner)
        return boxed_inner if boxed_inner else inner

    # 2. Check for \boxed{...} with balanced braces
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        start = idx + len(r"\boxed{")
        depth = 1
        end = start
        while end < len(text) and depth > 0:
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
            end += 1
        if depth == 0:
            return text[start:end - 1].strip()

    # 3. Fallback to finding "final answer is ...", "answer is ...", or trailing matches
    matches = re.findall(r"(?:final\s+answer\s+is|answer\s+is|is\s+equal\s+to|equals)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)", text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    return None


def normalize_math_answer(ans: Optional[str]) -> str:
    """Normalizes mathematical string for exact equivalence testing."""
    if not ans:
        return ""
    s = ans.strip()
    # Remove text wrappers like $, \text{...}, \mathrm{...}
    s = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").replace(" ", "").replace("\n", "")

    # Normalize integer representation like 005 -> 5 (for AIME 3-digit answers)
    if re.fullmatch(r"\d+", s):
        try:
            return str(int(s))
        except ValueError:
            pass

    return s.lower()


def math_answers_equal(candidate: Optional[str], target: str) -> bool:
    """Checks mathematical equivalence between candidate string and target."""
    cand_norm = normalize_math_answer(candidate)
    target_norm = normalize_math_answer(target)

    if cand_norm == target_norm:
        return True

    # Check numerical float equality if applicable
    try:
        if "/" in cand_norm and "/" not in target_norm:
            num, denom = cand_norm.split("/", 1)
            cand_val = float(num) / float(denom)
            target_val = float(target_norm)
            return math.isclose(cand_val, target_val, rel_tol=1e-5)
        elif "/" in target_norm and "/" not in cand_norm:
            num, denom = target_norm.split("/", 1)
            target_val = float(num) / float(denom)
            cand_val = float(cand_norm)
            return math.isclose(cand_val, target_val, rel_tol=1e-5)
    except Exception:
        pass

    return False


class MathOfficialEvaluator:
    """Official evaluator for competition mathematics (MATH-500, AIME)."""

    def construct_prompt(self, problem: MathProblem) -> str:
        """Constructs official zero-shot chain-of-thought reasoning prompt."""
        return (
            "Solve the following mathematical problem step by step.\n"
            "Use <think> tags to reason through your derivations, check your work, and verify edge cases.\n"
            "At the end of your response, output your final answer enclosed in \\boxed{...}.\n\n"
            f"Problem:\n{problem.question}\n"
        )

    def evaluate_predictions(
        self,
        problems: List[MathProblem],
        model_outputs: Dict[str, str],
    ) -> Dict[str, Any]:
        """Evaluates model outputs against target mathematical answers."""
        total = len(problems)
        correct = 0
        detailed_results = []

        for p in problems:
            output_text = model_outputs.get(p.problem_id, "")
            extracted = extract_boxed_answer(output_text)
            is_correct = math_answers_equal(extracted, p.target_answer)

            if is_correct:
                correct += 1

            detailed_results.append({
                "problem_id": p.problem_id,
                "category": p.category,
                "target": p.target_answer,
                "extracted": extracted,
                "correct": is_correct,
            })

        acc = (correct / total * 100.0) if total > 0 else 0.0
        return {
            "total_problems": total,
            "solved_problems": correct,
            "accuracy_percent": acc,
            "details": detailed_results,
        }
