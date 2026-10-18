"""
SAM-AI Website Content & Truthful Status Verification Suite
===========================================================
Verifies:
1. web/src/app/page.tsx has no mock/fake dialogue previews or unsupported verification guarantees.
2. Foundation model is clearly identified as DeepSeek-R1-Distill-Qwen-7B with adapters detached.
3. Multi-domain baseline score (84/95, 88.42%) matches certified audit in both page.tsx and research/page.tsx.
4. Experiment 001 failure (16/20 vs 19/20 base) is accurately documented and labeled as quarantined.
5. Phase 3B web research engine is explicitly described as implemented locally and NOT yet live in /api/chat.
6. Research page includes full 7-domain breakdown and HumanEval sandbox security disclosure.
7. Roadmap milestones are structured in correct sequential order.
"""

import html
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
SRC_DIR = WEB_DIR / "src"
APP_DIR = SRC_DIR / "app"


def test_homepage_truthful_content_and_status():
    """Verify that homepage contains accurate architecture, status badges, and no mock traces."""
    page_file = APP_DIR / "page.tsx"
    assert page_file.exists(), "web/src/app/page.tsx must exist"
    content = html.unescape(page_file.read_text(encoding="utf-8"))

    # 1. No fake dialogue preview or unsupported claims
    assert "Thought trace: verified 4-stage deduction" not in content
    assert "guarantees mathematical verification before answer emission" not in content

    # 2. Foundation model and adapter detachment disclosure
    assert "DeepSeek-R1-Distill-Qwen-7B" in content
    assert "Live Assistant (Base Model Only)" in content
    assert "Adapters Detached" in content
    assert "No Degraded Adapters" in content

    # 3. Web research explicitly labeled as local / not live in chat
    assert "Web Research Engine (Phase 3B Stage 1 & 2)" in content
    assert "Not yet wired into user chat" in content or "not yet wired into user chat" in content.lower()

    # 4. Accurate baseline and failure audit numbers
    assert "84/95" in content or "84 / 95" in content
    assert "16/20" in content or "16 / 20" in content
    assert "19/20" in content or "19 / 20" in content


def test_research_page_milestones_and_disclosures():
    """Verify that research page includes all 7 domains, parser corrections, sandbox disclosures, and roadmap."""
    research_file = APP_DIR / "research" / "page.tsx"
    assert research_file.exists(), "web/src/app/research/page.tsx must exist"
    content = html.unescape(research_file.read_text(encoding="utf-8"))

    # 1. Section A: Chat reliability
    assert "Chat Reliability & Production Gateway Hardening" in content
    assert "Conversation Hydration" in content
    assert "In-Flight Conversation Binding" in content
    assert "3.5s Stream Keepalives" in content or "Stream Keepalives" in content
    assert "Greeting Fast-Path" in content

    # 2. Section B: Phase 3A Baseline and all 7 domains
    domains = [
        "mathematics_gsm8k",
        "mathematics_math500",
        "coding_humaneval",
        "science_reasoning",
        "logic_constraints",
        "instruction_following",
        "multiturn_context",
    ]
    for domain in domains:
        assert domain in content, f"Missing domain {domain} in research page"

    assert "84 / 95" in content and "88.42%" in content
    assert "19 / 20" in content  # GSM8K
    assert "20 / 20" in content  # MATH-500
    assert "17 / 20" in content  # HumanEval
    assert "9 / 10" in content   # Science
    assert "7 / 10" in content   # Logic / IF
    assert "5 / 5" in content    # Multi-turn

    # LaTeX parser correction note
    assert "LaTeX Fraction Parser Correction" in content or "LaTeX" in content
    assert "\\dfrac" in content

    # HumanEval sandbox disclosure
    assert "Coding Security Validation" in content or "Sandbox Security Disclosures" in content
    assert "isolated" in content.lower() or "sandbox" in content.lower()

    # 3. Section C: Experiment 001 Failure Audit
    assert "Experiment 001: Failed Adapter Audit" in content or "Experiment 001" in content
    assert "16 / 20" in content
    assert "560.9" in content and "111.2" in content
    assert "shorter response is not automatically a smarter response" in content

    # 4. Section D: Phase 3B Web Research Engine
    assert "Phase 3B: Web Research Engine (Stage 1 & Stage 2)" in content
    assert "14 / 14 Passed" in content
    assert "Not Live in User Chat" in content or "not yet integrated into /api/chat" in content

    # 5. Section E: Roadmap in sequential order
    roadmap_steps = [
        "Website & Engine Publication",
        "Phase 3B Stage 3: Grounded Citation Synthesis",
        "Phase 3B Stage 4: Comparative Retrieval Evaluation",
        "Controlled Fine-Tuning Candidate Experiment",
        "Independent Generalization Gate",
    ]
    last_idx = -1
    for step in roadmap_steps:
        idx = content.find(step)
        assert idx != -1, f"Missing roadmap step: {step}"
        assert idx > last_idx, f"Roadmap step '{step}' out of order"
        last_idx = idx


def test_status_page_truthful_matrix():
    """Verify that status page capability matrix accurately categorizes all components."""
    status_file = APP_DIR / "status" / "page.tsx"
    assert status_file.exists(), "web/src/app/status/page.tsx must exist"
    content = status_file.read_text(encoding="utf-8")

    assert "Next.js Web Chat Interface" in content
    assert "Hugging Face Space Backend" in content
    assert "Phase 3A Multi-Domain Baseline" in content
    assert "Phase 3B Web Research Engine" in content
    assert "Experiment 001 Adapter" in content
    assert "Rejected" in content and "Quarantined" in content
    assert "Implemented locally" in content
    assert "Deployed and verified" in content
