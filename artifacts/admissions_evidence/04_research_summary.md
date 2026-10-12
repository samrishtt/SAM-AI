# Admissions Evidence Pack — 04: Research Summary

**Status:** PROPOSED METHODOLOGY & EVALUATION SPECIFICATION  

---

## 1. Core Research Questions
1. **Hypothesis H1 (Scaffolding vs. Weights):** Can deterministic action schemas and failure memory prevent hallucination in an open-weight 14B model without needing fine-tuning?
2. **Hypothesis H2 (Rejection-Sampling Fine-Tuning):** Can fine-tuning on a model's own verified-correct reasoning traces (EXP-SAM-RFT-01) raise task pass rates under a strict thinking token budget?

## 2. Experimental Stage Gates
- **G0 (Smoke Test):** Prove weight loading, prompt formatting, and single-action environment acceptance (5 steps).
- **G1 (Baseline Characterization):** Measure think-length distributions, parse rates, and VRAM throughput.
- **G2 (Runtime Upgrades):** Evaluate action grammar and decoding hygiene.
- **G3 (Adapter Comparison):** Controlled comparison between base 14B and checkpoint-25.
- **G4 (Fine-Tuning):** Targeted LoRA on verified traces only if a reasoning gap survives runtime fixes.

## 3. What Can Be Filled From Existing Repo Artifacts
- Methodology details from `research/SAM_AI_FINE_TUNING_PROPOSAL_v2.md`
- Precondition audit protocols from `research/P0_CHECKS.md`

## 4. NEEDS_USER Items
- [PLACEHOLDER: needs Samrish] Future publication targets (e.g. arXiv preprint, student symposium).
- [PLACEHOLDER: needs Samrish] Long-term vision for Parallax Intelligence Lab beyond ARC-AGI-3.
