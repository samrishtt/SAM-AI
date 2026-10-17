# Third-Party Licenses & Attribution Registry

Registry of all third-party code, datasets, models, tools, and ideas reused in SAM-AI / ARC-AGI-3.
Per workspace rule 6, third-party work must be credited with exact license and attribution in all docs and research papers.

> **Third-Party Code Notice:** `baselines/` and `notebooks/` contain third-party code under its original terms and that those terms are not yet verified, so the repository's Apache-2.0 declaration does not cover them.

---

## 1. Pennyroyal Wheelhouse
* **Author / Provider**: Daniel Franzen (`dfranzen`)
* **Kaggle Source**: `dfranzen/pennyroyal-v253`
* **License**: Apache 2.0 / MIT (see upstream wheelhouse metadata)
* **Usage**: Precompiled Python wheels for offline execution in Kaggle BYOD container.

## 2. TAAF Kaggle Source Bundle
* **Author / Provider**: Daniel Franzen (`dfranzen`)
* **Kaggle Source**: `dfranzen/taaf-kaggle-source-bundle-copy`
* **License**: Apache 2.0 / MIT
* **Usage**: ARC interactive environment harness and game interface.

## 3. Albucino & Intel Qwen 3.8B Weights
* **Author / Provider**: Daniel Franzen (`dfranzen`) / Intel / Qwen Team
* **Kaggle Models**:
  * `dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1`
  * `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1`
* **License**: Apache 2.0 (Qwen base license)
* **Usage**: Speculative drafting and quantized weights for high-throughput ARC action generation.

---

## 4. SAM-AI Sovereign Source Bundle (`dist/sam_ai_source_bundle.tar.gz`)
* **Author / Provider**: Parallax (Founder: Samrish B)
* **Declared Project License**: Apache 2.0 (documented in repository `README.md` and `research/paper_outline.md`)
* **Usage**: Sovereign agent controller (`SAMARC3Agent`), observation formatter, action parser, verification guards, and in-process `HuggingFaceModelRunner`.

---

## 5. SAM-AI v4 Checkpoint-25 LoRA Adapter Weights
* **Author / Provider**: Parallax (Founder: Samrish B)
* **Base Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (MIT License) / `unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit` (MIT License)
* **Model Card Declaration**: `checkpoint-25/README.md` declares `license: apache-2.0`.
* **Licensing Resolution**: `NEEDS_USER` verification. The model card frontmatter states `apache-2.0`, but the base model is MIT, and `checkpoint-50` notes `[More Information Needed]`. For private dataset upload, the user must decide whether the adapter is tagged `apache-2.0`, `mit`, or `other`.

