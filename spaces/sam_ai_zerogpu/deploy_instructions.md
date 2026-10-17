# SAM-AI ZeroGPU Space Deployment Guide

This guide describes how to deploy **SAM-AI Sovereign Assistant** to a free **Hugging Face Space** using **ZeroGPU** compute (NVIDIA RTX Pro 6000 Blackwell, 48 GB VRAM).

---

## 1. Prerequisites (100% Free / No Credit Card Required)

- A free Hugging Face account (email verified).
- No credit card or paid subscription is required.
- Free accounts in good standing can host up to **2 ZeroGPU Spaces** concurrently.

---

## 2. Option A: Pure Foundation Model Deployment (Recommended — Size: ~25 KB)

Since the Experiment 001 adapter is rejected, the Space deploys the pure **DeepSeek-R1-Distill-Qwen-7B** foundation model. This does not require uploading heavy adapter weights.

### Step 1: Create the Space on Hugging Face
1. Log in to [Hugging Face](https://huggingface.co).
2. Click **New Space** (or visit https://huggingface.co/new-space).
3. Set:
   - **Space name:** `sam-ai-assistant` (or your preferred name)
   - **License:** `apache-2.0`
   - **Space SDK:** **Gradio**
   - **Space hardware:** Select **ZeroGPU** (`zero-a10g` or `zero-gpu` / NVIDIA RTX Pro 6000)
   - **Visibility:** **Public**
4. Click **Create Space**.

### Step 2: Upload Files (Browser or Git)

**Via Browser (Fastest, < 1 minute):**
1. In your newly created Space, click **Files** → **Add file** → **Upload files**.
2. Drag and drop these 4 files from `spaces/sam_ai_zerogpu/`:
   - `app.py`
   - `model_loader.py`
   - `requirements.txt`
   - `README.md`
3. Click **Commit changes to main**.

**Via Git:**
```bash
git clone https://huggingface.co/spaces/<your-username>/sam-ai-assistant hf_space_repo
cp spaces/sam_ai_zerogpu/app.py spaces/sam_ai_zerogpu/model_loader.py spaces/sam_ai_zerogpu/requirements.txt spaces/sam_ai_zerogpu/README.md hf_space_repo/
cd hf_space_repo
git add .
git commit -m "Deploy SAM-AI Sovereign Assistant on ZeroGPU"
git push origin main
```

---

## 3. Option B: Separate Private Adapter Repository

If you prefer to keep your Space repository **Public** while keeping the adapter weights strictly **Private**:

1. Create a **Private Model Repository** on Hugging Face:
   - Repository type: **Model**
   - Repository name: `sam-ai-r1-7b-gsm8k-adapter`
   - Visibility: **Private**
2. Upload the 6 adapter files (`adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja`, `README.md`, `tokenizer.json`, `tokenizer_config.json`) into this private model repository.
3. In your Public Space:
   - Go to **Settings** → **Variables and secrets**.
   - Add a Secret:
     - Name: `HF_TOKEN`
     - Value: `<your-read-only-huggingface-token>`
   - Add a Variable:
     - Name: `SAM_AI_ADAPTER_REPO_ID`
     - Value: `<your-username>/sam-ai-r1-7b-gsm8k-adapter`
4. When `SAM_AI_ADAPTER_REPO_ID` is present, `model_loader.py` securely downloads the adapter using the secret `HF_TOKEN` during startup.

---

## 4. Verification and Startup Logs

Once pushed:
1. Open the Space page on Hugging Face.
2. Watch the **Building** status in the top right corner.
3. Check the **App Logs**:
   - `[*] Loading base model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B...`
   - `[*] Attaching verified SAM-AI adapter from ./adapter...`
   - `[OK] Ready: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B + sam_ai_r1_7b_gsm8k_adapter on cuda`
4. The status banner in the application will show:  
   `🟢 READY (SOVEREIGN ZERO-GPU) | Device: CUDA`

---

## 5. Troubleshooting & ZeroGPU Limits

- **"GPU quota exceeded"**: Free tier Hugging Face accounts receive 5 minutes of daily GPU inference time (2 minutes for unauthenticated visitors). When exhausted, wait for the daily quota reset or log in with your authenticated HF account.
- **"ZeroGPU queue timeout"**: If many users are accessing ZeroGPU simultaneously, requests enter a fair queue. Responses are capped at 60 seconds of GPU time.
- **Model Load Memory Error**: ZeroGPU's `large` tier provides 48 GB VRAM on NVIDIA RTX Pro 6000 Blackwell. DeepSeek 7B in 16-bit requires ~15 GB VRAM and easily fits within the 48 GB budget.
