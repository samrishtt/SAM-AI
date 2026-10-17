# SAM-AI Production Deployment Checklist & Human Handoff Guide
## Step-by-Step Instructions for Safe Public Deployment to Vercel Hobby & ZeroGPU

**Target Environments:**
- **Frontend / Web Platform:** Vercel Hobby (100% Free, Zero Credit Card Required)
- **Model Inference Server:** Hugging Face Spaces (ZeroGPU Free Tier, NVIDIA RTX Pro 6000 Blackwell)
- **Repository:** `samrishtt/SAM-AI`
- **Current Target Branch:** `integrate/claude-2026-10-12`

---

## 1. Prerequisites & Account Requirements

Both hosting targets utilize legitimate free tiers and require no payment methods or subscriptions:

1. **Vercel Account:**
   - Sign up at [vercel.com/signup](https://vercel.com/signup) using your GitHub account (`samrishtt`).
   - Hobby Tier is automatically provisioned for personal/hobby projects with zero billing setup.
2. **Hugging Face Account:**
   - Verified account (phone verification is standard on Hugging Face to unlock free ZeroGPU allocations).
3. **GitHub Permissions:**
   - Write/push permissions on `samrishtt/SAM-AI`.

---

## 2. Reviewing the Proposed Release Scope

Before staging or committing any code, review the classification in [`docs/website_release_audit.md`](file:///C:/Users/Sam%20Pavi/.gemini/antigravity/scratch/micro_agi_integrate_claude/docs/website_release_audit.md).

### Files Designated for Website Release
- **Core Web App:**
  - `web/package.json`
  - `web/.env.example`
  - `web/src/app/page.tsx`
  - `web/src/app/chat/page.tsx`
  - `web/src/app/research/page.tsx`
  - `web/src/app/status/page.tsx`
  - `web/src/app/api/chat/route.ts`
  - `web/src/app/api/status/route.ts`
  - `web/src/app/api/workspace/files/route.ts`
  - `web/src/app/api/workspace/file/route.ts`
  - `web/src/app/api/workspace/status/route.ts`
  - `web/src/components/Navbar.tsx`
  - `web/src/components/Header.tsx`
  - `web/src/components/ChatMessage.tsx`
  - `web/src/components/ChatInterface.tsx`
  - `web/src/components/NoticeBanners.tsx`
  - `web/src/components/CodeWorkspace.tsx`
  - `web/src/lib/model-adapter.ts`
  - `web/src/lib/security.ts`
  - `web/src/lib/storage.ts`
  - `web/src/types/chat.ts`
- **Release Documentation:**
  - `docs/website_launch_phase_1_report.md`
  - `docs/website_release_audit.md`
  - `docs/website_deployment_checklist.md`

### Files Strictly Excluded from Website Release
- **Do NOT add:** `dist/` (contains 144.7 MB zip).
- **Do NOT add:** `evidence/remote_r1_7b_run/` (contains 161.5 MB safetensors).
- **Do NOT add:** `curriculum/` (contains 250 OpenR1 training traces for offline training).
- **Do NOT add:** `notebooks/` or `sam_ai/foundation/` (offline research).
- **Do NOT add:** `web/.env.local` (local environment configuration).

---

## 3. Publishing Approved Changes to GitHub (Human Review Only)

> [!IMPORTANT]
> The AI agent will **NOT** execute Git staging, commit, or push operations. The human operator should review and run the following explicit commands:

### Step 3.1: Check Status and Working Tree
```powershell
git status --short
git branch --show-current
```

### Step 3.2: Stage Only the Designated Website Files
```powershell
# Stage all tracked and untracked web application files
git add web/package.json
git add web/.env.example
git add web/src/
git add docs/website_launch_phase_1_report.md
git add docs/website_release_audit.md
git add docs/website_deployment_checklist.md
```

### Step 3.3: Verify Staged Changes
Ensure no data files or binaries are staged:
```powershell
git diff --staged --stat
```
*(Verify that only `web/` and the 3 website documentation files are listed).*

### Step 3.4: Commit and Push
```powershell
git commit -m "feat(web): deploy multi-page research platform with Homepage, Research, Status, and Chat"
git push origin integrate/claude-2026-10-12
```

---

## 4. Importing the Repository into Vercel

### Step 4.1: Import Project in Vercel
1. Log in to [vercel.com/dashboard](https://vercel.com/dashboard).
2. Click **"Add New..."** → **"Project"**.
3. Locate `samrishtt/SAM-AI` and click **"Import"**.

### Step 4.2: Critical Configuration Settings
In the configuration screen, set:
- **Project Name:** `sam-ai` (or your choice).
- **Framework Preset:** `Next.js` (automatically detected).
- **Root Directory:** Click "Edit" and choose **`web`** *(CRITICAL: Do NOT leave at repository root)*.
- **Build Command:** `next build` (default).
- **Output Directory:** `.next` (default).
- **Install Command:** `npm install` (default).
- **Production Branch:** `integrate/claude-2026-10-12`.

### Step 4.3: Configure Initial Environment Variables
In the **Environment Variables** panel, set:

| Variable Name | Recommended Initial Value | Note |
| :--- | :--- | :--- |
| `SAM_AI_MODEL` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Authoritative foundation model identifier |
| `SAM_AI_BASE_URL` | *(Leave Empty)* | Leaves site in safe, truthful "Setup Needed" mode |
| `SAM_AI_RATE_LIMIT_PER_MINUTE` | `20` | Rate limiting per IP |
| `SAM_AI_MAX_OUTPUT_TOKENS` | `2048` | Max reasoning token length |
| `SAM_AI_TEMPERATURE` | `0.6` | Sampling temperature |

### Step 4.4: Trigger Deployment
Click **"Deploy"**. The build will complete in ~30–60 seconds, provisioning your public domain (e.g., `https://sam-ai-xxx.vercel.app`).

---

## 5. Checking Deployed Homepage and Primary Routes

Once Vercel reports the deployment is **Ready**, open the live domain and verify:
1. **Homepage (`/`):** Verify Parallax branding, research pillars, and links render cleanly.
2. **Research Chronicle (`/research`):** Verify Phase 3C (19/20 base vs 16/20 candidate) and Phase 4A audit findings are clearly displayed.
3. **Status Matrix (`/status`):** Verify the capability matrix displays accurate status tags, and the live status badge reports "Setup Needed" (honest unconfigured state).
4. **Chat & Workspace (`/chat`):** Verify the interface loads without errors, shows "Setup Needed", and entering a prompt returns a helpful setup explanation without fake mock text.
5. **Mobile Viewport:** Open on mobile; verify the navbar toggles cleanly and layout wraps properly.

---

## 6. Deploying the Space Separately (Human Approval)

Once the website is live, the sovereign model backend can be deployed independently to Hugging Face Spaces:

### Step 6.1: Create Space
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Configure:
   - **Space Name:** `sam-ai-r1-7b-backend`
   - **SDK:** `Gradio`
   - **SDK Version:** `4.44.0`
   - **Space Hardware:** **ZeroGPU** (Free tier, NVIDIA RTX Pro 6000 Blackwell).
   - **Visibility:** Public.

### Step 6.2: Deploy Code to Space
In a temporary directory:
```powershell
git clone https://huggingface.co/spaces/<your-username>/sam-ai-r1-7b-backend
cd sam-ai-r1-7b-backend

# Copy all files from spaces/sam_ai_zerogpu:
Copy-Item -Path "C:\Users\Sam Pavi\.gemini\antigravity\scratch\micro_agi_integrate_claude\spaces\sam_ai_zerogpu\*" -Destination "." -Recurse -Force

git add .
git commit -m "feat(zerogpu): deploy SAM-AI sovereign model server with DeepSeek-R1-Distill-Qwen-7B"
git push origin main
```

### Step 6.3: Verify ZeroGPU Space Initialization
- Wait for the Space container to build (`Building` → `Running`).
- Test an interactive prompt directly in the Space Gradio UI to confirm weights load into GPU memory.

---

## 7. Connecting Frontend to the Real Model Endpoint

Once the Space is confirmed active:
1. Copy the Space URL (e.g., `https://<your-username>-sam-ai-r1-7b-backend.hf.space`).
2. Go to **Vercel Dashboard** → Your Project → **Settings** → **Environment Variables**.
3. Add / Update:
   - `SAM_AI_BASE_URL`: `https://<your-username>-sam-ai-r1-7b-backend.hf.space/v1`
   - `SAM_AI_TRUSTED_HOSTS`: `<your-username>-sam-ai-r1-7b-backend.hf.space`
4. Go to **Deployments** → Select latest deployment → Click **"Redeploy"** to apply the updated environment variables.
5. Test live chat on your public Vercel domain; streaming tokens and visible reasoning `<think>` tags will now render directly from your sovereign ZeroGPU backend.
