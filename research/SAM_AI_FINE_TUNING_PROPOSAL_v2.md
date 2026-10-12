# SAM-AI Fine-Tuning Proposal v2 (evaluation first)



Status: PROPOSAL. Every experiment below is `NOT_RUN`. Nothing here may be run without Samrish's explicit, separate approval per stage.

Supersedes: `research/SAM_AI_FINE_TUNING_PROPOSAL.md` (EXP-SAM-FT-01, v1). Keep v1 as a historical record; mark it `SUPERSEDED by v2` at the top and do not delete it.

Written: 2026-10-12. Author of this revision: Claude (review by Samrish pending).



Labels used: `MEASURED` (raw output exists in the repo), `REPORTED` (stated by a tool or person, raw evidence not seen by the author of this document), `ASSUMPTION`, `ESTIMATE-UNMEASURED` (arithmetic from assumptions, not a benchmark), `UNVERIFIED`, `NOT_RUN`.



---



## 0. Bottom line



1. The repository has **no measured evidence** of any capability gap in the 14B model. No real inference has completed on any ARC-3 game or verified benchmark through the new agent. Two Kaggle attempts stopped before model load (Run 1 cancelled for an unknown reason; Run 2 halted on path discovery).

2. The task brief for v1 said: if evidence does not identify a gap, prepare an evaluation plan instead of inventing one. v1 invented one ("parse rate ~65%"). v2 follows the brief: **measure first, then decide whether to train.**

3. The format problem v1 wanted to train away (valid action JSON) is better solved at runtime (think-then-commit decoding with an action grammar, §6 U1). Weights should be trained only for a *reasoning* gap that survives runtime fixes.

4. If a fine-tune is justified, the defensible first one is **rejection-sampling fine-tuning on the model's own verified successes** (EXP-SAM-RFT-01, §5), not 2,000 hand-assembled pairs with identity data.



## 1. What is known and not known



| Claim | Status | Source / note |

|---|---|---|

| Base: DeepSeek-R1-Distill-Qwen-14B, 4-bit NF4 build `gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1`, ~9.94 GB | REPORTED | Kaggle catalog lookup by Antigravity; upstream HF commit `c73c785b...` |

| Mounted at `/kaggle/input/models/gowillgo/.../transformers/v1/1` | MEASURED | Run 2 log |

| GPU available: RTX PRO 6000 Blackwell, ~95 GB | MEASURED | Run 2 log |

| checkpoint-25: 137,716,248 B, SHA-256 `2fd12ef9...14af3`, LoRA r=16, alpha=32, 7 projections | REPORTED | Antigravity, local file |

| checkpoint-25 trained 25 of 1,200 planned GRPO steps; 3 of 5 logged grad norms = 0; 85% completions clipped at 512 tokens; reward 0.552 -> 0.200 | REPORTED | `trainer_state.json`, not seen by this author |

| checkpoint-50 is 0 bytes | REPORTED | unusable; preserved |

| Base model has produced valid ARC-3 actions | NOT_RUN | no inference has completed |

| Base-model parse rate, latency, tok/s, VRAM in the agent loop | NOT_RUN | v1 figures were invented |

| checkpoint-25 differs from the base in any measurable way | UNVERIFIED | LoRA starts at zero delta; after 25 steps with mostly zero gradients it is expected to be close to the base (ASSUMPTION) |

| Any ARC-3 submission score (9.50, 29.21, 23-28) reflects SAM-AI model ability | CONTRADICTED | those runs use Franzen's solver; no SAM-AI weights were loaded |



## 2. Corrections to v1



| v1 statement | Problem | v2 change |

|---|---|---|

| Parse rate "~65%" baseline, ">98%" target, "40% token waste" | No run exists; numbers have no source | Removed. Measured in E1b. |

| Eval cap 1,024 new tokens; training max length 2,048 | R1-style models think for thousands of tokens; this repeats the 512-token clipping (85% clipped) that damaged the GRPO run | Caps are set from the measured think-length distribution (E1a), not guessed |

| 400 training traces from MATH-500 and MATH-500 as the regression metric | The regression test is contaminated by training data | MATH-500, GSM8K test, HumanEval, LiveCodeBench are eval-only, never in training |

| 300 "sovereign identity" examples (15% of data) | Not a capability; adds nothing measurable; confounds attribution | Removed. Identity belongs in a system or user prompt. |

| LR 2e-5 for LoRA | About 5-10x below common LoRA values (1e-4 to 2e-4); a null result could be undertraining | Pre-registered 1e-4; optional 3-value pilot on validation only |

| 800 ARC-3 "spatial action pairs" split by trace | Traces from one game leak across splits; there are about 10 public games locally, so effective sample size is games | Split by game; ARC-3 SFT is deferred (see §5.3) |

| Throughput table (55-80 tok/s etc.) | A 14B model that has never run cannot have measured speeds; candidates (Qwen 2.5 32B/72B) were stale | Replaced with a measurement protocol (§4) |

| "Highest reasoning density per parameter" | Unsupported marketing claim | Removed |

| Arm A/B comparison under 1,024 cap | Measures truncation, not capability | Cap from E1a |

| Greedy decoding (temp 0.0) in the smoke test | Known to cause repetition loops in R1-distill models (per upstream model card; verify) | Smoke test uses sampling with a fixed seed; see prompt P1 |



## 3. Stage gates



```

G0  Integration smoke (5 env steps, base only)  -> proves load + generate + one valid action. Not a score.

G1  E1 baseline characterisation                -> report written, numbers recorded

G2  Runtime fixes U1/U2 applied and re-measured -> is a gap still there?

G3  EXP-SAM-EVAL-01 adapter A/B (optional)      -> closes the question on the published adapter

G4  EXP-SAM-RFT-01                              -> only if G2 shows a surviving reasoning gap

```



A stage may start only when the previous gate's report is committed and Samrish has approved the next stage in writing. Approval for one stage is not approval for the next.



## 4. Stage E1: baseline characterisation (NOT_RUN)



Base model only. No adapter. Offline. No competition submission. Decoding for all of E1: temperature 0.6, top-p 0.95, fixed seeds, thinking forced open with `<think>\n`, no system prompt (instructions in the user turn). These follow the upstream R1-distill model card guidance; verify against the card before use.



**E1a Think-length distribution.** 60 MATH train-split problems (Level 4-5), 60 code problems with tests (MBPP/APPS train splits), 60 agent-loop calls from public ARC-3 observations. Cap 8,192 new tokens. Record per call: tokens until `</think>`, whether `</think>` appeared, answer tokens, wall time. Report p50/p90/p95/p99 and the fraction unterminated at 1k/2k/4k/8k. Output feeds every later cap: `T_think` per task type = p95 rounded up, subject to the time budget.

Token ceiling: 180 x 8,192 = 1.47M tokens. ESTIMATE-UNMEASURED wall time: 2.0 h at 200 tok/s, 0.4 h at 1,000 tok/s, 0.14 h at 3,000 tok/s.



**E1b Action validity in the agent loop.** At least 400 model calls over at least 6 public ARC-3 games and 3 seeds (about 18 episodes), at most 25 env steps per episode, offline mode. Record parse-valid, schema-valid, legal-per-environment, repeated-state, truncation. Report the rate with a cluster bootstrap over games (the effective sample size is the number of games, not calls). Report per-seed SD.



**E1c Throughput and memory.** Batch 1, 8 and 32 (if memory allows): aggregate tok/s, time-to-first-token, peak VRAM, load time. This is the number that sizes every later plan; sampling for the later stages is bounded by it.



E1 outputs: `evidence/e1_baseline/{calls.jsonl,summary.json}` with model path, revision, adapter=None, commit hash, seeds, package versions. Nothing from E1 is a benchmark claim; it is a characterisation of one build in one harness.



## 5. Stages G3 and G4



### 5.1 EXP-SAM-EVAL-01: checkpoint-25 vs base (optional, low priority)



Purpose: close the loop on the published adapter, not to expect a gain. Prior: Arm A and Arm B are indistinguishable (ASSUMPTION, see §1).

- Arm A: base. Arm B: base + checkpoint-25 (loaded locally, never uploaded; assert adapter attached and SHA-256 matches).

- Same problems, prompts, decoding (0.6 / 0.95), seeds, token cap (from E1a), scoring.

- n = 300 problems held out from everything (150 math, 150 code), 4 seeds each; per-problem pass rate; paired bootstrap over problems.

- Token ceiling: 300 x 4 x 2 x 3,000 = 7.2M tokens. ESTIMATE-UNMEASURED: 10 h at 200 tok/s, 2 h at 1,000 tok/s, 0.7 h at 3,000 tok/s. Do not start if aggregate throughput is under about 800 tok/s.

- Report: difference with 95% CI. "No detectable difference" is a valid result and must be reported as such, not as a rejection of the adapter.

- Any statement on the Hugging Face model card about "SAM-AI Reasoning v4" must match this result.



### 5.2 EXP-SAM-RFT-01: rejection-sampling fine-tune (conditional on G2)



**Hypothesis (H1).** Fine-tuning the 14B base on its own verified-correct reasoning traces, for problems where it succeeds sometimes but not always, raises held-out pass@1 under the fixed budget `T_think` by at least 5 points without a regression worse than 5 points on regression sets. Rationale: shifts probability toward successful traces that fit the budget. No measured support exists yet; this is a hypothesis.



**Entry condition (G2).** E1 shows (a) base pass rate on the target distribution between 20% and 80% (otherwise there is no learning signal), and (b) a failure mode that remains after U1/U2. If not met, do not train; record the result.



**Data.**

- Problem sources, train splits only: GSM8K train, MATH train (Level 3-5), MBPP and APPS train with tests. Licences to be verified at the time of use and recorded in `research/licenses.md` (believed MIT / CC-BY-4.0; `UNVERIFIED`).

- Eval-only, never touched: MATH-500, GSM8K test, HumanEval, LiveCodeBench, all ARC-AGI-3 hidden-derived data.

- Held-out sets fixed and hashed (SHA-256 of the sorted ID list) before any sampling, then recorded in the experiment registry.

- Dedup: exact normalised match plus MinHash LSH (5-gram shingles, theta 0.8) of every training problem against every eval set; drop matches. Split by problem ID so all samples of a problem stay on one side.

- Known limit: the base model's own pretraining and distillation data are undisclosed; contamination of public benchmarks cannot be ruled out. Only paired within-model deltas are claimed.



**Sampling and selection.** 800 problems, k=8 samples each at 0.6 / 0.95 under `T_think`. Keep problems with 1 to 7 correct of 8. Per kept problem, keep up to 2 verified-correct traces chosen at random among correct ones (not the shortest; shortest-first selection may teach skipping verification, treat as an optional ablation). ASSUMPTION: about 40% of problems qualify, giving at most 640 traces.

- Sampling tokens: 800 x 8 x 3,000 = 19.2M. ESTIMATE-UNMEASURED: 26.7 h at 200 tok/s, 5.3 h at 1,000, 1.8 h at 3,000. **Sampling throughput is the binding constraint.** If E1c shows aggregate throughput under about 800 tok/s, shrink to at most 400 problems x k=4 or stop.



**Verification.**

- Math: final boxed answer checked by the hardened SymPy verifier (whitelisted AST, no `eval`).

- Code: hidden tests via the nonce harness in a subprocess. This verifier is not a sandbox: run model-generated code only inside the Kaggle container, never on a personal machine.

- Audit: Samrish manually reviews 50 randomly chosen accepted traces and records verifier precision. If precision is under 90%, stop and fix the verifier.



**Training.**

- QLoRA on the same 4-bit base. LoRA r=16, alpha=32, dropout 0.05, the same 7 projections (comparable with checkpoint-25).

- Loss: cross-entropy on completion tokens only (prompt masked with -100).

- LR 1e-4, cosine, 5% warmup. 2 epochs. Effective batch 16. bf16 compute, gradient checkpointing.

- Max sequence = prompt + `T_think` + answer, from E1a (expected to be well above v1's 2,048; to be measured).

- At least 2 training seeds, so training variance is estimated. With one seed, any claim is limited to "this run".

- Training tokens: at most 640 x 3,000 x 2 = 3.84M. ESTIMATE-UNMEASURED: 0.53 h at 2k tok/s, 0.27 h at 4k, 0.13 h at 8k. VRAM: v1's 18.5 GB assumed 2,048 tokens and is stale; expect more with longer sequences. UNMEASURED. Run a 3-step dry run first and record peak VRAM.

- Storage: adapter ~140 MB per checkpoint plus traces; well under quota. ESTIMATE.



**Checkpointing and recovery (lesson from the 0-byte checkpoint-50).**

- Check free disk before starting.

- Save every 50 steps to `/kaggle/working`. After each save: reopen the safetensors header, confirm tensor count and total bytes, compute SHA-256, append to the registry. Keep the last two. A save that fails verification stops the run with a clear error.

- Support resume from the last verified checkpoint. Save the final adapter twice.



**Evaluation (never used for sampling or selection).**

- Target: n >= 500 held-out paired problems (math + code), 4 seeds, token budget `T_think`.

- Regression: MATH-500 subset (200, none in training), GSM8K test (200), HumanEval (164), and ARC-3 agent action-validity on held-out games.

- Statistics: per-problem paired difference, paired bootstrap 95% CI; report per-seed results. Power: assumed 15% discordant pairs gives SE = sqrt(0.15/n): n=300 -> 2.2 pts (power for a +5 gain about 61%), n=500 -> 1.7 pts (about 82%), n=800 -> 1.4 pts (about 95%). Smaller effects than about 4 points at n=500 cannot be resolved; say so in the report.



**Pre-registered decision rule.**

- KEEP: target gain point estimate >= +5 and 95% CI lower bound > 0, in both training seeds with the same sign; every regression set non-inferior at a -5 point margin (CI lower bound > -5); truncation rate not higher than base.

- REJECT: any regression point estimate below -5, or the target CI entirely below 0.

- INCONCLUSIVE: anything else. Report as inconclusive. Do not relabel it.



**Risks.** Overfitting to the problem distribution; verifier false positives teaching wrong reasoning (audit above); length collapse or loss of the habit of verifying; forgetting general ability; contamination of the base; single-seed noise. A negative or inconclusive result is a valid, publishable-to-yourself outcome and goes in the paper as such.



### 5.3 Why not ARC-3 traces for SFT now



About 10 public games exist locally. Splitting by game leaves roughly 6 train / 1-2 validation / 2-3 test games, so any gain has a very wide interval, and traces may be restricted by competition rules (UNVERIFIED: read the rules before using environment data for training). Revisit only after E1b and U4 produce a per-game baseline.



## 6. Upgrades (runtime first)



Weights are for what the model cannot do. Runtime is for what code can guarantee. Priority order:



| ID | Upgrade | Layer | Gate | Done when |

|---|---|---|---|---|

| U0 | Content-based path discovery (`kaggle_path_discovery.py`) | runtime | before G0 | Run 3 log shows the real mount tree |

| U1 | **Think-then-commit decoding**: free thinking up to `T_think`; if `</think>` not emitted, inject `\n</think>\n\n`; then generate the action under a small grammar (action enum + x,y in 0..63) via a plain-transformers `LogitsProcessor` or a verified offline library wheel | runtime | G1 | parse-valid 100% by construction; truncation recorded separately; unit-tested against the real tokenizer (how it splits digits is `UNVERIFIED`) |

| U2 | Decoding hygiene: temp 0.6, top-p 0.95, no system prompt, `<think>\n` prefix, fixed seeds | runtime | G0 | settings recorded in every telemetry row |

| U3 | Telemetry: per call prompt/think/answer tokens, truncated, parse-valid, latency, seed, commit hash, adapter hash | runtime | G0 | JSONL schema test |

| U4 | Wire `SAMARC3Agent` into the seeded experiment runner (`scripts/arc3_experiment/runner.py`, patch 0003) as an `act(obs) -> (GameAction, data)` arm; compare with the random floor on the same games and seeds, paired CI | runtime | after G0 and the random-agent floor (A0) | per-game, per-seed table with CI |

| U5 | Failure-memory ablation (Arm C): memory on/off, single variable | runtime | after U4 | paired difference reported |

| U6 | Claims linter: fail CI if a markdown file states a percentage or "SOTA" without an `EVIDENCE_REGISTRY` ID | hygiene | any time | test passes on repo |

| U7 | Quarantine hand-written benchmark files (`phd_evaluation_suite.py`, `universal_benchmark_suite.py`): banner + exclude from any public bundle | hygiene | before any public release | bundle manifest check |

| U8 | Pinned environment capture (`pip freeze`, CUDA, driver) per run | reproducibility | G0 | file in every evidence folder |

| U9 | Real sandbox for code execution (container or microVM), model gateway, tool registry | runtime | after U4 | belongs to SAM-AI Code; not now |



## 7. Parameter scaling decision



Decision: **do not scale parameters now.** Reasons: the 14B baseline is unmeasured; no reasoning gap is established; larger models cost latency and memory, and nothing in the repo says they would fix a failure we have not yet observed.



A larger or newer model is tested only if E1/G2 shows a *reasoning* failure and the candidate passes all of: licence permits the intended use; weights obtainable as an offline Kaggle Model; fits memory for inference (and QLoRA if fine-tuning); throughput measured with the same E1c protocol; same harness and decoding. Candidate list must be re-verified on the day (the open-model field changes quickly; Apache-2.0 Qwen-family models are the first to check, `UNVERIFIED` here). No candidate speed or VRAM figure is quoted in this document because none has been measured.



Training a foundation model from scratch is out of scope; a long-term research study would need its own budget and proposal.



## 8. How this maps to the five SAM-AI parts



- **Weights (earned by measurement):** reasoning under a token budget, code and math correctness, self-checking habits.

- **Runtime (guaranteed by code):** action grammar, legality and repeat guards, memory, sandboxed execution, retrieval, telemetry, evaluation harness.

- **Assistant / Code / Research / Multimodal / Lab:** all share one model gateway, tool registry, safety layer, memory and evaluation harness. Build order: harness and telemetry first, then SAM-AI Code (verifiable by tests), then Assistant and Research, then Multimodal with specialist pretrained models, with the Lab running ablations over all of it.



## 9. What would persuade a compute provider or reviewer



A reproducible harness with paired statistics; a gap report of measured failures that runtime fixes did not remove; one small fine-tune with a pre-registered keep/reject and an honest result; a clear budget derived from measured throughput. Not: a headline score from another team's solver, a "100%" from a self-written suite, or an unmeasured throughput table.



## 10. Open items (NEEDS_USER)



1. Approval for each stage separately (G0 run 3, E1, E3, RFT).

2. Adapter licence decision (checkpoint-25 stays local until decided).

3. Read the ARC Prize rules on use of environment data for training and on licences before any ARC-derived data is used.

4. Manual audit of 50 accepted traces (Samrish personally).

5. Confirm whether the published HF adapter file is byte-identical to checkpoint-25 (compare SHA-256).
