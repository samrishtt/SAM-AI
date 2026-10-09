# Official ARC Prize 2026 / ARC-AGI-3 Rules & Scoring Analysis
**Agent 3: Competition Rules & Scoring Analyst Report**  
*Document Target*: `micro_agi/research/agent3_rules_and_scoring.md`  
*Analysis Date*: 2026-10-09  
*Author*: Agent 3 (Competition Rules & Scoring Analyst)  

---

## 1. Executive Summary

The **ARC Prize 2026 - ARC-AGI-3** (hosted on Kaggle as `arc-prize-2026-arc-agi-3`) marks a fundamental paradigm shift from static 2D grid transformation (ARC-1/ARC-2) to an **interactive, turn-based inductive reasoning benchmark**. In this track, autonomous agents interact with dynamic, closed-box puzzle environments through discrete action-observation cycles to infer latent world mechanics, formulate goals, and execute solutions.

The competition evaluates systems under **Relative Human Action Efficiency (RHAE)**, a metric designed to quantify "fluid skill-acquisition efficiency." 

Crucially, **internal reasoning compute is 100% free**—reasoning tokens, chain-of-thought (CoT), internal search trees, and simulation steps do not increment the action budget. Conversely, **environment actions are severely penalized via an inverse-squared power law** $\left(\frac{\text{Baseline}}{\text{Actions}}\right)^2$. Consequently, optimal agent design demands maximal compute expenditure prior to each environment action to ensure minimal physical exploration.

---

## 2. Nomenclature & Terminology Disambiguation

| Term | Full Name | Status / Source | Context & Interpretation |
| :--- | :--- | :--- | :--- |
| **RHAE** (Official) | **Relative Human Action Efficiency** | `OFFICIAL` ([docs.arcprize.org/methodology](https://docs.arcprize.org/methodology)) | The definitive metric name codified in the ARC Prize documentation and evaluation runtime. Measures agent action efficiency relative to an empirical human baseline. |
| **RHAE** (Variant / Academic) | **Relative Harmonic Action Efficiency** | `UNVERIFIED / SECONDARY` ([arXiv / Community Discussion](https://arxiv.org)) | A term appearing in academic preprints and forum discussions. Originates from the mathematical property of action efficiency: evaluating reciprocal action rates $\frac{H}{A}$ and calculating aggregate performance across heterogeneous tasks reflects a **harmonic mean regime** over agent actions, penalizing severe single-environment bottlenecks. |

*Finding*: While official documentation designates the metric as **Relative Human Action Efficiency**, the term **Relative Harmonic Action Efficiency** accurately characterizes its mathematical behavior: aggregation across reciprocal action quantities $(1 / A)$ behaves like a harmonic penalty on search inefficiency.

---

## 3. Exact Mathematical Formulation & Scoring Methodology

All scoring formulations below are extracted directly from the official ARC Prize specification ([docs.arcprize.org/methodology.md](https://docs.arcprize.org/methodology.md)) and verified against the runtime harness.

### 3.1. Primitives & Definitions

Let the benchmark consist of a set of games $G = \{g_1, g_2, \dots, g_{|G|}\}$. Each game $g$ contains $N_g$ sequential levels, indexed $l \in \{1, 2, \dots, N_g\}$.

1. **Level Completion ($C_{g, l}$)**:
   $$C_{g, l} \in \{0, 1\}$$
   * $C_{g, l} = 1$ if the agent satisfies the level win condition (`GameState.WIN`).
   * $C_{g, l} = 0$ if the agent triggers a terminal game over, times out, or reaches the action step ceiling without clearing the level.

2. **Discrete Action ($A_{g, l}$)**:
   The total number of environment-altering state interactions submitted by the agent on its exposure to level $l$:
   $$A_{g, l} \in \mathbb{N}_{\ge 1}$$
   * Counts every turn submitting `ACTION1`–`ACTION7` or `RESET`.
   * **Excludes** internal tool calls, reasoning steps, forward passes, MCTS expansions, and HTTP retries.

3. **Human Baseline ($H_{g, l}$)**:
   Established through controlled testing of first-time human players who have never seen the game:
   $$H_{g, l} = \text{UpperMedian}\left(\{h_{1}, h_{2}, \dots, h_{M}\}\right)$$
   where $h_1 \le h_2 \le \dots \le h_M$ are the action counts of $M$ human participants completing level $l$.
   * **Upper Median Definition**:
     * If $M$ is odd: $H_{g, l} = h_{\frac{M + 1}{2}}$.
     * If $M$ is even: $H_{g, l} = h_{\frac{M}{2} + 1}$ (the upper of the two middle entries; e.g., if $M = 4$, third place is selected).
   * **Design Justification**: Eliminates lucky speed-run outliers, grounds the baseline in typical competent human problem-solving, and stabilizes variances across test cohorts.

---

### 3.2. Per-Level Scoring Formula

For each level $l$ completed by the AI ($C_{g, l} = 1$):

$$\text{RawLevelScore}(g, l) = \left( \frac{H_{g, l}}{A_{g, l}} \right)^2$$

If $C_{g, l} = 0$, $\text{RawLevelScore}(g, l) = 0$.

#### Inverse-Quadratic Penalty Curve
The squaring operation deliberately enforces a steep power-law penalty against brute-force trial-and-error:
* If $A_{g, l} = H_{g, l} \implies \text{Score} = (1.0)^2 = 1.00$ (100.0%)
* If $A_{g, l} = 1.5 \times H_{g, l} \implies \text{Score} = (1 / 1.5)^2 \approx 0.444$ (44.4%)
* If $A_{g, l} = 2.0 \times H_{g, l} \implies \text{Score} = (1 / 2.0)^2 = 0.250$ (25.0%)
* If $A_{g, l} = 5.0 \times H_{g, l} \implies \text{Score} = (1 / 5.0)^2 = 0.040$ (4.0%)
* If $A_{g, l} = 10.0 \times H_{g, l} \implies \text{Score} = (1 / 10.0)^2 = 0.010$ (1.0%)
* If $A_{g, l} = 50.0 \times H_{g, l} \implies \text{Score} = (1 / 50.0)^2 = 0.0004$ (0.04%)

*Takeaway*: Taking just double the human action count destroys 75% of the points for that level. Taking 10x the steps collapses the score to near zero.

#### Per-Level Score Cap
To prevent superhuman shortcut exploits on trivial levels from skewing aggregate game scores, each level is capped at **1.15x**:

$$S_{g, l} = \min\left( 1.15, \, \left( \frac{H_{g, l}}{A_{g, l}} \right)^2 \right) \cdot C_{g, l}$$

---

### 3.3. Per-Game Aggregation & Strict Completion Gating

The overall score for game $g$ is computed as the **linearly weighted average** across all levels, weighted by the 1-indexed level index $l$:

$$S_g = \frac{\sum_{l=1}^{N_g} l \cdot S_{g, l}}{\sum_{l=1}^{N_g} l} = \frac{\sum_{l=1}^{N_g} l \cdot S_{g, l}}{\frac{N_g(N_g + 1)}{2}}$$

#### Mathematical Properties of Level Weighting
1. **Tutorial Underweighting / Mastery Overweighting**: Early tutorial levels (Level 1, Level 2) introduce simple mechanics and receive small weights ($l = 1, 2$). Deep levels require complex rule synthesis and receive high weights ($l = N_g$).
2. **Strict Completion Ceiling**: The maximum achievable game score is strictly bounded by the number of levels completed. If an agent completes only levels $1 \dots k$ ($k < N_g$), then:

$$\text{MaxGameScore}(k) = \frac{\sum_{l=1}^{k} l \cdot 1.15}{\frac{N_g(N_g + 1)}{2}} = 1.15 \cdot \frac{k(k + 1)}{N_g(N_g + 1)}$$

Under nominal human-parity efficiency ($S_{g, l} = 1.0$):

$$\text{NominalMaxGameScore}(k) = \frac{k(k + 1)}{N_g(N_g + 1)}$$

#### Concrete Derivation Example (5-Level Game)
Consider a game with $N_g = 5$ levels. Total weight denominator:
$$\sum_{l=1}^5 l = 1 + 2 + 3 + 4 + 5 = 15$$

* If the agent solves Levels 1 through 4 with human parity ($S_{g, l} = 1.0$), but fails Level 5 ($C_{g, 5} = 0$):
  $$S_g = \frac{1(1.0) + 2(1.0) + 3(1.0) + 4(1.0) + 5(0.0)}{15} = \frac{10}{15} \approx 66.7\%$$
  Even with perfect 1.15 cap on Levels 1–4:
  $$\text{Max } S_g = \frac{10 \times 1.15}{15} = \frac{11.5}{15} \approx 76.67\%$$
* If the agent solves Level 5 alone (e.g. by shortcut):
  $$S_g = \frac{5(1.0)}{15} \approx 33.3\%$$

*Implication*: To attain high scores, an agent **must complete all levels**. A solver that achieves 100% efficiency on 3 out of 5 levels will score lower than a solver that completes all 5 levels at 60% efficiency.

---

### 3.4. Aggregate Leaderboard Scoring

The global leaderboard score $S_{\text{total}}$ is the arithmetic mean of all game scores across the evaluated benchmark set $G$:

$$S_{\text{total}} = \frac{1}{|G|} \sum_{g \in G} S_g \times 100\%$$

* **Competition Mode Rule**: $G$ encompasses **all available environments** in the competition arcade. If an agent fails to interact with a game or errors out, $S_g = 0.0$ for that game.
* **Score Domain**: $S_{\text{total}} \in [0.0\%, 115.0\%]$. Real-world top competitors currently score in the 25%–35% range.

---

### 3.5. Harmonic Weighting Dynamics & Rate Interpretation

Why is RHAE associated with "Harmonic Efficiency"?

Consider efficiency as an action rate:
$$R_{g, l} = \frac{H_{g, l}}{A_{g, l}}$$
The harmonic mean of execution rates across $K$ tasks is defined as:
$$\mathcal{H}(R_1, \dots, R_K) = \frac{K}{\sum_{i=1}^K \frac{1}{R_i}} = \frac{K}{\sum_{i=1}^K \frac{A_i}{H_i}}$$

Notice that the denominator $\sum \frac{A_i}{H_i}$ is the total normalized action consumption. If an agent "flails" or enters an infinite loop in a single level ($A_i \to \infty$), the denominator explodes and the harmonic score collapses to 0. 

The RHAE formula uses an inverse quadratic $(H/A)^2 = R^2$, which is an even more aggressive penalty than a standard harmonic mean. It converts any action inefficiency into an immediate, irreversible point loss.

---

## 4. Environment Action Space & Cost Mechanics

### 4.1. Action Inventory

All ARC-AGI-3 environments expose a standardized action set of `RESET` plus up to 7 numbered actions ([docs.arcprize.org/actions.md](https://docs.arcprize.org/actions.md)):

| Action | Canonical Semantics | Cost to Action Budget | Notes & Constraints |
| :--- | :--- | :--- | :--- |
| `RESET` | Level / Game Restart | **1 Action Step** | In Competition Mode, only **Level Resets** are permitted; full Game Resets degrade to Level Resets. |
| `ACTION1` | Directional Up | **1 Action Step** | Simple action (no arguments). |
| `ACTION2` | Directional Down | **1 Action Step** | Simple action (no arguments). |
| `ACTION3` | Directional Left | **1 Action Step** | Simple action (no arguments). |
| `ACTION4` | Directional Right | **1 Action Step** | Simple action (no arguments). |
| `ACTION5` | Interact / Rotate / Select | **1 Action Step** | Varies by puzzle semantics (e.g. toggle switch, pick item). |
| `ACTION6` | Spatial Coordinate Action | **1 Action Step** | Complex action requiring `{"x": int, "y": int}`, with $0 \le x, y \le 63$. |
| `ACTION7` | Undo | **1 Action Step** | Reverts the previous state transition if supported by the game engine. |

### 4.2. Game States & The Game-Over Trap

Each returned frame exposes a `GameState`:
* `NOT_PLAYED`: Initial unstarted state.
* `NOT_FINISHED`: Game in progress, awaiting next action.
* `WIN`: Level/game completed successfully.
* `GAME_OVER`: Terminal failure state (hazard collision, step budget exhausted).

> [!CAUTION]
> **The Game-Over Trap (HTTP 400)**:  
> When a game enters `GAME_OVER`, **only `RESET` is valid**. Sending any other action (`ACTION1` through `ACTION7`) raises an immediate `HTTP 400 Bad Request`. An unhandled 400 error crashes the notebook evaluation loop. Agents must inspect `obs.state == GameState.GAME_OVER` and immediately dispatch `RESET`.

### 4.3. Competition Mode Constraints

Under `OperationMode.COMPETITION` ([docs.arcprize.org/toolkit/competition_mode.md](https://docs.arcprize.org/toolkit/competition_mode.md)):
1. **Gateway Mediation**: All environment interactions are executed via the Kaggle Gateway REST API (`arc_base_url`).
2. **Global Penalty for Skipped Games**: All games exposed by Arcade are included in the denominator $|G|$.
3. **No In-Flight Scoring**: Calling `get_scorecard()` during execution is blocked. Agents cannot inspect their live benchmark score to calibrate boldness.
4. **Single Session per Environment**: Calling `make()` on an environment is permitted only once per scorecard.
5. **Scorecard Lifecycle**: Scorecards auto-close after 15 minutes of idle time. The agent must continuously interact or finalize the scorecard via `/scorecard/close`.

---

## 5. Solver Design Implications: The "Free Reasoning, Costly Action" Theorem

The mathematical structure of RHAE dictates three foundational architectural principles for our solver:

```
+-------------------------------------------------------------------------+
|                          COMPUTE VS ACTION MATRIX                       |
+-------------------------------------------------------------------------+
|  Internal LLM Forward Passes (Tokens)          -->  COST: 0 ACTIONS     |
|  Chain-of-Thought Deliberation                 -->  COST: 0 ACTIONS     |
|  Monte Carlo Tree Search (Simulations)         -->  COST: 0 ACTIONS     |
|  A* / BFS Pathfinding Computations            -->  COST: 0 ACTIONS     |
|  Frame Delta Invariant Analysis                -->  COST: 0 ACTIONS     |
+-------------------------------------------------------------------------+
|  Single Environment Step (Move/Click)          -->  COST: 1 ACTION      |
|  Accidental Wall Bump / Oscillation            -->  COST: 1 ACTION      |
|  Blind Exploration Step                        -->  COST: 1 ACTION      |
|  Action Undo (ACTION7)                         -->  COST: 1 ACTION      |
|  Level Reset (RESET)                           -->  COST: 1 ACTION      |
+-------------------------------------------------------------------------+
```

### 5.1. Principle 1: Reasoning is Completely Free

* **Official Specification**: *"Internal operations that do not alter the environment (tool calls, reasoning steps, retries) are not counted as actions."* ([docs.arcprize.org/methodology.md](https://docs.arcprize.org/methodology.md)).
* **Reasoning Payload Injection**: The SDK allows passing arbitrary diagnostic metadata without scoring impact:
  ```python
  env.step(GameAction.ACTION1, reasoning={"thought": "...", "confidence": 0.95, "tokens": 4096})
  ```
* **Architectural Mandate**: 
  * Spend maximum compute FLOPs before submitting an action.
  * Perform deep visual feature extraction, frame-delta diffing, dynamic agent tracking, and multi-step lookahead in software.
  * Never use the environment as an exploratory scratchpad.

### 5.2. Principle 2: Environment Actions are Extremely Expensive

* Due to $(H/A)^2$, every non-essential action degrades the level score exponentially.
* **Oscillation Elimination**: A solver that oscillates between two cells $(r, c) \leftrightarrow (r, c+1)$ for 10 steps incurs a catastrophic efficiency drop.
* **Deterministic Macro Paths**: Instead of step-by-step guessing, identify verified objectives (key, door, switch) and compute deterministic BFS / A* shortest paths, executing the entire action sequence without exploratory detour.

### 5.3. Principle 3: Undo (`ACTION7`) and Reset (`RESET`) Economics

* **Undo is Not Free**: `ACTION7` consumes an action step in the level count $A_{g, l}$. An undo costs the step that made the mistake PLUS the step that reverted it (2 steps wasted).
* **When to Undo**: Only trigger `ACTION7` if the previous move caused irreversible damage (e.g. loss of health/energy) or trapped the avatar in a non-navigable configuration.
* **When to Reset**: Level reset restarts the level step counter or adds to the level action total depending on the game harness. Resets should be reserved strictly for unrecoverable failure states (`GAME_OVER` or dead-end traps).

---

## 6. Submission & Execution Constraints (Kaggle Container)

Analysis of competition rules ([Kaggle Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules)) and primary repository configurations (`kernel-metadata.json`, `arc-agi-3-milestone-2-solution.ipynb`):

### 6.1. Runtime Budget & Hardware Limits

| Parameter | Constraint | Operational Rule |
| :--- | :--- | :--- |
| **Execution Limit** | **9 Hours (540 minutes)** | Hard wall-clock limit enforced by Kaggle backend. The runner must implement a `soft_end_time` (e.g. 10–15 min before 9h) to guarantee graceful teardown. |
| **Internet Access** | **DISABLED (`enable_internet: false`)** | Completely offline container. No remote API calls (OpenAI, Anthropic, Gemini are prohibited). All models and wheels must be pre-bundled as Kaggle Datasets. |
| **Compute Accelerator** | **Nvidia RTX Pro 6000 (48GB VRAM)** | Provided exclusively for ARC-AGI-3 (`g4-standard-48`). High VRAM allows running 8B–14B quantized LLMs (e.g. Qwen 3.8B/7B/14B, GPT-OSS) locally with KV caching. |
| **CPU Fallback** | 12–16 vCPUs, ~60GB Host RAM | Available host resources for background Python BFS/MCTS and image processing. |

### 6.2. Submission Artifact Format: `submission.parquet`

The required output artifact is a Parquet file located at `/kaggle/working/submission.parquet`.

#### Official Dataframe Schema

```python
import pandas as pd

submission_df = pd.DataFrame(
    columns=["row_id", "game_id", "end_of_game", "score"],
    data=[
        ["1_0", "game_xyz", True, 0.85],
        # ...
    ]
)
submission_df.to_parquet("/kaggle/working/submission.parquet", index=False)
```

* **`row_id`** (String): Unique identifier per record (e.g. `"<game_idx>_<level_idx>"`).
* **`game_id`** (String): Canonical game ID string.
* **`end_of_game`** (Boolean): Indicates whether the game session concluded.
* **`score`** (Float): Recorded performance metric / level score for that game.

#### The Placeholder Parquet Pattern (Survival Defense)
Because Kaggle will fail a 9-hour run with `Submission Error` if `submission.parquet` is missing upon unexpected timeout or OOM crash, the solver notebook **must initialize a valid placeholder `submission.parquet` at startup**:

```python
# Cell 4 / Startup Hook: Write valid placeholder immediately
import pandas as pd
from pathlib import Path

WORKING_DIR = Path("/kaggle/working")
placeholder_df = pd.DataFrame(
    [["1_0", "1", True, 1]],
    columns=["row_id", "game_id", "end_of_game", "score"]
)
placeholder_df.to_parquet(WORKING_DIR / "submission.parquet", index=False)
```
During benchmark completion, the real results overwrite this placeholder.

### 6.3. Two-Phase Kaggle Execution Model

1. **Phase A (Save & Run All / Commit)**:
   * Kaggle executes the notebook on public/demo environments.
   * `KAGGLE_IS_COMPETITION_RERUN` is empty/false.
   * `TRUE_SUBMISSION = False`. Minimal diagnostics or full logs are generated.
2. **Phase B (Competition Rerun - Scored)**:
   * Triggered when clicking **Submit to Competition**.
   * Kaggle restarts the kernel with `KAGGLE_IS_COMPETITION_RERUN="true"`.
   * Gateway exposes hidden test games.
   * Minimal diagnostics mode enabled (suppresses heavy HTML/JSON logging to save disk I/O and latency).
   * Benchmark plays all hidden environments and saves final `submission.parquet`.

---

## 7. Multi-Pass Scoring & Leaderboard Empirical Evidence

### 7.1. Kaggle Best-of-4 Evaluation Mechanic

*Primary Code Evidence*: `micro_agi/scripts/parse_v4_scores.py` and `predictions/v4_output/.per_game_cards.pkl`.

In competition evaluation, each environment is played across **4 separate passes** (attempts):
$$\text{Official Game Score}(g) = \max_{p \in \{1, 2, 3, 4\}} S_g^{(p)}$$
The Kaggle leaderboard displays the mean of these **best passes**:
$$S_{\text{Leaderboard}} = \frac{1}{|G|} \sum_{g \in G} \max\left(S_g^{(1)}, S_g^{(2)}, S_g^{(3)}, S_g^{(4)}\right)$$

*Strategic Value*: 
* Pass 1 can be dedicated to careful exploration and invariant mapping.
* Passes 2–4 can exploit the learned rules to execute direct, minimal-action trajectories that maximize $\left(\frac{H}{A}\right)^2$.

### 7.2. Verified Historical Standing (Team Parallax / Samrish B)

*Source*: `micro_agi/evidence/SCORE_RECORD.md` (Generated from Kaggle API outputs on 2026-10-08):

| Metric | ARC-AGI-3 (Interactive Track) | ARC-AGI-2 (Static Track) |
| :--- | :--- | :--- |
| **Competition ID** | `arc-prize-2026-arc-agi-3` | `arc-prize-2026-arc-agi-2` |
| **All-Time Best Public Score** | **29.21** (Submission `56804468`, 2026-10-03) | **30.56** (Submission `55151535`, 2026-08-01) |
| **Second Best Score** | **28.29** (Submission `56866777`, 2026-10-06) | 30.14 (Submission `54210413`, 2026-06-30) |
| **Recent October Scores** | 26.04 (`56919253`), 25.76 (`56926771`) | 29.72 (`56038491`) |
| **Verified Leaderboard Rank** | **Rank 385 / 3,976 teams** (Top 9.7%) | **Rank 641 / 2,558 teams** (Top 25.0%) |

---

## 8. Summary Table: Solver Optimization Directives

| Rule / Constraint | Mathematical / Mechanical Reality | Solver Actionable Strategy |
| :--- | :--- | :--- |
| **RHAE Metric** | $S = \min\left(1.15, \left(\frac{H}{A}\right)^2\right)$ | **Minimize actions at all costs.** Never explore blindly. One wasted step loses quadratic points. |
| **Level Weighting** | Weights $w_l = l$. Completing $k < N$ levels caps max score at $\frac{k(k+1)}{N(N+1)}$. | **Prioritize game completion.** An agent must clear all levels to unlock the 100% score ceiling. |
| **Reasoning Budget** | Compute, LLM tokens, and local search do not count as actions. | **Inference-Heavy Architecture.** Use local Qwen/BFS/MCTS before every move to verify validity. |
| **Game Over State** | Sending any action other than `RESET` after game over returns HTTP 400. | **Immediate Reset Guard.** Catch `GameState.GAME_OVER` and dispatch `RESET` without calling model. |
| **Kaggle Environment** | 9-hour limit, offline container, Nvidia RTX Pro 6000. | **Pre-bundle all weights/wheels.** Track `soft_end_time` (8h 45m). Write placeholder `submission.parquet`. |
| **Multi-Pass Scoring** | Official score is Best-of-4 per game. | **Two-Phase Strategy.** Pass 1: discover mechanics. Passes 2–4: execute shortest-path exploit runs. |

---

## 9. Comprehensive Citations & Verification Ledger

All claims in this report are categorized using the strict evidentiary ledger below:

1. **[OFFICIAL]** ARC-AGI-3 Scoring Methodology  
   *URL*: `https://docs.arcprize.org/methodology.md`  
   *Access Date*: 2026-10-09  
   *Verified Facts*: Exact RHAE formula, Upper median human baseline, 1.15x per-level cap, 1-indexed level weighting, game completion capping.

2. **[OFFICIAL]** ARC-AGI-3 Actions & Input Interface  
   *URL*: `https://docs.arcprize.org/actions.md`  
   *Access Date*: 2026-10-09  
   *Verified Facts*: 8 discrete actions (`RESET`, `ACTION1`–`ACTION7`), keyboard mappings, HTTP 400 on post-game-over action.

3. **[OFFICIAL]** Competition Mode Specification  
   *URL*: `https://docs.arcprize.org/toolkit/competition_mode.md`  
   *Access Date*: 2026-10-09  
   *Verified Facts*: Prohibition of full game resets (degrade to level resets), single `make()` interaction per game, global arcade scoring denominator, blocking of in-flight scorecards.

4. **[OFFICIAL]** ARC Prize 2026 Starter & Kaggle Integration  
   *URL*: `https://docs.arcprize.org/arc-prize-2026.md`  
   *Access Date*: 2026-10-09  
   *Verified Facts*: `submission.parquet` requirement, RTX Pro 6000 hardware reservation, offline internet constraint, Python 3.12 requirement.

5. **[OFFICIAL]** ARC-AGI-3 Scorecards & API Lifecycle  
   *URL*: `https://docs.arcprize.org/scorecards.md`  
   *Access Date*: 2026-10-09  
   *Verified Facts*: 15-minute scorecard auto-close timeout, scorecard schema fields (`card_id`, `score`, `environments`).

6. **[OFFICIAL / PRIMARY CODE]** Kaggle Notebook Kernel Metadata & Submission Harness  
   *File Path*: `micro_agi/notebooks/samrish_solver_v7/kernel-metadata.json`, `arc-agi-3-milestone-2-solution.ipynb`  
   *Access Date*: 2026-10-09  
   *Verified Facts*: Competition ID `arc-prize-2026-arc-agi-3`, `enable_internet: false`, `machine_shape: NvidiaRtxPro6000`, `TRUE_SUBMISSION` flag check via `KAGGLE_IS_COMPETITION_RERUN`, placeholder parquet fallback schema.

7. **[PRIMARY CODE]** Kaggle API Submissions Log & Score Record  
   *File Path*: `micro_agi/evidence/SCORE_RECORD.md`, `evidence/arc_agi_3_submissions_raw.txt`  
   *Access Date*: 2026-10-09  
   *Verified Facts*: Top score 29.21 (Submission `56804468`), Rank 385 / 3,976 on ARC-3, Rank 641 / 2,558 on ARC-2.

8. **[PRIMARY CODE]** Multi-Pass Diagnostic Log & Scoring Parser  
   *File Path*: `micro_agi/scripts/parse_v4_scores.py`, `predictions/v4_output/.per_game_cards.pkl`  
   *Access Date*: 2026-10-09  
   *Verified Facts*: Kaggle evaluates 4 passes per game and banks the maximum score pass.

9. **[COMMUNITY / SECONDARY - UNVERIFIED]** "Relative Harmonic Action Efficiency" Alternative Terminology  
   *Sources*: Academic preprints and community discussions on ARC-AGI-3 rate aggregations.  
   *Verification Status*: **UNVERIFIED AS OFFICIAL NAME**. Confirmed that official documentation designates the metric as "Relative Human Action Efficiency", while the term "Relative Harmonic Action Efficiency" describes the harmonic rate properties and inverse-quadratic aggregation behavior.
