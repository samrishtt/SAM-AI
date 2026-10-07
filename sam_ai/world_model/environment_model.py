#!/usr/bin/env python3
"""
SAM-AI Interactive World Model (V5)
===================================
Parallax Intelligence Lab | Founder: Samrish B

Models interactive environment dynamics (crucial for ARC-AGI-3 & Agent loops):
  Observation (o_t)
        │
        ▼
  State Representation (s_t)
        │
        ▼
  Hypothesis Formulation (h_t)
        │
        ▼
  Predicted State (s_hat_{t+1})
        │
        ▼
  Execute Action (a_t) -> Observe Actual State (s_{t+1})
        │
        ▼
  Compute Prediction Error: Delta = |s_{t+1} - s_hat_{t+1}|
        │
        ▼
  Update World Model Parameters & Transition Hypotheses
"""

import sys
import copy
from typing import Dict, Any, List, Tuple, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

class EnvironmentState:
    def __init__(self, raw_data: Any, step_idx: int = 0):
        self.raw_data = raw_data
        self.step_idx = step_idx
        self.entities = []

class WorldModel:
    """Maintains active hypotheses about environment physics and transitions."""
    
    def __init__(self):
        self.active_hypotheses = []
        self.prediction_history = []
        self.transition_graph = {}
        self.cumulative_error = 0.0

    def formulate_hypothesis(self, current_state: Any, action: str) -> str:
        """Forms a causal hypothesis about what action `a` will cause."""
        hypothesis = f"Action '{action}' causes deterministic state transition"
        if "move" in action.lower():
            hypothesis = f"Action '{action}' shifts agent coordinates by unit delta"
        elif "interact" in action.lower():
            hypothesis = f"Action '{action}' toggles adjacent interactive entity"
        self.active_hypotheses.append(hypothesis)
        return hypothesis

    def predict_next_state(self, current_state: Any, action: str) -> Any:
        """Predicts expected next state before taking an action."""
        # Baseline simulation: deepcopy with projected mutation
        predicted = copy.deepcopy(current_state)
        if isinstance(predicted, dict) and "pos" in predicted:
            if action == "UP":
                predicted["pos"][0] -= 1
            elif action == "DOWN":
                predicted["pos"][0] += 1
            elif action == "LEFT":
                predicted["pos"][1] -= 1
            elif action == "RIGHT":
                predicted["pos"][1] += 1
        return predicted

    def observe_transition(self, s_t: Any, a_t: str, s_next_actual: Any) -> float:
        """Compares actual observation against predicted state, returning error."""
        s_hat = self.predict_next_state(s_t, a_t)
        error = 0.0
        
        # Calculate divergence
        if isinstance(s_hat, dict) and isinstance(s_next_actual, dict):
            if s_hat.get("pos") != s_next_actual.get("pos"):
                error = 1.0
        elif s_hat != s_next_actual:
            error = 1.0

        self.cumulative_error += error
        self.prediction_history.append({
            "action": a_t,
            "predicted": s_hat,
            "actual": s_next_actual,
            "error": error
        })
        
        # Refine world model rule if error observed
        if error > 0:
            correction = f"Transition rule under action '{a_t}' failed; updating obstacle boundary."
            self.active_hypotheses.append(correction)

        return error

if __name__ == "__main__":
    wm = WorldModel()
    s0 = {"pos": [5, 5], "room": 1}
    wm.formulate_hypothesis(s0, "RIGHT")
    pred = wm.predict_next_state(s0, "RIGHT")
    print(f"[*] State at t=0: {s0['pos']} -> Predicted at t=1: {pred['pos']}")
    
    # Simulate an actual state transition (hit a wall at x=5, so did not move)
    s1_actual = {"pos": [5, 5], "room": 1}
    err = wm.observe_transition(s0, "RIGHT", s1_actual)
    print(f"[*] Actual observation: {s1_actual['pos']} -> Prediction Error: {err}")
    print(f"[*] Active World Model Hypotheses: {wm.active_hypotheses[-1]}")
