#!/usr/bin/env python3
"""
SAM-AI ARC Compositional Reasoning Engine (V5)
==============================================
Parallax Intelligence Lab | Founder: Samrish B

Hypothesis Generator + Structured Search + ArcVerifier in the Loop:
1. Feature & Dimension Rule Extraction
2. Hypothesis Enumeration across Primitive & Composite Transforms
3. Exact Verification against all training pairs (Pruning invalid hypotheses)
4. Occam's Razor Ranking
5. Test-Time Execution
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import sam_ai.arc.transforms as T
from sam_ai.verifiers.arc_verifier import ArcVerifier

Grid = List[List[int]]

class RuleCandidate:
    def __init__(self, name: str, fn: Callable[[Grid], Grid], complexity: int = 1):
        self.name = name
        self.fn = fn
        self.complexity = complexity

class ArcCompositionalEngine:
    """Compositional Program Search for ARC Grid Transformations."""

    def __init__(self):
        self.verifier = ArcVerifier()

    def generate_hypotheses(self, train_pairs: List[Dict]) -> List[RuleCandidate]:
        """Generates candidate hypotheses conditioned on training pairs."""
        candidates = []
        if not train_pairs:
            return candidates

        p0 = train_pairs[0]
        h_in, w_in = len(p0["input"]), len(p0["input"][0])
        h_out, w_out = len(p0["output"]), len(p0["output"][0])
        
        # Family 1: D4 Symmetries (Level 0)
        d4_modes = ["identity", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_transpose"]
        for m in d4_modes:
            candidates.append(RuleCandidate(f"d4_{m}", lambda g, m=m: T.d4_transform(g, m), complexity=1))

        # Family 2: Gravity (Level 1)
        for direction in ["DOWN", "UP", "LEFT", "RIGHT"]:
            candidates.append(RuleCandidate(f"gravity_{direction}", lambda g, d=direction: T.apply_gravity(g, d), complexity=2))

        # Family 3: Crop Bounding Box (Level 1)
        candidates.append(RuleCandidate("crop_non_background", lambda g: T.crop_non_background(g), complexity=2))

        # Family 4: Periodic Tiling & Zoom (Level 2)
        if h_out % h_in == 0 and w_out % w_in == 0 and (h_out != h_in or w_out != w_in):
            sr = h_out // h_in
            sc = w_out // w_in
            candidates.append(RuleCandidate(f"tile_{sr}x{sc}", lambda g, sr=sr, sc=sc: T.tile_grid(g, sr, sc), complexity=2))
            if sr == sc:
                candidates.append(RuleCandidate(f"zoom_{sr}x", lambda g, sr=sr: T.zoom_grid(g, sr), complexity=2))

        # Family 5: Flood Fill Enclosed Regions (Level 2)
        # Check colors present in output
        out_colors = set(c for pair in train_pairs for row in pair["output"] for c in row if c != 0)
        for fc in out_colors:
            candidates.append(RuleCandidate(f"flood_fill_{fc}", lambda g, fc=fc: T.flood_fill_enclosed(g, fc), complexity=3))

        # Family 6: Connect Aligned Dots (Level 2)
        candidates.append(RuleCandidate("connect_aligned_dots", lambda g: T.connect_aligned_dots(g), complexity=3))

        # Family 7: Color Substitution (Level 1)
        # Attempt to induce 1-to-1 color map
        color_map = {}
        consistent_map = True
        for pair in train_pairs:
            inp, out = pair["input"], pair["output"]
            if len(inp) == len(out) and len(inp[0]) == len(out[0]):
                for r in range(len(inp)):
                    for c in range(len(inp[0])):
                        ci, co = inp[r][c], out[r][c]
                        if ci in color_map and color_map[ci] != co:
                            consistent_map = False
                            break
                        color_map[ci] = co
            else:
                consistent_map = False
                break
        if consistent_map and color_map:
            def apply_col_map(g, cmap=color_map):
                return [[cmap.get(c, c) for c in row] for row in g]
            candidates.append(RuleCandidate("color_substitution", apply_col_map, complexity=2))

        # Family 8: Composite (Two-step) Pipelines (Level 3)
        # e.g. Crop then D4, or Flood Fill then Gravity
        for m in ["rot90", "rot180", "rot270", "flip_h", "flip_v"]:
            candidates.append(RuleCandidate(
                f"crop_then_{m}",
                lambda g, m=m: T.d4_transform(T.crop_non_background(g), m),
                complexity=4
            ))

        return candidates

    def solve(self, train_pairs: List[Dict], test_input: Grid) -> Tuple[Optional[Grid], Dict[str, Any]]:
        """
        Executes hypothesis search with ArcVerifier in the loop.
        Returns: (predicted_test_grid, search_metrics)
        """
        candidates = self.generate_hypotheses(train_pairs)
        verifier_calls = 0
        surviving_rules = []

        for rule in candidates:
            rule_valid = True
            for pair in train_pairs:
                inp = pair["input"]
                expected_out = pair["output"]
                verifier_calls += 1
                try:
                    candidate_out = rule.fn(inp)
                    ok, _ = self.verifier.verify_grid(candidate_out, expected_out)
                    if not ok:
                        rule_valid = False
                        break
                except Exception:
                    rule_valid = False
                    break
                    
            if rule_valid:
                surviving_rules.append(rule)

        # Occam's Razor: Select rule with lowest complexity
        if surviving_rules:
            surviving_rules.sort(key=lambda r: r.complexity)
            best_rule = surviving_rules[0]
            try:
                prediction = best_rule.fn(test_input)
            except Exception:
                prediction = [row[:] for row in test_input]
            return prediction, {
                "verified": True,
                "rule_name": best_rule.name,
                "complexity": best_rule.complexity,
                "verifier_calls": verifier_calls,
                "surviving_count": len(surviving_rules)
            }
        else:
            # Fallback to test input
            return [row[:] for row in test_input], {
                "verified": False,
                "rule_name": "fallback_identity",
                "complexity": 999,
                "verifier_calls": verifier_calls,
                "surviving_count": 0
            }
