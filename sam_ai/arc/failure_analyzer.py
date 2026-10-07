#!/usr/bin/env python3
"""
SAM-AI ARC Failure Classification & Diagnostic Engine (V5)
===========================================================
Parallax Intelligence Lab | Founder: Samrish B

Analyzes failed ARC predictions against ground truth to categorize
failure modes and generate structured data for targeted RLVR / SFT:
Categories:
- dimension_expansion: Target grid dimensions differ from input (e.g. scale, crop, tile)
- color_distribution: Colors present in target do not exist in candidate
- object_relation: Target requires inter-object spatial reasoning (alignment, containment)
- pattern_tiling: Alternating subgrid or Kronecker mosaic pattern
- spatial_inversion: Negative space / background inversion
- complex_composite: Multi-step composition unreached by search depth
"""

import sys
import json
from typing import Dict, Any, List, Tuple
from sam_ai.arc.representation import GridRepresentation

Grid = List[List[int]]

class ArcFailureAnalyzer:
    """Classifies the root cause of an ARC task prediction failure."""

    @staticmethod
    def classify_failure(
        task_id: str,
        train_pairs: List[Dict],
        test_input: Grid,
        predicted: Grid,
        ground_truth: Grid
    ) -> Dict[str, Any]:
        h_pred, w_pred = len(predicted), len(predicted[0]) if predicted else 0
        h_gt, w_gt = len(ground_truth), len(ground_truth[0]) if ground_truth else 0
        h_in, w_in = len(test_input), len(test_input[0]) if test_input else 0

        # 1. Dimension Mismatch
        if (h_pred, w_pred) != (h_gt, w_gt):
            if h_gt % h_in == 0 and w_gt % w_in == 0 and (h_gt != h_in or w_gt != w_in):
                cat = "dimension_expansion_factor"
                detail = f"Target is factor scaled ({h_gt // h_in}x{w_gt // w_in}), but candidate predicted ({h_pred}x{w_pred})"
            elif h_gt < h_in or w_gt < w_in:
                cat = "dimension_cropping"
                detail = f"Target is cropped ({h_gt}x{w_gt}) from input ({h_in}x{w_in})"
            else:
                cat = "dimension_arbitrary"
                detail = f"Target dimension ({h_gt}x{w_gt}) differs from input ({h_in}x{w_in})"
            return {
                "task_id": task_id,
                "category": cat,
                "primary_failure": "dimension_mismatch",
                "detail": detail
            }

        # 2. Color Space Divergence
        pred_colors = set(c for r in predicted for c in r)
        gt_colors = set(c for r in ground_truth for c in r)
        missing_colors = gt_colors - pred_colors
        if missing_colors:
            return {
                "task_id": task_id,
                "category": "color_space_divergence",
                "primary_failure": "color_generation",
                "detail": f"Target contains colors {list(missing_colors)} absent from candidate"
            }

        # 3. Object-level Spatial Structure
        rep_gt = GridRepresentation(ground_truth)
        rep_pred = GridRepresentation(predicted)
        
        if len(rep_gt.objects) != len(rep_pred.objects):
            return {
                "task_id": task_id,
                "category": "object_count_mismatch",
                "primary_failure": "object_segmentation",
                "detail": f"Target has {len(rep_gt.objects)} discrete objects, candidate produced {len(rep_pred.objects)}"
            }

        # 4. Pixel-level Arrangement
        diff_count = sum(
            1 for r in range(h_gt) for c in range(w_gt)
            if predicted[r][c] != ground_truth[r][c]
        )
        total_pixels = h_gt * w_gt
        error_ratio = round(diff_count / total_pixels, 3)

        return {
            "task_id": task_id,
            "category": "spatial_arrangement_error",
            "primary_failure": "transformation_inference",
            "detail": f"{diff_count}/{total_pixels} pixels mismatched ({error_ratio*100:.1f}%)"
        }

if __name__ == "__main__":
    t_in = [[1, 2], [3, 4]]
    pred = [[1, 2], [3, 4]]
    gt = [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
    res = ArcFailureAnalyzer.classify_failure("sample_001", [], t_in, pred, gt)
    print(f"[*] Sample Failure Analysis: {res}")
