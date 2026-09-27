"""Unit tests for SAM-AI Task-Conditioned A11y Compressor."""

import pytest
from sam_ai.agents.task_conditioned_a11y import TaskConditionedA11yCompressor


def test_task_conditioned_ranking_preserves_target():
    """
    Demonstrates that task-conditioned ranking prioritizes elements matching the task goal,
    even if they appear late (e.g. at index 100) in the raw DOM tree.
    """
    compressor = TaskConditionedA11yCompressor(max_tokens_budget=10)

    # Construct 120 raw elements where the first 80 are irrelevant desktop icons/labels,
    # and element 105 is the target "Save Document As" button.
    raw_elements = []
    for i in range(120):
        raw_elements.append({
            "role": "button",
            "name": f"Generic Toolbar Tool {i}",
            "bbox": [10, 10 + i * 5, 20, 20],
        })

    # The actual task target
    raw_elements[105] = {
        "role": "button",
        "name": "Save Document As PDF",
        "bbox": [200, 300, 100, 30],
    }

    # Naive slicing [:10] would completely drop element 105!
    # Task-conditioned compression must rank it at the top.
    compressed = compressor.compress(
        raw_elements=raw_elements,
        task_description="Please click Save Document As PDF to export the report.",
        active_window_title="LibreOffice Writer",
    )

    assert len(compressed) <= 10
    top_element = compressed[0]
    assert top_element.name == "Save Document As PDF"
    assert top_element.relevance_score > 3.0


def test_invisible_and_zero_geometry_filtering():
    """Verifies that off-screen and zero-size elements are pruned."""
    compressor = TaskConditionedA11yCompressor(max_tokens_budget=10)
    raw_elements = [
        {"role": "button", "name": "Zero Width", "bbox": [0, 0, 0, 50]},
        {"role": "button", "name": "Off Screen", "bbox": [-500, 10, 50, 50]},
        {"role": "button", "name": "Valid Button", "bbox": [50, 50, 80, 30]},
    ]

    compressed = compressor.compress(raw_elements, task_description="Click valid button")
    assert len(compressed) == 1
    assert compressed[0].name == "Valid Button"
