"""WebWorld: Predictive Mental Sandbox for Web Navigation.

Implements the World-Model-Augmented (WMA) web agent paradigm:
simulates expected informational yield, dead-end risks, and paywall/login
bottlenecks before issuing physical network requests.
"""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

from core.world_model.latent_simulator import WorldModel, WorldState, PredictedTransition


class WebWorldModel(WorldModel):
    """Predictive dynamics model tailored for open-world web navigation."""

    DEAD_END_DOMAINS = {"facebook.com", "instagram.com", "pinterest.com", "tiktok.com"}
    HIGH_CREDIBILITY_DOMAINS = {"wikipedia.org", "arxiv.org", "nature.com", "science.org", "github.com", "nih.gov", "mit.edu"}

    def __init__(self):
        super().__init__()
        self.visited_urls: set[str] = set()

    def simulate_web_action(self, state: Optional[WorldState], action_desc: str) -> PredictedTransition:
        """Predicts the consequence of a web action (search, navigate, extract)."""
        action_clean = action_desc.strip()
        next_state = state.copy() if state is not None else WorldState(state_id="sim_root")
        reward = 0.1
        cost = 0.05
        is_safe = True
        violation = None

        # 1. Simulate Search
        if "search" in action_clean.lower():
            cost = 0.1
            reward = 0.4
            explanation = "Search query expected to yield candidate domain URLs."

        # 2. Simulate Navigation to URL
        elif "navigate" in action_clean.lower() or "http" in action_clean.lower():
            # Extract target URL or domain
            urls = re.findall(r"https?://[^\s'\"]+", action_clean)
            target_url = urls[0] if urls else action_clean

            # Check if already visited
            if target_url in self.visited_urls:
                reward = -0.5
                explanation = "Redundant navigation: URL already in episodic memory."
            # Check dead end / low credibility
            elif any(d in target_url.lower() for d in self.DEAD_END_DOMAINS):
                reward = -0.7
                is_safe = False
                violation = f"Navigating to low-yield social media dead end: {target_url}"
                explanation = "Blocked uninformative domain."
            # Boost high credibility domains
            elif any(d in target_url.lower() for d in self.HIGH_CREDIBILITY_DOMAINS):
                reward = 0.8
                explanation = f"High-yield verified academic/encyclopedic domain: {target_url}"
            else:
                reward = 0.3
                explanation = f"General web navigation to: {target_url}"

        # 3. Simulate Fact Extraction & Cross-Referencing
        elif "extract" in action_clean.lower() or "cross_reference" in action_clean.lower():
            reward = 0.6
            explanation = "Distilling verified knowledge triples into Semantic Graph."

        elif "conclude" in action_clean.lower() or "synthesize" in action_clean.lower():
            reward = 1.0
            explanation = "Final report synthesis with grounded citations."

        else:
            reward = 0.2
            explanation = "General exploratory cognitive action."

        return PredictedTransition(
            next_state=next_state,
            predicted_reward=reward,
            predicted_cost=cost,
            is_safe=is_safe,
            safety_violation=violation,
            confidence=0.88,
            explanation=explanation,
        )
