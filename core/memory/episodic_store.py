"""Episodic Memory Subsystem (Hippocampal Store).

Stores high-dimensional episodic traces of past experiences, decisions,
and reward outcomes. Uses vector similarity retrieval based on random
orthogonal projection hashing and cosine similarity metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import re
import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Episode:
    id: str
    context: str
    goal: str
    action_sequence: List[str]
    outcome: str
    reward: float  # In range [-1.0, 1.0]
    reflection: str = ""
    domain: str = "general"
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "context": self.context,
            "goal": self.goal,
            "action_sequence": self.action_sequence,
            "outcome": self.outcome,
            "reward": self.reward,
            "reflection": self.reflection,
            "domain": self.domain,
            "timestamp": self.timestamp,
        }


class SemanticEmbedder:
    """Deterministic, lightweight semantic embedding engine.

    Maps text to a normalized d-dimensional unit vector using n-gram tokenization
    and stable hash projections. Completely self-contained without external ML runtimes.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec

        # Bag of words + character 3-grams
        tokens = list(words)
        for w in words:
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    tokens.append(w[i:i+3])

        for token in tokens:
            # Deterministic MD5 hash bucket
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec


class EpisodicStore:
    """Hippocampal-inspired fast instance store with cosine similarity retrieval."""

    def __init__(self, embedding_dim: int = 64, max_capacity: int = 1000):
        self.max_capacity = max_capacity
        self.embedder = SemanticEmbedder(dim=embedding_dim)
        self.episodes: List[Episode] = []

    def record_episode(
        self,
        goal: str,
        context: str,
        action_sequence: List[str],
        outcome: str,
        reward: float,
        reflection: str = "",
        domain: str = "general",
    ) -> Episode:
        """Stores a new experiential episode."""
        ep_id = f"ep_{len(self.episodes) + 1}_{int(time.time() * 1000) % 10000}"
        combined_text = f"{goal} {context} {' '.join(action_sequence)} {reflection}"
        emb = self.embedder.embed(combined_text)

        episode = Episode(
            id=ep_id,
            goal=goal,
            context=context,
            action_sequence=action_sequence,
            outcome=outcome,
            reward=reward,
            reflection=reflection,
            domain=domain,
            embedding=emb,
        )

        if len(self.episodes) >= self.max_capacity:
            # Evict lowest-reward or oldest episode
            min_idx = int(np.argmin([e.reward for e in self.episodes]))
            self.episodes.pop(min_idx)

        self.episodes.append(episode)
        return episode

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.1,
        min_reward: Optional[float] = None,
        domain: Optional[str] = None,
    ) -> List[Tuple[Episode, float]]:
        """Retrieves top-k relevant episodes ranked by cosine similarity."""
        if not self.episodes:
            return []

        q_vec = self.embedder.embed(query)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-9:
            return []

        results: List[Tuple[Episode, float]] = []

        for ep in self.episodes:
            if domain is not None and ep.domain != domain:
                continue
            if min_reward is not None and ep.reward < min_reward:
                continue
            if ep.embedding is None:
                continue

            sim = float(np.dot(q_vec, ep.embedding))
            if sim >= min_similarity:
                results.append((ep, sim))

        # Sort descending by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        self.episodes.clear()

    def __len__(self) -> int:
        return len(self.episodes)
