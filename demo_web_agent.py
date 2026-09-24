"""Interactive Demonstration of Micro-AGI in the Open-World Web Environment.

Demonstrates autonomous web research, counterfactual trajectory evaluation,
HTML sanitization, and live knowledge graph distillation.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Ensure micro_agi root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.web_agent import WebResearchAgent, GroundedResearchReport


def print_banner():
    banner = """
==============================================================================
   __  __ _                     _    ____ ___ 
  |  \/  (_) ___ _ __ ___      / \  / ___|_ _|
  | |\/| | |/ __| '__/ _ \    / _ \| |  _ | | 
  | |  | | | (__| | | (_) |  / ___ \ |_| || | 
  |_|  |_|_|\___|_|  \___/  /_/   \_\____|___|
                                              
  Autonomous Open-World Web Cognitive Architecture
  Grounded in Kahneman's Dual-Process Theory & WebWorld Latent Modeling
==============================================================================
"""
    print(banner)


def run_demo(topic: str = "Artificial General Intelligence"):
    print_banner()
    print(f"[*] Target Research Objective: {topic}\n")

    agent = WebResearchAgent(max_search_depth=3)

    print("[*] Stage 1: Initializing Working Memory & Goal Hierarchy...")
    print(f"    Active Goal Stack: [Root: Investigate '{topic}']")
    print(f"    Memory Subsystems: Episodic Store (0 traces) | Semantic Graph (0 axioms)")
    print("-" * 78)

    print("[*] Stage 2: Autonomous Web Navigation & Latent Trajectory Evaluation...")
    report: GroundedResearchReport = agent.research_topic(topic)

    print("\n" + "=" * 78)
    print("                    AUTONOMOUS RESEARCH INTELLIGENCE REPORT")
    print("=" * 78)
    print(f"Research Topic    : {report.query}")
    print(f"Execution Latency : {report.execution_time_sec:.2f} seconds")
    print(f"Domains Consulted : {len(report.sources_consulted)}")
    print(f"Triples Induced   : {report.knowledge_triples_induced}")
    print("-" * 78)

    print("\n[+] Consulted Authoritative Web Sources:")
    for idx, s in enumerate(report.sources_consulted, 1):
        print(f"    [{idx}] {s['title']}")
        print(f"        URL    : {s['url']}")
        print(f"        Snippet: {s['snippet']}")

    print("\n[+] Distilled Factual Assertions:")
    for idx, fact in enumerate(report.key_findings, 1):
        print(f"    {idx}. {fact}")

    print("\n[+] Consolidated Semantic Knowledge Graph Triples:")
    triples = agent.semantic_graph.get_all_facts()
    for t in triples[:6]:
        print(f"    -> {t}")

    print("\n" + "=" * 78)
    print(" [COMPLETED] Epistemic Grounding and Memory Consolidation Succeeded.")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Micro-AGI Web Research Agent Demo")
    parser.add_argument(
        "--topic",
        type=str,
        default="Artificial General Intelligence",
        help="Research topic query to investigate on the live web.",
    )
    args = parser.parse_args()
    run_demo(args.topic)
