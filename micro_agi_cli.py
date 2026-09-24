"""Micro-AGI Unified Command-Line Interface.

Entry point supporting:
- chat: Interactive pair-programming & reasoning REPL
- repomap: High-density codebase indexing
- mcp: Autonomous Model Context Protocol server creation
- research: Open-world live web intelligence
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Ensure micro_agi root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.chat.repl import InteractiveAgiREPL
from core.coding.repo_map import RepoMap
from core.coding.code_editor import CodeEditor
from core.mcp.mcp_generator import MCPGenerator, ToolSpec
from core.web_agent import WebResearchAgent


def main():
    parser = argparse.ArgumentParser(
        description="Micro-AGI: Autonomous Cognitive Architecture & Developer Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: chat
    subparsers.add_parser("chat", help="Start interactive conversational coding & reasoning REPL")

    # Command: repomap
    repo_parser = subparsers.add_parser("repomap", help="Render dense AST structural map of repository")
    repo_parser.add_argument("--dir", type=str, default=".", help="Directory to map")

    # Command: mcp
    mcp_parser = subparsers.add_parser("mcp", help="Model Context Protocol commands")
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_subcommand")
    create_mcp = mcp_sub.add_parser("create", help="Generate a standalone MCP server")
    create_mcp.add_argument("name", type=str, help="Name of MCP server")
    create_mcp.add_argument("--out", type=str, default=None, help="Output python file path")

    # Command: research
    res_parser = subparsers.add_parser("research", help="Perform autonomous open-world web research")
    res_parser.add_argument("topic", type=str, help="Research topic or scientific query")

    args = parser.parse_args()

    if args.command == "chat" or not args.command:
        repl = InteractiveAgiREPL()
        repl.start_cli()

    elif args.command == "repomap":
        rm = RepoMap(args.dir)
        rm.scan_repository()
        print(rm.render_map())

    elif args.command == "mcp" and args.mcp_subcommand == "create":
        server_name = args.name
        tools = [
            ToolSpec(
                name="execute_query",
                description="Executes a database or API query",
                parameters={"query": {"type": "string", "description": "Query string"}},
                implementation_code="return f'Executed query: {query}'",
            )
        ]
        out_path = args.out or f"mcp_{server_name.lower()}_server.py"
        MCPGenerator.generate_server_script(server_name, tools, output_file_path=out_path)
        print(f"[SUCCESS] Synthesized MCP server at: {out_path}")

    elif args.command == "research":
        agent = WebResearchAgent()
        report = agent.research_topic(args.topic)
        print(report.executive_synthesis)


if __name__ == "__main__":
    main()
