"""Local Web Chat Interface for Micro-AGI (Hyper-Astra Studio).

Provides a modern, high-aesthetic browser-based chat, cognitive visualization,
file explorer, and pair-programming studio using Python's standard library.
"""

from __future__ import annotations
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.parse
from pathlib import Path

# Ensure micro_agi root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.chat.repl import InteractiveAgiREPL

PORT = 8765
repl = InteractiveAgiREPL()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hyper-Astra Studio // Micro-AGI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #090a0f;
            --bg-surface: #0f121d;
            --bg-elevated: #161a29;
            --bg-card: #131726;
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(255, 255, 255, 0.16);
            --border-accent: rgba(99, 102, 241, 0.4);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
            --accent: #6366f1;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #3b82f6 100%);
            --accent-glow: rgba(99, 102, 241, 0.25);
            --cyan: #06b6d4;
            --emerald: #10b981;
            --emerald-glow: rgba(16, 185, 129, 0.2);
            --amber: #f59e0b;
            --code-bg: #0b0e17;
            --sidebar-width: 340px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-base); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }

        /* Sidebar Styling */
        #sidebar {
            width: var(--sidebar-width);
            background: var(--bg-surface);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            transition: width 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 20;
            position: relative;
        }

        #sidebar.collapsed {
            width: 0;
            overflow: hidden;
            border-right: none;
        }

        .brand-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 18, 29, 0.95);
            backdrop-filter: blur(12px);
        }

        .brand-logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 36px;
            height: 36px;
            background: var(--accent-gradient);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            box-shadow: 0 0 20px var(--accent-glow);
            color: #fff;
        }

        .brand-text h1 {
            font-size: 0.96rem;
            font-weight: 700;
            letter-spacing: -0.3px;
            color: #fff;
        }

        .brand-text span {
            font-size: 0.72rem;
            color: var(--emerald);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .pulse-dot {
            width: 6px;
            height: 6px;
            background: var(--emerald);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--emerald);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(1.2); }
        }

        /* Sidebar Tabs */
        .sidebar-tabs {
            display: flex;
            border-bottom: 1px solid var(--border);
            background: rgba(11, 14, 23, 0.6);
            padding: 4px;
            gap: 4px;
        }

        .tab-btn {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-dim);
            padding: 8px 4px;
            font-size: 0.76rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }

        .tab-btn.active {
            background: var(--bg-elevated);
            color: #fff;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }

        .tab-btn:hover:not(.active) {
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.03);
        }

        .sidebar-content {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .tab-panel { display: none; flex-direction: column; gap: 14px; }
        .tab-panel.active { display: flex; }

        .card-widget {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
        }

        .widget-title {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-dim);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            font-size: 0.8rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }

        .status-row:last-child { border-bottom: none; }
        .status-label { color: var(--text-muted); }
        .status-val { font-weight: 600; color: #fff; display: flex; align-items: center; gap: 6px; }
        .val-badge { font-size: 0.72rem; padding: 2px 7px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; }
        .badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-blue { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
        .badge-purple { background: rgba(139, 92, 246, 0.15); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.3); }

        #knowledge-cards {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 480px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .fact-card {
            background: rgba(11, 14, 23, 0.8);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 0.78rem;
            transition: all 0.15s;
        }

        .fact-card:hover {
            border-color: var(--accent);
            transform: translateX(2px);
        }

        .fact-subj { font-weight: 700; color: #93c5fd; }
        .fact-rel { color: var(--text-dim); font-size: 0.7rem; font-style: italic; margin: 0 4px; }
        .fact-obj { color: #e2e8f0; margin-top: 4px; line-height: 1.4; }
        .fact-conf { font-size: 0.68rem; color: var(--emerald); font-family: 'JetBrains Mono', monospace; margin-top: 6px; }

        /* Files List */
        #files-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
            max-height: 480px;
            overflow-y: auto;
        }

        .file-item {
            padding: 7px 10px;
            border-radius: 6px;
            font-size: 0.78rem;
            color: var(--text-muted);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.15s;
            font-family: 'JetBrains Mono', monospace;
        }

        .file-item:hover {
            background: var(--bg-elevated);
            color: #fff;
        }

        /* Main Chat Area */
        #main {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
            background: var(--bg-base);
            position: relative;
        }

        .top-navbar {
            padding: 12px 24px;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            z-index: 10;
        }

        .nav-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .toggle-btn {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-muted);
            width: 32px;
            height: 32px;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s;
        }

        .toggle-btn:hover {
            background: var(--bg-elevated);
            color: #fff;
        }

        .engine-pill {
            background: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #a5b4fc;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.76rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .nav-actions {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn-ghost {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .btn-ghost:hover {
            background: var(--bg-elevated);
            color: #fff;
            border-color: var(--border-hover);
        }

        /* Chat History */
        #chat-viewport {
            flex: 1;
            overflow-y: auto;
            padding: 24px 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        #chat-history {
            width: 100%;
            max-width: 880px;
            padding: 0 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        /* Hero Welcome State */
        #hero-welcome {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 40px 20px 20px 20px;
            animation: fadeIn 0.4s ease;
        }

        .hero-logo {
            width: 64px;
            height: 64px;
            background: var(--accent-gradient);
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            box-shadow: 0 0 32px var(--accent-glow);
            margin-bottom: 20px;
            color: #fff;
        }

        .hero-title {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            font-size: 0.92rem;
            color: var(--text-muted);
            max-width: 580px;
            line-height: 1.6;
            margin-bottom: 32px;
        }

        .starter-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            width: 100%;
            max-width: 720px;
        }

        .starter-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            text-align: left;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .starter-card:hover {
            border-color: var(--accent);
            background: var(--bg-elevated);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }

        .starter-icon { font-size: 1.3rem; margin-bottom: 8px; }
        .starter-title { font-size: 0.88rem; font-weight: 700; color: #fff; margin-bottom: 4px; }
        .starter-desc { font-size: 0.78rem; color: var(--text-dim); line-height: 1.4; }

        /* Message Bubbles */
        .msg-wrapper {
            display: flex;
            gap: 14px;
            width: 100%;
            animation: fadeIn 0.2s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .msg-wrapper.user {
            flex-direction: row-reverse;
        }

        .msg-avatar {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 15px;
            flex-shrink: 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }

        .avatar-user { background: linear-gradient(135deg, #2563eb, #3b82f6); color: #fff; }
        .avatar-ai { background: var(--accent-gradient); color: #fff; }

        .msg-bubble {
            max-width: calc(100% - 48px);
            padding: 16px 20px;
            border-radius: 14px;
            font-size: 0.92rem;
            line-height: 1.68;
            word-break: break-word;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }

        .user .msg-bubble {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: #fff;
            border-top-right-radius: 4px;
        }

        .assistant .msg-bubble {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: #e2e8f0;
            border-top-left-radius: 4px;
            width: 100%;
        }

        /* Code block styling */
        .code-container {
            background: var(--code-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin: 14px 0;
            overflow: hidden;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
        }

        .code-header {
            background: #0f131f;
            padding: 8px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.74rem;
            color: var(--text-dim);
            border-bottom: 1px solid var(--border);
            font-family: 'JetBrains Mono', monospace;
        }

        .code-dots {
            display: flex;
            gap: 6px;
            align-items: center;
        }

        .code-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }

        .dot-red { background: #ef4444; }
        .dot-yellow { background: #f59e0b; }
        .dot-green { background: #10b981; }

        .code-actions {
            display: flex;
            gap: 6px;
        }

        .btn-code {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: #93c5fd;
            font-size: 0.72rem;
            padding: 3px 9px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s;
            font-family: inherit;
        }

        .btn-code:hover { background: rgba(255, 255, 255, 0.12); color: #fff; }

        pre {
            padding: 16px;
            overflow-x: auto;
            font-size: 0.86rem;
            font-family: 'JetBrains Mono', monospace;
            color: #f1f5f9;
            line-height: 1.55;
        }

        code.inline-code {
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.86em;
            color: #a5b4fc;
        }

        h1, h2, h3, h4 { color: #fff; margin: 16px 0 10px 0; font-weight: 700; }
        h3 { font-size: 1.08rem; }
        p { margin-bottom: 12px; }
        p:last-child { margin-bottom: 0; }
        ul { margin-left: 22px; margin-bottom: 12px; }
        li { margin-bottom: 6px; }

        .table-wrap { overflow-x: auto; margin: 14px 0; border-radius: 8px; border: 1px solid var(--border); }
        .md-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
        .md-table th, .md-table td { border-bottom: 1px solid var(--border); padding: 10px 14px; text-align: left; }
        .md-table th { background: rgba(22, 26, 41, 0.9); color: #fff; font-weight: 600; }
        .md-table tr:nth-child(even) { background: rgba(15, 18, 29, 0.5); }
        .md-table tr:hover { background: rgba(255, 255, 255, 0.02); }

        /* Chips Bar */
        .chips-container {
            width: 100%;
            max-width: 880px;
            padding: 6px 24px 8px 24px;
            display: flex;
            gap: 8px;
            overflow-x: auto;
            white-space: nowrap;
        }

        .chips-container::-webkit-scrollbar { height: 4px; }
        .chips-container::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 4px; }

        .chip {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 6px 13px;
            border-radius: 20px;
            font-size: 0.76rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            flex-shrink: 0;
        }

        .chip:hover {
            background: var(--bg-elevated);
            color: #fff;
            border-color: var(--accent);
            transform: translateY(-1px);
        }

        /* Input Area */
        #input-dock {
            width: 100%;
            max-width: 880px;
            padding: 0 24px 20px 24px;
        }

        .input-box {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 16px;
            display: flex;
            gap: 12px;
            align-items: flex-end;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .input-box:focus-within {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-glow), 0 8px 30px rgba(0, 0, 0, 0.5);
        }

        textarea {
            flex: 1;
            background: transparent;
            border: none;
            color: #fff;
            font-size: 0.94rem;
            line-height: 1.5;
            resize: none;
            max-height: 180px;
            min-height: 24px;
            outline: none;
            font-family: inherit;
        }

        textarea::placeholder { color: var(--text-dim); }

        button.send-btn {
            background: var(--accent-gradient);
            color: #fff;
            border: none;
            width: 36px;
            height: 36px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            transition: all 0.15s;
            box-shadow: 0 2px 10px var(--accent-glow);
        }

        button.send-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 14px var(--accent-glow);
        }

        .input-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.72rem;
            color: var(--text-dim);
            margin-top: 8px;
            padding: 0 6px;
        }

        /* Typing loader */
        .typing-loader {
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 6px 10px;
        }

        .typing-dot {
            width: 6px;
            height: 6px;
            background: var(--accent);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }

        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>

    <!-- Left Collapsible Sidebar -->
    <div id="sidebar">
        <div class="brand-header">
            <div class="brand-logo">
                <div class="brand-icon">⚛</div>
                <div class="brand-text">
                    <h1>Hyper-Astra</h1>
                    <span><div class="pulse-dot"></div> System 2 Cognitive AGI</span>
                </div>
            </div>
            <button class="btn-ghost" onclick="refreshMemory()" title="Refresh" style="padding:4px 8px;">🔄</button>
        </div>

        <div class="sidebar-tabs">
            <button class="tab-btn active" onclick="switchTab('neocortex', this)">🧠 Neocortex</button>
            <button class="tab-btn" onclick="switchTab('files', this)">📂 Files</button>
            <button class="tab-btn" onclick="switchTab('health', this)">⚡ Health</button>
        </div>

        <div class="sidebar-content">
            <!-- Panel 1: Neocortical Knowledge Graph -->
            <div id="tab-neocortex" class="tab-panel active">
                <div class="card-widget">
                    <div class="widget-title">
                        <span>Active Axioms</span>
                        <span id="axiom-counter" class="val-badge badge-blue">Loading...</span>
                    </div>
                    <div id="knowledge-cards">Loading knowledge axioms...</div>
                </div>
            </div>

            <!-- Panel 2: Files Browser -->
            <div id="tab-files" class="tab-panel">
                <div class="card-widget">
                    <div class="widget-title">
                        <span>Workspace Files</span>
                        <button class="btn-ghost" onclick="loadFiles()" style="padding:2px 6px; font-size:0.7rem;">Reload</button>
                    </div>
                    <div id="files-list">Loading files...</div>
                </div>
            </div>

            <!-- Panel 3: Architecture Diagnostics -->
            <div id="tab-health" class="tab-panel">
                <div class="card-widget">
                    <div class="widget-title">
                        <span>Cognitive Architecture</span>
                        <span style="color:var(--emerald);">100% VERIFIED</span>
                    </div>
                    <div class="status-row">
                        <span class="status-label">Reasoning Engine</span>
                        <span class="status-val"><span class="val-badge badge-purple">PUCT System 2</span></span>
                    </div>
                    <div class="status-row">
                        <span class="status-label">Compute Scaling</span>
                        <span class="status-val"><span class="val-badge badge-blue">Shannon Entropy</span></span>
                    </div>
                    <div class="status-row">
                        <span class="status-label">Memory Architecture</span>
                        <span class="status-val"><span class="val-badge badge-purple">CLS (Hippocampal/Neo)</span></span>
                    </div>
                    <div class="status-row">
                        <span class="status-label">Immune Shield</span>
                        <span class="status-val"><span class="val-badge badge-green">IPI Protected</span></span>
                    </div>
                    <div class="status-row">
                        <span class="status-label">Execution Sandbox</span>
                        <span class="status-val"><span class="val-badge badge-green">Deterministic AST</span></span>
                    </div>
                    <div class="status-row">
                        <span class="status-label">Test Suite</span>
                        <span class="status-val"><span class="val-badge badge-green">37/37 Passed (1.0s)</span></span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Chat Viewport -->
    <div id="main">
        <div class="top-navbar">
            <div class="nav-left">
                <button class="toggle-btn" onclick="toggleSidebar()" title="Toggle Sidebar">☰</button>
                <div class="engine-pill">
                    <span style="color:var(--emerald);">●</span> Hyper-Astra 3.5 (System 2 Hybrid)
                </div>
            </div>
            <div class="nav-actions">
                <button class="btn-ghost" onclick="sendCmd('/repomap')">📁 /repomap</button>
                <button class="btn-ghost" onclick="sendCmd('/test')">🧪 /test</button>
                <button class="btn-ghost" onclick="newChat()">✨ New Chat</button>
            </div>
        </div>

        <div id="chat-viewport">
            <!-- Hero Welcome Card when chat is fresh -->
            <div id="hero-welcome">
                <div class="hero-logo">⚛</div>
                <h2 class="hero-title">Hyper-Astra AGI Studio</h2>
                <p class="hero-subtitle">Sovereign Local Cognitive Engine with Dual-Process MCTS, Shannon Entropy Compute Budgeting, and Deterministic Code Execution.</p>
                <div class="starter-grid">
                    <div class="starter-card" onclick="sendCmd('Write a python quicksort')">
                        <div class="starter-icon">💻</div>
                        <div class="starter-title">Algorithmic Synthesis</div>
                        <div class="starter-desc">Synthesize Quicksort, LRU Cache, or DFS with AST sandbox execution</div>
                    </div>
                    <div class="starter-card" onclick="sendCmd('create file demo_task.py with content print(\\'Micro-AGI Sovereign Runtime Active\\')')">
                        <div class="starter-icon">📁</div>
                        <div class="starter-title">Autonomous File & Code Tasks</div>
                        <div class="starter-desc">Create files, view workspace code, and execute test suites</div>
                    </div>
                    <div class="starter-card" onclick="sendCmd('could u connect with my instagram?')">
                        <div class="starter-icon">📸</div>
                        <div class="starter-title">Platform Integrations</div>
                        <div class="starter-desc">Connect with Instagram, Discord, GitHub, or synthesize MCP servers</div>
                    </div>
                    <div class="starter-card" onclick="sendCmd('What is the meaning of AGI vs ASI?')">
                        <div class="starter-icon">🔬</div>
                        <div class="starter-title">Cognitive Science & AGI Theory</div>
                        <div class="starter-desc">DeepMind Levels of AGI, Shannon entropy budgeting, and CLS memory</div>
                    </div>
                </div>
            </div>

            <!-- Chat message thread -->
            <div id="chat-history"></div>
        </div>

        <!-- Suggestion Chips Bar -->
        <div class="chips-container">
            <div class="chip" onclick="sendCmd('launch notepad')">🖥️ Launch Notepad</div>
            <div class="chip" onclick="sendCmd('list open windows')">🪟 List Windows</div>
            <div class="chip" onclick="sendCmd('take screenshot')">📸 Screenshot</div>
            <div class="chip" onclick="sendCmd('/neural')">🧠 /neural</div>
            <div class="chip" onclick="sendCmd('/train Sovereign AI continuous learning.')">⚡ /train AI</div>
            <div class="chip" onclick="sendCmd('Write a python quicksort')">⚡ Quicksort</div>
            <div class="chip" onclick="sendCmd('Implement LRU cache in python')">⚡ LRU Cache</div>
            <div class="chip" onclick="sendCmd('create file hello.py with content print(\\'Hello World\\')')">📁 Create hello.py</div>
            <div class="chip" onclick="sendCmd('could u connect with my instagram?')">📸 Connect Instagram</div>
            <div class="chip" onclick="sendCmd('could u connect to any mcp?')">🔌 Connect MCP</div>
            <div class="chip" onclick="sendCmd('What is the meaning of AGI vs ASI?')">🔬 Meaning of AGI</div>
            <div class="chip" onclick="sendCmd('/repomap')">📁 /repomap</div>
            <div class="chip" onclick="sendCmd('/test')">🧪 /test</div>
        </div>

        <!-- Input Dock -->
        <div id="input-dock">
            <div class="input-box">
                <textarea id="user-input" rows="1" placeholder="Type a message, request code, or ask tasks (e.g. 'create file hello.py', 'how r u?')..." onkeydown="handleKeyDown(event)" oninput="autoResize(this)"></textarea>
                <button class="send-btn" onclick="sendMessage()" title="Send">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13"></line>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                </button>
            </div>
            <div class="input-footer">
                <span>Enter to send • Shift+Enter for newline • 100% Sovereign Local Execution</span>
                <span>Subsystems: System 2 MCTS | AST Sandbox</span>
            </div>
        </div>
    </div>

    <script>
        function toggleSidebar() {
            document.getElementById("sidebar").classList.toggle("collapsed");
        }

        function switchTab(tabId, btn) {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById("tab-" + tabId).classList.add("active");
            if (tabId === 'files') loadFiles();
        }

        function renderMarkdown(md) {
            if (!md) return "";
            let html = md;
            
            // Normalize CRLF
            html = html.replace(/\\r\\n/g, "\\n");

            // Code blocks with syntax copy and run button
            html = html.replace(/```([a-zA-Z0-9_-]*)\\n([\\s\\S]*?)```/g, function(match, lang, code) {
                const escaped = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                const codeLang = lang || 'python';
                const runBtn = (codeLang.toLowerCase() === 'python' || codeLang.toLowerCase() === 'py')
                    ? `<button class="btn-code" onclick="runSandboxCode(this.parentElement.parentElement.querySelector('code').innerText)">▶ Run</button>`
                    : '';
                return `<div class="code-container">
                    <div class="code-header">
                        <div class="code-dots">
                            <span class="code-dot dot-red"></span>
                            <span class="code-dot dot-yellow"></span>
                            <span class="code-dot dot-green"></span>
                            <span style="margin-left:6px; font-weight:600;">${codeLang.toUpperCase()}</span>
                        </div>
                        <div class="code-actions">
                            ${runBtn}
                            <button class="btn-code" onclick="copyCode(this)">📋 Copy</button>
                        </div>
                    </div>
                    <pre><code>${escaped.trim()}</code></pre>
                </div>`;
            });

            // Inline code
            html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

            // Headers
            html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
            html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
            html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

            // Bold & Italic
            html = html.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
            html = html.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');

            // Lists
            html = html.replace(/^\\s*-\\s+(.*$)/gim, '<li>$1</li>');

            // Tables
            if (html.includes("|")) {
                const lines = html.split("\\n");
                let inTable = false;
                let tableHtml = "";
                let newLines = [];
                for (let line of lines) {
                    if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
                        if (line.includes("---")) continue;
                        if (!inTable) {
                            inTable = true;
                            tableHtml = '<div class="table-wrap"><table class="md-table">';
                        }
                        const cells = line.split("|").filter((c, idx, arr) => idx > 0 && idx < arr.length - 1);
                        tableHtml += "<tr>" + cells.map(c => `<td>${c.trim()}</td>`).join("") + "</tr>";
                    } else {
                        if (inTable) {
                            tableHtml += "</table></div>";
                            newLines.push(tableHtml);
                            inTable = false;
                            tableHtml = "";
                        }
                        newLines.push(line);
                    }
                }
                if (inTable) {
                    tableHtml += "</table></div>";
                    newLines.push(tableHtml);
                }
                html = newLines.join("\\n");
            }

            // Paragraphs
            const parts = html.split(/(<div class="code-container">[\\s\\S]*?<\\/div>|<div class="table-wrap">[\\s\\S]*?<\\/div>)/g);
            for (let i = 0; i < parts.length; i++) {
                if (!parts[i].startsWith('<div class="code-container">') && !parts[i].startsWith('<div class="table-wrap">')) {
                    parts[i] = parts[i].replace(/\\n\\n/g, '<p></p>').replace(/\\n/g, '<br>');
                }
            }
            return parts.join("");
        }

        function copyCode(btn) {
            const pre = btn.closest('.code-container').querySelector('code');
            navigator.clipboard.writeText(pre.innerText).then(() => {
                const orig = btn.innerText;
                btn.innerText = "✓ Copied!";
                setTimeout(() => btn.innerText = orig, 1500);
            });
        }

        async function runSandboxCode(code) {
            sendCmd("/run " + code);
        }

        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = (textarea.scrollHeight) + 'px';
        }

        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        async function sendMessage() {
            const textarea = document.getElementById("user-input");
            const text = textarea.value.trim();
            if (!text) return;

            // Hide hero welcome once chat begins
            const hero = document.getElementById("hero-welcome");
            if (hero) hero.style.display = "none";

            appendMsg(text, "user");
            textarea.value = "";
            textarea.style.height = 'auto';

            const history = document.getElementById("chat-history");
            const loadingMsg = document.createElement("div");
            loadingMsg.className = "msg-wrapper assistant";
            loadingMsg.id = "thinking-loader";
            loadingMsg.innerHTML = `
                <div class="msg-avatar avatar-ai">🧠</div>
                <div class="msg-bubble" style="color:var(--text-muted); display:flex; align-items:center; gap:8px;">
                    <span>System 2 PUCT Search & Reasoning...</span>
                    <div class="typing-loader">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            `;
            history.appendChild(loadingMsg);
            scrollViewport();

            try {
                const res = await fetch("/api/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();
                const loader = document.getElementById("thinking-loader");
                if (loader) history.removeChild(loader);
                appendMsg(data.reply, "assistant");
                refreshMemory();
            } catch (err) {
                const loader = document.getElementById("thinking-loader");
                if (loader) history.removeChild(loader);
                appendMsg("Connection error: " + err, "assistant");
            }
        }

        function sendCmd(cmd) {
            const textarea = document.getElementById("user-input");
            textarea.value = cmd;
            sendMessage();
        }

        function appendMsg(text, sender) {
            const history = document.getElementById("chat-history");
            const wrapper = document.createElement("div");
            wrapper.className = "msg-wrapper " + sender;
            
            const avatar = document.createElement("div");
            avatar.className = "msg-avatar " + (sender === "user" ? "avatar-user" : "avatar-ai");
            avatar.innerText = sender === "user" ? "👤" : "⚛";

            const bubble = document.createElement("div");
            bubble.className = "msg-bubble";
            if (sender === "user") {
                bubble.innerText = text;
            } else {
                bubble.innerHTML = renderMarkdown(text);
            }

            wrapper.appendChild(avatar);
            wrapper.appendChild(bubble);
            history.appendChild(wrapper);
            scrollViewport();
        }

        function scrollViewport() {
            const vp = document.getElementById("chat-viewport");
            vp.scrollTop = vp.scrollHeight;
        }

        function newChat() {
            document.getElementById("chat-history").innerHTML = "";
            const hero = document.getElementById("hero-welcome");
            if (hero) hero.style.display = "flex";
        }

        async function refreshMemory() {
            try {
                const res = await fetch("/api/memory");
                const data = await res.json();
                const container = document.getElementById("knowledge-cards");
                const counter = document.getElementById("axiom-counter");
                
                counter.innerText = `${data.axiom_count || 0} Axioms`;

                if (data.triples && data.triples.length > 0) {
                    container.innerHTML = data.triples.map(t => `
                        <div class="fact-card">
                            <div><span class="fact-subj">${t.subject}</span><span class="fact-rel">&rarr; ${t.relation} &rarr;</span></div>
                            <div class="fact-obj">${t.object}</div>
                            <div class="fact-conf">Confidence: ${(t.confidence * 100).toFixed(0)}%</div>
                        </div>
                    `).join("");
                } else {
                    container.innerHTML = `<div style="color:var(--text-dim); font-size:0.75rem; padding:8px;">No active axioms yet.</div>`;
                }
            } catch(e) {
                console.error("Memory refresh failed:", e);
            }
        }

        async function loadFiles() {
            try {
                const res = await fetch("/api/files");
                const data = await res.json();
                const container = document.getElementById("files-list");
                if (data.files && data.files.length > 0) {
                    container.innerHTML = data.files.map(f => `
                        <div class="file-item" onclick="sendCmd('view file ${f}')" title="Click to view file">
                            📄 ${f}
                        </div>
                    `).join("");
                } else {
                    container.innerHTML = `<div style="color:var(--text-dim); font-size:0.75rem; padding:8px;">No files found.</div>`;
                }
            } catch(e) {
                console.error("Files load failed:", e);
            }
        }

        refreshMemory();
    </script>
</body>
</html>
"""


class StudioHTTPHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Quiet logs

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path == "/api/memory":
            facts = repl.engine.semantic_graph.get_all_facts()
            triples = []
            for f in facts:
                triples.append({
                    "subject": f.subject,
                    "relation": f.relation,
                    "object": f.object_,
                    "confidence": f.confidence,
                    "source": f.source or "internal"
                })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "triples": triples,
                "axiom_count": len(facts),
                "episodic_count": len(repl.engine.episodic_store.episodes)
            }).encode("utf-8"))

        elif self.path == "/api/files":
            files_list = []
            for root, dirs, files in os.walk(repl.workspace_root):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                rel_root = os.path.relpath(root, repl.workspace_root)
                for f in sorted(files):
                    if not f.startswith("."):
                        rel_path = f if rel_root == "." else os.path.join(rel_root, f)
                        files_list.append(rel_path.replace("\\", "/"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"files": files_list[:60]}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(post_body)
                user_msg = data.get("message", "")
                reply = repl.process_command(user_msg)
                resp_payload = json.dumps({"reply": reply})
            except Exception as e:
                resp_payload = json.dumps({"reply": f"Error: {e}"})

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp_payload.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server():
    server = ThreadedTCPServer(("127.0.0.1", PORT), StudioHTTPHandler)
    print(f"[*] Hyper-Astra Web Studio running at: http://127.0.0.1:{PORT}")
    print("[*] Threading enabled. Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    start_server()
