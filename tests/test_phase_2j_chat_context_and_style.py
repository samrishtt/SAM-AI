"""
SAM-AI Phase 2J: Conversation Context, Response Style, and Explanation Panel Acceptance Suite
=============================================================================================
Validates:
1. Concise-by-default response policy in both web and space system prompts.
2. formatGradioChatHistory deterministic serialization, road-trip isolation, and edge-case handling.
3. Fresh conversation context isolation (sending 'hi' sends strictly zero old history).
4. Follow-up multi-turn context retention in existing conversations.
5. In-flight conversation isolation and query prefill cleanup safeguards.
6. Server-side reasoning filter suppression of <think> traces across chunks and whole texts.
7. Optional explanation API route (/api/chat/explain) structure, trivial greeting bypass, and security.
8. Accessibility attributes and collapsed-by-default contract for explanation panel.
"""

import re
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
SPACES_DIR = REPO_ROOT / "spaces" / "sam_ai_zerogpu"


def test_system_prompt_concise_policy_in_web_and_space():
    """Verify that both web/model-adapter.ts and spaces/model_loader.py implement the concise-by-default policy."""
    web_adapter_file = WEB_DIR / "src" / "lib" / "model-adapter.ts"
    space_loader_file = SPACES_DIR / "model_loader.py"

    assert web_adapter_file.exists(), "web/src/lib/model-adapter.ts must exist"
    assert space_loader_file.exists(), "spaces/sam_ai_zerogpu/model_loader.py must exist"

    web_content = web_adapter_file.read_text(encoding="utf-8")
    space_content = space_loader_file.read_text(encoding="utf-8")

    required_keywords = [
        "Concise by default",
        "Greetings",
        "Direct Q&A",
        "Mathematics and science",
        "Grounded and honest",
        "acknowledge the mismatch briefly",
        "Ambiguity",
        "Multi-turn context",
        "prioritize the intent of the latest user message",
        "Identity and integrity",
    ]

    for kw in required_keywords:
        assert kw.lower() in web_content.lower(), f"Keyword '{kw}' missing from web/src/lib/model-adapter.ts"
        assert kw.lower() in space_content.lower(), f"Keyword '{kw}' missing from spaces/sam_ai_zerogpu/model_loader.py"


def test_format_gradio_chat_history_isolation_via_node():
    """Verify formatGradioChatHistory serialization and isolation using Node.js script."""
    node_test_script = """
    const path = require('path');
    const { formatGradioChatHistory } = require('./web/src/lib/gradio-connector');
    """
    # Test cases defined in JSON and verified via standalone Python logic mirroring the TypeScript function
    # 1. Fresh conversation with 'hi'
    fresh_messages = [{"role": "user", "content": "hi"}]
    # 2. Road-trip history followed by 'hi'
    roadtrip_messages = [
        {"role": "user", "content": "To calculate the total cost of a road trip to Dallas..."},
        {"role": "assistant", "content": "The road trip will cost approximately $340."},
        {"role": "user", "content": "hi"}
    ]
    # 3. Consecutive user messages
    consecutive_user = [
        {"role": "user", "content": "Part 1 of problem"},
        {"role": "user", "content": "Part 2 of problem"}
    ]
    # 4. System + Orphaned assistant + User + Assistant + Follow-up
    mixed_messages = [
        {"role": "system", "content": "You are SAM-AI."},
        {"role": "assistant", "content": "Orphaned greeting"},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "and multiply by 3?"}
    ]

    connector_path = str(WEB_DIR / "src" / "lib" / "gradio-connector.ts").replace("\\", "/")
    script = f"""
    const fs = require('fs');
    const connectorCode = fs.readFileSync('{connector_path}', 'utf8');

    // Extract formatGradioChatHistory function body
    const match = connectorCode.match(/export function formatGradioChatHistory[\\s\\S]*?\\n\\}}/);
    if (!match) {{
        console.error("Could not find formatGradioChatHistory");
        process.exit(1);
    }}

    // Clean TypeScript types for plain node execution
    let jsCode = match[0]
        .replace(/export function/g, 'function')
        .replace(/: Array<[^>]+>/g, '')
        .replace(/: \\{{[^}}]+\\}}/g, '')
        .replace(/: Array<\\[string, string\\]>/g, '')
        .replace(/: string/g, '')
        .replace(/: number/g, '')
        .replace(/: "user" \\| "assistant"/g, '');

    eval(jsCode + '; global.formatGradioChatHistory = formatGradioChatHistory;');

    // Test 1: Fresh conversation
    const r1 = formatGradioChatHistory({json.dumps(fresh_messages)});
    if (r1.userMessage !== 'hi' || r1.history.length !== 0) {{
        console.error('Test 1 Failed:', r1);
        process.exit(1);
    }}

    // Test 2: Road-trip followed by hi
    const r2 = formatGradioChatHistory({json.dumps(roadtrip_messages)});
    if (r2.userMessage !== 'hi' || r2.history.length !== 1 || !r2.history[0][0].includes('road trip')) {{
        console.error('Test 2 Failed:', r2);
        process.exit(2);
    }}

    // Test 3: Consecutive user messages
    const r3 = formatGradioChatHistory({json.dumps(consecutive_user)});
    if (!r3.userMessage.includes('Part 1') || !r3.userMessage.includes('Part 2') || r3.history.length !== 0) {{
        console.error('Test 3 Failed:', r3);
        process.exit(3);
    }}

    // Test 4: Mixed messages with system and orphaned assistant
    const r4 = formatGradioChatHistory({json.dumps(mixed_messages)});
    if (r4.userMessage !== 'and multiply by 3?' || r4.history.length !== 1 || r4.history[0][0] !== 'What is 2+2?' || r4.history[0][1] !== '4') {{
        console.error('Test 4 Failed:', r4);
        process.exit(4);
    }}

    console.log('ALL_GRADIO_FORMATTER_TESTS_PASSED');
    """

    res = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert res.returncode == 0, f"Node test execution failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "ALL_GRADIO_FORMATTER_TESTS_PASSED" in res.stdout


def test_server_side_reasoning_filter_defense_in_depth():
    import sys
    space_dir = str(REPO_ROOT / "spaces" / "sam_ai_zerogpu")
    if space_dir not in sys.path:
        sys.path.insert(0, space_dir)
    from model_loader import ReasoningStreamFilter, filter_reasoning_from_text

    # 1. Complete text filter
    raw_with_think = "<think>\nCalculating 2+2 = 4 step by step...\n</think>\n4"
    assert filter_reasoning_from_text(raw_with_think) == "4"

    # 2. Text without think tags
    raw_clean = "Hello! I am SAM-AI. How can I assist you?"
    assert filter_reasoning_from_text(raw_clean) == raw_clean

    # 3. Interrupted reasoning (fail closed)
    interrupted = "<think>\nComputing incomplete trajectory..."
    filtered_interrupted = filter_reasoning_from_text(interrupted)
    assert "<think>" not in filtered_interrupted
    assert "interrupted" in filtered_interrupted.lower()

    # 4. Streaming filter with tag split across chunks
    stream_filter = ReasoningStreamFilter()
    chunks = [
        "<th",
        "ink>\nStep 1: Analyz",
        "ing user prompt.\n</th",
        "ink>\nHello! Nice to meet you.",
    ]
    emitted = []
    for c in chunks:
        out = stream_filter.process_chunk(c)
        if out:
            emitted.append(out)
    final = stream_filter.finalize()
    if final:
        emitted.append(final)

    full_emitted = "".join(emitted).strip()
    assert "<think>" not in full_emitted
    assert "</think>" not in full_emitted
    assert "Analyzing" not in full_emitted
    assert "Hello! Nice to meet you." in full_emitted


def test_chat_interface_context_and_query_safeguards():
    """Verify that ChatInterface.tsx implements inFlightConvIdRef, useChat hydration, and URL query cleanup."""
    chat_interface_file = WEB_DIR / "src" / "components" / "ChatInterface.tsx"
    assert chat_interface_file.exists()
    content = chat_interface_file.read_text(encoding="utf-8")

    # 1. inFlightConvIdRef present to guard in-flight chat switches
    assert "inFlightConvIdRef = useRef<string | null>(null)" in content
    assert "inFlightConvIdRef.current = targetId" in content
    assert "targetId = inFlightConvIdRef.current" in content

    # 2. Hydration synchronizes useChat messages
    assert "setMessages(currentActive.messages as any)" in content

    # 3. URL query parameter cleanup
    assert "url.searchParams.delete" in content or "cleanUrl.searchParams.delete" in content
    assert "window.history.replaceState" in content

    # 4. New conversation clears messages and in-flight ref
    assert "inFlightConvIdRef.current = null" in content
    assert "setMessages([])" in content

    # 5. userPrompt passed to ChatMessage for on-demand explanation
    assert "userPrompt=" in content


def test_chat_message_explanation_panel_accessibility():
    """Verify that ChatMessage.tsx implements accessible, collapsed-by-default explanation disclosure."""
    chat_message_file = WEB_DIR / "src" / "components" / "ChatMessage.tsx"
    assert chat_message_file.exists()
    content = chat_message_file.read_text(encoding="utf-8")

    # 1. Collapsed by default
    assert "isExplanationOpen, setIsExplanationOpen] = useState(false)" in content

    # 2. ARIA accessibility attributes
    assert 'aria-expanded={isExplanationOpen}' in content
    assert 'aria-controls=' in content
    assert 'role="region"' in content
    assert 'aria-label="Reasoning explanation"' in content

    # 3. Disabled during request to prevent duplicates
    assert "disabled={isLoadingExplanation}" in content

    # 4. Greeting bypass logic present
    assert "isGreeting" in content

    # 5. Calls /api/chat/explain on-demand
    assert '"/api/chat/explain"' in content


def test_explain_api_route_contract_and_security():
    """Verify that web/src/app/api/chat/explain/route.ts exists and enforces security and trivial bypass."""
    explain_route_file = WEB_DIR / "src" / "app" / "api" / "chat" / "explain" / "route.ts"
    assert explain_route_file.exists(), "Explain API route must exist at web/src/app/api/chat/explain/route.ts"
    content = explain_route_file.read_text(encoding="utf-8")

    # 1. Password gate validation
    assert "validateSitePassword" in content
    assert "UNAUTHORIZED" in content

    # 2. Rate limiting
    assert "checkRateLimit" in content
    assert "RATE_LIMITED" in content

    # 3. Trivial / greeting bypass
    assert "isTrivial" in content
    assert "isGreeting" in content

    # 4. Reasoning filter defense-in-depth
    assert "filterReasoningFromText" in content
