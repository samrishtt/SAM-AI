"""
SAM-AI Cloud Interface (ZeroGPU Space Edition)
==============================================
Dual-mode Gradio application offering SAM-AI Chat and SAM-AI Code, powered
exclusively by sovereign open-weight foundation model DeepSeek-R1-Distill-Qwen-7B.
"""

import os
import sys
import time
import gradio as gr
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from model_loader import engine, ModelLifecycleStatus, SAM_AI_SYSTEM_PROMPT

# ZeroGPU requires model loading at module level
print("[*] Starting SAM-AI ZeroGPU Space...")
try:
    engine.load()
except Exception as exc:
    print(f"[!] Warning: Initial model loading encountered exception: {exc}")


SYSTEM_PROMPT_CHAT = SAM_AI_SYSTEM_PROMPT

SYSTEM_PROMPT_CODE = (
    "You are SAM-AI Code, a specialized sovereign software engineering assistant developed "
    "by Parallax Intelligence Lab. You assist with coding, debugging, refactoring, and code review. "
    "When suggesting changes, present code clearly and use standard unified diff format where appropriate. "
    "Never invent fake library APIs or pretend unexecuted commands ran successfully."
)


try:
    import spaces
except ImportError:
    class _MockSpaces:
        def GPU(self, duration=60):
            def decorator(fn):
                return fn
            return decorator
    spaces = _MockSpaces()


@spaces.GPU(duration=60)
def chat_respond(user_message, chat_history, temperature, top_p, max_tokens):
    """Handles multi-turn conversation streaming in SAM-AI Chat."""
    if not user_message or not user_message.strip():
        yield chat_history, ""
        return

    # Build history in OpenAI-style format
    messages = []
    for turn in (chat_history or []):
        if turn[0]:
            messages.append({"role": "user", "content": turn[0]})
        if turn[1]:
            messages.append({"role": "assistant", "content": turn[1]})

    messages.append({"role": "user", "content": user_message})

    # Append pending message to history (with deduplication guard)
    history_list = [list(turn) for turn in (chat_history or [])]
    if history_list and history_list[-1][0] == user_message and not history_list[-1][1]:
        new_history = history_list
    else:
        new_history = history_list + [[user_message, ""]]

    for partial_response in engine.generate_stream(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=int(max_tokens),
        system_prompt=SYSTEM_PROMPT_CHAT
    ):
        new_history[-1][1] = partial_response
        yield new_history, ""


@spaces.GPU(duration=60)
def code_respond(user_prompt, code_context, task_type, temperature, max_tokens):
    """Handles coding tasks, explanations, and diff generation in SAM-AI Code."""
    if not user_prompt or not user_prompt.strip():
        yield "Please provide a coding prompt or question."
        return

    full_content = f"### Task: {task_type}\n\n"
    if code_context and code_context.strip():
        full_content += f"```\n{code_context.strip()}\n```\n\n"
    full_content += f"**User Request:**\n{user_prompt.strip()}"

    messages = [{"role": "user", "content": full_content}]

    for partial_response in engine.generate_stream(
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_new_tokens=int(max_tokens),
        system_prompt=SYSTEM_PROMPT_CODE
    ):
        yield partial_response


def get_status_markdown():
    """Generates real-time status banner for header."""
    diag = engine.get_diagnostics()
    status = diag["status"]
    is_loaded = diag["is_loaded"]
    device = diag["device"].upper()
    adapter_name = Path(diag["adapter_dir"]).name if diag.get("adapter_dir") else "None (Pure Base)"
    adapter_label = f" + <code>{adapter_name}</code>" if diag.get("adapter_dir") else " <em>(Pure Foundation Model)</em>"

    if is_loaded:
        badge = "🟢 **READY (SOVEREIGN ZERO-GPU)**"
        color = "green"
    elif status == ModelLifecycleStatus.LOADING_BASE_MODEL or status == ModelLifecycleStatus.ATTACHING_ADAPTER:
        badge = "🟡 **LOADING WEIGHTS...**"
        color = "orange"
    else:
        badge = f"🔴 **NOT READY: {status}**"
        color = "red"

    return (
        f"<div style='border: 1px solid #E8E2DA; border-radius: 8px; padding: 12px 16px; background-color: #FDFCFA; color: #28231F; margin-bottom: 12px; font-family: sans-serif;'>"
        f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
        f"<div><strong>Model:</strong> <code>{diag['base_model']}</code>{adapter_label}</div>"
        f"<div><strong>Status:</strong> {badge} | <strong>Device:</strong> <code>{device}</code></div>"
        f"</div>"
        f"<div style='font-size: 0.85em; color: #6B5E55; margin-top: 4px;'>{diag['status_detail']}</div>"
        f"</div>"
    )


# Build Gradio UI with matching warm light theme (Ivory / Terracotta / Charcoal)
custom_theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="stone",
    neutral_hue="stone",
).set(
    body_background_fill="#FAF8F5",
    body_text_color="#28231F",
    block_background_fill="#FFFFFF",
    block_border_color="#E8E2DA",
    button_primary_background_fill="#D97745",
    button_primary_background_fill_hover="#C86332",
    button_primary_text_color="#FFFFFF",
)

with gr.Blocks(theme=custom_theme, title="SAM-AI Sovereign Assistant") as demo:
    gr.Markdown(
        """
        # 🧠 SAM-AI Sovereign Assistant
        **Self-Hosted Open-Weight Intelligence | DeepSeek-R1-Distill-Qwen-7B (Pure Sovereign Foundation Model)**\
        *Running on Hugging Face Spaces ZeroGPU (NVIDIA RTX Pro 6000 Blackwell)*
        """
    )

    status_header = gr.HTML(value=get_status_markdown)

    with gr.Tabs() as tabs:
        # =====================================================================
        # Tab 1: SAM-AI Chat
        # =====================================================================
        with gr.Tab("💬 SAM-AI Chat", id="tab_chat"):
            chatbot = gr.Chatbot(
                height=520,
                show_copy_button=True,
                bubble_full_width=False,
                render_markdown=True,
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask a question, mathematical problem, or reasoning challenge...",
                    show_label=False,
                    scale=8,
                    lines=2,
                )
                with gr.Column(scale=2, min_width=120):
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

            with gr.Accordion("⚙️ Inference Parameters", open=False):
                with gr.Row():
                    chat_temp = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.6, step=0.05, label="Temperature (Creativity / Randomness)"
                    )
                    chat_top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-P (Nucleus Sampling)"
                    )
                    chat_max_tokens = gr.Slider(
                        minimum=256, maximum=4096, value=2048, step=128, label="Max New Tokens"
                    )

            # Event wiring
            chat_msg_event = chat_input.submit(
                chat_respond,
                inputs=[chat_input, chatbot, chat_temp, chat_top_p, chat_max_tokens],
                outputs=[chatbot, chat_input],
                api_name="chat_respond",
            )
            send_btn.click(
                chat_respond,
                inputs=[chat_input, chatbot, chat_temp, chat_top_p, chat_max_tokens],
                outputs=[chatbot, chat_input],
                api_name="chat_respond",
            )
            clear_btn.click(lambda: None, None, chatbot, queue=False)

        # =====================================================================
        # Tab 2: SAM-AI Code
        # =====================================================================
        with gr.Tab("💻 SAM-AI Code", id="tab_code"):
            with gr.Row():
                with gr.Column(scale=5):
                    code_task = gr.Radio(
                        choices=["Generate Code", "Debug & Fix", "Refactor & Optimize", "Generate Unified Diff"],
                        value="Generate Code",
                        label="Coding Task Type"
                    )
                    code_prompt = gr.Textbox(
                        label="Coding Instructions / Problem Description",
                        placeholder="e.g. Implement a thread-safe LRU cache with eviction telemetry in Python...",
                        lines=4,
                    )
                    code_context = gr.Code(
                        label="Existing Code / Context (Optional)",
                        language="python",
                        lines=12,
                    )
                    run_code_btn = gr.Button("Generate Solution", variant="primary")

                with gr.Column(scale=5):
                    code_output = gr.Markdown(
                        label="SAM-AI Code Response",
                        value="*Click 'Generate Solution' to run SAM-AI Code on the assigned ZeroGPU.*"
                    )

            with gr.Accordion("⚙️ Code Generation Parameters", open=False):
                with gr.Row():
                    code_temp = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Temperature (Lower = More Deterministic)"
                    )
                    code_max_tokens = gr.Slider(
                        minimum=512, maximum=4096, value=2048, step=128, label="Max Tokens"
                    )

            run_code_btn.click(
                code_respond,
                inputs=[code_prompt, code_context, code_task, code_temp, code_max_tokens],
                outputs=[code_output],
                api_name="code_respond",
            )

        # =====================================================================
        # Tab 3: Diagnostics & Sovereign Architecture
        # =====================================================================
        with gr.Tab("🛡️ Sovereign Diagnostics", id="tab_diag"):
            refresh_diag_btn = gr.Button("Refresh Telemetry")
            diag_json = gr.JSON(value=engine.get_diagnostics, label="Engine Diagnostic State")
            refresh_diag_btn.click(engine.get_diagnostics, outputs=[diag_json])

    gr.Markdown(
        """
        ---
        **Parallax Intelligence Lab | SAM-AI Project**  
        *Zero commercial third-party inference APIs. Pure open-weight execution.*
        """
    )


# Configure Gradio queue for ZeroGPU concurrency management
demo.queue(
    max_size=20,
    default_concurrency_limit=1,
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
