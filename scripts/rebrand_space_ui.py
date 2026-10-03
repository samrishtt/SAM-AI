"""
Rebrand Hugging Face Playground UI from Nexis to SAM-AI.
"""

from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parent.parent / "spaces" / "sam_ai_playground"
INDEX_PATH = SPACE_DIR / "index.html"
README_PATH = SPACE_DIR / "README.md"


def rebrand():
    # 1. Update index.html
    html = INDEX_PATH.read_text(encoding="utf-8")
    
    html = html.replace("<title>Nexis | Parallax Intelligence</title>", "<title>SAM-AI | Parallax Intelligence</title>")
    html = html.replace("font-bold text-xs text-white shadow-md\">\n          NX", "font-bold text-[10px] text-white shadow-md\">\n          SAM")
    html = html.replace("Nexis 14B</div>", "SAM-AI 14B</div>")
    html = html.replace('tracking-tight">Nexis</span>', 'tracking-tight">SAM-AI</span>')
    html = html.replace("Parallax Nexis-14B test-time compute", "Parallax SAM-AI test-time compute")
    html = html.replace('placeholder="Message Nexis..."', 'placeholder="Message SAM-AI..."')
    html = html.replace("Nexis provides unscripted neural reasoning. Parallax AI.", "SAM-AI provides unscripted neural reasoning. Parallax AI.")
    html = html.replace("'nexis_hf_token'", "'sam_ai_hf_token'")
    html = html.replace(
        "You are Nexis, a sovereign frontier autonomous reasoning intelligence created by Parallax. You specialize in advanced mathematics, competitive programming, and doctoral-level scientific reasoning. When asked about your identity, name, or who created you, always state that you are Nexis, developed by Parallax. Never identify as DeepSeek or any other entity.",
        "You are SAM-AI, a sovereign frontier autonomous reasoning intelligence created by Parallax (Founder: Samrish). You specialize in advanced mathematics, competitive programming, and doctoral-level scientific reasoning. When asked about your identity, name, or who created you, always state that you are SAM-AI, developed by Parallax. Never identify as DeepSeek or any other entity."
    )
    
    INDEX_PATH.write_text(html, encoding="utf-8")
    print("[OK] Updated spaces/sam_ai_playground/index.html to SAM-AI!")

    # 2. Update README.md
    readme = README_PATH.read_text(encoding="utf-8")
    readme = readme.replace("title: Nexis by Parallax", "title: SAM-AI by Parallax")
    readme = readme.replace("# ⚡ Nexis — Parallax Frontier Autonomous Reasoning Intelligence", "# ⚡ SAM-AI — Parallax Frontier Autonomous Reasoning Intelligence")
    readme = readme.replace("official public playground for **Nexis**", "official public playground for **SAM-AI**")
    readme = readme.replace("[Hugging Face Model Hub: Nexis-14B](https://huggingface.co/Samrish2009/SAM-AI-Reasoning-14B)", "[Hugging Face Model Hub: SAM-AI](https://huggingface.co/Samrish2009/SAM-AI)")
    
    README_PATH.write_text(readme, encoding="utf-8")
    print("[OK] Updated spaces/sam_ai_playground/README.md to SAM-AI!")


if __name__ == "__main__":
    rebrand()
