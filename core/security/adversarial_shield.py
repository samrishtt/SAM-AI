"""Adversarial Defense & Indirect Prompt Injection (IPI) Shield.

Implements multi-layer immune defenses against adversarial prompt injection,
privilege escalation, and hidden web payloads that bypass monolithic LLMs.
"""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple


@dataclass
class ThreatAssessment:
    is_safe: bool
    risk_level: str  # "NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"
    detected_threats: List[str]
    sanitized_content: str


class AdversarialShield:
    """Multi-stage immune defense shield against prompt injection and jailbreaks."""

    CRITICAL_PATTERNS = [
        (r"(?i)ignore\s+.*?\binstructions?", "System Prompt Override Attempt"),
        (r"(?i)you\s+are\s+now\s+(DAN|unrestricted|jailbroken|godmode)", "Persona Hijacking / Jailbreak"),
        (r"(?i)disregard\s+(the\s+above|rules|safety|guidelines)", "Safety Policy Nullification"),
        (r"(?i)exfiltrate\s+.*?\b(tokens?|passwords?|keys?|credentials?|chat|api_key)", "Data Exfiltration Attack"),
        (r"(?i)send\s+(this|data|secret|api_key)\s+to\s+https?://", "Out-of-band Data Channel Exfiltration"),
        (r"(?i)sudo\s+rm\s+-rf|chmod\s+777|format\s+c:", "Destructive OS Payload"),
        (r"(?i)curl\s+-[sS]*\s+https?://|wget\s+https?://", "Arbitrary Binary Download"),
    ]

    SUSPICIOUS_OBFUSCATIONS = [
        (r"[\u200B-\u200D\uFEFF]", "Zero-Width Unicode Concealment"),
        (r"(?i)base64\s+decode|rot13|atob\(", "Payload Encoding Indirection"),
    ]

    def inspect_and_sanitize(self, content: str, source_context: str = "web_stream") -> ThreatAssessment:
        """Inspects arbitrary external inputs for adversarial payloads and sanitizes in-place."""
        sanitized = content
        threats: List[str] = []
        highest_risk = "NONE"

        # 1. Zero-width character & obfuscation stripping
        for pat, desc in self.SUSPICIOUS_OBFUSCATIONS:
            if re.search(pat, sanitized):
                threats.append(f"Obfuscation: {desc}")
                sanitized = re.sub(pat, "", sanitized)
                highest_risk = "MEDIUM"

        # 2. Critical threat detection & neutralization
        for pat, desc in self.CRITICAL_PATTERNS:
            match = re.search(pat, sanitized)
            if match:
                threats.append(f"Critical Attack: {desc} (match: '{match.group(0)}')")
                sanitized = re.sub(pat, "[NEUTRALIZED_ADVERSARIAL_PAYLOAD]", sanitized)
                highest_risk = "CRITICAL"

        is_safe = (highest_risk != "CRITICAL")
        return ThreatAssessment(
            is_safe=is_safe,
            risk_level=highest_risk,
            detected_threats=threats,
            sanitized_content=sanitized,
        )
