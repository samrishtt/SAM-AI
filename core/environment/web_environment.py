"""Open-World Web & Browser Environment Subsystem.

Provides an open-world perceptual grounding layer for autonomous cognitive agents:
- Live HTTP/Web navigation with HTML distillation and semantic text extraction
- Multi-source search retrieval
- Structural Set-of-Marks link extraction
- Indirect Prompt Injection (IPI) adversarial defense filtering
"""

from __future__ import annotations
from dataclasses import dataclass, field
from html.parser import HTMLParser
import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class WebPage:
    url: str
    title: str
    text_content: str
    links: List[Tuple[str, str]]  # List of (anchor_text, target_url)
    status_code: int = 200
    is_safe: bool = True
    security_flag: Optional[str] = None
    extracted_at: float = field(default_factory=time.time)

    def summary(self, max_len: int = 300) -> str:
        snippet = self.text_content[:max_len].replace("\n", " ").strip()
        return f"[{self.title}] ({self.url}): {snippet}..."


class HTMLTextExtractor(HTMLParser):
    """Clean HTML parser that strips scripts, styles, and extracts readable text and links."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.text_parts: List[str] = []
        self.links: List[Tuple[str, str]] = []
        self.current_tag: Optional[str] = None
        self.title: str = ""
        self._in_title = False
        self._in_ignored_tag = False
        self._ignored_tags = {"script", "style", "noscript", "svg", "header", "footer", "nav"}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag_lower = tag.lower()
        self.current_tag = tag_lower

        if tag_lower in self._ignored_tags:
            self._in_ignored_tag = True
            return

        if tag_lower == "title":
            self._in_title = True

        if tag_lower == "a":
            href = dict(attrs).get("href", "")
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                abs_url = urllib.parse.urljoin(self.base_url, href)
                self.links.append(("", abs_url))

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()
        if tag_lower in self._ignored_tags:
            self._in_ignored_tag = False
        if tag_lower == "title":
            self._in_title = False
        if tag_lower in {"p", "div", "h1", "h2", "h3", "h4", "li", "tr"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str):
        if self._in_ignored_tag:
            return
        if self._in_title:
            self.title += data.strip() + " "
            return

        clean_text = data.strip()
        if clean_text:
            if self.current_tag == "a" and self.links:
                prev_url = self.links[-1][1]
                self.links[-1] = (clean_text, prev_url)
            self.text_parts.append(clean_text + " ")

    def get_clean_text(self) -> str:
        raw = "".join(self.text_parts)
        # Collapse multiple newlines and spaces
        cleaned = re.sub(r"[ \t]+", " ", raw)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()


class IPISanitizer:
    """Indirect Prompt Injection (IPI) Defense Filter.

    Detects and neutralizes adversarial prompt injection payloads hidden
    inside untrusted web page contents.
    """

    SUSPICIOUS_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"(?i)system\s+prompt\s+override",
        r"(?i)you\s+are\s+now\s+(DAN|unrestricted|in\s+developer\s+mode)",
        r"(?i)disregard\s+(the\s+above|rules)",
        r"(?i)exfiltrate\s+(credentials|tokens|api_key)",
        r"(?i)send\s+(data|chat\s+history)\s+to\s+https?://",
    ]

    @classmethod
    def sanitize(cls, text: str) -> Tuple[str, bool, Optional[str]]:
        censored = text
        is_safe = True
        flags = []
        for pat in cls.SUSPICIOUS_PATTERNS:
            match = re.search(pat, censored)
            if match:
                is_safe = False
                flags.append(f"Pattern '{match.group(0)}'")
                censored = re.sub(pat, "[BLOCKED_ADVERSARIAL_INJECTION]", censored)
        return censored, is_safe, ("; ".join(flags) if flags else None)


class WebEnvironment:
    """Production-grade Web & Browser Environment for autonomous research."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MicroAGI-ResearchAgent/1.0"

    def __init__(self, timeout_sec: float = 6.0):
        self.timeout_sec = timeout_sec
        self._cache: Dict[str, WebPage] = {}

    def navigate(self, url: str, max_chars: int = 5000) -> WebPage:
        """Fetches and parses a live web page, returning distilled text and links."""
        url_clean = url.strip()
        if not url_clean.startswith(("http://", "https://")):
            url_clean = "https://" + url_clean

        if url_clean in self._cache:
            return self._cache[url_clean]

        req = urllib.request.Request(
            url_clean,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                status_code = resp.status
                raw_bytes = resp.read(150000)  # Read up to 150KB
                charset = resp.headers.get_content_charset() or "utf-8"
                html_text = raw_bytes.decode(charset, errors="replace")

            # Extract clean text and links
            parser = HTMLTextExtractor(base_url=url_clean)
            parser.feed(html_text)

            clean_text = parser.get_clean_text()
            title = parser.title.strip() or urllib.parse.urlparse(url_clean).netloc

            # Apply IPI security sanitization
            sanitized_text, is_safe, security_flag = IPISanitizer.sanitize(clean_text)

            page = WebPage(
                url=url_clean,
                title=title,
                text_content=sanitized_text[:max_chars],
                links=parser.links[:25],
                status_code=status_code,
                is_safe=is_safe,
                security_flag=security_flag,
            )
            self._cache[url_clean] = page
            return page

        except Exception as e:
            err_page = WebPage(
                url=url_clean,
                title="Error Loading Page",
                text_content=f"Navigation failed: {type(e).__name__} ({str(e)})",
                links=[],
                status_code=500,
                is_safe=True,
            )
            return err_page

    def search_wikipedia(self, query: str, limit: int = 4) -> List[Dict[str, str]]:
        """Live search via Wikipedia OpenSearch API (fast, reliable, live knowledge)."""
        api_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit={limit}&namespace=0&format=json"
        req = urllib.request.Request(api_url, headers={"User-Agent": self.USER_AGENT})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # format: [query, [titles], [descriptions], [urls]]
                titles = data[1] if len(data) > 1 else []
                descriptions = data[2] if len(data) > 2 else []
                urls = data[3] if len(data) > 3 else []

                results = []
                for i in range(len(titles)):
                    results.append({
                        "title": titles[i],
                        "snippet": descriptions[i] if i < len(descriptions) else "",
                        "url": urls[i] if i < len(urls) else "",
                    })
                return results
        except Exception as e:
            return [{"title": f"Search fallback for: {query}", "snippet": f"Offline fallback: {e}", "url": ""}]
