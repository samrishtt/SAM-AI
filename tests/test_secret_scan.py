"""Unit tests for secret scanning and reversed-secret detection."""

import sys
from pathlib import Path

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.pre_commit_secret_scan import scan_line, scan_content


def test_reversed_string_literal_flagged():
    """String literals >= 30 chars reversed with [::-1] must be flagged."""
    long_secret = "a" * 30
    line_double = f'secret = "{long_secret}"[::-1]'
    matches = scan_line(line_double)
    assert "Reversed String Secret" in matches

    line_single = f"secret = '{long_secret}'[::-1]"
    matches = scan_line(line_single)
    assert "Reversed String Secret" in matches

    # Even longer secret
    very_long = "x" * 64
    line_very_long = f'token = "{very_long}"[::-1]'
    assert "Reversed String Secret" in scan_line(line_very_long)


def test_short_string_literal_not_flagged():
    """String literals < 30 chars reversed with [::-1] should NOT be flagged."""
    short_line = 'rev = "short_str"[::-1]'
    assert "Reversed String Secret" not in scan_line(short_line)

    boundary_29 = 'boundary = "' + ('y' * 29) + '"[::-1]'
    assert "Reversed String Secret" not in scan_line(boundary_29)


def test_variable_slicing_not_flagged():
    """Reversing variables or lists with [::-1] must NOT be flagged as secrets."""
    benign_lines = [
        "return grid[::-1]",
        "return [row[::-1] for row in grid]",
        "return [list(r) for r in zip(*grid[::-1])]",
        "my_very_long_variable_name_exceeding_thirty_chars[::-1]",
        "words = ' '.join(s.split()[::-1])",
        "h_sym = all(self.grid[r] == self.grid[r][::-1] for r in range(self.height))",
    ]
    for line in benign_lines:
        assert "Reversed String Secret" not in scan_line(line), f"Falsely flagged: {line}"


def test_standard_secret_patterns_still_detected():
    """Standard secret patterns (HF, OpenAI, etc.) must still trigger detection."""
    # Constructed dynamically so the test source file itself does not trigger pre-commit
    hf_token = "hf_" + "0" * 30
    assert "Hugging Face Token" in scan_line(f'TOKEN = "{hf_token}"')

    openai_key = "sk-" + "proj-" + "1" * 30
    assert "OpenAI API Key" in scan_line(f'KEY = "{openai_key}"')


def test_scan_content_reports_line_numbers():
    """scan_content should return filepath, 1-indexed line number, and pattern name."""
    content = "\n".join([
        "# Safe comment",
        "x = [1, 2, 3][::-1]",
        f'bad = "{"k" * 32}"[::-1]',
        "y = 'safe'",
    ])
    violations = scan_content(content, filepath="dummy.py")
    assert len(violations) == 1
    assert violations[0] == ("dummy.py", 3, "Reversed String Secret")
