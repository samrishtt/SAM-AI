"""OSWorld Automation Bridge.

Enables full operating system automation (terminal shell execution, filesystem
manipulation, process telemetry, and multi-application workflows) matching
the OSWorld 2.0 benchmark specification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from core.execution.ast_sandbox import PythonASTSandbox
from core.security.adversarial_shield import AdversarialShield, ThreatAssessment


@dataclass
class OSTaskResult:
    command: str
    return_code: int
    stdout: str
    stderr: str
    execution_time_ms: float
    is_safe: bool
    security_flag: Optional[str] = None


class OSWorldBridge:
    """Interfaces cognitive architecture with local/sandboxed operating system operations."""

    BLOCKED_SHELL_PATTERNS = [
        r"(?i)rm\s+-rf\s+/",
        r"(?i)format\s+c:",
        r"(?i):(){ :\|:& };:",  # fork bomb
        r"(?i)dd\s+if=/dev/zero",
        r"(?i)shutdown|reboot",
    ]

    def __init__(self, working_dir: Optional[str] = None, timeout_sec: float = 10.0):
        self.working_dir = working_dir or os.getcwd()
        self.timeout_sec = timeout_sec
        self.shield = AdversarialShield()
        self.system_info = {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
        }

    def inspect_safety(self, command: str) -> Tuple[bool, Optional[str]]:
        """Verifies shell command against catastrophic destruction patterns."""
        import re
        for pat in self.BLOCKED_SHELL_PATTERNS:
            if re.search(pat, command):
                return False, f"Blocked catastrophic shell command pattern: '{pat}'"
        return True, None

    def execute_terminal_command(self, command: str) -> OSTaskResult:
        """Executes verified terminal command, capturing stdout, stderr, and execution time."""
        is_safe, flag = self.inspect_safety(command)
        if not is_safe:
            return OSTaskResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=flag or "Security violation",
                execution_time_ms=0.0,
                is_safe=False,
                security_flag=flag,
            )

        start_time = time.perf_counter()
        try:
            # Execute command using platform-appropriate shell
            use_shell = True
            proc = subprocess.run(
                command,
                shell=use_shell,
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_sec,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return OSTaskResult(
                command=command,
                return_code=proc.returncode,
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
                execution_time_ms=elapsed_ms,
                is_safe=True,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return OSTaskResult(
                command=command,
                return_code=-2,
                stdout="",
                stderr=f"Command timed out after {self.timeout_sec}s",
                execution_time_ms=elapsed_ms,
                is_safe=True,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return OSTaskResult(
                command=command,
                return_code=-3,
                stdout="",
                stderr=f"{type(e).__name__}: {str(e)}",
                execution_time_ms=elapsed_ms,
                is_safe=True,
            )

    def file_system_operation(self, op_type: str, path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """Manages OS filesystem operations (read, write, list, delete)."""
        target_path = os.path.abspath(os.path.join(self.working_dir, path))

        try:
            if op_type == "write":
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content or "")
                return {"success": True, "path": target_path, "bytes_written": len(content or "")}

            elif op_type == "read":
                if not os.path.exists(target_path):
                    return {"success": False, "error": f"File '{path}' does not exist."}
                with open(target_path, "r", encoding="utf-8") as f:
                    data = f.read()
                return {"success": True, "path": target_path, "content": data}

            elif op_type == "list":
                if not os.path.exists(target_path):
                    return {"success": False, "error": f"Directory '{path}' does not exist."}
                entries = os.listdir(target_path)
                return {"success": True, "path": target_path, "entries": entries}

            else:
                return {"success": False, "error": f"Unsupported filesystem operation: '{op_type}'"}

        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}
