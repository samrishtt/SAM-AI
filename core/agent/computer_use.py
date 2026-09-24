"""Windows OS Computer-Use Agent Subsystem (Astra & Claude Computer-Use Parity).

Enables the autonomous agent to interact directly with the operating system:
- Detecting screen dimensions and active applications
- Launching desktop applications (Notepad, Browser, Calculator, Terminal, etc.)
- Controlling mouse pointer coordinates and synthetic click events
- Typing keyboard input via Windows input subsystem
- Taking desktop screenshots
- Querying and managing system processes
"""

from __future__ import annotations
import ctypes
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class WindowInfo:
    pid: int
    process_name: str
    title: str


@dataclass
class ComputerUseResult:
    success: bool
    action: str
    details: str
    output: Optional[str] = None


class WindowsComputerUseAgent:
    """Autonomous Windows desktop automation and computer-use controller."""

    # Mouse event flags
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_MIDDLEDOWN = 0x0020
    MOUSEEVENTF_MIDDLEUP = 0x0040

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.user32 = getattr(ctypes, "windll", None)
        if self.user32:
            self.user32 = self.user32.user32

    def get_screen_resolution(self) -> Tuple[int, int]:
        """Returns the primary monitor resolution (width, height)."""
        if self.user32:
            w = self.user32.GetSystemMetrics(0)
            h = self.user32.GetSystemMetrics(1)
            return (w, h)
        return (1920, 1080)

    def get_cursor_position(self) -> Tuple[int, int]:
        """Returns current mouse cursor coordinates."""
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        if self.user32:
            pt = POINT()
            self.user32.GetCursorPos(ctypes.byref(pt))
            return (pt.x, pt.y)
        return (0, 0)

    def move_mouse(self, x: int, y: int) -> ComputerUseResult:
        """Moves cursor to exact (x, y) desktop coordinates."""
        if not self.user32:
            return ComputerUseResult(False, "move_mouse", "Windows user32 API not available")
        w, h = self.get_screen_resolution()
        x_clamped = max(0, min(x, w - 1))
        y_clamped = max(0, min(y, h - 1))
        self.user32.SetCursorPos(x_clamped, y_clamped)
        return ComputerUseResult(True, "move_mouse", f"Cursor moved to ({x_clamped}, {y_clamped})")

    def click_mouse(self, button: str = "left", double: bool = False) -> ComputerUseResult:
        """Triggers synthetic mouse click event."""
        if not self.user32:
            return ComputerUseResult(False, "click_mouse", "Windows user32 API not available")

        btn = button.lower()
        down_flag = self.MOUSEEVENTF_LEFTDOWN if btn == "left" else self.MOUSEEVENTF_RIGHTDOWN
        up_flag = self.MOUSEEVENTF_LEFTUP if btn == "left" else self.MOUSEEVENTF_RIGHTUP

        # Single or double click
        clicks = 2 if double else 1
        for _ in range(clicks):
            self.user32.mouse_event(down_flag, 0, 0, 0, 0)
            time.sleep(0.05)
            self.user32.mouse_event(up_flag, 0, 0, 0, 0)
            if clicks > 1:
                time.sleep(0.1)

        pos = self.get_cursor_position()
        return ComputerUseResult(True, "click_mouse", f"{button.capitalize()} {'double-' if double else ''}click at {pos}")

    def type_text(self, text: str) -> ComputerUseResult:
        """Sends keystrokes to active window using Windows WScript.Shell."""
        # Escape characters for SendKeys: +, ^, %, ~, (, ), {, }
        escaped = ""
        for c in text:
            if c in "+^%~(){}[]":
                escaped += f"{{{c}}}"
            elif c == "\n":
                escaped += "{ENTER}"
            elif c == "\t":
                escaped += "{TAB}"
            else:
                escaped += c

        ps_cmd = f'$ws = New-Object -ComObject WScript.Shell; $ws.SendKeys("{escaped}")'
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True, timeout=5)
            return ComputerUseResult(True, "type_text", f"Typed {len(text)} characters into active window")
        except Exception as e:
            return ComputerUseResult(False, "type_text", f"Keystroke transmission failed: {str(e)}")

    def launch_app(self, app_name: str) -> ComputerUseResult:
        """Launches target Windows desktop application."""
        app_map = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "calc": "calc.exe",
            "explorer": "explorer.exe",
            "terminal": "powershell.exe",
            "cmd": "cmd.exe",
            "chrome": "start chrome",
            "edge": "start msedge",
            "vscode": "code",
            "code": "code",
        }

        target = app_map.get(app_name.lower().strip(), app_name.strip())
        try:
            if target.startswith("start "):
                subprocess.Popen(target, shell=True)
            else:
                subprocess.Popen([target], shell=True)
            return ComputerUseResult(True, "launch_app", f"Launched application: {target}")
        except Exception as e:
            return ComputerUseResult(False, "launch_app", f"Failed to launch '{app_name}': {str(e)}")

    def list_open_windows(self) -> List[WindowInfo]:
        """Lists active running graphical windows on the desktop."""
        ps_script = (
            "Get-Process | Where-Object { $_.MainWindowTitle -ne '' } | "
            "Select-Object Id, ProcessName, MainWindowTitle | "
            "ConvertTo-Json -Compress"
        )
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=5,
            )
            raw = res.stdout.strip()
            if not raw:
                return []
            import json
            data = json.loads(raw)
            if isinstance(data, dict):
                data = [data]
            windows = []
            for item in data:
                windows.append(
                    WindowInfo(
                        pid=item.get("Id", 0),
                        process_name=item.get("ProcessName", ""),
                        title=item.get("MainWindowTitle", ""),
                    )
                )
            return windows
        except Exception:
            return []

    def capture_screenshot(self, filename: str = "screenshot.png") -> ComputerUseResult:
        """Captures full-desktop screenshot and saves to file."""
        out_path = os.path.join(self.workspace_root, filename)
        clean_path = out_path.replace("\\", "/")
        ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$bitmap.Save("{clean_path}")
$graphics.Dispose()
$bitmap.Dispose()
"""
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True, timeout=10)
            if os.path.exists(out_path):
                return ComputerUseResult(True, "capture_screenshot", f"Screenshot saved to {out_path}", output=out_path)
            return ComputerUseResult(False, "capture_screenshot", "Screenshot file not found after capture")
        except Exception as e:
            return ComputerUseResult(False, "capture_screenshot", f"Screen capture error: {str(e)}")
