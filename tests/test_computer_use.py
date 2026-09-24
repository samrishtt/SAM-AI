"""Unit tests for Windows Computer-Use Agent subsystem."""

import pytest
import os
from core.agent.computer_use import WindowsComputerUseAgent, ComputerUseResult


@pytest.fixture
def agent(tmp_path):
    return WindowsComputerUseAgent(workspace_root=str(tmp_path))


def test_screen_resolution_detection(agent):
    w, h = agent.get_screen_resolution()
    assert w > 0
    assert h > 0
    print(f"Detected screen resolution: {w}x{h}")


def test_cursor_position_and_movement(agent):
    pos = agent.get_cursor_position()
    assert isinstance(pos, tuple)
    assert len(pos) == 2

    res = agent.move_mouse(100, 100)
    assert res.success
    assert "Cursor moved to" in res.details


def test_list_open_windows(agent):
    windows = agent.list_open_windows()
    assert isinstance(windows, list)
    print(f"Discovered {len(windows)} active desktop windows.")


def test_mouse_click_synthetic(agent):
    # Click should succeed without throwing exceptions
    res = agent.click_mouse(button="left", double=False)
    assert res.success
    assert "Left click at" in res.details
