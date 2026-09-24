"""Unit tests for OSWorldBridge operating system automation."""

import pytest
import os
from core.operator.os_world_bridge import OSWorldBridge, OSTaskResult


def test_os_world_bridge_terminal_execution():
    bridge = OSWorldBridge()
    # Test simple safe command
    res: OSTaskResult = bridge.execute_terminal_command("python --version")
    assert res.is_safe
    assert res.return_code == 0
    assert "Python" in (res.stdout + res.stderr)


def test_os_world_bridge_blocks_dangerous_command():
    bridge = OSWorldBridge()
    dangerous_cmd = "rm -rf /"
    res: OSTaskResult = bridge.execute_terminal_command(dangerous_cmd)
    assert not res.is_safe
    assert res.return_code == -1
    assert "Blocked catastrophic shell command" in res.stderr


def test_os_world_bridge_filesystem_operations(tmp_path):
    bridge = OSWorldBridge(working_dir=str(tmp_path))

    # Test write
    write_res = bridge.file_system_operation("write", "test_file.txt", "Hello OSWorld AGI")
    assert write_res["success"]

    # Test read
    read_res = bridge.file_system_operation("read", "test_file.txt")
    assert read_res["success"]
    assert read_res["content"] == "Hello OSWorld AGI"

    # Test list
    list_res = bridge.file_system_operation("list", ".")
    assert list_res["success"]
    assert "test_file.txt" in list_res["entries"]
