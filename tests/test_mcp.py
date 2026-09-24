"""Unit tests for Model Context Protocol (MCP) server, client, and generator."""

import pytest
import json
from core.mcp.protocol import MCPTool, MCPProtocolHandler
from core.mcp.server import MCPServer
from core.mcp.client import MCPClient
from core.mcp.mcp_generator import MCPGenerator, ToolSpec
from core.execution.tool_registry import ToolRegistry


def test_mcp_server_client_roundtrip():
    server = MCPServer(name="TestAnalyticsServer")

    # Register custom tool
    def _add_numbers(a: int, b: int) -> int:
        return a + b

    server.register_tool(
        name="add_numbers",
        description="Adds two integers together",
        input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
        handler=_add_numbers,
    )

    # Register resource
    server.register_resource(
        uri="resource://metrics/active",
        name="Active Metrics",
        description="Live operational metrics",
        reader=lambda: "System Status: Nominal, Load: 0.12",
    )

    # Connect client
    client = MCPClient(server_target=server)
    conn_res = client.connect()
    assert conn_res["success"]
    assert conn_res["toolCount"] == 1
    assert conn_res["resourceCount"] == 1
    assert "add_numbers" in client.discovered_tools

    # Call tool via client
    call_res = client.call_tool("add_numbers", {"a": 40, "b": 2})
    assert call_res["success"]
    assert call_res["output"] == "42"

    # Attach to ToolRegistry
    reg = ToolRegistry()
    client.attach_to_tool_registry(reg)
    tool_exec_res = reg.execute_tool("add_numbers", a=10, b=5)
    assert tool_exec_res["success"]
    assert tool_exec_res["result"] == "15"


def test_mcp_generator(tmp_path):
    tools = [
        ToolSpec(
            name="multiply",
            description="Multiplies two numbers",
            parameters={"x": {"type": "integer"}, "y": {"type": "integer"}},
            implementation_code="return x * y",
        )
    ]
    out_script = tmp_path / "math_mcp_server.py"
    script_str = MCPGenerator.generate_server_script("MathServer", tools, output_file_path=str(out_script))

    assert "def multiply(x, y):" in script_str
    assert "TOOLS_SCHEMA" in script_str
    assert out_script.exists()
