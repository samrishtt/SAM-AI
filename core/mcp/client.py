"""Model Context Protocol (MCP) Client Implementation.

Allows Micro-AGI to connect to local or remote MCP servers, discover available
tools and resources dynamically, and execute MCP tools inside its cognitive loop.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, List, Optional

from core.mcp.protocol import MCPProtocolHandler
from core.mcp.server import MCPServer
from core.execution.tool_registry import Tool, ToolParameter, ToolRegistry


class MCPClient:
    """Client connecting to and consuming services from an MCP server."""

    def __init__(self, server_target: Optional[MCPServer] = None):
        self.server = server_target
        self.handler = MCPProtocolHandler()
        self.discovered_tools: Dict[str, Dict[str, Any]] = {}
        self.discovered_resources: Dict[str, Dict[str, Any]] = {}
        self._request_counter = 0

    def connect(self) -> Dict[str, Any]:
        """Performs initial handshake and discovers capabilities."""
        if not self.server:
            return {"success": False, "error": "No server configured"}

        self._request_counter += 1
        req = self.handler.create_request("initialize", {}, req_id=self._request_counter)
        resp_str = self.server.handle_request(req)
        resp = json.loads(resp_str)

        # Discover tools
        self._request_counter += 1
        req_tools = self.handler.create_request("tools/list", {}, req_id=self._request_counter)
        tools_resp = json.loads(self.server.handle_request(req_tools))
        if "result" in tools_resp and "tools" in tools_resp["result"]:
            for t in tools_resp["result"]["tools"]:
                self.discovered_tools[t["name"]] = t

        # Discover resources
        self._request_counter += 1
        req_res = self.handler.create_request("resources/list", {}, req_id=self._request_counter)
        res_resp = json.loads(self.server.handle_request(req_res))
        if "result" in res_resp and "resources" in res_resp["result"]:
            for r in res_resp["result"]["resources"]:
                self.discovered_resources[r["uri"]] = r

        return {
            "success": True,
            "serverInfo": resp.get("result", {}).get("serverInfo", {}),
            "toolCount": len(self.discovered_tools),
            "resourceCount": len(self.discovered_resources),
        }

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a tool on the connected MCP server."""
        if not self.server:
            return {"success": False, "error": "No server connected."}

        self._request_counter += 1
        req = self.handler.create_request("tools/call", {"name": name, "arguments": arguments}, req_id=self._request_counter)
        resp_str = self.server.handle_request(req)
        resp = json.loads(resp_str)

        if "error" in resp:
            return {"success": False, "error": resp["error"]}

        content_list = resp.get("result", {}).get("content", [])
        text_output = "\n".join(c.get("text", "") for c in content_list if c.get("type") == "text")
        return {
            "success": not resp.get("result", {}).get("isError", False),
            "output": text_output,
        }

    def attach_to_tool_registry(self, registry: ToolRegistry) -> None:
        """Exposes all discovered MCP tools as native tools in the ToolRegistry."""
        for tool_name, schema in self.discovered_tools.items():
            desc = schema.get("description", "")
            params_schema = schema.get("inputSchema", {}).get("properties", {})
            required_params = schema.get("inputSchema", {}).get("required", [])

            tool_params = []
            for p_name, p_def in params_schema.items():
                tool_params.append(
                    ToolParameter(
                        name=p_name,
                        param_type=p_def.get("type", "string"),
                        description=p_def.get("description", ""),
                        required=(p_name in required_params),
                    )
                )

            # Handler closure
            def _create_mcp_handler(t_name: str):
                def _handler(**kwargs):
                    res = self.call_tool(t_name, kwargs)
                    if not res["success"]:
                        raise RuntimeError(f"MCP Tool '{t_name}' failed: {res.get('error')}")
                    return res["output"]
                return _handler

            registry.register_tool(
                Tool(
                    name=tool_name,
                    description=f"[MCP] {desc}",
                    parameters=tool_params,
                    handler=_create_mcp_handler(tool_name),
                )
            )
