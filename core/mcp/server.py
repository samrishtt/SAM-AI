"""Model Context Protocol (MCP) Server Implementation.

Allows Micro-AGI to host and serve custom tools, file resources, and prompts
over standard MCP JSON-RPC 2.0 interfaces.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import sys
from typing import Any, Callable, Dict, List, Optional, Union

from core.mcp.protocol import MCPTool, MCPResource, MCPProtocolHandler


class MCPServer:
    """Full-featured Model Context Protocol (MCP) Server."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self.handler = MCPProtocolHandler()

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> MCPTool:
        """Registers an executable tool on the MCP server."""
        tool = MCPTool(name=name, description=description, inputSchema=input_schema, handler=handler)
        self._tools[name] = tool
        return tool

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        reader: Callable[[], str],
        mime_type: str = "text/plain",
    ) -> MCPResource:
        """Registers a readable resource manifest on the MCP server."""
        resource = MCPResource(uri=uri, name=name, description=description, reader=reader, mimeType=mime_type)
        self._resources[uri] = resource
        return resource

    def handle_request(self, raw_json: str) -> str:
        """Processes an incoming JSON-RPC 2.0 MCP request and returns the serialized response."""
        msg = self.handler.parse_message(raw_json)
        if "error" in msg:
            return json.dumps(msg)

        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})

        # 1. MCP initialize handshake
        if method == "initialize":
            init_res = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": False, "listChanged": True},
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version,
                },
            }
            return self.handler.create_response(init_res, req_id=req_id)

        # 2. tools/list
        elif method == "tools/list":
            tools_list = [tool.to_dict() for tool in self._tools.values()]
            return self.handler.create_response({"tools": tools_list}, req_id=req_id)

        # 3. tools/call
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name not in self._tools:
                return self.handler.create_error(
                    code=-32601,
                    message=f"Tool '{tool_name}' not found on server '{self.name}'.",
                    req_id=req_id,
                )

            tool = self._tools[tool_name]
            try:
                output = tool.handler(**arguments) if tool.handler else None
                return self.handler.create_response({
                    "content": [{"type": "text", "text": str(output)}],
                    "isError": False,
                }, req_id=req_id)
            except Exception as e:
                return self.handler.create_response({
                    "content": [{"type": "text", "text": f"ExecutionError: {type(e).__name__} ({str(e)})"}],
                    "isError": True,
                }, req_id=req_id)

        # 4. resources/list
        elif method == "resources/list":
            resources_list = [res.to_dict() for res in self._resources.values()]
            return self.handler.create_response({"resources": resources_list}, req_id=req_id)

        # 5. resources/read
        elif method == "resources/read":
            uri = params.get("uri")
            if uri not in self._resources:
                return self.handler.create_error(code=-32602, message=f"Resource '{uri}' not found.", req_id=req_id)
            res_obj = self._resources[uri]
            content_text = res_obj.reader() if res_obj.reader else ""
            return self.handler.create_response({
                "contents": [{"uri": uri, "mimeType": res_obj.mimeType, "text": content_text}],
            }, req_id=req_id)

        # Fallback: Method not found
        else:
            return self.handler.create_error(code=-32601, message=f"Method '{method}' not implemented.", req_id=req_id)
