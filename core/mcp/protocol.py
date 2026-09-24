"""Model Context Protocol (MCP) Core Definitions & JSON-RPC 2.0 Engine.

Implements the standard Anthropic/OpenAI Model Context Protocol specifications:
JSON-RPC 2.0 messages, Tool schemas, Resource manifests, and Prompt templates.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class MCPToolParameter:
    type: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)


@dataclass
class MCPTool:
    name: str
    description: str
    inputSchema: Dict[str, Any]
    handler: Optional[Callable[..., Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


@dataclass
class MCPResource:
    uri: str
    name: str
    description: str
    mimeType: str = "text/plain"
    reader: Optional[Callable[[], str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mimeType,
        }


class MCPProtocolHandler:
    """Processes incoming and outgoing JSON-RPC 2.0 MCP payloads."""

    @staticmethod
    def create_request(method: str, params: Optional[Dict[str, Any]] = None, req_id: Union[int, str] = 1) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }
        return json.dumps(payload)

    @staticmethod
    def create_response(result: Any, req_id: Union[int, str] = 1) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }
        return json.dumps(payload)

    @staticmethod
    def create_error(code: int, message: str, req_id: Optional[Union[int, str]] = None, data: Any = None) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": code,
                "message": message,
                "data": data,
            },
        }
        return json.dumps(payload)

    @staticmethod
    def parse_message(raw_json: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError as e:
            return {"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {e}"}}
