from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from .agent_loop import KernelTool

logger = logging.getLogger("ambient.mcp")


@dataclass
class MCPServerSpec:
    name: str
    description: str
    transport: str  # "stdio" | "url"
    command: str = ""
    args: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str = ""


class DynamicMCPHub:
    """
    Universal Model Context Protocol (MCP) Bridge with Anti-Bloat Dynamic Discovery.

    Follows Pi's minimal context principle:
      - Regardless of how many MCP servers are configured in `mcp_servers.json`,
        only a compact directory summary is placed in the Kernel system prompt.
      - The Agent dynamically discovers tool schemas via `mcp_discover` and invokes
        any stdio / SSE / Streamable HTTP MCP tool via `mcp_call`.
    """

    def __init__(self, config_path: Path, project_root: Path | None = None) -> None:
        self.config_path = config_path
        self.project_root = project_root or config_path.resolve().parent.parent
        self._specs: dict[str, MCPServerSpec] = {}
        self._schema_cache: dict[str, list[dict[str, Any]]] = {}
        self.reload_config()

    def reload_config(self) -> None:
        self._specs.clear()
        self._schema_cache.clear()
        if not self.config_path.exists():
            return
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse MCP config %s: %s", self.config_path, exc)
            return

        servers = raw.get("mcpServers", {})
        for name, cfg in servers.items():
            if not isinstance(cfg, dict) or cfg.get("enabled", True) is False:
                continue
            desc = str(cfg.get("description", f"MCP Server '{name}'"))
            if "url" in cfg and cfg["url"]:
                self._specs[name] = MCPServerSpec(
                    name=name,
                    description=desc,
                    transport="url",
                    url=str(cfg["url"]),
                )
            elif "command" in cfg and cfg["command"]:
                cmd = str(cfg["command"])
                # Resolve generic "python" to current venv interpreter for seamless execution
                if cmd in ("python", "python3"):
                    cmd = sys.executable
                merged_env = dict(os.environ)
                merged_env["PYTHONPATH"] = str(self.project_root) + os.pathsep + merged_env.get("PYTHONPATH", "")
                if isinstance(cfg.get("env"), dict):
                    for k, v in cfg["env"].items():
                        merged_env[str(k)] = str(v)
                self._specs[name] = MCPServerSpec(
                    name=name,
                    description=desc,
                    transport="stdio",
                    command=cmd,
                    args=[str(a) for a in cfg.get("args", [])],
                    env=merged_env,
                    cwd=str(cfg.get("cwd") or self.project_root),
                )

    def render_directory_summary(self) -> str:
        if not self._specs:
            return "【已挂载 MCP 服务目录】当前未配置外部 MCP 服务。"
        lines = ["【已挂载 MCP 服务目录（使用 `mcp_discover` 查询具体工具参数，再用 `mcp_call` 调用）】"]
        for spec in self._specs.values():
            lines.append(f"- `{spec.name}`: {spec.description}")
        return "\n".join(lines)

    def _build_client_target(self, spec: MCPServerSpec) -> StdioServerParameters | str:
        if spec.transport == "url":
            return spec.url
        return StdioServerParameters(
            command=spec.command,
            args=spec.args or [],
            env=spec.env,
            cwd=spec.cwd,
        )

    async def list_server_tools(self, server_name: str, force_refresh: bool = False) -> list[dict[str, Any]]:
        if server_name not in self._specs:
            raise ValueError(f"MCP server '{server_name}' not found. Configured: {list(self._specs.keys())}")
        if not force_refresh and server_name in self._schema_cache:
            return self._schema_cache[server_name]

        spec = self._specs[server_name]
        target = self._build_client_target(spec)
        async with Client(target, read_timeout_seconds=20.0) as client:
            res = await client.list_tools()
            tools_out: list[dict[str, Any]] = []
            for t in res.tools:
                schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None) or {}
                tools_out.append({
                    "server": server_name,
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": schema,
                })
            self._schema_cache[server_name] = tools_out
            return tools_out

    async def discover_tools(self, query: str = "", server: str = "") -> str:
        self.reload_config() if not self._specs else None
        target_servers = [server] if server else list(self._specs.keys())
        matched: list[dict[str, Any]] = []
        q_lower = query.strip().lower()

        for srv in target_servers:
            if srv not in self._specs:
                continue
            try:
                tools = await self.list_server_tools(srv)
                for t in tools:
                    haystack = f"{t['server']} {t['name']} {t['description']}".lower()
                    if not q_lower or any(tok in haystack for tok in q_lower.split()):
                        matched.append(t)
            except Exception as exc:
                matched.append({"server": srv, "error": str(exc)})

        if not matched and q_lower:
            # If keyword was too specific, return all available tools across target servers
            for srv in target_servers:
                if srv in self._schema_cache:
                    matched.extend(self._schema_cache[srv])

        return json.dumps({"tools": matched}, ensure_ascii=False, indent=2)

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        if server_name not in self._specs:
            return f"Error: Unknown MCP server '{server_name}'. Available: {list(self._specs.keys())}"
        spec = self._specs[server_name]
        target = self._build_client_target(spec)
        async with Client(target, read_timeout_seconds=30.0) as client:
            result = await client.call_tool(tool_name, arguments=arguments or {})
            parts: list[str] = []
            for block in getattr(result, "content", []):
                text = getattr(block, "text", None)
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(str(block))
            is_err = getattr(result, "isError", False) or getattr(result, "is_error", False)
            joined = "\n".join(parts) if parts else json.dumps(str(result), ensure_ascii=False)
            return f"[MCP Error] {joined}" if is_err else joined

    def build_kernel_tools(self) -> list[KernelTool]:
        async def _exec_discover(args: dict[str, Any]) -> str:
            return await self.discover_tools(
                query=str(args.get("query", "")),
                server=str(args.get("server", "")),
            )

        async def _exec_call(args: dict[str, Any]) -> str:
            raw_tool_args = args.get("arguments") or {}
            if isinstance(raw_tool_args, str):
                try:
                    raw_tool_args = json.loads(raw_tool_args)
                except Exception:
                    raw_tool_args = {}
            return await self.call_tool(
                server_name=str(args.get("server", "")),
                tool_name=str(args.get("tool_name", "")),
                arguments=raw_tool_args if isinstance(raw_tool_args, dict) else {},
            )

        return [
            KernelTool(
                name="mcp_discover",
                description="动态发现或搜索已挂载的 MCP Server 工具列表及 JSON 参数定义。",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词（留空则列出全部可用工具）"},
                        "server": {"type": "string", "description": "可选：指定 MCP Server 名称"},
                    },
                },
                execute=_exec_discover,
            ),
            KernelTool(
                name="mcp_call",
                description="调用指定 MCP Server 上的具体工具。",
                parameters={
                    "type": "object",
                    "properties": {
                        "server": {"type": "string", "description": "MCP Server 名称"},
                        "tool_name": {"type": "string", "description": "工具名称"},
                        "arguments": {"type": "object", "description": "传给该 MCP 工具的参数对象"},
                    },
                    "required": ["server", "tool_name"],
                },
                execute=_exec_call,
            ),
        ]
