from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

logger = logging.getLogger("ambient.kernel")


@dataclass
class ToolOutput:
    """Rich tool execution output supporting text + optional multimodal image attachments."""
    text: str
    image_b64_list: list[str] = field(default_factory=list)


@dataclass
class KernelTool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], Awaitable[ToolOutput | str]]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class KernelEvent:
    type: str  # "agent_start" | "turn_start" | "tool_start" | "tool_end" | "turn_end" | "agent_end"
    step: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class PiKernel:
    """
    Ultra-thin, event-driven Agent Kernel inspired by `@mariozechner/pi-agent-core`.

    Design Philosophy:
      - Zero bloated prompt templates (< 500 tokens core prompt).
      - Minimal irreducible primitives + dynamic MCP discovery proxy.
      - Native multimodal inspection: tools can attach live/historical camera JPEGs
        directly into the reasoning context for `qwen3.8-omni-flash`.
    """

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        model: str,
        max_steps: int = 10,
        system_prompt_provider: Callable[[], str] | None = None,
    ) -> None:
        self.llm = llm_client
        self.model = model
        self.max_steps = max_steps
        self._system_prompt_provider = system_prompt_provider
        self._tools: dict[str, KernelTool] = {}
        self._subscribers: list[Callable[[KernelEvent], None]] = []

    def register_tool(self, tool: KernelTool) -> None:
        self._tools[tool.name] = tool

    def subscribe(self, listener: Callable[[KernelEvent], None]) -> None:
        self._subscribers.append(listener)

    def _emit(self, ev_type: str, step: int = 0, **detail: Any) -> None:
        ev = KernelEvent(type=ev_type, step=step, detail=detail)
        for sub in list(self._subscribers):
            try:
                sub(ev)
            except Exception:
                pass

    def _build_system_prompt(self) -> str:
        base = (
            "你是后端极薄智能体内核 (PiKernel)，负责精确执行复杂推理、数学验算、高清/历史视觉检视、"
            "长期记忆管理、动态主动意图调度以及通用 MCP 外部工具调用。\n"
            "准则：\n"
            "1. 遇到数学、统计、逻辑计算必须使用 `python_exec` 验证，绝不靠口算猜测。\n"
            "2. 需要调用外部服务时，先用 `mcp_discover` 查看可用工具参数，再用 `mcp_call` 执行。\n"
            "3. 最终回复将被送回实时语音前端播报给主人，请直接给出准确、简洁、口语友好的结论。"
        )
        if self._system_prompt_provider:
            extra = self._system_prompt_provider().strip()
            if extra:
                base = f"{base}\n\n{extra}"
        return base

    async def run(
        self,
        goal: str,
        initial_images_b64: list[str] | None = None,
        enable_thinking: bool = True,
    ) -> str:
        """Execute a complete autonomous agent loop for `goal` and return synthesized answer."""
        self._emit("agent_start", step=0, goal=goal)

        user_content: str | list[dict[str, Any]]
        if initial_images_b64:
            blocks: list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                for img in initial_images_b64
                if img
            ]
            blocks.append({"type": "text", "text": goal})
            user_content = blocks
        else:
            user_content = goal

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        tool_schemas = [t.to_openai_schema() for t in self._tools.values()]

        final_text = ""
        for step in range(1, self.max_steps + 1):
            self._emit("turn_start", step=step)
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1200,
                "extra_body": {"enable_thinking": enable_thinking},
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas

            resp = await self.llm.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            assistant_text = (msg.content or "").strip()
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                final_text = assistant_text
                self._emit("turn_end", step=step, finished=True, text=final_text[:200])
                break

            # Append assistant tool_calls turn
            assistant_dict: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_dict)

            pending_visual_attachments: list[str] = []

            for tc in tool_calls:
                t_name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    parsed_args = json.loads(raw_args)
                    if not isinstance(parsed_args, dict):
                        parsed_args = {"value": parsed_args}
                except Exception:
                    parsed_args = {}

                self._emit("tool_start", step=step, tool=t_name, args=parsed_args)
                tool_obj = self._tools.get(t_name)
                if tool_obj is None:
                    out_text = f"Error: Unknown tool '{t_name}'. Available: {list(self._tools.keys())}"
                else:
                    try:
                        raw_out = await tool_obj.execute(parsed_args)
                        if isinstance(raw_out, ToolOutput):
                            out_text = raw_out.text
                            pending_visual_attachments.extend(raw_out.image_b64_list)
                        else:
                            out_text = str(raw_out)
                    except Exception as exc:
                        logger.warning("Tool %s failed: %s", t_name, exc)
                        out_text = f"Tool execution error: {exc}"

                # Clamp excessive output (pi-agent-kernel style context protection)
                if len(out_text) > 4000:
                    out_text = out_text[:3800] + "\n...[output truncated]..."

                self._emit("tool_end", step=step, tool=t_name, result_preview=out_text[:200])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": out_text,
                })

            # If any tool returned camera frames, inject them cleanly as a multimodal observation message
            if pending_visual_attachments:
                img_blocks: list[dict[str, Any]] = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    for b64 in pending_visual_attachments[:4]
                ]
                img_blocks.append({"type": "text", "text": "以上是刚调取的摄像头视觉帧，请结合画面继续分析并完成目标。"})
                messages.append({"role": "user", "content": img_blocks})

            self._emit("turn_end", step=step, finished=False)

        if not final_text:
            final_text = "后台任务已处理完成。"
        self._emit("agent_end", step=step, answer=final_text[:300])
        return final_text
