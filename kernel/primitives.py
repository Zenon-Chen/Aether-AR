from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import Any, Awaitable, Callable

from memory.hot_context import HotWorkingContext
from memory.hy_memory_bus import HyMemorySidecarBus
from memory.outbox import DurableOutboxAndEpisodeStore
from .agent_loop import KernelTool, ToolOutput


def build_core_primitives(
    hot_context: HotWorkingContext,
    memory_bus: HyMemorySidecarBus,
    store: DurableOutboxAndEpisodeStore,
    request_hd_frame_cb: Callable[[], Awaitable[str | None]] | None = None,
    sync_intents_cb: Callable[[], Awaitable[None]] | None = None,
    python_timeout_sec: float = 15.0,
) -> list[KernelTool]:
    """
    Construct the 4 irreducible, infinitely generalizable primitives of PiKernel:
      1. `python_exec`: Exact math, statistics, logic, data processing via subprocess Python.
      2. `inspect_visual_memory`: Live HD camera capture, recent temporal keyframes, or historical episode snapshots.
      3. `memory_ops`: Deep recall, fact pinning, user profile update, or 0ms proactive briefing cache staging.
      4. `manage_autonomous_intents`: Self-programming `cron` background jobs & `visual_watch` proactive triggers.
    """

    # 1. Python Execution Primitive
    async def _python_exec(args: dict[str, Any]) -> str:
        code = str(args.get("code", "")).strip()
        if not code:
            return "Error: `code` is empty."
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=python_timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: Python execution timed out after {python_timeout_sec}s."

        out_str = stdout.decode("utf-8", errors="replace").strip()
        err_str = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            return f"Exit code {proc.returncode}\nSTDOUT:\n{out_str}\nSTDERR:\n{err_str}".strip()
        if err_str:
            return f"{out_str}\n[stderr]: {err_str}".strip()
        return out_str or "(executed successfully with no stdout; use print() to output results)"

    # 2. Multimodal Visual & Spatio-Temporal Inspection Primitive
    async def _inspect_visual_memory(args: dict[str, Any]) -> ToolOutput:
        mode = str(args.get("mode", "recent_frames"))
        query = str(args.get("query", ""))
        max_frames = max(1, min(4, int(args.get("max_frames", 2))))

        if mode == "live_hd":
            hd_b64: str | None = None
            if request_hd_frame_cb is not None:
                try:
                    hd_b64 = await request_hd_frame_cb()
                except Exception:
                    hd_b64 = None
            if not hd_b64:
                latest = store.get_latest_frame(prefer_hd=True)
                hd_b64 = latest.jpeg_b64 if latest else None
            if not hd_b64:
                return ToolOutput(text="当前没有可用的实时摄像头画面（边缘端尚未上传图像帧）。")
            return ToolOutput(
                text="已获取当前摄像头最新画面，图像已附加到上下文中供检视。",
                image_b64_list=[hd_b64],
            )

        if mode == "search_episodes":
            episodes = store.search_visual_episodes(query=query, limit=5)
            if not episodes:
                return ToolOutput(text=f"未在空间视觉事件库中找到与 '{query}' 匹配的记录。")
            imgs: list[str] = []
            lines = ["检索到以下历史空间/动作观察记录："]
            for ep in episodes:
                lines.append(f"- [{ep['iso_time']}] {ep['summary']}")
                if len(imgs) < max_frames and ep.get("frame_rel_path"):
                    b64 = store.load_episode_frame_b64(ep["frame_rel_path"])
                    if b64:
                        imgs.append(b64)
            return ToolOutput(text="\n".join(lines), image_b64_list=imgs)

        # Default: recent_frames
        recent = store.get_recent_frames(count=max_frames)
        if not recent:
            return ToolOutput(text="环形缓冲区暂无近期摄像头画面。")
        lines = [f"已调取最近 {len(recent)} 帧房间时序画面："]
        imgs = []
        for r in recent:
            lines.append(f"- [{r.iso_time}] motion_score={r.motion_score:.3f}")
            imgs.append(r.jpeg_b64)
        return ToolOutput(text="\n".join(lines), image_b64_list=imgs)

    # 3. Long-Term Memory & Proactive Briefing Staging Primitive
    async def _memory_ops(args: dict[str, Any]) -> str:
        action = str(args.get("action", "search"))
        content = str(args.get("content", "")).strip()
        key = str(args.get("key", "default")).strip() or "default"

        if action == "search":
            hits = await memory_bus.deep_recall(content, limit=8)
            if not hits:
                return "未检索到相关长期记忆。"
            return "\n".join(f"- {h}" for h in hits)

        if action == "pin_fact":
            await memory_bus.pin_explicit_fact(content)
            return f"已将重要事实异步归档至 Hy-Memory 2.0 并写入热上下文: {content}"

        if action == "update_profile":
            await hot_context.update_user_profile(content)
            await memory_bus.pin_explicit_fact(f"[用户核心偏好更新] {content}")
            return "已更新主人核心画像。"

        if action == "stage_briefing":
            await hot_context.set_briefing_cache(key, content)
            return f"已将预编译简报写入热块缓存 [{key}]，实时语音会话可零延迟直接播报。"

        return f"Unknown memory_ops action: {action}"

    # 4. Autonomous Proactive Intent & Cron Trigger Primitive
    async def _manage_intents(args: dict[str, Any]) -> str:
        action = str(args.get("action", "list"))
        if action == "list":
            rows = store.list_intents(only_enabled=True)
            if not rows:
                return "当前没有激活的自主定时或视觉守望意图。"
            return json.dumps(rows, ensure_ascii=False, indent=2)

        if action == "remove":
            iid = str(args.get("intent_id", "")).strip()
            removed = store.remove_intent(iid)
            if sync_intents_cb:
                await sync_intents_cb()
            return f"已移除意图 {iid}" if removed else f"未找到意图 {iid}"

        if action == "add":
            iid = str(args.get("intent_id", "")).strip() or f"intent_{uuid.uuid4().hex[:8]}"
            trigger_type = str(args.get("trigger_type", "visual_watch"))
            spec = str(args.get("spec", "")).strip()
            goal = str(args.get("goal", "")).strip()
            one_shot = bool(args.get("one_shot", False))
            if not spec or not goal:
                return "Error: `spec` and `goal` are required when adding an intent."
            store.upsert_intent(
                intent_id=iid,
                trigger_type=trigger_type,
                spec=spec,
                goal=goal,
                one_shot=one_shot,
            )
            if sync_intents_cb:
                await sync_intents_cb()
            return f"已激活自主意图 [{iid}] ({trigger_type}): spec='{spec}' -> goal='{goal}'"

        return f"Unknown action: {action}"

    return [
        KernelTool(
            name="python_exec",
            description="在隔离 Python 环境中运行代码并返回 stdout。用于精确数学/微积分/统计计算、日期换算或数据处理（记得用 print 输出结果）。",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的完整 Python 3 代码"},
                },
                "required": ["code"],
            },
            execute=_python_exec,
        ),
        KernelTool(
            name="inspect_visual_memory",
            description="检视房间摄像头画面或搜索历史空间视觉观察记录（用于看清细节、读手写题/标签、找东西放在哪了）。",
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["live_hd", "recent_frames", "search_episodes"],
                        "description": "live_hd=抓拍当前高清帧; recent_frames=查看最近几帧动作序列; search_episodes=按关键词搜索过去空间观察与存档快照",
                    },
                    "query": {"type": "string", "description": "当 mode=search_episodes 时的检索词（如'钥匙'、'药瓶'）"},
                    "max_frames": {"type": "integer", "description": "最多附加几张图片进入视觉上下文 (1~4)"},
                },
                "required": ["mode"],
            },
            execute=_inspect_visual_memory,
        ),
        KernelTool(
            name="memory_ops",
            description="操作长期记忆总线与零延迟简报缓存：深度检索 Hy-Memory 2.0、固定关键事实、更新用户画像或预存早报缓存。",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "pin_fact", "update_profile", "stage_briefing"],
                    },
                    "content": {"type": "string", "description": "查询语句、事实内容、画像描述或简报正文"},
                    "key": {"type": "string", "description": "当 action=stage_briefing 时的缓存键名（如 morning_briefing）"},
                },
                "required": ["action", "content"],
            },
            execute=_memory_ops,
        ),
        KernelTool(
            name="manage_autonomous_intents",
            description="管理 Agent 自主触发意图：支持 `cron`（5段式定时后台任务，如清早预编译天气早报）和 `visual_watch`（自然语言视觉观察触发条件，如看到主人起床/拿快递进门时主动提醒）。",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "remove", "list"]},
                    "intent_id": {"type": "string", "description": "意图唯一ID（add时可省略自动生成）"},
                    "trigger_type": {"type": "string", "enum": ["cron", "visual_watch"]},
                    "spec": {
                        "type": "string",
                        "description": "cron 时填标准5段表达式如 '30 6 * * *'；visual_watch 时填自然语言视觉条件如 '清晨看到主人睡醒坐起'",
                    },
                    "goal": {"type": "string", "description": "触发后要执行的目标或主动交互内容"},
                    "one_shot": {"type": "boolean", "description": "是否触发一次后自动关闭"},
                },
                "required": ["action"],
            },
            execute=_manage_intents,
        ),
    ]
