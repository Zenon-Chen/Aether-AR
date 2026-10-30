from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from config.loader import AppConfig
from kernel.agent_loop import PiKernel
from kernel.mcp_bridge import DynamicMCPHub
from kernel.primitives import build_core_primitives
from kernel.scheduler import AutonomousIntentScheduler
from memory.hot_context import HotWorkingContext
from memory.hy_memory_bus import HyMemorySidecarBus
from memory.outbox import DurableOutboxAndEpisodeStore


class MockToolCall:
    def __init__(self, tc_id: str, name: str, arguments: dict[str, Any]) -> None:
        self.id = tc_id
        self.function = type("Fn", (), {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)})()


class ScriptedLLM:
    """Deterministic OpenAI-compatible mock driving multi-step PiKernel tool loops."""

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self._turns = list(turns)
        self.captured_messages: list[Any] = []
        self.chat = type("Chat", (), {"completions": self})()

    async def create(self, **kwargs: Any) -> Any:
        self.captured_messages.append(kwargs.get("messages", []))
        step = self._turns.pop(0)
        msg = type(
            "Msg",
            (),
            {
                "content": step.get("content", ""),
                "tool_calls": step.get("tool_calls", []),
            },
        )()
        return type("Resp", (), {"choices": [type("Choice", (), {"message": msg})()]})()


def _sample_jpeg_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (80, 60), (40, 200, 100)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_pi_kernel_python_math_and_visual_inspection(tmp_path: Path) -> None:
    cfg = AppConfig(
        omni_api_base="http://localhost",
        omni_api_key="",
        omni_model="mock-omni",
        realtime_ws_url="ws://localhost",
        realtime_model="mock-rt",
        data_dir=tmp_path,
    )
    hot_ctx = HotWorkingContext()
    store = DurableOutboxAndEpisodeStore(data_dir=tmp_path)
    store.record_visual_episode("主人把蓝色护照夹放在书架第三层右侧", importance=2, jpeg_b64=_sample_jpeg_b64())

    scripted_llm = ScriptedLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    MockToolCall("c1", "python_exec", {"code": "print(sum(i*i for i in range(1, 21)))"}),
                    MockToolCall("c2", "inspect_visual_memory", {"mode": "search_episodes", "query": "护照"}),
                ],
            },
            {
                "content": "平方和等于 2870，另外你的蓝色护照夹在书架第三层右侧。",
                "tool_calls": [],
            },
        ]
    )

    bus = HyMemorySidecarBus(cfg, hot_ctx, store, llm_client=scripted_llm)  # type: ignore[arg-type]
    kernel = PiKernel(llm_client=scripted_llm, model="mock-omni")  # type: ignore[arg-type]
    for t in build_core_primitives(hot_ctx, bus, store):
        kernel.register_tool(t)

    events: list[str] = []
    kernel.subscribe(lambda ev: events.append(ev.type))

    ans = await kernel.run("算一下1到20的平方和，并找一下我的护照放在哪")
    assert "2870" in ans
    assert "书架第三层" in ans
    assert "agent_start" in events and "tool_start" in events and "agent_end" in events

    # Verify multimodal image attachment from historical episode was injected into second LLM call!
    second_call_msgs = scripted_llm.captured_messages[1]
    assert any(
        isinstance(m.get("content"), list)
        and any(b.get("type") == "image_url" for b in m["content"])
        for m in second_call_msgs
    )
    store.close()


@pytest.mark.asyncio
async def test_dynamic_mcp_hub_stdio_discovery_and_call(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    mcp_cfg_file = tmp_path / "mcp_servers.json"
    mcp_cfg_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "ambient_utilities": {
                        "command": "python",
                        "args": ["-m", "kernel.builtin_mcp_demo"],
                        "description": "内置演示扩展 MCP 服务",
                        "enabled": True,
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    hub = DynamicMCPHub(config_path=mcp_cfg_file, project_root=project_root)
    summary = hub.render_directory_summary()
    assert "ambient_utilities" in summary

    discovered_json = await hub.discover_tools(query="clock")
    discovered = json.loads(discovered_json)
    assert any(t["name"] == "get_environment_clock" for t in discovered["tools"])

    clock_res = await hub.call_tool("ambient_utilities", "get_environment_clock", {})
    clock_data = json.loads(clock_res)
    assert "iso_datetime" in clock_data and "weekday" in clock_data

    note_res = await hub.call_tool(
        "ambient_utilities",
        "add_home_note",
        {"title": "测试便签", "content": "买牛奶和咖啡豆"},
    )
    assert "已保存家庭便签" in note_res


@pytest.mark.asyncio
async def test_autonomous_intent_scheduler_cron_and_visual_watch(tmp_path: Path) -> None:
    cfg = AppConfig(
        omni_api_base="http://localhost",
        omni_api_key="",
        omni_model="mock-omni",
        realtime_ws_url="ws://localhost",
        realtime_model="mock-rt",
        data_dir=tmp_path,
    )
    hot_ctx = HotWorkingContext()
    store = DurableOutboxAndEpisodeStore(data_dir=tmp_path)

    scripted_llm = ScriptedLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    MockToolCall(
                        "c_brief",
                        "memory_ops",
                        {
                            "action": "stage_briefing",
                            "key": "scheduled_briefing",
                            "content": "今日晴朗 24度，上午无会议。",
                        },
                    )
                ],
            },
            {"content": "早间简报已预编译完成。", "tool_calls": []},
        ]
    )
    bus = HyMemorySidecarBus(cfg, hot_ctx, store, llm_client=scripted_llm)  # type: ignore[arg-type]
    kernel = PiKernel(llm_client=scripted_llm, model="mock-omni")  # type: ignore[arg-type]

    scheduler = AutonomousIntentScheduler(store=store, hot_context=hot_ctx, kernel=kernel, poll_interval_sec=0.1)
    for t in build_core_primitives(hot_ctx, bus, store, sync_intents_cb=scheduler.sync_visual_watches_to_hot_context):
        kernel.register_tool(t)

    # 1. Add visual watch intent & verify L0 HotWorkingContext syncs automatically
    store.upsert_intent(
        intent_id="wake_watch",
        trigger_type="visual_watch",
        spec="清晨看到主人起床坐起",
        goal="温柔问早并播报缓存中的早间简报",
        one_shot=False,
    )
    await scheduler.sync_visual_watches_to_hot_context()
    assert any("wake_watch" in i for i in hot_ctx.active_intentions)

    # 2. Add an already-due cron intent (backdated created_at) and verify scheduler runs PiKernel -> stages briefing!
    store.upsert_intent(
        intent_id="morning_prep",
        trigger_type="cron",
        spec="* * * * *",
        goal="查询天气并预存今日简报",
        one_shot=True,
    )
    # Backdate created_at by 120s so croniter `* * * * *` triggers immediately
    store._conn.execute("UPDATE autonomous_intents SET created_at = created_at - 120 WHERE id = 'morning_prep'")
    store._conn.commit()

    await scheduler._check_due_crons()
    import asyncio
    await asyncio.sleep(0.2)

    assert "scheduled_briefing" in hot_ctx.briefing_cache
    assert "24度" in hot_ctx.briefing_cache["scheduled_briefing"]
    store.close()
