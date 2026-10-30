from __future__ import annotations

import asyncio
import base64
import io
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from config.loader import AppConfig, derive_realtime_model, derive_realtime_ws_url
from memory.hot_context import HotWorkingContext
from memory.hy_memory_bus import HyMemorySidecarBus
from memory.outbox import DurableOutboxAndEpisodeStore


class SimulatedSlowHyMemoryClient:
    """Simulates a heavy cognitive memory engine (e.g. 300ms add, 150ms search) to prove 0ms main-thread blocking."""

    def __init__(self) -> None:
        self.stored: list[str] = []
        self.closed = False

    async def async_add(self, data: Any, *, user_id: str = "", **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.25)
        if isinstance(data, list):
            txt = " | ".join(f"{m['role']}: {m['content']}" for m in data)
        else:
            txt = str(data)
        self.stored.append(f"[{user_id}] {txt}")
        return {"success": True, "ops_summary": {"add": 1}}

    async def async_search(
        self, query: str | None = None, *, user_ids: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        await asyncio.sleep(0.12)
        q = (query or "").strip()
        hits = []
        for idx, item in enumerate(self.stored):
            if not q or any(ch in item for ch in q if ch.strip()):
                hits.append({"memory_id": f"m_{idx}", "content": item, "score": 0.88})
        return {"memories": {"proactive": [], "schema": [], "normal": hits}}

    async def async_get_user_profile(self, *, user_id: str) -> dict[str, Any]:
        return {"summary": "主人偏好极简科技风，早晨喝冰美式"}

    def close(self) -> None:
        self.closed = True


class DummyCompletionChoice:
    def __init__(self, content: str) -> None:
        self.message = type("Msg", (), {"content": content, "tool_calls": None})()


class DummyChatCompletions:
    async def create(self, **kwargs: Any) -> Any:
        return type("Resp", (), {"choices": [DummyCompletionChoice("滚动压缩后的近期对话摘要：已讨论钥匙位置与早报设置。")]})()


class DummyLLMClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": DummyChatCompletions()})()


def _make_jpeg_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (90, 140, 210)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_url_and_model_derivation() -> None:
    ws = derive_realtime_ws_url("https://maas.qianwenaiapi.com/compatible-mode/v1")
    assert ws == "wss://maas.qianwenaiapi.com/api-ws/v1/realtime"
    assert derive_realtime_model("qwen3.8-omni-flash") == "qwen3.8-omni-flash-realtime"
    assert derive_realtime_model("qwen3.8-omni-flash-realtime") == "qwen3.8-omni-flash-realtime"


@pytest.mark.asyncio
async def test_hot_context_atomic_swap_and_budget() -> None:
    ctx = HotWorkingContext(max_chars=1200, max_prefetched=3, compaction_turn_threshold=5)
    swaps: list[tuple[int, str]] = []

    async def listener(instr: str, ver: int) -> None:
        swaps.append((ver, instr))

    ctx.on_hot_swap(listener)
    await ctx.update_user_profile("喜欢科幻电影与自动化编程")
    await ctx.set_briefing_cache("morning", "杭州晴 22°C，10:00 产品评审")
    changed = await ctx.set_prefetched_memories(["备用钥匙在书房抽屉", "猫粮放在阳台柜子", "重复项", "备用钥匙在书房抽屉"])
    assert changed is True
    # Duplicate call should not trigger redundant hot-swap
    changed_again = await ctx.set_prefetched_memories(["备用钥匙在书房抽屉", "猫粮放在阳台柜子", "重复项"])
    assert changed_again is False

    await asyncio.sleep(0.05)
    assert len(swaps) >= 3
    rendered = ctx.render_realtime_instructions()
    assert len(rendered) <= 1200
    assert "备用钥匙在书房抽屉" in rendered
    assert "杭州晴 22°C" in rendered


@pytest.mark.asyncio
async def test_zero_blocking_memory_bus_and_outbox_recovery(tmp_path: Path) -> None:
    cfg = AppConfig(
        omni_api_base="https://maas.qianwenaiapi.com/compatible-mode/v1",
        omni_api_key="test_key",
        omni_model="qwen3.8-omni-flash",
        realtime_ws_url="wss://maas.qianwenaiapi.com/api-ws/v1/realtime",
        realtime_model="qwen3.8-omni-flash-realtime",
        data_dir=tmp_path,
        settings={"memory": {"outbox_flush_interval_sec": 0.2, "compaction_turn_threshold": 4}},
    )
    hot_ctx = HotWorkingContext(compaction_turn_threshold=4)
    store = DurableOutboxAndEpisodeStore(data_dir=tmp_path)
    slow_hy = SimulatedSlowHyMemoryClient()
    bus = HyMemorySidecarBus(
        config=cfg,
        hot_context=hot_ctx,
        store=store,
        hy_client=slow_hy,
        llm_client=DummyLLMClient(),  # type: ignore[arg-type]
    )
    await bus.start()

    try:
        # 1. Measure main-path write latency: must be < 15ms despite 250ms Hy-Memory backend!
        t0 = time.perf_counter()
        await bus.record_visual_observation(
            "用户把黑色降噪耳机放在了显示器左下方",
            importance=2,
            jpeg_b64=_make_jpeg_b64(),
        )
        await bus.record_dialogue_turn("user", "记住我周三晚上八点要打羽毛球")
        await bus.record_dialogue_turn("assistant", "好，已经记下周三晚八点羽毛球。")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 25.0, f"Write path blocked caller for {elapsed_ms:.2f}ms!"

        # 2. Immediate spatial episode recall works in <5ms BEFORE Hy-Memory even finishes flushing!
        immediate_hits = await bus.deep_recall("降噪耳机放哪了")
        assert any("黑色降噪耳机" in h for h in immediate_hits)

        # 3. Wait for background outbox worker to drain into Hy-Memory 2.0
        await asyncio.sleep(0.8)
        assert store.pending_outbox_count() == 0
        assert len(slow_hy.stored) >= 2
        assert "冰美式" in hot_ctx.user_profile

        # 4. Trigger speculative prefetch and verify L0 HotWorkingContext gets populated asynchronously
        task = bus.trigger_speculative_prefetch("羽毛球")
        assert task is not None
        await task
        assert any("羽毛球" in m for m in hot_ctx.prefetched_memories)

        # 5. Trigger turn overflow compaction
        for i in range(4):
            await bus.record_dialogue_turn("user", f"测试多轮对话句子 {i}")
        await asyncio.sleep(0.1)
        assert "滚动压缩" in hot_ctx.rolling_dialogue_summary
    finally:
        await bus.stop()
        store.close()
