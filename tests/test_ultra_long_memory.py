from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from config.loader import AppConfig, load_config
from memory.hot_context import HotWorkingContext
from memory.hy_memory_bus import HyMemorySidecarBus
from memory.outbox import DurableOutboxAndEpisodeStore


class BatchTrackingHyMemoryClient:
    """Tracks coalesced multi-turn batch writes and simulates slow LLM/vector indexing."""

    def __init__(self) -> None:
        self.batches: list[Any] = []
        self.stored_texts: list[str] = []

    async def async_add(self, data: Any, *, user_id: str = "", **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.08)
        self.batches.append(data)
        if isinstance(data, list):
            for m in data:
                self.stored_texts.append(f"{m.get('role')}: {m.get('content')}")
        else:
            self.stored_texts.append(str(data))
        return {"success": True, "ops_summary": {"add": 1}}

    async def async_search(
        self, query: str | None = None, *, user_ids: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        q = (query or "").strip()
        hits = []
        for idx, txt in enumerate(self.stored_texts):
            if q and any(tok in txt for tok in q.split() if len(tok) >= 2):
                hits.append({"memory_id": f"m_{idx}", "content": txt, "score": 0.91})
        return {"memories": {"proactive": [], "schema": [], "normal": hits[:6]}}

    async def async_get_user_profile(self, *, user_id: str) -> dict[str, Any]:
        return {"summary": "长期记忆压力测试用户"}

    def close(self) -> None:
        pass


class SequentialCompactionMockLLM:
    """Verifies that sidecar compactions never overlap and preserve cumulative wave markers."""

    def __init__(self) -> None:
        self.wave = 0
        self.in_flight = 0
        self.max_concurrent = 0
        self.chat = type("Chat", (), {"completions": self})()

    async def create(self, **kwargs: Any) -> Any:
        self.in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self.in_flight)
        await asyncio.sleep(0.03)
        self.wave += 1
        prompt = kwargs["messages"][0]["content"]
        # Extract previous wave tag if present to prove sequential accumulation
        prev_marker = ""
        for w in range(1, self.wave):
            if f"[Wave#{w}]" in prompt:
                prev_marker += f"[Wave#{w}]"
        summary = f"{prev_marker}[Wave#{self.wave}] 已累计压缩 {self.wave} 批历史对话（含早中晚期关键事实）。"
        self.in_flight -= 1
        msg = type("Msg", (), {"content": summary, "tool_calls": None})()
        return type("Resp", (), {"choices": [type("C", (), {"message": msg})()]})()


@pytest.mark.asyncio
async def test_ultra_long_60_turns_compaction_coalescing_and_budget_guard(tmp_path: Path) -> None:
    cfg = AppConfig(
        omni_api_base="http://localhost",
        omni_api_key="test_key",
        omni_model="mock-omni",
        realtime_ws_url="ws://localhost",
        realtime_model="mock-rt",
        data_dir=tmp_path,
        settings={"memory": {"outbox_flush_interval_sec": 0.1, "compaction_turn_threshold": 8}},
    )
    hot_ctx = HotWorkingContext(max_chars=2200, max_prefetched=6, compaction_turn_threshold=8)
    store = DurableOutboxAndEpisodeStore(data_dir=tmp_path)
    hy_mock = BatchTrackingHyMemoryClient()
    llm_mock = SequentialCompactionMockLLM()

    bus = HyMemorySidecarBus(
        config=cfg,
        hot_context=hot_ctx,
        store=store,
        hy_client=hy_mock,
        llm_client=llm_mock,  # type: ignore[arg-type]
    )
    await bus.start()

    try:
        # Populate all L0 slots heavily to test priority-preserving budget protection
        await hot_ctx.update_user_profile("资深全栈架构师，偏好极简代码，对芒果与花生过敏，家里有一只名叫Rocket的边牧。")
        for i in range(8):
            await hot_ctx.add_spatial_observation(f"空间视觉记录 #{i}: 书桌右侧放置了测试仪器与第 {i} 号样品盒")
        await hot_ctx.set_active_intentions([f"[intent_{i}] 当看到主人拿起第 {i} 号样品盒时提醒记录实验数据" for i in range(6)])
        await hot_ctx.set_briefing_cache("morning_briefing", "今日多云转晴 21°C，10:00 架构评审，14:00 算力集群巡检，16:30 牙医预约。")

        # Feed 60 turns (30 user + 30 assistant) with Needle facts at Turn 1, Turn 30, and Turn 59
        t0 = time.perf_counter()
        for pair_idx in range(30):
            if pair_idx == 0:
                u_txt = "记住：地下室服务器机柜的应急解锁码是 ZX-9081-OMEGA，千万别忘。"
                a_txt = "好的，已记下地下室服务器机柜应急解锁码为 ZX-9081-OMEGA。"
            elif pair_idx == 14:
                u_txt = "顺便提一下，下个月去慕尼黑出差的航班号是 LH727，座位选了 14A。"
                a_txt = "收到，慕尼黑出差航班 LH727，座位 14A。"
            elif pair_idx == 29:
                u_txt = "最后确认一下，祖母绿宝石戒指收在了主卧衣帽间最上层的丝绒盒子里。"
                a_txt = "明白，祖母绿宝石戒指在主卧衣帽间最上层丝绒盒内。"
            else:
                u_txt = (
                    f"这是第 {pair_idx + 1} 轮超长背景讨论：我们正在探讨分布式存储引擎在海量写入下的 WAL 日志合并策略、"
                    f"向量索引 HNSW 图构建参数 M 与 efConstruction 的权衡，以及边缘设备低功耗唤醒的滤波算法细节。"
                )
                a_txt = (
                    f"第 {pair_idx + 1} 轮回应：在高频写入场景中，采用批量合并（Batch Coalescing）与异步 Compaction "
                    f"可以将尾延迟降低一个数量级，同时保持 L0 热上下文窗口恒定。"
                )
            await bus.record_dialogue_turn("user", u_txt)
            await bus.record_dialogue_turn("assistant", a_txt)

        avg_turn_ms = ((time.perf_counter() - t0) * 1000.0) / 60.0
        assert avg_turn_ms < 5.0, f"Per-turn write latency too high: {avg_turn_ms:.2f}ms"

        # Wait for serialized sidecar compaction waves and coalesced outbox draining
        deadline = time.monotonic() + 5.0
        while (store.pending_outbox_count() > 0 or llm_mock.in_flight > 0) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.15)

        # 1. Verify Sidecar Compaction was strictly serialized (max_concurrent == 1) and lost zero waves!
        assert llm_mock.max_concurrent == 1
        assert llm_mock.wave >= 10
        assert "[Wave#1]" in hot_ctx.rolling_dialogue_summary
        assert f"[Wave#{llm_mock.wave}]" in hot_ctx.rolling_dialogue_summary

        # 2. Verify Outbox Batch Coalescing reduced 30 dialogue pairs into <= 10 batched Hy-Memory writes
        assert store.pending_outbox_count() == 0
        assert 1 <= len(hy_mock.batches) <= 10

        # 3. Verify early, middle, and late needles are all recallable
        early_hits = await bus.deep_recall("服务器机柜的应急解锁码是多少？")
        mid_hits = await bus.deep_recall("慕尼黑出差的航班号和座位")
        late_hits = await bus.deep_recall("祖母绿宝石戒指放在哪了")
        assert any("ZX-9081-OMEGA" in h for h in early_hits)
        assert any("LH727" in h for h in mid_hits)
        assert any("丝绒盒" in h for h in late_hits)

        # 4. Verify L0 HotWorkingContext instructions stay strictly <= 2200 chars AND keep all critical sections intact!
        await hot_ctx.set_prefetched_memories(early_hits + mid_hits + late_hits)
        rendered = hot_ctx.render_realtime_instructions()
        assert len(rendered) <= 2200
        assert "【关联长期记忆(Hy-Memory 2.0)】" in rendered
        assert "【近期对话背景摘要】" in rendered
        assert "【最近几句对话】" in rendered
        assert "【交互与工具准则】" in rendered
        assert "delegate_to_kernel" in rendered
    finally:
        await bus.stop()
        store.close()


@pytest.mark.asyncio
async def test_live_smoke_ultra_long_context_memory_qianwen(tmp_path: Path) -> None:
    """
    Live Cloud Smoke Test against real `qwen3.8-omni-flash` + `text-embedding-v4` + `Hy-Memory 2.0`:
      - Feeds 24 turns (~6,500 chars) of multi-topic conversation with 3 buried needles (Early, Middle, Late)
        plus a spatial visual observation.
      - Triggers multiple real LLM sidecar rolling compactions (`_compact_turns_sidecar`).
      - Verifies 0ms immediate recall, coalesced `Hy-Memory 2.0` Chroma extraction/indexing,
        and post-compaction L0 HotWorkingContext integrity.
    """
    cfg = load_config()
    if not cfg.omni_api_key:
        pytest.skip("OMNI_API_KEY not configured")

    cfg.data_dir = tmp_path / "live_ultra_long"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.memory_user_id = f"smoke_long_user_{int(time.time())}"
    cfg.settings.setdefault("memory", {})["outbox_flush_interval_sec"] = 0.5
    cfg.settings["memory"]["compaction_turn_threshold"] = 8

    hot_ctx = HotWorkingContext(max_chars=2200, max_prefetched=6, compaction_turn_threshold=8)
    store = DurableOutboxAndEpisodeStore(data_dir=cfg.data_dir)
    bus = HyMemorySidecarBus(config=cfg, hot_context=hot_ctx, store=store)
    await bus.start()

    try:
        script: list[tuple[str, str]] = [
            # Early Needle (Turn 1-2)
            (
                "你好！先帮我记住一个非常重要的数据：我家书房保险箱的备用密码是 7392-Alpha，另外我对奇异果严重过敏。",
                "好的，我已经牢记：书房保险箱备用密码是 7392-Alpha，并且你对奇异果严重过敏。",
            ),
            # Distractor turns 3-10 (Wave 1 compaction triggers here)
            (
                "今天我们来聊聊手冲咖啡的萃取率吧，如果用92度水温冲浅烘埃塞俄比亚豆，研磨度应该怎么调？",
                "浅烘埃塞豆密度较高，建议采用中细研磨（类似细砂糖颗粒），水粉比1:15，总萃取时间控制在2分10秒左右。",
            ),
            (
                "如果觉得尾段有一点涩感，是应该调粗研磨还是降低水温？",
                "尾段涩感通常来自过度萃取的木质纤维杂味，可以先将研磨度微调粗半格，或者在最后注水阶段提前移开滤杯。",
            ),
            (
                "顺便问下，Python 3.11 里的 TaskGroup 相比 gather 有什么优势？",
                "TaskGroup 提供了结构化并发（Structured Concurrency），当组内任一子任务抛出异常时会自动取消其余兄弟任务，避免悬挂协程。",
            ),
            (
                "那在 SQLite 的 WAL 模式下，为什么读操作不会阻塞写操作？",
                "因为 WAL 将修改追加写入单独的预写日志文件中，读取器只需根据快照点读取旧页，因此并发读与单写可以并行。",
            ),
            # Middle Needle (Turn 11-12)
            (
                "插播一条日程：下周二上午九点半我要去瑞金医院门诊楼四楼复查甲状腺，主治医生姓林。",
                "记下了：下周二上午 9:30 去瑞金医院门诊楼四楼找林医生复查甲状腺。",
            ),
            # Distractor turns 13-22 (Wave 2 & Wave 3 compactions trigger here)
            (
                "继续聊技术，如果要在树莓派或者 RK3588 上跑轻量级视觉差分门控，为什么缩放到 64x36 灰度图就够了？",
                "因为房间宏观动作（如有人进门、起床、挥手）在低频空间分量上非常显著，64x36 能滤除传感器噪点且 CPU 占用不到 0.5%。",
            ),
            (
                "周末我想在家做一道勃艮第红酒炖牛肉，选牛肋条还是牛腱子更好？",
                "传统勃艮第炖牛肉推荐用牛肩肉或带筋膜的牛肋条，慢炖三小时后胶质融化口感最软糯。",
            ),
            (
                "炖的时候需要放多少百里香和月桂叶？",
                "一公斤牛肉搭配两枝新鲜百里香、两片香叶即可，出锅前记得夹出香料束。",
            ),
            (
                "最近在看科幻小说《海伯利安》，里面伯劳鸟的时间坟墓设定很有意思。",
                "是的，时间坟墓逆时间流而行的设定与各位朝圣者的故事线交织得非常精妙。",
            ),
            (
                "如果给客厅配一套家庭影院音箱，5.1.2 杜比全景声的吸顶音箱装在沙发正上方还是稍微靠前一点？",
                "5.1.2 的两只天空声道建议安装在聆听位置正上方偏前约 70 到 80 度仰角的位置，声场包围感最自然。",
            ),
            # Late Needle (Turn 23-24)
            (
                "最后再记一下，我刚才把那把银色保时捷备用车钥匙放进玄关左手边第二个胡桃木抽屉里了。",
                "明白，银色保时捷备用车钥匙放在玄关左手边第二个胡桃木抽屉里。",
            ),
        ]

        t0 = time.perf_counter()
        await bus.record_visual_observation(
            "主人把黑色徕卡相机放在了书房飘窗的灰色软垫上",
            importance=2,
        )
        for u_msg, a_msg in script:
            await bus.record_dialogue_turn("user", u_msg)
            await bus.record_dialogue_turn("assistant", a_msg)

        write_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"\n[Smoke Test] 24 turns + 1 visual obs recorded in {write_elapsed_ms:.2f}ms ({write_elapsed_ms / 25:.2f}ms/op)")
        assert write_elapsed_ms < 120.0

        # Phase A: Immediate 0-wait recall (while background LLM compactor & Hy-Memory outbox are still running!)
        imm_safe = await bus.deep_recall("书房保险箱的备用密码是多少？对什么水果过敏？")
        imm_hosp = await bus.deep_recall("下周二去哪家医院复查什么？找哪位医生？")
        imm_key = await bus.deep_recall("保时捷备用车钥匙放在哪个抽屉了？")
        imm_cam = await bus.deep_recall("徕卡相机放在哪儿了？")
        print("[Phase A Immediate Recall - Early Needle]:", imm_safe)
        print("[Phase A Immediate Recall - Middle Needle]:", imm_hosp)
        print("[Phase A Immediate Recall - Late Needle]:", imm_key)
        print("[Phase A Immediate Recall - Visual Needle]:", imm_cam)
        assert any("7392-Alpha" in h or "奇异果" in h for h in imm_safe)
        assert any("瑞金医院" in h or "甲状腺" in h for h in imm_hosp)
        assert any("保时捷" in h or "胡桃木抽屉" in h for h in imm_key)
        assert any("徕卡相机" in h for h in imm_cam)

        # Phase B: Wait for background Sidecar Compactor & Coalesced Hy-Memory 2.0 Outbox to finish
        t_drain_start = time.monotonic()
        deadline = t_drain_start + 90.0
        while (store.pending_outbox_count() > 0 or bus._compaction_lock.locked()) and time.monotonic() < deadline:
            await asyncio.sleep(1.0)
        drain_sec = time.monotonic() - t_drain_start
        print(f"[Phase B Background Drain] Completed in {drain_sec:.1f}s (pending={store.pending_outbox_count()})")
        assert store.pending_outbox_count() == 0

        # Verify real LLM rolling dialogue summary was generated and compacted cleanly
        print("[Phase B Compacted Rolling Summary]:", hot_ctx.rolling_dialogue_summary)
        assert len(hot_ctx.rolling_dialogue_summary) > 15

        # Phase C: Post-indexing Hy-Memory 2.0 + Hybrid Recall & Speculative Prefetch into L0 HotWorkingContext
        post_safe = await bus.deep_recall("保险箱备用密码与食物过敏禁忌")
        post_hosp = await bus.deep_recall("下周二医院复查安排与医生姓氏")
        post_key = await bus.deep_recall("银色备用车钥匙收纳位置")
        print("[Phase C Post-HyMemory Recall - Early]:", post_safe)
        print("[Phase C Post-HyMemory Recall - Middle]:", post_hosp)
        print("[Phase C Post-HyMemory Recall - Late]:", post_key)

        assert any("7392" in h or "奇异果" in h for h in post_safe)
        assert any("瑞金" in h or "甲状腺" in h or "林" in h for h in post_hosp)
        assert any("钥匙" in h or "抽屉" in h for h in post_key)

        pf_task = bus.trigger_speculative_prefetch("书房保险箱密码和瑞金医院复查")
        if pf_task:
            await pf_task

        rendered = hot_ctx.render_realtime_instructions()
        print(f"[Phase C L0 HotWorkingContext Rendered] length={len(rendered)} / {hot_ctx.max_chars} chars")
        assert len(rendered) <= hot_ctx.max_chars
        assert "【关联长期记忆(Hy-Memory 2.0)】" in rendered
        assert "【近期对话背景摘要】" in rendered
        assert "【交互与工具准则】" in rendered
    finally:
        await bus.stop()
        store.close()
