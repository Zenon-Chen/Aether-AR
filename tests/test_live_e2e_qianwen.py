from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from config.loader import load_config
from omni.coordinator import AmbientOmniCoordinator
from memory.outbox import FrameRecord


def _encode_jpeg_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _draw_bedroom_wakeup_sequence() -> list[FrameRecord]:
    """Render 3 clear visual frames depicting transition from sleeping in dark bedroom -> waking up in morning light."""
    now = time.time()

    # Frame 1: Dark bedroom, person lying flat under blanket asleep
    f1 = Image.new("RGB", (640, 480), (18, 22, 38))
    d1 = ImageDraw.Draw(f1)
    d1.rectangle([420, 40, 580, 180], fill=(28, 35, 58), outline=(60, 70, 95), width=3)  # Dark window
    d1.rectangle([60, 290, 560, 430], fill=(70, 55, 45))  # Bed frame
    d1.rectangle([75, 260, 175, 295], fill=(210, 210, 220))  # Pillow
    d1.ellipse([105, 240, 165, 290], fill=(225, 190, 165))  # Head lying flat
    d1.rectangle([155, 255, 520, 340], fill=(45, 85, 150))  # Blanket covering sleeping person
    d1.text((180, 195), "Zzz... (Sleeping in bed)", fill=(150, 170, 220))

    # Frame 2: Morning light in window, blanket pushed aside, torso rising
    f2 = Image.new("RGB", (640, 480), (135, 145, 155))
    d2 = ImageDraw.Draw(f2)
    d2.rectangle([420, 40, 580, 180], fill=(255, 235, 165), outline=(220, 190, 110), width=3)  # Sunny window
    d2.rectangle([60, 290, 560, 430], fill=(110, 85, 65))
    d2.rectangle([75, 260, 175, 295], fill=(240, 240, 245))
    d2.polygon([(170, 290), (235, 195), (275, 290)], fill=(95, 160, 210))  # Torso rising up
    d2.ellipse([205, 145, 265, 205], fill=(240, 200, 175))
    d2.rectangle([280, 265, 530, 340], fill=(55, 105, 180))  # Blanket pushed back

    # Frame 3: Bright morning room, person sitting upright on bed stretching arms wide awake!
    f3 = Image.new("RGB", (640, 480), (215, 220, 210))
    d3 = ImageDraw.Draw(f3)
    d3.rectangle([420, 40, 580, 180], fill=(255, 245, 180), outline=(230, 200, 120), width=4)
    d3.ellipse([475, 70, 535, 130], fill=(255, 200, 50))  # Morning sun
    d3.rectangle([60, 290, 560, 430], fill=(130, 95, 70))
    # Person sitting upright on bed edge stretching arms up
    d3.rectangle([230, 185, 305, 295], fill=(80, 155, 215))  # Upright torso
    d3.ellipse([238, 120, 298, 180], fill=(245, 205, 180))  # Awake head upright
    d3.line([(235, 205), (170, 115)], fill=(80, 155, 215), width=14)  # Left arm stretching up
    d3.line([(300, 205), (365, 115)], fill=(80, 155, 215), width=14)  # Right arm stretching up
    d3.text((160, 75), "Waking up & stretching in morning!", fill=(40, 50, 60))

    return [
        FrameRecord(timestamp=now - 8, iso_time="2026-10-30 07:29:52", jpeg_b64=_encode_jpeg_b64(f1), motion_score=0.01),
        FrameRecord(timestamp=now - 3, iso_time="2026-10-30 07:29:57", jpeg_b64=_encode_jpeg_b64(f2), motion_score=0.19),
        FrameRecord(timestamp=now, iso_time="2026-10-30 07:30:00", jpeg_b64=_encode_jpeg_b64(f3), motion_score=0.24),
    ]


def _draw_math_whiteboard_b64() -> str:
    """Render a crisp whiteboard image with an arithmetic expression to verify visual OCR + Python execution."""
    img = Image.new("RGB", (800, 360), (250, 252, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 780, 340], outline=(40, 60, 90), width=5)
    try:
        font_big = ImageFont.truetype("arial.ttf", 52)
        font_sub = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font_big = ImageFont.load_default()
        font_sub = font_big

    draw.text((60, 65), "WHITEBOARD FORMULA:", fill=(70, 90, 120), font=font_sub)
    draw.text((60, 150), "(247 * 389) + 1584 = ?", fill=(15, 20, 30), font=font_big)
    return _encode_jpeg_b64(img)


@pytest.mark.asyncio
async def test_live_1_zero_hardcode_morning_wakeup_and_anti_nag(tmp_path: Path) -> None:
    cfg = load_config()
    cfg.data_dir = tmp_path / "live_t1"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.omni_api_key:
        pytest.skip("OMNI_API_KEY not configured")

    # Use lightweight mock hy_client here so test 1 isolates PerceptualPulse + Realtime Duplex speed
    from tests.test_hot_context_and_bus import SimulatedSlowHyMemoryClient

    coord = AmbientOmniCoordinator(cfg, hy_client=SimulatedSlowHyMemoryClient())
    audio_bytes_received = 0
    spoken_sentences: list[str] = []

    async def on_edge_audio(pcm: bytes) -> None:
        nonlocal audio_bytes_received
        audio_bytes_received += len(pcm)

    async def on_edge_json(ev: dict) -> None:
        if ev.get("type") == "transcript" and ev.get("role") == "assistant" and ev.get("final"):
            spoken_sentences.append(ev.get("text", ""))

    coord.send_edge_audio = on_edge_audio
    coord.send_edge_json = on_edge_json
    await coord.start()

    try:
        await coord.hot_context.update_scene_state("凌晨至清早，主人一直在床上盖被子熟睡。")
        await coord.hot_context.set_briefing_cache(
            "morning_briefing",
            "杭州今天晴转多云 19到25度，上午10点有全模态Agent项目评审。",
        )
        coord.store.upsert_intent(
            intent_id="morning_wake",
            trigger_type="visual_watch",
            spec="清晨看到原本熟睡的主人醒来坐起伸懒腰",
            goal="温柔问早并简短播报缓存中的早间简报",
            one_shot=False,
        )
        await coord.scheduler.sync_visual_watches_to_hot_context()

        frames = _draw_bedroom_wakeup_sequence()
        for f in frames[:-1]:
            coord.store.push_frame(f.jpeg_b64, motion_score=f.motion_score)

        # Evaluate final wakeup frame with real qwen3.8-omni-flash!
        decision = await coord.pulse_observer.evaluate_frames(
            frames,
            bypass_rate_limit=True,
            override_now_str="2026-10-30 07:30:05 (Friday Morning)",
        )
        assert decision is not None
        print("[Live Test 1] Wakeup PulseDecision:", decision)
        assert decision.action == "proactive_engage", f"Expected proactive_engage on morning wakeup, got {decision}"

        # Trigger full-duplex proactive speech via real qwen3.8-omni-flash-realtime!
        await coord._apply_pulse_decision(decision, latest_jpeg_b64=frames[-1].jpeg_b64)
        assert coord.state == "LIVE_DUPLEX" and coord._realtime is not None
        await coord._realtime.wait_until_response_idle(timeout=25.0)
        await asyncio.sleep(0.3)

        print(f"[Live Test 1] Synthesized PCM24k bytes={audio_bytes_received}, Spoken={spoken_sentences}")
        assert audio_bytes_received > 8000
        assert len(spoken_sentences) >= 1

        # Verify Social Anti-Nagging: 1 minute later user is still sitting on bed, greeting already in recent_turns
        follow_up = await coord.pulse_observer.evaluate_frames(
            [frames[-1]],
            bypass_rate_limit=True,
            override_now_str="2026-10-30 07:31:10 (Friday Morning)",
        )
        assert follow_up is not None
        print("[Live Test 1] Follow-up Anti-Nag Decision:", follow_up)
        assert follow_up.action != "proactive_engage"
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_live_2_hy_memory_2_zero_blocking_write_and_recall(tmp_path: Path) -> None:
    cfg = load_config()
    cfg.data_dir = tmp_path / "live_t2"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.memory_user_id = f"e2e_user_{int(time.time())}"
    cfg.settings.setdefault("memory", {})["outbox_flush_interval_sec"] = 0.5
    if not cfg.omni_api_key:
        pytest.skip("OMNI_API_KEY not configured")

    coord = AmbientOmniCoordinator(cfg)
    await coord.start()

    try:
        # 1. Non-blocking write benchmark on live Hy-Memory 2.0 bus
        t0 = time.perf_counter()
        await coord.memory_bus.record_visual_observation(
            "主人把银色钛合金U盘放在了显示器支架右侧的小木盒里",
            importance=2,
        )
        await coord.memory_bus.record_dialogue_turn("user", "记住我对花生过敏，而且周六下午三点要看牙医。")
        await coord.memory_bus.record_dialogue_turn("assistant", "记下了：花生过敏，周六下午三点看牙医。")
        write_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[Live Test 2] Main-path write latency: {write_ms:.2f}ms")
        assert write_ms < 50.0

        # 2. Wait for background Outbox worker to finish real HyMemoryClient.async_add extraction & Chroma indexing
        deadline = time.monotonic() + 75.0
        while coord.store.pending_outbox_count() > 0 and time.monotonic() < deadline:
            await asyncio.sleep(1.5)
        assert coord.store.pending_outbox_count() == 0, "Hy-Memory 2.0 outbox did not finish draining within 75s"

        # 3. Query real Hy-Memory 2.0 + local episode hybrid recall
        usb_hits = await coord.memory_bus.deep_recall("银色U盘放在哪儿了？")
        allergy_hits = await coord.memory_bus.deep_recall("主人有什么食物过敏或者周末安排？")
        print("[Live Test 2] USB Recall:", usb_hits)
        print("[Live Test 2] Allergy/Schedule Recall:", allergy_hits)

        assert any("U盘" in h or "木盒" in h for h in usb_hits)
        assert any("花生" in h or "牙医" in h for h in allergy_hits)

        # 4. Verify speculative prefetch updates L0 HotWorkingContext
        pf_task = coord.memory_bus.trigger_speculative_prefetch("我对什么食物过敏")
        if pf_task:
            await pf_task
        assert any("花生" in m for m in coord.hot_context.prefetched_memories)
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_live_3_visual_math_and_mcp_delegation_over_realtime_duplex(tmp_path: Path) -> None:
    cfg = load_config()
    cfg.data_dir = tmp_path / "live_t3"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.omni_api_key:
        pytest.skip("OMNI_API_KEY not configured")

    from tests.test_hot_context_and_bus import SimulatedSlowHyMemoryClient

    coord = AmbientOmniCoordinator(cfg, hy_client=SimulatedSlowHyMemoryClient())
    audio_bytes = 0
    assistant_finals: list[str] = []
    kernel_tools_called: list[str] = []

    async def on_audio(pcm: bytes) -> None:
        nonlocal audio_bytes
        audio_bytes += len(pcm)

    async def on_json(ev: dict) -> None:
        if ev.get("type") == "transcript" and ev.get("role") == "assistant" and ev.get("final"):
            assistant_finals.append(ev.get("text", ""))
        elif ev.get("type") == "kernel_event" and ev.get("event") == "tool_start":
            kernel_tools_called.append(ev.get("detail", {}).get("tool", ""))

    coord.send_edge_audio = on_audio
    coord.send_edge_json = on_json
    await coord.start()

    try:
        # Push whiteboard image `(247 * 389) + 1584 = ?` (exact answer = 97667) into camera buffer
        wb_b64 = _draw_math_whiteboard_b64()
        coord.store.push_frame(wb_b64, motion_score=0.5, is_hd=True)
        coord.store.push_frame(wb_b64, motion_score=0.5, is_hd=False)

        await coord.handle_user_text_input(
            "请调用后端内核看清当前画面白板上的算式并用Python算出精确数值，"
            "同时用MCP工具把结果记入家庭便签（标题填'白板验算'），最后把答案读给我听。"
        )

        # Wait up to 45s for Realtime -> delegate_to_kernel -> PiKernel (OCR + python_exec + MCP) -> Spoken output!
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            combined = " ".join(assistant_finals)
            if ("97667" in combined or "97,667" in combined or "九万七千六百六十七" in combined) and not (
                coord._realtime and coord._realtime.is_busy
            ):
                break
            await asyncio.sleep(1.0)

        print("[Live Test 3] Kernel tools invoked:", kernel_tools_called)
        print("[Live Test 3] Spoken transcripts:", assistant_finals)
        print("[Live Test 3] Audio PCM24k bytes:", audio_bytes)

        combined_spoken = " ".join(assistant_finals)
        assert "python_exec" in kernel_tools_called
        assert "mcp_call" in kernel_tools_called
        assert audio_bytes > 8000
        assert any(tok in combined_spoken for tok in ("97667", "97,667", "九万七千六百六十七", "九七六六七"))
    finally:
        await coord.stop()
