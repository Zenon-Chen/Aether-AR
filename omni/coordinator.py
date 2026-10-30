from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from config.loader import AppConfig
from kernel.agent_loop import KernelEvent, PiKernel
from kernel.mcp_bridge import DynamicMCPHub
from kernel.primitives import build_core_primitives
from kernel.scheduler import AutonomousIntentScheduler
from memory.hot_context import HotWorkingContext
from memory.hy_memory_bus import HyMemorySidecarBus
from memory.outbox import DurableOutboxAndEpisodeStore
from protocol.events import compute_pcm16_rms
from .dashscope_realtime import QwenOmniRealtimeSession
from .perceptual_pulse import PerceptualPulseObserver, PulseDecision

logger = logging.getLogger("ambient.coordinator")


class AmbientOmniCoordinator:
    """
    Central Brain Coordinator orchestrating:
      - `AMBIENT_WATCH` <-> `LIVE_DUPLEX` state transitions
      - Edge audio pre-roll ring buffer & speech wake gate
      - Zero-hardcode `PerceptualPulseObserver` visual awareness
      - Full-duplex `QwenOmniRealtimeSession` (`qwen3.8-omni-flash-realtime`)
      - Thin `PiKernel` + Universal `DynamicMCPHub` + `AutonomousIntentScheduler`
      - Zero-blocking `HyMemorySidecarBus` (`Hy-Memory 2.0`)
    """

    def __init__(
        self,
        config: AppConfig,
        hy_client: Any | None = None,
        llm_client: AsyncOpenAI | None = None,
        realtime_session_factory: Callable[..., QwenOmniRealtimeSession] | None = None,
    ) -> None:
        self.config = config
        self.llm = llm_client or AsyncOpenAI(
            api_key=config.omni_api_key or "EMPTY",
            base_url=config.omni_api_base,
        )

        ast_cfg = config.settings.get("assistant", {})
        mem_cfg = config.settings.get("memory", {})
        perc_cfg = config.settings.get("perception", {})
        dup_cfg = config.settings.get("duplex", {})
        kern_cfg = config.settings.get("kernel", {})

        self.hot_context = HotWorkingContext(
            assistant_name=str(ast_cfg.get("name", "Jarvis")),
            persona=str(ast_cfg.get("persona", "敏锐、温暖、克制的科幻级全模态智能管家。")),
            max_chars=int(mem_cfg.get("max_hot_context_chars", 2400)),
            max_prefetched=int(mem_cfg.get("max_prefetched_items", 6)),
            compaction_turn_threshold=int(mem_cfg.get("compaction_turn_threshold", 8)),
        )
        self.store = DurableOutboxAndEpisodeStore(data_dir=config.data_dir)
        self.memory_bus = HyMemorySidecarBus(
            config=config,
            hot_context=self.hot_context,
            store=self.store,
            hy_client=hy_client,
            llm_client=self.llm,
        )

        self.mcp_hub = DynamicMCPHub(config_path=config.mcp_config_path)
        self.kernel = PiKernel(
            llm_client=self.llm,
            model=config.omni_model,
            max_steps=int(kern_cfg.get("max_tool_steps", 10)),
            system_prompt_provider=self._render_kernel_context,
        )
        self.scheduler = AutonomousIntentScheduler(
            store=self.store,
            hot_context=self.hot_context,
            kernel=self.kernel,
            on_cron_result=self._on_scheduled_cron_completed,
        )

        # Register 4 irreducible primitives + 2 universal MCP dynamic discovery tools
        for tool in build_core_primitives(
            hot_context=self.hot_context,
            memory_bus=self.memory_bus,
            store=self.store,
            request_hd_frame_cb=self._request_edge_hd_frame,
            sync_intents_cb=self.scheduler.sync_visual_watches_to_hot_context,
            python_timeout_sec=float(kern_cfg.get("python_exec_timeout_sec", 15.0)),
        ):
            self.kernel.register_tool(tool)
        for mcp_tool in self.mcp_hub.build_kernel_tools():
            self.kernel.register_tool(mcp_tool)

        self.kernel.subscribe(self._on_kernel_event)

        self.pulse_observer = PerceptualPulseObserver(
            llm_client=self.llm,
            model=config.omni_model,
            hot_context=self.hot_context,
            min_interval_sec=float(perc_cfg.get("pulse_min_interval_sec", 4.0)),
            proactive_cooldown_sec=float(perc_cfg.get("proactive_min_cooldown_sec", 20.0)),
        )
        self.temporal_grid_frames = int(perc_cfg.get("temporal_grid_frames", 3))

        # Duplex & Audio Pre-roll parameters
        self.audio_wake_rms = float(dup_cfg.get("audio_wake_rms_threshold", 0.018))
        self.duplex_idle_sleep_sec = float(dup_cfg.get("duplex_idle_sleep_sec", 45.0))
        self.session_rotate_sec = float(dup_cfg.get("session_rotate_sec", 840.0))
        preroll_chunks = max(4, int(dup_cfg.get("preroll_buffer_ms", 800)) // 40)
        self._preroll_audio: deque[bytes] = deque(maxlen=preroll_chunks)
        self._consecutive_voice_chunks = 0

        self.state: str = "AMBIENT_WATCH"  # "AMBIENT_WATCH" | "LIVE_DUPLEX"
        self._realtime: QwenOmniRealtimeSession | None = None
        self._realtime_factory = realtime_session_factory
        self._state_lock = asyncio.Lock()
        self._pulse_busy = False
        self._last_duplex_frame_sent_ts = 0.0
        self._standby_after_current_response = False

        # Edge transport callbacks (populated by server gateway)
        self.send_edge_audio: Callable[[bytes], Awaitable[None]] | None = None
        self.send_edge_json: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self._hd_frame_waiter: asyncio.Future[str] | None = None

        self._housekeeping_task: asyncio.Task[None] | None = None
        self._running = False

        # Wire L0 HotWorkingContext hot-swaps directly to active Realtime session
        self.hot_context.on_hot_swap(self._on_hot_context_updated)

    def _render_kernel_context(self) -> str:
        snap = self.hot_context.snapshot_dict()
        lines = [
            f"【主人画像】{snap['user_profile']}",
            f"【当前环境状态】{snap['scene_and_spatial_state']}",
            self.mcp_hub.render_directory_summary(),
        ]
        if snap["spatial_observations"]:
            lines.append("【近期空间观察】" + "；".join(snap["spatial_observations"][-4:]))
        return "\n".join(lines)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self.memory_bus.start()
        await self.scheduler.start()
        self._housekeeping_task = asyncio.create_task(self._housekeeping_loop(), name="omni_housekeeping")
        logger.info("AmbientOmniCoordinator started in %s state.", self.state)

    async def stop(self) -> None:
        self._running = False
        if self._housekeeping_task:
            self._housekeeping_task.cancel()
            try:
                await self._housekeeping_task
            except asyncio.CancelledError:
                pass
        await self._close_duplex_session()
        await self.scheduler.stop()
        await self.memory_bus.stop()
        self.store.close()

    # -------------------------------------------------------------------------
    # Edge Uplink Entry Points (Audio / Video / Text)
    # -------------------------------------------------------------------------
    async def handle_edge_audio_pcm16(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return

        if self.state == "LIVE_DUPLEX" and self._realtime and self._realtime.connected:
            await self._realtime.append_audio_pcm16(pcm_bytes)
            return

        # In AMBIENT_WATCH: buffer pre-roll audio & monitor acoustic energy to wake seamlessly
        self._preroll_audio.append(pcm_bytes)
        rms = compute_pcm16_rms(pcm_bytes)
        if rms >= self.audio_wake_rms:
            self._consecutive_voice_chunks += 1
            if self._consecutive_voice_chunks >= 2:
                self._consecutive_voice_chunks = 0
                await self._wake_duplex_from_voice()
        else:
            self._consecutive_voice_chunks = 0

    async def handle_edge_video_frame(
        self,
        jpeg_b64: str,
        motion_score: float = 0.0,
        is_hd: bool = False,
        bypass_pulse_rate_limit: bool = False,
    ) -> PulseDecision | None:
        if not jpeg_b64:
            return None

        self.store.push_frame(jpeg_b64=jpeg_b64, motion_score=motion_score, is_hd=is_hd)

        if is_hd and self._hd_frame_waiter and not self._hd_frame_waiter.done():
            self._hd_frame_waiter.set_result(jpeg_b64)
            return None

        # In LIVE_DUPLEX: stream live frames directly (~1 fps max) into qwen3.8-omni-flash-realtime
        if self.state == "LIVE_DUPLEX" and self._realtime and self._realtime.connected:
            now = time.monotonic()
            if (now - self._last_duplex_frame_sent_ts) >= 1.0:
                self._last_duplex_frame_sent_ts = now
                await self._realtime.append_video_frame(jpeg_b64)
            return None

        # In AMBIENT_WATCH: run Perceptual Pulse asynchronously (non-reentrant)
        if self._pulse_busy and not bypass_pulse_rate_limit:
            return None

        self._pulse_busy = True
        try:
            frames = self.store.get_recent_frames(count=self.temporal_grid_frames)
            decision = await self.pulse_observer.evaluate_frames(
                frames,
                bypass_rate_limit=bypass_pulse_rate_limit,
            )
            if decision:
                await self._apply_pulse_decision(decision, latest_jpeg_b64=jpeg_b64)
            return decision
        finally:
            self._pulse_busy = False

    async def handle_user_text_input(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        await self.memory_bus.record_dialogue_turn("user", cleaned)
        await self._emit_edge_event({"type": "transcript", "role": "user", "text": cleaned, "final": True})
        await self.ensure_duplex_session()
        latest_frame = self.store.get_latest_frame()
        if self._realtime:
            await self._realtime.inject_event_and_speak(
                cleaned,
                frame_jpeg_b64=latest_frame.jpeg_b64 if latest_frame else None,
            )

    # -------------------------------------------------------------------------
    # Perceptual Pulse Decision Handling (Zero Hardcoding)
    # -------------------------------------------------------------------------
    async def _apply_pulse_decision(self, decision: PulseDecision, latest_jpeg_b64: str) -> None:
        await self.hot_context.update_scene_state(decision.scene_state, notify_realtime=False)

        if decision.observation:
            save_img = latest_jpeg_b64 if decision.importance >= 1 else None
            await self.memory_bus.record_visual_observation(
                summary=decision.observation,
                importance=decision.importance,
                jpeg_b64=save_img,
            )

        if decision.fulfilled_intent_id:
            clean_fid = decision.fulfilled_intent_id.strip().strip("[]").strip()
            # Check if one-shot intent should be marked completed
            for it in self.store.list_intents(trigger_type="visual_watch", only_enabled=True):
                if it["id"] == clean_fid:
                    self.store.mark_intent_triggered(it["id"], disable=bool(it["one_shot"]))
                    await self.scheduler.sync_visual_watches_to_hot_context()
                    break

        await self._emit_edge_event(
            {
                "type": "pulse_decision",
                "action": decision.action,
                "scene_state": decision.scene_state,
                "observation": decision.observation,
            }
        )

        if decision.action == "dispatch_kernel" and decision.kernel_goal:
            asyncio.create_task(self._run_background_kernel_and_announce(decision.kernel_goal, speak_result=False))

        elif decision.action == "proactive_engage":
            prompt_hint = decision.proactive_prompt or f"画面观察到：{decision.scene_state}。请自然主动向主人开口互动。"
            # Speculatively prefetch associative memories (bounded to 500ms so proactive greeting stays snappy!)
            prefetch_task = self.memory_bus.trigger_speculative_prefetch(
                f"{decision.scene_state} {prompt_hint}"
            )
            if prefetch_task:
                try:
                    await asyncio.wait_for(asyncio.shield(prefetch_task), timeout=0.5)
                except asyncio.TimeoutError:
                    pass

            await self.ensure_duplex_session()
            if self._realtime:
                trigger_msg = (
                    f"【视觉主动交互触发】当前画面状态：{decision.scene_state}。\n"
                    f"交互要点：{prompt_hint}\n"
                    "请直接用自然亲切的口语对主人开口说话（不要复述系统提示词）。"
                )
                await self._realtime.inject_event_and_speak(trigger_msg, frame_jpeg_b64=latest_jpeg_b64)

    # -------------------------------------------------------------------------
    # Full-Duplex Session Lifecycle & Tool Execution Bridge
    # -------------------------------------------------------------------------
    async def _wake_duplex_from_voice(self) -> None:
        preroll = list(self._preroll_audio)
        self._preroll_audio.clear()
        logger.info("Acoustic wake triggered! Transitioning AMBIENT_WATCH -> LIVE_DUPLEX (%d pre-roll chunks)", len(preroll))
        await self.ensure_duplex_session()
        if self._realtime and self._realtime.connected:
            for chunk in preroll:
                await self._realtime.append_audio_pcm16(chunk)
            latest = self.store.get_latest_frame()
            if latest:
                await self._realtime.append_video_frame(latest.jpeg_b64)

    async def ensure_duplex_session(self) -> QwenOmniRealtimeSession:
        async with self._state_lock:
            if self._realtime and self._realtime.connected:
                self.state = "LIVE_DUPLEX"
                return self._realtime

            factory = self._realtime_factory or QwenOmniRealtimeSession
            sess = factory(
                ws_url=self.config.realtime_ws_url,
                api_key=self.config.omni_api_key,
                model=self.config.realtime_model,
                voice=self.config.omni_voice,
                on_audio_delta=self._on_realtime_audio_delta,
                on_barge_in=self._on_realtime_barge_in,
                on_user_transcript=self._on_realtime_user_transcript,
                on_assistant_transcript=self._on_realtime_assistant_transcript,
                tool_executor=self._execute_realtime_meta_tool,
                on_session_closed=self._on_realtime_closed,
            )
            await sess.connect(self.hot_context.render_realtime_instructions())
            self._realtime = sess
            self._standby_after_current_response = False
            await self._set_state("LIVE_DUPLEX")
            return sess

    async def _close_duplex_session(self) -> None:
        async with self._state_lock:
            sess = self._realtime
            self._realtime = None
            if sess is not None:
                await sess.close()
            await self._set_state("AMBIENT_WATCH")

    async def _set_state(self, new_state: str) -> None:
        if self.state != new_state:
            self.state = new_state
            await self._emit_edge_event({"type": "state_sync", "state": self.state})

    async def _on_hot_context_updated(self, new_instructions: str, _ver: int) -> None:
        if self.state == "LIVE_DUPLEX" and self._realtime and self._realtime.connected:
            await self._realtime.hot_swap_instructions(new_instructions)

    async def _on_realtime_audio_delta(self, pcm_bytes: bytes) -> None:
        if self.send_edge_audio:
            await self.send_edge_audio(pcm_bytes)

    async def _on_realtime_barge_in(self) -> None:
        await self._emit_edge_event({"type": "control", "action": "clear_audio"})

    async def _on_realtime_user_transcript(self, text: str, is_final: bool) -> None:
        await self._emit_edge_event({"type": "transcript", "role": "user", "text": text, "final": is_final})
        if is_final:
            await self.memory_bus.record_dialogue_turn("user", text)
        elif len(text.strip()) >= 6:
            # Speculative streaming prefetch BEFORE the user even finishes their sentence!
            self.memory_bus.trigger_speculative_prefetch(text)

    async def _on_realtime_assistant_transcript(self, text: str, is_final: bool) -> None:
        await self._emit_edge_event({"type": "transcript", "role": "assistant", "text": text, "final": is_final})
        if is_final:
            await self.memory_bus.record_dialogue_turn("assistant", text)
            if self._standby_after_current_response:
                self._standby_after_current_response = False
                asyncio.create_task(self._delayed_standby(1.2))

    async def _delayed_standby(self, delay_sec: float) -> None:
        await asyncio.sleep(delay_sec)
        await self._close_duplex_session()

    async def _on_realtime_closed(self) -> None:
        if self._realtime is not None and not self._realtime.connected:
            self._realtime = None
            await self._set_state("AMBIENT_WATCH")

    # -------------------------------------------------------------------------
    # Realtime Meta-Tools -> PiKernel & Hy-Memory 2.0
    # -------------------------------------------------------------------------
    async def _execute_realtime_meta_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "recall_deep_memory":
            query = str(args.get("query", ""))
            hits = await self.memory_bus.deep_recall(query, limit=6)
            return json.dumps(
                {"memories": hits or ["未找到相关历史记录"]},
                ensure_ascii=False,
            )

        if name == "pin_memory_note":
            note = str(args.get("note", ""))
            await self.memory_bus.pin_explicit_fact(note)
            return json.dumps({"saved": True, "note": note}, ensure_ascii=False)

        if name == "enter_ambient_watch":
            self._standby_after_current_response = True
            asyncio.create_task(self._delayed_standby(2.0))
            return "__ENTER_STANDBY__"

        if name == "delegate_to_kernel":
            goal = str(args.get("goal", ""))
            attach_frame = bool(args.get("attach_live_frame", False))
            init_imgs: list[str] = []
            if attach_frame:
                hd = await self._request_edge_hd_frame()
                if hd:
                    init_imgs.append(hd)
                else:
                    latest = self.store.get_latest_frame(prefer_hd=True)
                    if latest:
                        init_imgs.append(latest.jpeg_b64)

            kernel_task = asyncio.create_task(self.kernel.run(goal, initial_images_b64=init_imgs or None))
            done, _ = await asyncio.wait({kernel_task}, timeout=2.6)
            if kernel_task in done:
                ans = kernel_task.result()
                return json.dumps({"status": "completed", "result": ans}, ensure_ascii=False)

            # Long-running multi-step task: let voice reply naturally now, inject result when done!
            asyncio.create_task(self._await_and_inject_kernel_task(goal, kernel_task))
            return json.dumps(
                {
                    "status": "running_in_background",
                    "instruction": "后台内核正在执行该多步任务，请先用简短自然的一句话告知主人正在处理，结果出来后会立即播报。",
                },
                ensure_ascii=False,
            )

        return json.dumps({"error": f"Unknown meta tool: {name}"}, ensure_ascii=False)

    async def _await_and_inject_kernel_task(self, goal: str, task: asyncio.Task[str]) -> None:
        try:
            ans = await task
        except Exception as exc:
            ans = f"后台任务执行遇到异常：{exc}"

        await self.ensure_duplex_session()
        if self._realtime and self._realtime.connected:
            await self._realtime.inject_event_and_speak(
                f"【后端 PiKernel 任务已完成】原任务：{goal}\n执行结果：{ans}\n请用自然简练的口语把结果告诉主人。"
            )

    async def _run_background_kernel_and_announce(self, goal: str, speak_result: bool = False) -> str:
        ans = await self.kernel.run(goal)
        if speak_result:
            await self.ensure_duplex_session()
            if self._realtime:
                await self._realtime.inject_event_and_speak(
                    f"【后台主动任务汇报】{ans}\n请自然播报给主人。"
                )
        return ans

    async def _on_scheduled_cron_completed(self, intent_id: str, result: str) -> None:
        logger.info("Scheduled cron [%s] completed: %s", intent_id, result[:120])
        await self._emit_edge_event({"type": "cron_completed", "intent_id": intent_id, "summary": result[:200]})

    def _on_kernel_event(self, ev: KernelEvent) -> None:
        if self.send_edge_json:
            asyncio.create_task(
                self._emit_edge_event(
                    {"type": "kernel_event", "event": ev.type, "step": ev.step, "detail": ev.detail}
                )
            )

    async def _request_edge_hd_frame(self) -> str | None:
        if not self.send_edge_json:
            return None
        loop = asyncio.get_running_loop()
        self._hd_frame_waiter = loop.create_future()
        await self._emit_edge_event({"type": "control", "action": "capture_hd_frame"})
        try:
            return await asyncio.wait_for(self._hd_frame_waiter, timeout=2.5)
        except asyncio.TimeoutError:
            return None
        finally:
            self._hd_frame_waiter = None

    async def _emit_edge_event(self, payload: dict[str, Any]) -> None:
        if self.send_edge_json:
            try:
                await self.send_edge_json(payload)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Housekeeping: Idle Standby & Infinite Seamless Session Rotation
    # -------------------------------------------------------------------------
    async def _housekeeping_loop(self) -> None:
        while self._running:
            await asyncio.sleep(3.0)
            if self.state != "LIVE_DUPLEX" or self._realtime is None or not self._realtime.connected:
                continue

            now = time.monotonic()
            idle_sec = now - self._realtime.last_activity_ts
            age_sec = now - self._realtime.created_at_ts

            if not self._realtime.is_busy and idle_sec >= self.duplex_idle_sleep_sec:
                logger.info("LIVE_DUPLEX idle for %.1fs -> returning to AMBIENT_WATCH", idle_sec)
                await self._close_duplex_session()
            elif not self._realtime.is_busy and idle_sec >= 3.0 and age_sec >= self.session_rotate_sec:
                logger.info("Rotating LIVE_DUPLEX session seamlessly at age %.1fs", age_sec)
                async with self._state_lock:
                    old = self._realtime
                    self._realtime = None
                    if old:
                        await old.close()
                await self.ensure_duplex_session()

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "realtime_connected": bool(self._realtime and self._realtime.connected),
            "pending_outbox_items": self.store.pending_outbox_count(),
            "recent_frames_buffered": len(self.store.get_recent_frames(count=30)),
            "hot_context": self.hot_context.snapshot_dict(),
            "active_intents_db": self.store.list_intents(only_enabled=True),
        }
