from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, Awaitable, Callable

import websockets

logger = logging.getLogger("ambient.realtime")


REALTIME_META_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "delegate_to_kernel",
        "description": (
            "仅在需要以下深度能力时调用后端极薄内核 (PiKernel)："
            "(1) 精确数学/微积分/代码逻辑计算；(2) 抓拍1080p超清图辨认极小文字或回查过去历史画面记录；"
            "(3) 调用外部 MCP 工具（天气、日历、便签、智能家居、网页等）；(4) 注册定时/视觉守望意图。"
            "注意：对于当前实时摄像头画面已能看清的颜色、人物状态、动作问候或普通聊天，请直接用你自身的原生视觉与语音能力立即回答，不要调用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "清晰描述需要内核完成的任务目标"},
                "attach_live_frame": {
                    "type": "boolean",
                    "description": "若任务涉及辨认当前画面微小文字或公式细节，设为 true",
                },
            },
            "required": ["goal"],
        },
    },
    {
        "type": "function",
        "name": "recall_deep_memory",
        "description": "当主人询问过去的具体事实、物品位置或偏好，且当前热上下文尚未包含时，检索腾讯 Hy-Memory 2.0 与空间事件库。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用于检索长期记忆的查询语句或关键词"},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "pin_memory_note",
        "description": "当主人明确要求记住某个重要事实、约定或长期个人偏好时，立即固定到长期记忆中。",
        "parameters": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "需要长久记住的自包含事实陈述"},
            },
            "required": ["note"],
        },
    },
    {
        "type": "function",
        "name": "enter_ambient_watch",
        "description": "当主人明确表示对话结束（例如'先不聊了'、'我去睡了'、'谢啦先忙了'），在简短道别时调用此工具退回静默观察态。",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "退回静默观察态的原因"},
            },
        },
    },
]


class QwenOmniRealtimeSession:
    """
    Full-Duplex Multimodal Realtime Client for `qwen3.8-omni-flash-realtime`.

    Features:
      - Streaming 16kHz PCM audio uplink + live JPEG camera frame buffer (`input_image_buffer.append`).
      - Streaming 24kHz PCM audio downlink + live transcript deltas.
      - Server-side Semantic VAD with instant `<20ms` barge-in speaker buffer flush callback.
      - Safe zero-blocking `session.update` hot-swapping of L0 `HotWorkingContext`.
      - Non-blocking tool execution bridge (`delegate_to_kernel`, `recall_deep_memory`, etc.).
    """

    def __init__(
        self,
        ws_url: str,
        api_key: str,
        model: str = "qwen3.8-omni-flash-realtime",
        voice: str = "Tina",
        on_audio_delta: Callable[[bytes], Awaitable[None]] | None = None,
        on_barge_in: Callable[[], Awaitable[None]] | None = None,
        on_user_transcript: Callable[[str, bool], Awaitable[None]] | None = None,
        on_assistant_transcript: Callable[[str, bool], Awaitable[None]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[str]] | None = None,
        on_session_closed: Callable[[], Awaitable[None]] | None = None,
        ws_connect_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.ws_url = ws_url
        self.api_key = api_key
        self.model = model
        self.voice = voice

        self.on_audio_delta = on_audio_delta
        self.on_barge_in = on_barge_in
        self.on_user_transcript = on_user_transcript
        self.on_assistant_transcript = on_assistant_transcript
        self.tool_executor = tool_executor
        self.on_session_closed = on_session_closed
        self._ws_connect_factory = ws_connect_factory

        self._ws: Any = None
        self._recv_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()

        self.connected = False
        self.created_at_ts: float = 0.0
        self.last_activity_ts: float = 0.0
        self._has_sent_audio = False
        self._response_in_progress = False
        self._pending_tool_calls = 0
        self._session_ready_event = asyncio.Event()
        self._pending_instructions: str | None = None
        self._pending_injections: list[tuple[str, str | None]] = []
        self._response_idle_event = asyncio.Event()
        self._response_idle_event.set()

    @property
    def is_busy(self) -> bool:
        return self._response_in_progress

    def _full_endpoint(self) -> str:
        sep = "&" if "?" in self.ws_url else "?"
        if "model=" in self.ws_url:
            return self.ws_url
        return f"{self.ws_url}{sep}model={self.model}"

    async def connect(self, initial_instructions: str) -> None:
        endpoint = self._full_endpoint()
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                if self._ws_connect_factory is not None:
                    self._ws = await self._ws_connect_factory(endpoint, self.api_key)
                else:
                    self._ws = await websockets.connect(
                        endpoint,
                        additional_headers={"Authorization": f"Bearer {self.api_key}"},
                        open_timeout=25.0,
                        ping_interval=20.0,
                        ping_timeout=20.0,
                        max_size=16 * 1024 * 1024,
                    )
                first_raw = await asyncio.wait_for(self._ws.recv(), timeout=15.0)
                first_ev = json.loads(first_raw)
                logger.info(
                    "Realtime session connected (%s): session_id=%s",
                    first_ev.get("type"),
                    first_ev.get("session", {}).get("id", "unknown"),
                )
                break
            except Exception as exc:
                last_err = exc
                logger.warning("Realtime WS connect attempt %d/3 failed: %s", attempt, exc)
                if attempt < 3:
                    await asyncio.sleep(0.8 * attempt)
        else:
            raise ConnectionError(f"Failed to connect Realtime WebSocket after 3 attempts: {last_err}")

        self.connected = True
        self._has_sent_audio = False
        self._pending_tool_calls = 0
        self._session_ready_event.clear()
        self.created_at_ts = time.monotonic()
        self.last_activity_ts = time.monotonic()

        await self._send_json(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": self.voice,
                    "instructions": initial_instructions,
                    "input_audio_format": "pcm",
                    "output_audio_format": "pcm",
                    "input_audio_transcription": {"model": "qwen3-asr-flash-realtime"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 750,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                    "tools": REALTIME_META_TOOLS,
                },
            }
        )
        self._recv_task = asyncio.create_task(self._recv_loop(), name="realtime_ws_recv")
        try:
            await asyncio.wait_for(self._session_ready_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

    async def close(self) -> None:
        self.connected = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if not self.connected or self._ws is None:
            return
        raw = json.dumps(payload, ensure_ascii=False)
        async with self._send_lock:
            if self._ws is not None:
                await self._ws.send(raw)

    # -------------------------------------------------------------------------
    # Streaming Media & Hot-Swap Operations
    # -------------------------------------------------------------------------
    async def append_audio_pcm16(self, pcm_bytes: bytes) -> None:
        if not self.connected or not pcm_bytes:
            return
        b64 = base64.b64encode(pcm_bytes).decode("ascii")
        await self._send_json({"type": "input_audio_buffer.append", "audio": b64})
        self._has_sent_audio = True

    async def append_video_frame(self, jpeg_b64: str) -> None:
        if not self.connected or not jpeg_b64:
            return
        # DashScope Realtime requires audio timeline initialization prior to appending image frames
        if not self._has_sent_audio:
            await self.append_audio_pcm16(b"\x00" * 1280)
        await self._send_json({"type": "input_image_buffer.append", "image": jpeg_b64})

    async def hot_swap_instructions(self, new_instructions: str) -> None:
        """
        Update session instructions with 0ms interruption:
        If the model is currently speaking/generating, defer until `response.done`.
        """
        if not self.connected:
            return
        if self._response_in_progress:
            self._pending_instructions = new_instructions
            return
        self._pending_instructions = None
        await self._send_json(
            {
                "type": "session.update",
                "session": {"instructions": new_instructions},
            }
        )

    async def inject_event_and_speak(self, prompt_text: str, frame_jpeg_b64: str | None = None) -> None:
        """
        Proactively initiate speech or inject an asynchronous PiKernel result into the live session.
        """
        if not self.connected:
            return
        if self._response_in_progress:
            self._pending_injections.append((prompt_text, frame_jpeg_b64))
            return

        self.last_activity_ts = time.monotonic()
        if frame_jpeg_b64:
            await self.append_video_frame(frame_jpeg_b64)
        await self._send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt_text}],
                },
            }
        )
        self._response_in_progress = True
        self._response_idle_event.clear()
        await self._send_json({"type": "response.create"})

    async def wait_until_response_idle(self, timeout: float = 20.0) -> None:
        await asyncio.wait_for(self._response_idle_event.wait(), timeout=timeout)

    # -------------------------------------------------------------------------
    # Server Event Loop
    # -------------------------------------------------------------------------
    async def _recv_loop(self) -> None:
        assistant_transcript_buf: list[str] = []
        handled_call_ids: set[str] = set()

        try:
            while self.connected and self._ws is not None:
                raw = await self._ws.recv()
                ev = json.loads(raw)
                ev_type = ev.get("type", "")

                if ev_type == "session.updated":
                    self._session_ready_event.set()

                elif ev_type == "input_audio_buffer.speech_started":
                    self.last_activity_ts = time.monotonic()
                    logger.info("Semantic VAD detected user speech start (barge-in)")
                    if self.on_barge_in:
                        await self.on_barge_in()

                elif ev_type == "conversation.item.input_audio_transcription.delta":
                    delta = ev.get("delta", "")
                    if delta and self.on_user_transcript:
                        await self.on_user_transcript(delta, False)

                elif ev_type == "conversation.item.input_audio_transcription.completed":
                    self.last_activity_ts = time.monotonic()
                    transcript = (ev.get("transcript") or "").strip()
                    if transcript and self.on_user_transcript:
                        await self.on_user_transcript(transcript, True)

                elif ev_type == "response.created":
                    self._response_in_progress = True
                    self._response_idle_event.clear()
                    assistant_transcript_buf.clear()

                elif ev_type in ("response.audio.delta", "response.output_audio.delta"):
                    self.last_activity_ts = time.monotonic()
                    b64_chunk = ev.get("delta", "")
                    if b64_chunk and self.on_audio_delta:
                        pcm = base64.b64decode(b64_chunk)
                        await self.on_audio_delta(pcm)

                elif ev_type in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                    delta = ev.get("delta", "")
                    if delta:
                        assistant_transcript_buf.append(delta)
                        if self.on_assistant_transcript:
                            await self.on_assistant_transcript(delta, False)

                elif ev_type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                    full_text = (ev.get("transcript") or "".join(assistant_transcript_buf)).strip()
                    assistant_transcript_buf.clear()
                    if full_text and self.on_assistant_transcript:
                        await self.on_assistant_transcript(full_text, True)

                elif ev_type == "response.function_call_arguments.done":
                    call_id = ev.get("call_id", "")
                    name = ev.get("name", "")
                    raw_args = ev.get("arguments", "{}")
                    if call_id and call_id not in handled_call_ids:
                        handled_call_ids.add(call_id)
                        self._pending_tool_calls += 1
                        self._response_idle_event.clear()
                        asyncio.create_task(self._execute_and_reply_tool(call_id, name, raw_args))

                elif ev_type == "response.output_item.done":
                    item = ev.get("item") or {}
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id", "")
                        name = item.get("name", "")
                        raw_args = item.get("arguments", "{}")
                        if call_id and call_id not in handled_call_ids:
                            handled_call_ids.add(call_id)
                            self._pending_tool_calls += 1
                            self._response_idle_event.clear()
                            asyncio.create_task(self._execute_and_reply_tool(call_id, name, raw_args))

                elif ev_type == "response.done":
                    self._response_in_progress = False
                    self.last_activity_ts = time.monotonic()
                    # Flush any hot-swapped instructions first
                    if self._pending_instructions is not None:
                        instr = self._pending_instructions
                        self._pending_instructions = None
                        await self.hot_swap_instructions(instr)
                    # Flush any queued async kernel injections
                    if self._pending_injections and self._pending_tool_calls == 0:
                        next_prompt, next_frame = self._pending_injections.pop(0)
                        await self.inject_event_and_speak(next_prompt, next_frame)
                    elif self._pending_tool_calls == 0:
                        self._response_idle_event.set()

                elif ev_type == "error":
                    logger.warning("Realtime server error event: %s", ev.get("error"))
                    self._response_in_progress = False
                    self._response_idle_event.set()

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.connected:
                logger.warning("Realtime WS connection dropped: %s", exc)
        finally:
            self.connected = False
            self._response_in_progress = False
            self._response_idle_event.set()
            if self.on_session_closed:
                asyncio.create_task(self.on_session_closed())

    async def _execute_and_reply_tool(self, call_id: str, name: str, raw_args: str) -> None:
        try:
            try:
                args = json.loads(raw_args) if raw_args else {}
                if not isinstance(args, dict):
                    args = {"value": args}
            except Exception:
                args = {}

            logger.info("Realtime tool invoked: %s(%s)", name, args)
            output_str = ""
            if self.tool_executor:
                try:
                    output_str = await self.tool_executor(name, args)
                except Exception as exc:
                    output_str = f"Error: {exc}"

            # Wait briefly if the initial tool_call response hasn't emitted response.done yet
            for _ in range(30):
                if not self._response_in_progress:
                    break
                await asyncio.sleep(0.05)

            if not self.connected:
                return

            await self._send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output_str,
                    },
                }
            )
            if name == "enter_ambient_watch" and output_str == "__ENTER_STANDBY__":
                return

            self._response_in_progress = True
            self._response_idle_event.clear()
            await self._send_json({"type": "response.create"})
        finally:
            self._pending_tool_calls = max(0, self._pending_tool_calls - 1)
            if self._pending_tool_calls == 0 and not self._response_in_progress:
                self._response_idle_event.set()
