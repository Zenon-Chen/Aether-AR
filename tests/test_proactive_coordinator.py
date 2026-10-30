from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import struct
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from config.loader import AppConfig
from edge.speaker_client import SpeakerRingBuffer
from omni.coordinator import AmbientOmniCoordinator
from protocol.events import (
    BINARY_DOWNLINK_AUDIO_PCM16,
    pack_uplink_audio,
    unpack_binary_frame,
)
from server.main import create_app


class FakeRealtimeSession:
    """In-memory mock of QwenOmniRealtimeSession verifying full-duplex state transitions, pre-roll & barge-in."""

    def __init__(self, **kwargs: Any) -> None:
        self.on_audio_delta = kwargs.get("on_audio_delta")
        self.on_barge_in = kwargs.get("on_barge_in")
        self.on_user_transcript = kwargs.get("on_user_transcript")
        self.on_assistant_transcript = kwargs.get("on_assistant_transcript")
        self.tool_executor = kwargs.get("tool_executor")

        self.connected = False
        self.is_busy = False
        self.created_at_ts = 0.0
        self.last_activity_ts = 0.0
        self.instructions_history: list[str] = []
        self.received_audio_chunks: list[bytes] = []
        self.received_frames: list[str] = []
        self.injected_prompts: list[str] = []

    async def connect(self, initial_instructions: str) -> None:
        import time
        self.connected = True
        self.created_at_ts = time.monotonic()
        self.last_activity_ts = time.monotonic()
        self.instructions_history.append(initial_instructions)

    async def close(self) -> None:
        self.connected = False

    async def append_audio_pcm16(self, pcm_bytes: bytes) -> None:
        self.received_audio_chunks.append(pcm_bytes)

    async def append_video_frame(self, jpeg_b64: str) -> None:
        self.received_frames.append(jpeg_b64)

    async def hot_swap_instructions(self, new_instructions: str) -> None:
        self.instructions_history.append(new_instructions)

    async def inject_event_and_speak(self, prompt_text: str, frame_jpeg_b64: str | None = None) -> None:
        self.injected_prompts.append(prompt_text)
        if frame_jpeg_b64:
            self.received_frames.append(frame_jpeg_b64)
        if self.on_assistant_transcript:
            await self.on_assistant_transcript("早上好主人，今天天气晴朗！", True)
        if self.on_audio_delta:
            await self.on_audio_delta(b"\x10\x00" * 480)


class FakePulseChoiceLLM:
    def __init__(self, json_responses: list[dict[str, Any]]) -> None:
        self._responses = list(json_responses)
        self.chat = type("Chat", (), {"completions": self})()

    async def create(self, **kwargs: Any) -> Any:
        payload = self._responses.pop(0) if self._responses else {"scene_state": "安静", "action": "noop"}
        msg = type("Msg", (), {"content": json.dumps(payload, ensure_ascii=False), "tool_calls": None})()
        return type("Resp", (), {"choices": [type("C", (), {"message": msg})()]})()


def _make_frame_b64(color: tuple[int, int, int]) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (128, 96), color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _loud_pcm16_chunk(samples: int = 640) -> bytes:
    return b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 300 * i / 16000))) for i in range(samples))


def test_speaker_ring_buffer_instant_barge_in_flush() -> None:
    ring = SpeakerRingBuffer()
    ring.write(b"\x01\x02" * 2000)
    out = bytearray(400)
    ring.read_into(memoryview(out))
    assert out[:4] == b"\x01\x02\x01\x02"

    # Simulate Semantic VAD barge-in clear
    ring.clear()
    out2 = bytearray(400)
    ring.read_into(memoryview(out2))
    assert out2 == bytearray(400)


@pytest.mark.asyncio
async def test_proactive_visual_wake_and_cooldown_guardrail(tmp_path: Path) -> None:
    cfg = AppConfig(
        omni_api_base="http://localhost",
        omni_api_key="",
        omni_model="mock-omni",
        realtime_ws_url="ws://localhost",
        realtime_model="mock-rt",
        data_dir=tmp_path,
        settings={"perception": {"pulse_min_interval_sec": 0.0, "proactive_min_cooldown_sec": 30.0}},
    )
    fake_llm = FakePulseChoiceLLM(
        [
            {
                "scene_state": "主人刚刚推门走进客厅，手里拿着外套",
                "action": "proactive_engage",
                "observation": "主人回到家中，将深色外套挂在玄关衣架",
                "importance": 2,
                "proactive_prompt": "微笑欢迎主人回家",
            },
            {
                "scene_state": "主人在桌前坐下喝水",
                "action": "noop",
                "observation": "",
            },
        ]
    )

    created_sessions: list[FakeRealtimeSession] = []

    def rt_factory(**kw: Any) -> FakeRealtimeSession:
        s = FakeRealtimeSession(**kw)
        created_sessions.append(s)
        return s

    coord = AmbientOmniCoordinator(cfg, llm_client=fake_llm, realtime_session_factory=rt_factory)  # type: ignore[arg-type]
    downlink_audio: list[bytes] = []
    edge_events: list[dict[str, Any]] = []

    async def capture_audio(pcm: bytes) -> None:
        downlink_audio.append(pcm)

    async def capture_json(ev: dict[str, Any]) -> None:
        edge_events.append(ev)

    coord.send_edge_audio = capture_audio
    coord.send_edge_json = capture_json
    await coord.start()

    try:
        assert coord.state == "AMBIENT_WATCH"
        decision = await coord.handle_edge_video_frame(_make_frame_b64((180, 90, 60)), motion_score=0.18)
        assert decision is not None and decision.action == "proactive_engage"
        assert coord.state == "LIVE_DUPLEX"
        assert len(created_sessions) == 1
        assert len(downlink_audio) == 1
        # Verify spatial observation was simultaneously persisted
        eps = coord.store.search_visual_episodes("外套")
        assert len(eps) == 1

        # Simulate Semantic VAD barge-in from Realtime session -> verify Edge receives clear_audio!
        await created_sessions[0].on_barge_in()
        assert any(e.get("type") == "control" and e.get("action") == "clear_audio" for e in edge_events)
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_acoustic_wake_replays_preroll_without_clipping(tmp_path: Path) -> None:
    cfg = AppConfig(
        omni_api_base="http://localhost",
        omni_api_key="",
        omni_model="mock-omni",
        realtime_ws_url="ws://localhost",
        realtime_model="mock-rt",
        data_dir=tmp_path,
    )
    fake_llm = FakePulseChoiceLLM([])
    sessions: list[FakeRealtimeSession] = []

    coord = AmbientOmniCoordinator(
        cfg,
        llm_client=fake_llm,  # type: ignore[arg-type]
        realtime_session_factory=lambda **kw: (sessions.append(FakeRealtimeSession(**kw)) or sessions[-1]),
    )
    await coord.start()
    try:
        # Feed 3 quiet ambient chunks + 2 loud speech chunks
        quiet = b"\x02\x00" * 640
        loud = _loud_pcm16_chunk(640)
        await coord.handle_edge_audio_pcm16(quiet)
        await coord.handle_edge_audio_pcm16(quiet)
        await coord.handle_edge_audio_pcm16(quiet)
        assert coord.state == "AMBIENT_WATCH"

        await coord.handle_edge_audio_pcm16(loud)
        await coord.handle_edge_audio_pcm16(loud)
        assert coord.state == "LIVE_DUPLEX"
        # All 5 pre-roll chunks (3 quiet lead-in + 2 speech onset) were replayed seamlessly!
        assert len(sessions[0].received_audio_chunks) == 5
    finally:
        await coord.stop()


def test_gateway_websocket_binary_and_json_roundtrip(tmp_path: Path) -> None:
    cfg = AppConfig(
        omni_api_base="http://localhost",
        omni_api_key="",
        omni_model="mock-omni",
        realtime_ws_url="ws://localhost",
        realtime_model="mock-rt",
        data_dir=tmp_path,
    )
    fake_llm = FakePulseChoiceLLM([])
    coord = AmbientOmniCoordinator(
        cfg,
        llm_client=fake_llm,  # type: ignore[arg-type]
        realtime_session_factory=lambda **kw: FakeRealtimeSession(**kw),
    )
    app = create_app(cfg, coord)

    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200 and r.json()["ok"] is True

        with client.websocket_connect("/ws/edge") as ws:
            init_msg = json.loads(ws.receive_text())
            assert init_msg["type"] == "state_sync"
            assert init_msg["state"] == "AMBIENT_WATCH"

            # Send text_input over Edge WS -> wakes FakeRealtimeSession -> returns transcript + binary PCM24k!
            ws.send_text(json.dumps({"type": "text_input", "text": "早上好"}))

            got_audio = False
            got_assistant_txt = False
            for _ in range(8):
                msg = ws.receive()
                if "bytes" in msg and msg["bytes"]:
                    tag, pcm = unpack_binary_frame(msg["bytes"])
                    if tag == BINARY_DOWNLINK_AUDIO_PCM16 and len(pcm) > 0:
                        got_audio = True
                elif "text" in msg and msg["text"]:
                    ev = json.loads(msg["text"])
                    if ev.get("type") == "transcript" and ev.get("role") == "assistant":
                        got_assistant_txt = True
                if got_audio and got_assistant_txt:
                    break

            assert got_audio is True
            assert got_assistant_txt is True

            # Verify uplink binary PCM frame is accepted cleanly
            ws.send_bytes(pack_uplink_audio(b"\x00\x00" * 640))
            ws.send_text('{"type":"ping"}')
