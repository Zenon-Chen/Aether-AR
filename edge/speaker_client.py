#!/usr/bin/env python3
"""
Standalone Ultra-Lightweight Smart Speaker / RK3588 / Raspberry Pi / Camo / PC Edge Client.

Design Goal:
  - ZERO dependency on backend Agent/Memory/LLM libraries (< 30MB RAM footprint).
  - Supports both Linux V4L2 cameras and Windows DirectShow virtual cameras (Camo Studio, OBS).
  - Includes Split-Device Acoustic Echo Guard so separate Mic (e.g. Camo) + Speaker (e.g. Redmi)
    setups never self-interrupt from speaker bleed while preserving loud human barge-in!
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import struct
import sys
import threading
import time
from typing import Any

import websockets

BINARY_UPLINK_AUDIO = 0x01
BINARY_DOWNLINK_AUDIO = 0x02


def _pcm16_rms(pcm_bytes: bytes) -> float:
    n = len(pcm_bytes) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm_bytes[: n * 2])
    return math.sqrt(sum(s * s for s in samples) / n) / 32768.0


class SpeakerRingBuffer:
    """Thread-safe byte buffer feeding hardware audio output callback with instant barge-in flush & echo tail tracking."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._last_active_play_ts: float = 0.0

    def write(self, pcm_bytes: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm_bytes)
            self._last_active_play_ts = time.monotonic()

    def read_into(self, out_view: memoryview) -> None:
        needed = len(out_view)
        with self._lock:
            avail = min(needed, len(self._buf))
            if avail > 0:
                out_view[:avail] = self._buf[:avail]
                del self._buf[:avail]
                self._last_active_play_ts = time.monotonic()
            if avail < needed:
                out_view[avail:] = b"\x00" * (needed - avail)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._last_active_play_ts = 0.0

    def is_playing_or_tail(self, tail_sec: float = 0.32) -> bool:
        with self._lock:
            if len(self._buf) > 0:
                return True
            return (time.monotonic() - self._last_active_play_ts) < tail_sec


def open_camera_cross_platform(camera_index: int) -> Any:
    """Open physical or virtual camera (Camo Studio DirectShow on Windows, V4L2/default on Linux)."""
    import cv2
    import numpy as np

    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            got_any = False
            # Warm up virtual camera stream (Camo/OBS often yields black frames for first 3~6 reads)
            for _ in range(15):
                ok, f = cap.read()
                if ok and f is not None:
                    got_any = True
                    if float(np.std(f)) > 1.5:
                        return cap
                time.sleep(0.04)
            if got_any:
                return cap
            cap.release()
    return None


class EdgeTerminalDaemon:
    def __init__(
        self,
        server_url: str,
        camera_index: int = 0,
        enable_audio: bool = True,
        enable_camera: bool = True,
        motion_threshold: float = 0.025,
        heartbeat_sec: float = 30.0,
        hardware_aec: bool = False,
        barge_in_rms: float = 0.12,
    ) -> None:
        self.server_url = server_url
        self.camera_index = camera_index
        self.enable_audio = enable_audio
        self.enable_camera = enable_camera
        self.motion_threshold = motion_threshold
        self.heartbeat_sec = heartbeat_sec
        self.hardware_aec = hardware_aec
        self.barge_in_rms = barge_in_rms

        self.speaker_buf = SpeakerRingBuffer()
        self._hd_capture_event = asyncio.Event()
        self._running = True

    async def run(self) -> None:
        while self._running:
            try:
                print(f"[Edge] Connecting to Brain Gateway: {self.server_url} ...", flush=True)
                async with websockets.connect(
                    self.server_url,
                    ping_interval=20.0,
                    ping_timeout=20.0,
                    max_size=16 * 1024 * 1024,
                ) as ws:
                    print("[Edge] Connected! Ambient observation & full-duplex audio active.", flush=True)
                    tasks = [asyncio.create_task(self._recv_loop(ws))]
                    if self.enable_audio:
                        tasks.append(asyncio.create_task(self._audio_io_loop(ws)))
                    if self.enable_camera:
                        tasks.append(asyncio.create_task(self._camera_gate_loop(ws)))

                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for p in pending:
                        p.cancel()
                    for d in done:
                        if d.exception():
                            raise d.exception()  # type: ignore[misc]
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"[Edge] Disconnected ({exc}), retrying in 2.5s...", flush=True)
                await asyncio.sleep(2.5)

    async def _recv_loop(self, ws: Any) -> None:
        async for msg in ws:
            if isinstance(msg, bytes):
                if len(msg) > 1 and msg[0] == BINARY_DOWNLINK_AUDIO:
                    self.speaker_buf.write(msg[1:])
            else:
                data = json.loads(msg)
                mtype = data.get("type", "")
                if mtype == "control":
                    action = data.get("action")
                    if action == "clear_audio":
                        self.speaker_buf.clear()
                        print("\n[Edge] ⚡ Barge-in -> Speaker buffer flushed!", flush=True)
                    elif action == "capture_hd_frame":
                        self._hd_capture_event.set()
                elif mtype == "state_sync":
                    print(f"\n[Brain State] ==> {data.get('state')}", flush=True)
                elif mtype == "pulse_decision":
                    print(
                        f"[Pulse] action={data.get('action')} | scene={data.get('scene_state')}"
                        + (f" | obs={data.get('observation')}" if data.get("observation") else ""),
                        flush=True,
                    )
                elif mtype == "transcript" and data.get("final"):
                    role = "🧑 You" if data.get("role") == "user" else "🤖 Jarvis"
                    print(f"{role}: {data.get('text')}", flush=True)

    async def _audio_io_loop(self, ws: Any) -> None:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        uplink_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
        silence_frame = bytes([BINARY_UPLINK_AUDIO]) + (b"\x00" * 1280)

        def mic_callback(indata: bytes, _frames: int, _time_info: Any, _status: Any) -> None:
            raw_pcm = bytes(indata)
            # Split-device echo guard: suppress speaker bleed unless hardware AEC is enabled or loud human barge-in occurs
            if not self.hardware_aec and self.speaker_buf.is_playing_or_tail():
                if _pcm16_rms(raw_pcm) < self.barge_in_rms:
                    pkt = silence_frame
                else:
                    pkt = bytes([BINARY_UPLINK_AUDIO]) + raw_pcm
            else:
                pkt = bytes([BINARY_UPLINK_AUDIO]) + raw_pcm

            loop.call_soon_threadsafe(
                lambda: uplink_q.put_nowait(pkt) if not uplink_q.full() else None
            )

        def spk_callback(outdata: bytearray, _frames: int, _time_info: Any, _status: Any) -> None:
            self.speaker_buf.read_into(memoryview(outdata))

        with (
            sd.RawInputStream(samplerate=16000, blocksize=640, dtype="int16", channels=1, callback=mic_callback),
            sd.RawOutputStream(samplerate=24000, blocksize=960, dtype="int16", channels=1, callback=spk_callback),
        ):
            while True:
                pkt = await uplink_q.get()
                await ws.send(pkt)

    async def _camera_gate_loop(self, ws: Any) -> None:
        import cv2
        import numpy as np

        cap = await asyncio.to_thread(open_camera_cross_platform, self.camera_index)
        if cap is None:
            print(f"[Edge] Warning: Camera #{self.camera_index} could not be opened. Running audio-only.", flush=True)
            return

        prev_tiny: Any = None
        last_sent_ts = 0.0
        first_frame = True

        try:
            while True:
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok or frame is None:
                    await asyncio.sleep(0.5)
                    continue

                if first_frame:
                    h, w = frame.shape[:2]
                    print(f"[Edge] Camera #{self.camera_index} streaming active ({w}x{h})", flush=True)
                    first_frame = False

                # On-demand native HD frame capture requested by PiKernel (zero DirectShow pin re-init stall!)
                if self._hd_capture_event.is_set():
                    self._hd_capture_event.clear()
                    _, hd_jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    await ws.send(
                        json.dumps(
                            {
                                "type": "video_frame",
                                "jpeg_b64": base64.b64encode(hd_jpg.tobytes()).decode("ascii"),
                                "motion_score": 1.0,
                                "is_hd": True,
                            }
                        )
                    )

                # 5-line ultra-lightweight pixel-diff gatekeeper (<1% CPU)
                small_frame = cv2.resize(frame, (640, 480))
                tiny_gray = cv2.resize(cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY), (64, 36))
                motion_score = (
                    1.0
                    if prev_tiny is None
                    else float(np.mean(cv2.absdiff(tiny_gray, prev_tiny))) / 255.0
                )
                now = time.monotonic()

                if motion_score >= self.motion_threshold or (now - last_sent_ts) >= self.heartbeat_sec:
                    prev_tiny = tiny_gray
                    last_sent_ts = now
                    _, jpg = cv2.imencode(".jpg", small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 74])
                    await ws.send(
                        json.dumps(
                            {
                                "type": "video_frame",
                                "jpeg_b64": base64.b64encode(jpg.tobytes()).decode("ascii"),
                                "motion_score": round(motion_score, 4),
                                "is_hd": False,
                            }
                        )
                    )

                await asyncio.sleep(0.5)
        finally:
            cap.release()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Ambient Omni-Agent Lightweight Edge Terminal")
    parser.add_argument("--server", default="ws://127.0.0.1:8765/ws/edge", help="Brain Gateway WebSocket URL")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--no-audio", action="store_true", help="Disable mic/speaker streams")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera stream")
    parser.add_argument("--hardware-aec", action="store_true", help="Set if using a speakerphone with built-in DSP AEC")
    parser.add_argument("--motion-threshold", type=float, default=0.025, help="Pixel-diff gate threshold (0~1)")
    args = parser.parse_args()

    daemon = EdgeTerminalDaemon(
        server_url=args.server,
        camera_index=args.camera,
        enable_audio=not args.no_audio,
        enable_camera=not args.no_camera,
        motion_threshold=args.motion_threshold,
        hardware_aec=args.hardware_aec,
    )
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
