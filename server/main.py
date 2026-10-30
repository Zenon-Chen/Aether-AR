from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from config.loader import AppConfig, load_config
from omni.coordinator import AmbientOmniCoordinator
from protocol.events import (
    BINARY_UPLINK_AUDIO_PCM16,
    pack_downlink_audio,
    unpack_binary_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ambient.server")


class EdgeConnectionHub:
    """Manages active WebSocket connections from lightweight room edge terminals."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.info("Edge terminal connected (active=%d)", len(self._clients))

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        logger.info("Edge terminal disconnected (active=%d)", len(self._clients))

    async def broadcast_audio_pcm24k(self, pcm_bytes: bytes) -> None:
        frame = pack_downlink_audio(pcm_bytes)
        async with self._lock:
            targets = list(self._clients)
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_bytes(frame)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.remove(ws)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False)
        async with self._lock:
            targets = list(self._clients)
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_text(raw)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.remove(ws)


class TextInputRequest(BaseModel):
    text: str


def create_app(
    config: AppConfig | None = None,
    coordinator: AmbientOmniCoordinator | None = None,
) -> FastAPI:
    cfg = config or load_config()
    coord = coordinator or AmbientOmniCoordinator(cfg)
    hub = EdgeConnectionHub()

    coord.send_edge_audio = hub.broadcast_audio_pcm24k
    coord.send_edge_json = hub.broadcast_json

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await coord.start()
        yield
        await coord.stop()

    app = FastAPI(title="Aether-AR Brain Gateway", version="0.1.0", lifespan=lifespan)
    app.state.config = cfg
    app.state.coordinator = coord
    app.state.hub = hub

    web_index = Path(__file__).resolve().parent.parent / "edge" / "mobile_web" / "index.html"

    @app.get("/")
    async def serve_web_terminal():
        if web_index.exists():
            return FileResponse(str(web_index), media_type="text/html")
        return HTMLResponse("<h1>Aether-AR Gateway Running</h1>")

    @app.get("/api/health")
    async def health_check():
        return {"ok": True, "state": coord.state, "model": cfg.omni_model, "realtime_model": cfg.realtime_model}

    @app.get("/api/state")
    async def inspect_state():
        return JSONResponse(coord.status_snapshot())

    @app.post("/api/text")
    async def submit_text(req: TextInputRequest):
        await coord.handle_user_text_input(req.text)
        return {"ok": True, "state": coord.state}

    @app.websocket("/ws/edge")
    async def edge_websocket(ws: WebSocket):
        await hub.add(ws)
        await ws.send_text(
            json.dumps(
                {
                    "type": "state_sync",
                    "state": coord.state,
                    "assistant": coord.hot_context.assistant_name,
                },
                ensure_ascii=False,
            )
        )
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                if "bytes" in msg and msg["bytes"]:
                    tag, payload = unpack_binary_frame(msg["bytes"])
                    if tag == BINARY_UPLINK_AUDIO_PCM16 and payload:
                        await coord.handle_edge_audio_pcm16(payload)

                elif "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    mtype = data.get("type", "")
                    if mtype == "video_frame":
                        # Run visual evaluation without blocking incoming audio packets on this socket
                        asyncio.create_task(
                            coord.handle_edge_video_frame(
                                jpeg_b64=str(data.get("jpeg_b64", "")),
                                motion_score=float(data.get("motion_score", 0.0)),
                                is_hd=bool(data.get("is_hd", False)),
                                bypass_pulse_rate_limit=bool(data.get("force_eval", False)),
                            )
                        )
                    elif mtype == "text_input":
                        asyncio.create_task(coord.handle_user_text_input(str(data.get("text", ""))))
                    elif mtype == "ping":
                        await ws.send_text('{"type":"pong"}')

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Edge WebSocket session exception: %s", exc)
        finally:
            await hub.remove(ws)

    return app


def main() -> None:
    cfg = load_config()
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
