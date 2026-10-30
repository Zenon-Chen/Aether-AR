#!/usr/bin/env python3
"""
Aether-AR Quick Launcher & Connectivity Verifier (GUI + CLI).

Features:
  - Configure API Endpoint (`OMNI_API_BASE`), API Key (`OMNI_API_KEY`), Model ID (`OMNI_MODEL`),
    Embedding model, optional local vLLM `Hy-Memory-30B-A3B` endpoint, port, and camera index.
  - Automatic 4-stage live availability verification:
      [1/4] Multimodal Chat & Tool Endpoint (`POST /chat/completions`)
      [2/4] Memory Embedding Endpoint & Auto-Dimension Calibration (`POST /embeddings`)
      [3/4] Full-Duplex Realtime Speech WebSocket (`WSS /api-ws/v1/realtime`)
      [4/4] Local Edge Audio/Camera Hardware Self-Check
  - Persists validated configuration to `.env` (gitignored, local-only).
  - One-click launch & clean shutdown of Brain Server (`server.main`) and Edge Client (`edge/speaker_client.py`)
    with real-time log streaming.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import websockets

from config.loader import derive_realtime_model, derive_realtime_ws_url

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"


# =============================================================================
# 1. .env Reader / Writer
# =============================================================================
DEFAULT_ENV_VALUES: dict[str, str] = {
    "OMNI_API_BASE": "https://maas.qianwenaiapi.com/compatible-mode/v1",
    "OMNI_API_KEY": "",
    "OMNI_MODEL": "qwen3.8-omni-flash",
    "OMNI_REALTIME_WS_URL": "",
    "OMNI_REALTIME_MODEL": "",
    "OMNI_VOICE": "Tina",
    "MEMORY_MODE": "pro",
    "MEMORY_USER_ID": "default_home_user",
    "MEMORY_LLM_BASE_URL": "",
    "MEMORY_LLM_API_KEY": "",
    "MEMORY_LLM_MODEL": "",
    "MEMORY_EMBEDDING_MODEL": "text-embedding-v4",
    "MEMORY_EMBEDDING_DIMS": "1024",
    "HOST": "0.0.0.0",
    "PORT": "8765",
    "DATA_DIR": "./data",
}


def load_env_dict(env_path: Path = ENV_PATH) -> dict[str, str]:
    values = dict(DEFAULT_ENV_VALUES)
    source = env_path if env_path.exists() else (ENV_EXAMPLE_PATH if ENV_EXAMPLE_PATH.exists() else None)
    if source is not None:
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if source == ENV_EXAMPLE_PATH and v == "your_api_key_here":
                v = ""
            values[k] = v
    return values


def save_env_dict(values: dict[str, str], env_path: Path = ENV_PATH) -> None:
    merged = dict(DEFAULT_ENV_VALUES)
    merged.update({k: str(v).strip() for k, v in values.items()})
    lines = [
        "# Generated & verified by Aether-AR Launcher",
        f"OMNI_API_BASE={merged['OMNI_API_BASE']}",
        f"OMNI_API_KEY={merged['OMNI_API_KEY']}",
        f"OMNI_MODEL={merged['OMNI_MODEL']}",
        f"OMNI_REALTIME_WS_URL={merged['OMNI_REALTIME_WS_URL']}",
        f"OMNI_REALTIME_MODEL={merged['OMNI_REALTIME_MODEL']}",
        f"OMNI_VOICE={merged['OMNI_VOICE']}",
        "",
        f"MEMORY_MODE={merged['MEMORY_MODE']}",
        f"MEMORY_USER_ID={merged['MEMORY_USER_ID']}",
        f"MEMORY_LLM_BASE_URL={merged['MEMORY_LLM_BASE_URL']}",
        f"MEMORY_LLM_API_KEY={merged['MEMORY_LLM_API_KEY']}",
        f"MEMORY_LLM_MODEL={merged['MEMORY_LLM_MODEL']}",
        f"MEMORY_EMBEDDING_MODEL={merged['MEMORY_EMBEDDING_MODEL']}",
        f"MEMORY_EMBEDDING_DIMS={merged['MEMORY_EMBEDDING_DIMS']}",
        "",
        f"HOST={merged['HOST']}",
        f"PORT={merged['PORT']}",
        f"DATA_DIR={merged['DATA_DIR']}",
        "",
    ]
    env_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 2. Live Multi-Stage Endpoint & Hardware Verification Engine
# =============================================================================
@dataclass
class CheckItemResult:
    name: str
    ok: bool
    latency_ms: float
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    checks: list[CheckItemResult]
    detected_embedding_dims: int | None = None

    @property
    def core_api_ok(self) -> bool:
        # Core API checks are chat_omni, embedding, and realtime_ws
        core = [c for c in self.checks if c.name != "本地音视频硬件自检"]
        return bool(core) and all(c.ok for c in core)


def _http_json_post(url: str, api_key: str, payload: dict[str, Any], timeout: float = 15.0) -> tuple[int, dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Aether-AR-Launcher/0.1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"error": {"message": body[:300]}}


async def _probe_realtime_ws(ws_url: str, api_key: str, rt_model: str) -> tuple[bool, float, str]:
    sep = "&" if "?" in ws_url else "?"
    full_url = ws_url if "model=" in ws_url else f"{ws_url}{sep}model={rt_model}"
    t0 = time.perf_counter()
    last_err = ""
    for attempt in range(1, 3):
        try:
            async with websockets.connect(
                full_url,
                additional_headers={"Authorization": f"Bearer {api_key}"},
                open_timeout=15.0,
                close_timeout=3.0,
            ) as ws:
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                ev = json.loads(raw)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if ev.get("type") == "session.created":
                    sess = ev.get("session", {})
                    return (
                        True,
                        dt_ms,
                        f"握手成功 ({rt_model}, session={sess.get('id', 'ok')[:14]}..., default_voice={sess.get('voice', 'Tina')})",
                    )
                if ev.get("type") == "error":
                    return False, dt_ms, f"服务端返回错误: {ev.get('error')}"
                return True, dt_ms, f"连接已建立 (首包事件: {ev.get('type')})"
        except Exception as exc:
            last_err = str(exc)
            if attempt < 2:
                await asyncio.sleep(0.5)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return False, dt_ms, f"WebSocket 握手失败: {last_err}"


def _probe_local_hardware(camera_index: int = 0) -> CheckItemResult:
    t0 = time.perf_counter()
    parts: list[str] = []
    ok_any = True

    try:
        import sounddevice as sd

        in_idx, out_idx = sd.default.device
        devs = sd.query_devices()
        in_name = devs[in_idx]["name"] if in_idx is not None and in_idx >= 0 else "未检测到麦克风"
        out_name = devs[out_idx]["name"] if out_idx is not None and out_idx >= 0 else "未检测到扬声器"
        parts.append(f"麦克风='{in_name}' | 扬声器='{out_name}'")
    except Exception as exc:
        parts.append(f"音频设备未就绪({exc})")

    try:
        from edge.speaker_client import open_camera_cross_platform

        cap = open_camera_cross_platform(camera_index)
        if cap is not None:
            ok_read, frame = cap.read()
            if ok_read and frame is not None:
                h, w = frame.shape[:2]
                parts.append(f"摄像头#{camera_index}=正常({w}x{h})")
            else:
                parts.append(f"摄像头#{camera_index}=已打开但无画面")
            cap.release()
        else:
            parts.append(f"摄像头#{camera_index}=未连接(可仅用语音或手机Web端)")
    except Exception as exc:
        parts.append(f"摄像头自检跳过({exc})")

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return CheckItemResult(
        name="本地音视频硬件自检",
        ok=ok_any,
        latency_ms=dt_ms,
        detail=" ；".join(parts),
    )


async def verify_configuration(
    env_cfg: dict[str, str],
    camera_index: int = 0,
    include_hardware: bool = True,
) -> VerificationReport:
    api_base = env_cfg.get("OMNI_API_BASE", "").strip().rstrip("/")
    api_key = env_cfg.get("OMNI_API_KEY", "").strip()
    omni_model = env_cfg.get("OMNI_MODEL", "qwen3.8-omni-flash").strip()
    emb_model = env_cfg.get("MEMORY_EMBEDDING_MODEL", "text-embedding-v4").strip()
    ws_url = derive_realtime_ws_url(api_base, env_cfg.get("OMNI_REALTIME_WS_URL", ""))
    rt_model = derive_realtime_model(omni_model, env_cfg.get("OMNI_REALTIME_MODEL", ""))

    checks: list[CheckItemResult] = []
    detected_dims: int | None = None

    if not api_base or not api_key:
        checks.append(
            CheckItemResult(
                name="基础参数校验",
                ok=False,
                latency_ms=0.0,
                detail="API 端点 (OMNI_API_BASE) 与 API 密钥 (OMNI_API_KEY) 不能为空。",
            )
        )
        return VerificationReport(checks=checks)

    # 1. Check Chat / Omni Model
    t0 = time.perf_counter()
    try:
        st, body = await asyncio.to_thread(
            _http_json_post,
            f"{api_base}/chat/completions",
            api_key,
            {
                "model": omni_model,
                "messages": [{"role": "user", "content": "Reply OK"}],
                "max_tokens": 10,
                "enable_thinking": False,
            },
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if st == 200 and "choices" in body:
            reply_preview = (body["choices"][0]["message"].get("content") or "").strip()[:30]
            checks.append(
                CheckItemResult(
                    name="全模态推理模型 (Chat/Omni)",
                    ok=True,
                    latency_ms=dt_ms,
                    detail=f"可用 ({omni_model} -> '{reply_preview}')",
                )
            )
        else:
            err_msg = body.get("error", {}).get("message", str(body)[:180])
            checks.append(
                CheckItemResult(
                    name="全模态推理模型 (Chat/Omni)",
                    ok=False,
                    latency_ms=dt_ms,
                    detail=f"HTTP {st}: {err_msg}",
                )
            )
    except Exception as exc:
        checks.append(
            CheckItemResult(
                name="全模态推理模型 (Chat/Omni)",
                ok=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                detail=f"请求异常: {exc}",
            )
        )

    # 2. Check Embedding Model & Auto-Detect Dimensions
    t1 = time.perf_counter()
    try:
        st, body = await asyncio.to_thread(
            _http_json_post,
            f"{api_base}/embeddings",
            api_key,
            {"model": emb_model, "input": "Aether-AR memory vector probe"},
        )
        dt_ms = (time.perf_counter() - t1) * 1000.0
        if st == 200 and body.get("data"):
            vec = body["data"][0].get("embedding", [])
            detected_dims = len(vec)
            checks.append(
                CheckItemResult(
                    name="长期记忆向量模型 (Embeddings)",
                    ok=detected_dims > 0,
                    latency_ms=dt_ms,
                    detail=f"可用 ({emb_model}, 自动校准维度={detected_dims})",
                    extra={"dims": detected_dims},
                )
            )
        else:
            err_msg = body.get("error", {}).get("message", str(body)[:180])
            checks.append(
                CheckItemResult(
                    name="长期记忆向量模型 (Embeddings)",
                    ok=False,
                    latency_ms=dt_ms,
                    detail=f"HTTP {st}: {err_msg}",
                )
            )
    except Exception as exc:
        checks.append(
            CheckItemResult(
                name="长期记忆向量模型 (Embeddings)",
                ok=False,
                latency_ms=(time.perf_counter() - t1) * 1000.0,
                detail=f"请求异常: {exc}",
            )
        )

    # 3. Check Realtime Full-Duplex WebSocket
    ws_ok, ws_ms, ws_detail = await _probe_realtime_ws(ws_url, api_key, rt_model)
    checks.append(
        CheckItemResult(
            name="全双工实时语音流 (Realtime WS)",
            ok=ws_ok,
            latency_ms=ws_ms,
            detail=ws_detail,
        )
    )

    # 4. Optional Custom Hy-Memory LLM Endpoint Check (e.g. local vLLM Hy-Memory-30B-A3B)
    mem_llm_base = env_cfg.get("MEMORY_LLM_BASE_URL", "").strip().rstrip("/")
    if mem_llm_base and mem_llm_base != api_base:
        mem_key = env_cfg.get("MEMORY_LLM_API_KEY", "").strip() or api_key
        mem_model = env_cfg.get("MEMORY_LLM_MODEL", "").strip() or omni_model
        t_m = time.perf_counter()
        try:
            st, body = await asyncio.to_thread(
                _http_json_post,
                f"{mem_llm_base}/chat/completions",
                mem_key,
                {"model": mem_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8},
            )
            dt_ms = (time.perf_counter() - t_m) * 1000.0
            checks.append(
                CheckItemResult(
                    name="独立记忆写入模型 (Hy-Memory LLM)",
                    ok=(st == 200),
                    latency_ms=dt_ms,
                    detail=f"可用 ({mem_model} @ {mem_llm_base})" if st == 200 else f"HTTP {st}: {body}",
                )
            )
        except Exception as exc:
            checks.append(
                CheckItemResult(
                    name="独立记忆写入模型 (Hy-Memory LLM)",
                    ok=False,
                    latency_ms=(time.perf_counter() - t_m) * 1000.0,
                    detail=f"连接失败: {exc}",
                )
            )

    # 5. Local Hardware Self-Check
    if include_hardware:
        hw_res = await asyncio.to_thread(_probe_local_hardware, camera_index)
        checks.append(hw_res)

    return VerificationReport(checks=checks, detected_embedding_dims=detected_dims)


# =============================================================================
# 3. Desktop Sci-Fi GUI Launcher (Tkinter)
# =============================================================================
def run_gui_launcher() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    env_data = load_env_dict()

    root = tk.Tk()
    root.title("Aether-AR · 全模态主动式语音助手启动器")
    root.geometry("880x680")
    root.minsize(780, 600)
    root.configure(bg="#0b101b")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#0b101b")
    style.configure("Card.TFrame", background="#131c2e", relief="flat")
    style.configure("TLabel", background="#131c2e", foreground="#e2eaf4", font=("Microsoft YaHei UI", 10))
    style.configure("Header.TLabel", background="#0b101b", foreground="#4de1ff", font=("Microsoft YaHei UI", 15, "bold"))
    style.configure("Sub.TLabel", background="#0b101b", foreground="#8a9bb5", font=("Microsoft YaHei UI", 9))
    style.configure("Hint.TLabel", background="#131c2e", foreground="#7d92b0", font=("Consolas", 9))
    style.configure("TCheckbutton", background="#131c2e", foreground="#e2eaf4", font=("Microsoft YaHei UI", 9))
    style.map("TCheckbutton", background=[("active", "#131c2e")])
    style.configure("Accent.TButton", background="#0088bb", foreground="#ffffff", font=("Microsoft YaHei UI", 10, "bold"), padding=8)
    style.map("Accent.TButton", background=[("active", "#00aadd")])
    style.configure("Launch.TButton", background="#00a86b", foreground="#ffffff", font=("Microsoft YaHei UI", 10, "bold"), padding=8)
    style.map("Launch.TButton", background=[("active", "#00c880")])
    style.configure("Stop.TButton", background="#c93b4b", foreground="#ffffff", font=("Microsoft YaHei UI", 10, "bold"), padding=8)
    style.map("Stop.TButton", background=[("active", "#e04d5f")])

    # Header
    hdr = ttk.Frame(root)
    hdr.pack(fill="x", padx=18, pady=(14, 6))
    ttk.Label(hdr, text="AETHER-AR · 快捷配置与全链路启动台", style="Header.TLabel").pack(side="left")
    status_lbl = ttk.Label(hdr, text="● 待验证 / 未启动", style="Sub.TLabel")
    status_lbl.pack(side="right", pady=4)

    # Config Card
    card = ttk.Frame(root, style="Card.TFrame", padding=16)
    card.pack(fill="x", padx=18, pady=6)
    card.columnconfigure(1, weight=1)
    card.columnconfigure(3, weight=1)

    var_api_base = tk.StringVar(value=env_data.get("OMNI_API_BASE", ""))
    var_api_key = tk.StringVar(value=env_data.get("OMNI_API_KEY", ""))
    var_show_key = tk.BooleanVar(value=False)
    var_model = tk.StringVar(value=env_data.get("OMNI_MODEL", "qwen3.8-omni-flash"))
    var_emb_model = tk.StringVar(value=env_data.get("MEMORY_EMBEDDING_MODEL", "text-embedding-v4"))
    var_emb_dims = tk.StringVar(value=env_data.get("MEMORY_EMBEDDING_DIMS", "1024"))
    var_voice = tk.StringVar(value=env_data.get("OMNI_VOICE", "Tina"))
    var_port = tk.StringVar(value=env_data.get("PORT", "8765"))
    var_cam_idx = tk.StringVar(value="0")
    var_mem_base = tk.StringVar(value=env_data.get("MEMORY_LLM_BASE_URL", ""))
    var_start_edge = tk.BooleanVar(value=True)
    var_derived_hint = tk.StringVar(value="")

    def _entry(parent: Any, text_var: tk.StringVar, show: str = "") -> tk.Entry:
        e = tk.Entry(
            parent,
            textvariable=text_var,
            show=show,
            bg="#090e18",
            fg="#edf4fc",
            insertbackground="#4de1ff",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#263859",
            highlightcolor="#4de1ff",
            font=("Consolas", 10),
        )
        return e

    # Row 0: API Base
    ttk.Label(card, text="API 端点 (Base URL):").grid(row=0, column=0, sticky="w", pady=5, padx=(0, 8))
    ent_base = _entry(card, var_api_base)
    ent_base.grid(row=0, column=1, columnspan=3, sticky="ew", pady=5)

    # Row 1: API Key
    ttk.Label(card, text="API 密钥 (API Key):").grid(row=1, column=0, sticky="w", pady=5, padx=(0, 8))
    ent_key = _entry(card, var_api_key, show="•")
    ent_key.grid(row=1, column=1, columnspan=2, sticky="ew", pady=5)
    ttk.Checkbutton(
        card,
        text="显示密钥",
        variable=var_show_key,
        command=lambda: ent_key.configure(show="" if var_show_key.get() else "•"),
    ).grid(row=1, column=3, sticky="w", padx=(10, 0))

    # Row 2: Model & Embedding Model
    ttk.Label(card, text="全模态模型 ID:").grid(row=2, column=0, sticky="w", pady=5, padx=(0, 8))
    _entry(card, var_model).grid(row=2, column=1, sticky="ew", pady=5, padx=(0, 12))
    ttk.Label(card, text="向量模型 (Hy-Memory):").grid(row=2, column=2, sticky="w", pady=5, padx=(6, 8))
    _entry(card, var_emb_model).grid(row=2, column=3, sticky="ew", pady=5)

    # Row 3: Voice, Port, Camera, Optional Local Memory LLM
    ttk.Label(card, text="语音音色 / 网关端口:").grid(row=3, column=0, sticky="w", pady=5, padx=(0, 8))
    vp_frame = ttk.Frame(card, style="Card.TFrame")
    vp_frame.grid(row=3, column=1, sticky="ew", pady=5, padx=(0, 12))
    _entry(vp_frame, var_voice).pack(side="left", fill="x", expand=True)
    ttk.Label(vp_frame, text=" 端口:").pack(side="left", padx=4)
    _entry(vp_frame, var_port).pack(side="left", width=7)

    ttk.Label(card, text="摄像头序号 / 记忆端点:").grid(row=3, column=2, sticky="w", pady=5, padx=(6, 8))
    cm_frame = ttk.Frame(card, style="Card.TFrame")
    cm_frame.grid(row=3, column=3, sticky="ew", pady=5)
    _entry(cm_frame, var_cam_idx).pack(side="left", width=4)
    ttk.Label(cm_frame, text=" vLLM(可选):").pack(side="left", padx=4)
    _entry(cm_frame, var_mem_base).pack(side="left", fill="x", expand=True)

    # Row 4: Derived Realtime WS Preview & Edge toggle
    def _update_derived(*_: Any) -> None:
        ws = derive_realtime_ws_url(var_api_base.get())
        rt = derive_realtime_model(var_model.get())
        var_derived_hint.set(f"自动推导全双工流: {ws}?model={rt}  |  向量维度: {var_emb_dims.get()}")

    var_api_base.trace_add("write", _update_derived)
    var_model.trace_add("write", _update_derived)
    var_emb_dims.trace_add("write", _update_derived)
    _update_derived()

    ttk.Label(card, textvariable=var_derived_hint, style="Hint.TLabel").grid(
        row=4, column=0, columnspan=3, sticky="w", pady=(6, 0)
    )
    ttk.Checkbutton(
        card,
        text="同时启动本机音箱/Camo边缘守护进程",
        variable=var_start_edge,
    ).grid(row=4, column=3, sticky="e", pady=(6, 0))

    # Action Bar
    bar = ttk.Frame(root)
    bar.pack(fill="x", padx=18, pady=8)

    log_q: queue.Queue[str] = queue.Queue()
    running_procs: list[subprocess.Popen[Any]] = []

    def log_msg(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        log_q.put(f"[{ts}] {msg}\n")

    def collect_current_env() -> dict[str, str]:
        d = load_env_dict()
        d["OMNI_API_BASE"] = var_api_base.get().strip()
        d["OMNI_API_KEY"] = var_api_key.get().strip()
        d["OMNI_MODEL"] = var_model.get().strip()
        d["MEMORY_EMBEDDING_MODEL"] = var_emb_model.get().strip()
        d["MEMORY_EMBEDDING_DIMS"] = var_emb_dims.get().strip() or "1024"
        d["OMNI_VOICE"] = var_voice.get().strip() or "Tina"
        d["PORT"] = var_port.get().strip() or "8765"
        d["MEMORY_LLM_BASE_URL"] = var_mem_base.get().strip()
        return d

    def do_verify_async(on_success_launch: bool = False) -> None:
        cfg_snapshot = collect_current_env()
        try:
            cam_i = int(var_cam_idx.get().strip() or "0")
        except ValueError:
            cam_i = 0

        btn_verify.configure(state="disabled")
        btn_launch.configure(state="disabled")
        status_lbl.configure(text="● 正在执行全链路可用性验证...", foreground="#ffb347")
        log_msg("开始验证 API 端点、模型、向量库、Realtime 双工流及本地硬件...")

        def worker() -> None:
            report = asyncio.run(verify_configuration(cfg_snapshot, camera_index=cam_i, include_hardware=True))
            for item in report.checks:
                mark = "✅ PASS" if item.ok else "❌ FAIL"
                log_msg(f"{mark} | {item.name} ({item.latency_ms:.0f}ms) -> {item.detail}")

            if report.detected_embedding_dims:
                cfg_snapshot["MEMORY_EMBEDDING_DIMS"] = str(report.detected_embedding_dims)
                root.after(0, lambda d=str(report.detected_embedding_dims): var_emb_dims.set(d))

            if report.core_api_ok:
                save_env_dict(cfg_snapshot)
                log_msg("✨ 全部核心接口验证通过！配置已安全保存至本地 .env。")
                root.after(0, lambda: status_lbl.configure(text="● 验证通过 (配置已保存)", foreground="#38ef7d"))
                if on_success_launch:
                    root.after(100, start_processes)
            else:
                log_msg("⚠️ 存在未通过的核心 API 检查项，请检查端点地址、API 密钥或模型名称。")
                root.after(0, lambda: status_lbl.configure(text="● 验证未通过，请检查日志", foreground="#ff5c77"))
                if on_success_launch:
                    root.after(0, lambda: messagebox.showerror("验证未通过", "核心 API 可用性验证失败，请查看下方诊断日志修正后再启动。"))

            root.after(0, lambda: btn_verify.configure(state="normal"))
            root.after(0, lambda: btn_launch.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _pipe_reader(proc: subprocess.Popen[Any], prefix: str) -> None:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            log_q.put(f"{prefix} {line}")

    def start_processes() -> None:
        if running_procs:
            log_msg("服务已在运行中。")
            return
        port = var_port.get().strip() or "8765"
        cam = var_cam_idx.get().strip() or "0"
        py_exe = sys.executable
        child_env = dict(os.environ)
        child_env["PYTHONIOENCODING"] = "utf-8"

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        log_msg(f"正在启动后端智能中枢 (http://127.0.0.1:{port}) ...")
        srv_proc = subprocess.Popen(
            [py_exe, "-u", "-m", "server.main"],
            cwd=str(PROJECT_ROOT),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        running_procs.append(srv_proc)
        threading.Thread(target=_pipe_reader, args=(srv_proc, "[Server]"), daemon=True).start()

        if var_start_edge.get():
            def _delayed_edge() -> None:
                time.sleep(2.2)
                if not running_procs:
                    return
                log_msg(f"正在启动本机边缘采集客户端 (Camera #{cam}) ...")
                edge_proc = subprocess.Popen(
                    [py_exe, "-u", "edge/speaker_client.py", "--server", f"ws://127.0.0.1:{port}/ws/edge", "--camera", cam],
                    cwd=str(PROJECT_ROOT),
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creationflags,
                )
                running_procs.append(edge_proc)
                threading.Thread(target=_pipe_reader, args=(edge_proc, "[Edge]  "), daemon=True).start()

            threading.Thread(target=_delayed_edge, daemon=True).start()

        status_lbl.configure(text=f"● 运行中 (Port {port})", foreground="#38ef7d")

    def stop_processes() -> None:
        if not running_procs:
            log_msg("当前没有正在运行的服务进程。")
            return
        log_msg("正在停止所有 Aether-AR 服务进程...")
        while running_procs:
            p = running_procs.pop()
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        status_lbl.configure(text="● 已停止", foreground="#8a9bb5")
        log_msg("所有进程已安全关闭，摄像头与音频设备已释放。")

    btn_verify = ttk.Button(bar, text="🔍 验证 API 可用性并保存", style="Accent.TButton", command=lambda: do_verify_async(False))
    btn_verify.pack(side="left", padx=(0, 10))

    btn_launch = ttk.Button(bar, text="🚀 验证并一键启动 Aether-AR", style="Launch.TButton", command=lambda: do_verify_async(True))
    btn_launch.pack(side="left", padx=(0, 10))

    btn_stop = ttk.Button(bar, text="⏹ 停止服务", style="Stop.TButton", command=stop_processes)
    btn_stop.pack(side="left", padx=(0, 10))

    ttk.Button(
        bar,
        text="🌐 打开 Web 终端",
        style="Accent.TButton",
        command=lambda: webbrowser.open(f"http://127.0.0.1:{var_port.get().strip() or '8765'}"),
    ).pack(side="right")

    # Console Output Card
    console_frame = ttk.Frame(root, style="Card.TFrame", padding=10)
    console_frame.pack(fill="both", expand=True, padx=18, pady=(4, 16))
    txt_console = tk.Text(
        console_frame,
        bg="#060911",
        fg="#bfe6ff",
        insertbackground="#4de1ff",
        relief="flat",
        font=("Consolas", 9),
        wrap="word",
    )
    scroll = ttk.Scrollbar(console_frame, command=txt_console.yview)
    txt_console.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    txt_console.pack(side="left", fill="both", expand=True)

    log_msg("欢迎使用 Aether-AR 启动台。请确认上方 API 端点与密钥，点击【验证 API 可用性并保存】或【验证并一键启动】。")

    def _poll_logs() -> None:
        while True:
            try:
                line = log_q.get_nowait()
            except queue.Empty:
                break
            txt_console.insert("end", line)
            txt_console.see("end")
        root.after(100, _poll_logs)

    def _on_close() -> None:
        stop_processes()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.after(100, _poll_logs)
    root.mainloop()


# =============================================================================
# 4. Headless CLI Wizard & Automated `--verify-only` Mode
# =============================================================================
def run_cli_wizard(verify_only: bool = False, cli_overrides: dict[str, str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    env_cfg = load_env_dict()
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                env_cfg[k] = v

    if not verify_only:
        print("=" * 66)
        print("  Aether-AR · 交互式配置与连通性自检向导 (CLI Mode)")
        print("=" * 66)
        cur_base = env_cfg.get("OMNI_API_BASE", "")
        inp_base = input(f"1. API 端点 Base URL [{cur_base}]: ").strip()
        if inp_base:
            env_cfg["OMNI_API_BASE"] = inp_base

        cur_key = env_cfg.get("OMNI_API_KEY", "")
        masked = (cur_key[:8] + "..." + cur_key[-4:]) if len(cur_key) > 14 else ("已设置" if cur_key else "未设置")
        inp_key = getpass.getpass(f"2. API 密钥 API Key [{masked}, 直接回车保持不变]: ").strip()
        if inp_key:
            env_cfg["OMNI_API_KEY"] = inp_key

        cur_model = env_cfg.get("OMNI_MODEL", "qwen3.8-omni-flash")
        inp_model = input(f"3. 核心全模态模型 ID [{cur_model}]: ").strip()
        if inp_model:
            env_cfg["OMNI_MODEL"] = inp_model

    print("\n正在执行全链路可用性验证...")
    report = asyncio.run(verify_configuration(env_cfg, include_hardware=True))
    for item in report.checks:
        icon = "[PASS]" if item.ok else "[FAIL]"
        print(f"  {icon} {item.name:<28} ({item.latency_ms:>5.0f} ms) -> {item.detail}")

    if report.detected_embedding_dims:
        env_cfg["MEMORY_EMBEDDING_DIMS"] = str(report.detected_embedding_dims)

    if not report.core_api_ok:
        print("\n[!] 核心 API 验证未全部通过，请检查端点、密钥或模型权限。")
        return 1

    save_env_dict(env_cfg)
    print("\n[OK] 全部核心接口验证通过，配置已保存至 .env！")

    if verify_only:
        return 0

    ans = input("\n是否立即启动 Aether-AR 中枢与本机边缘端？[Y/n]: ").strip().lower()
    if ans in ("", "y", "yes"):
        port = env_cfg.get("PORT", "8765")
        srv = subprocess.Popen([sys.executable, "-u", "-m", "server.main"], cwd=str(PROJECT_ROOT))
        time.sleep(2.2)
        edge = subprocess.Popen(
            [sys.executable, "-u", "edge/speaker_client.py", "--server", f"ws://127.0.0.1:{port}/ws/edge"],
            cwd=str(PROJECT_ROOT),
        )
        try:
            srv.wait()
        except KeyboardInterrupt:
            edge.terminate()
            srv.terminate()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Aether-AR Quick Launcher & API Verifier")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal wizard instead of GUI")
    parser.add_argument("--verify-only", action="store_true", help="Verify current/provided API config and exit")
    parser.add_argument("--api-base", help="Override OMNI_API_BASE")
    parser.add_argument("--api-key", help="Override OMNI_API_KEY")
    parser.add_argument("--model", help="Override OMNI_MODEL")
    args = parser.parse_args()

    overrides: dict[str, str] = {}
    if args.api_base:
        overrides["OMNI_API_BASE"] = args.api_base
    if args.api_key:
        overrides["OMNI_API_KEY"] = args.api_key
    if args.model:
        overrides["OMNI_MODEL"] = args.model

    if args.verify_only or args.cli:
        sys.exit(run_cli_wizard(verify_only=args.verify_only, cli_overrides=overrides))

    # Default to GUI if display is available, otherwise gracefully fall back to CLI wizard
    try:
        run_gui_launcher()
    except Exception as exc:
        print(f"[Launcher] GUI unavailable ({exc}), falling back to CLI wizard...")
        sys.exit(run_cli_wizard(verify_only=False, cli_overrides=overrides))


if __name__ == "__main__":
    main()
