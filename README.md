# AetherMate · 科幻级主动式全模态语音助手

一套像科幻电影智能管家一样自然融入物理房间的 **全天候主动视觉感知 + 毫秒级全双工实时语音 + 零阻塞长期认知记忆 + 无限 MCP 泛化能力** 智能体系统。

硬件仅需 **一个摄像头观察环境 + 一个扬声器发声 + 麦克风收音**（支持智能音箱、RK3588、树莓派、Camo Studio 虚拟镜头或任意手机浏览器）。

---

## 核心架构与设计原则

1. **端侧极致精简 (`edge/`)**：房间边缘端不运行任何重型视觉检测模型、不存储业务状态，仅通过 5 行纯像素差分门控抽帧 + 16kHz 音频采集 + `<20ms` 瞬时打断清空的 24kHz 环形缓冲播放队列与中枢连接（内存占用 `< 30MB`）。
2. **原生全模态零硬编码感知 (`omni/perceptual_pulse.py`)**：绝不编写 `if pose == "wake_up"` 等死板规则。由 `qwen3.8-omni-flash` 结合 **时序关键帧网格拼图 (Temporal Contact Sheet)**、时间与记忆热块自主决策：静默观察 (`noop`)、静默记录空间/物品记忆 (`record_observation`)、后台调度内核 (`dispatch_kernel`) 或主动自然开口问候 (`proactive_engage`)，内置社交防打扰冷却锁。
3. **快慢双脑协同 (`omni/` + `kernel/`)**：
   - **快脑 (System 1)**：`qwen3.8-omni-flash-realtime` 全双工实时语音视频流，内置 `Semantic VAD` 防误触打断与 `800ms` Pre-roll 零吞字唤醒，仅挂载 4 个极简元工具确保 `< 400ms` 交互灵敏度。
   - **慢脑 (System 2)**：致敬 `@mariozechner/pi-agent-core` 的 `< 300` 行事件驱动薄内核 `PiKernel`，仅用 **4 个不可约简原语**（`python_exec`、`inspect_visual_memory`、`memory_ops`、`manage_autonomous_intents`）+ **动态防膨胀 MCP 代理 (`DynamicMCPHub`)** 涌现出精确数学验算、1080p/历史画面回溯、动态定时早报预编译及无限第三方 MCP 工具能力。
4. **腾讯 `Hy-Memory 2.0` 零阻塞旁路记忆总线 (`memory/`)**：原生集成 `hy-memory` (`HyMemoryClient`)，通过 **L0 内存热上下文块 (`0ms` 读取) + 流式联想预测预取 (`TTLLRUCache`) + `session.update` 无感热替换 + SQLite WAL 崩溃安全异步写入队列 (`Outbox`)**，彻底消除认知记忆提炼与重排对实时语音的延迟影响。

---

## 一键快捷启动与 API 可用性自动验证（推荐）

项目内置图形化与命令行双模启动台 `launcher.py`，支持填入 **API 端点 (`OMNI_API_BASE`)** 与 **密钥 (`OMNI_API_KEY`)** 后自动完成 4 阶段全链路可用性体检：
1. **全模态推理模型 (`POST /chat/completions`)** 连通性与延迟验证
2. **长期记忆向量库 (`POST /embeddings`)** 可用性检测与 **向量维度自动校准**
3. **全双工实时语音流 (`WSS /api-ws/v1/realtime`)** 握手与会话创建测试
4. **本地音视频硬件自检**（麦克风、扬声器、摄像头 / Camo Studio 跨平台探测）

### Windows 一键启动（自动创建虚拟环境并打开深色桌面启动台）

双击运行或在终端执行：

```powershell
.\start_aether_mate.bat
```

### Linux / macOS / RK3588 一键启动

```bash
chmod +x start_aether_mate.sh
./start_aether_mate.sh
```

### 无头服务器交互向导 (CLI) 与命令行静默验证

```bash
# 终端交互式填入端点/密钥并验证启动
python launcher.py --cli

# 仅执行全链路 API 与硬件验证（适合 CI 或部署自检）
python launcher.py --verify-only --api-base https://maas.qianwenaiapi.com/compatible-mode/v1 --api-key sk-xxx
```

> 验证通过后，配置会自动写入本地 `.env`（已被 `.gitignore` 严格排除，不会上传至版本库）。

---

## 分离部署：智能中枢 + 极简房间边缘端

### 1. 启动后端智能中枢

```bash
python -m server.main
```

中枢默认监听 `0.0.0.0:8765`：
- Web / 手机全双工终端界面：`http://127.0.0.1:8765`
- 边缘设备极简 WebSocket 接入点：`ws://<中枢IP>:8765/ws/edge`
- 实时记忆热块与状态检视：`http://127.0.0.1:8765/api/state`

### 2. 启动房间边缘终端（二选一或同时多端接入）

- **方式 A：智能音箱 / RK3588 / 树莓派 / Camo Studio PC 守护进程（单文件零后端依赖）**
  可将 `edge/speaker_client.py` 单独拷贝至任意边缘设备（仅需 `pip install websockets numpy opencv-python-headless sounddevice`）：
  ```bash
  python edge/speaker_client.py --server ws://<中枢IP>:8765/ws/edge --camera 0
  ```
- **方式 B：手机 / 平板 / 电脑浏览器终端（内置 WebRTC 硬件级回声消除 AEC）**
  浏览器打开 `http://<中枢IP>:8765`，点击 **启动房间感知终端** 即可将手机立在桌面充当全模态智能音箱。

---

## 挂载任意 MCP 工具服务

编辑 `config/mcp_servers.json`（与 Claude Desktop / Cursor 标准配置完全兼容），无需修改任何 Python 代码，`PiKernel` 会自动通过 `mcp_discover` 按需发现并调用：

```json
{
  "mcpServers": {
    "ambient_utilities": {
      "command": "python",
      "args": ["-m", "kernel.builtin_mcp_demo"],
      "description": "内置世界时钟、免Key实时天气预报与家庭便签服务"
    }
  }
}
```

---

## 切换本地 `Hy-Memory-30B-A3B` 专用记忆模型

如果你在本地 GPU 主机上使用 vLLM 启动了 `tencent/Hy-Memory-30B-A3B`，可在启动器的 `vLLM(可选)` 输入框或 `.env` 中填入：

```ini
MEMORY_LLM_BASE_URL=http://127.0.0.1:8000/v1
MEMORY_LLM_API_KEY=EMPTY
MEMORY_LLM_MODEL=hy-memory
```

---

## 极简边缘协议规范 (`/ws/edge`)

若后续移植到 ESP32-S3 / C++ / Rust 智能音箱固件，仅需对接以下 4 种帧：
1. **上行音频（Binary）**：`0x01` + `16000Hz 16-bit Signed LE Mono PCM`
2. **下行音频（Binary）**：`0x02` + `24000Hz 16-bit Signed LE Mono PCM`
3. **上行图像（JSON）**：`{"type": "video_frame", "jpeg_b64": "...", "motion_score": 0.08, "is_hd": false}`
4. **下行控制（JSON）**：`{"type": "control", "action": "clear_audio" | "capture_hd_frame"}`

---

## 运行测试套件

```bash
# 离线单元与网关集成测试（含启动器可用性验证测试）
pytest tests/test_hot_context_and_bus.py tests/test_pi_kernel_mcp.py tests/test_proactive_coordinator.py tests/test_launcher.py -v

# 真实云端全模态 + Hy-Memory 2.0 + MCP 端到端实景测试（需配置 .env）
pytest tests/test_live_e2e_qianwen.py -v -s
```

---

## License

MIT License
