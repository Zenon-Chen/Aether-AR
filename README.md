# Aether-AR · 自主多模态语音助手

**Aether-AR** 是一个支持环境视觉感知、主动交互、长期记忆与工具调用的自主多模态语音助手。只需配备基础的摄像头、麦克风与扬声器，即可在房间内实现自然的实时语音交流与日常任务协助。

---

## 技术栈概览

- **全模态交互模型 (Omni Model)**：基于 `Qwen-Omni`（`qwen3.8-omni-flash` & `realtime`）提供视觉环境感知与全双工实时语音对话。
- **长期记忆系统 (Long-Term Memory)**：集成 `Hy-Memory 2.0` 与本地 SQLite/向量存储，在后台异步完成对话总结、用户偏好提炼与历史记忆检索。
- **轻量 Agent 后端 (Agent Kernel & MCP)**：内置精简的 `Pi` 风格 Agent 执行循环，支持 Python 代码执行、定时任务调度，并兼容标准 `MCP (Model Context Protocol)` 工具扩展（通过 `config/mcp_servers.json` 配置）。
- **轻量化端侧部署 (Edge Deployment)**：计算与状态集中在服务端，端侧仅负责基础音视频采集与播放，资源占用极低，支持：
  - **独立硬件终端**：单文件脚本 `edge/speaker_client.py`，可轻松部署在树莓派、RK3588、迷你主机或带摄像头的智能音箱上。
  - **手机 / 浏览器终端**：直接通过浏览器打开内置 Web 页面，即可调用手机或电脑的摄像头与麦克风接入系统。

---

## 一键快捷启动与 API 可用性自动验证

项目内置图形化与命令行双模启动台 `launcher.py`，支持填入 **API 端点 (`OMNI_API_BASE`)** 与 **密钥 (`OMNI_API_KEY`)** 后自动完成 4 项可用性检测：
1. **全模态推理模型 (`POST /chat/completions`)** 连通性与延迟验证
2. **长期记忆向量库 (`POST /embeddings`)** 可用性检测与 **向量维度自动校准**
3. **全双工实时语音流 (`WSS /api-ws/v1/realtime`)** 握手与会话创建测试
4. **本地音视频硬件自检**（麦克风、扬声器、摄像头跨平台探测）

### Windows 一键启动

双击运行或在终端执行（首次运行会自动创建 `.venv` 虚拟环境并安装依赖）：

```powershell
.\start_aether_mate.bat
```

### Linux / macOS / 边缘设备一键启动

```bash
chmod +x start_aether_mate.sh
./start_aether_mate.sh
```

### 命令行交互向导 (CLI) 与静默验证

```bash
# 终端交互式填入端点/密钥并验证启动
python launcher.py --cli

# 仅执行 API 与硬件可用性验证
python launcher.py --verify-only --api-base https://maas.qianwenaiapi.com/compatible-mode/v1 --api-key sk-xxx
```

> 验证通过后，配置会自动保存至本地 `.env`（已被 `.gitignore` 排除，不会上传至版本库）。

---

## 分离部署（服务端 + 端侧设备）

若将服务端与端侧设备分开部署：

```bash
# 1. 在服务端启动中枢（默认监听 0.0.0.0:8765）
python -m server.main

# 2. 在端侧设备启动音视频客户端（或直接用手机浏览器访问 http://<服务端IP>:8765）
python edge/speaker_client.py --server ws://<服务端IP>:8765/ws/edge --camera 0
```

---

## License

MIT License
