from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv


def derive_realtime_ws_url(api_base: str, explicit_ws_url: str = "") -> str:
    """Derive DashScope / Qianwen Realtime WebSocket URL from OpenAI-compatible base URL."""
    if explicit_ws_url.strip():
        return explicit_ws_url.strip()
    parsed = urlparse(api_base.strip())
    scheme = "wss" if parsed.scheme in ("https", "wss", "") else "ws"
    host = parsed.netloc or parsed.path.split("/")[0]
    return f"{scheme}://{host}/api-ws/v1/realtime"


def derive_realtime_model(omni_model: str, explicit_realtime_model: str = "") -> str:
    """Derive full-duplex speech model ID from base omni model ID."""
    if explicit_realtime_model.strip():
        return explicit_realtime_model.strip()
    m = omni_model.strip()
    if m.endswith("-realtime") or "-realtime-" in m:
        return m
    return f"{m}-realtime"


@dataclass
class AppConfig:
    # Core API
    omni_api_base: str
    omni_api_key: str
    omni_model: str
    realtime_ws_url: str
    realtime_model: str
    omni_voice: str = "Tina"

    # Hy-Memory 2.0
    memory_mode: str = "pro"
    memory_user_id: str = "default_home_user"
    memory_llm_base_url: str = ""
    memory_llm_api_key: str = ""
    memory_llm_model: str = ""
    memory_embedding_model: str = "text-embedding-v4"
    memory_embedding_dims: int = 1024

    # Server & Storage
    host: str = "0.0.0.0"
    port: int = 8765
    data_dir: Path = Path("./data")
    mcp_config_path: Path = Path("./config/mcp_servers.json")

    # YAML runtime settings
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.memory_llm_base_url:
            self.memory_llm_base_url = self.omni_api_base
        if not self.memory_llm_api_key:
            self.memory_llm_api_key = self.omni_api_key
        if not self.memory_llm_model:
            self.memory_llm_model = self.omni_model


def load_config(project_root: Path | None = None) -> AppConfig:
    root = project_root or Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)

    settings_file = root / "config" / "settings.yaml"
    settings: dict[str, Any] = {}
    if settings_file.exists():
        with settings_file.open("r", encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}

    api_base = os.getenv("OMNI_API_BASE", "https://maas.qianwenaiapi.com/compatible-mode/v1").rstrip("/")
    api_key = os.getenv("OMNI_API_KEY", "")
    omni_model = os.getenv("OMNI_MODEL", "qwen3.8-omni-flash")

    ws_url = derive_realtime_ws_url(api_base, os.getenv("OMNI_REALTIME_WS_URL", ""))
    rt_model = derive_realtime_model(omni_model, os.getenv("OMNI_REALTIME_MODEL", ""))

    data_dir_raw = os.getenv("DATA_DIR", "./data")
    data_dir = Path(data_dir_raw) if Path(data_dir_raw).is_absolute() else (root / data_dir_raw)
    data_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        omni_api_base=api_base,
        omni_api_key=api_key,
        omni_model=omni_model,
        realtime_ws_url=ws_url,
        realtime_model=rt_model,
        omni_voice=os.getenv("OMNI_VOICE", "Tina"),
        memory_mode=os.getenv("MEMORY_MODE", "pro"),
        memory_user_id=os.getenv("MEMORY_USER_ID", "default_home_user"),
        memory_llm_base_url=os.getenv("MEMORY_LLM_BASE_URL", "").rstrip("/"),
        memory_llm_api_key=os.getenv("MEMORY_LLM_API_KEY", ""),
        memory_llm_model=os.getenv("MEMORY_LLM_MODEL", ""),
        memory_embedding_model=os.getenv("MEMORY_EMBEDDING_MODEL", "text-embedding-v4"),
        memory_embedding_dims=int(os.getenv("MEMORY_EMBEDDING_DIMS", "1024")),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8765")),
        data_dir=data_dir,
        mcp_config_path=root / "config" / "mcp_servers.json",
        settings=settings,
    )
