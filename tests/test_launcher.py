from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import launcher


def test_env_load_and_save_roundtrip(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    launcher.save_env_dict(
        {
            "OMNI_API_BASE": "https://example.com/compatible-mode/v1",
            "OMNI_API_KEY": "sk-test-dummy-123456",
            "OMNI_MODEL": "qwen3.8-omni-flash",
            "MEMORY_EMBEDDING_DIMS": "1536",
            "PORT": "9000",
        },
        env_path=env_file,
    )
    loaded = launcher.load_env_dict(env_path=env_file)
    assert loaded["OMNI_API_BASE"] == "https://example.com/compatible-mode/v1"
    assert loaded["OMNI_API_KEY"] == "sk-test-dummy-123456"
    assert loaded["MEMORY_EMBEDDING_DIMS"] == "1536"
    assert loaded["PORT"] == "9000"


@pytest.mark.asyncio
async def test_verify_configuration_rejects_empty_credentials() -> None:
    report = await launcher.verify_configuration(
        {"OMNI_API_BASE": "", "OMNI_API_KEY": ""},
        include_hardware=False,
    )
    assert report.core_api_ok is False
    assert len(report.checks) == 1
    assert report.checks[0].ok is False


@pytest.mark.asyncio
async def test_verify_configuration_auto_calibrates_embedding_dims(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_http_post(url: str, api_key: str, payload: dict[str, Any], timeout: float = 15.0) -> tuple[int, dict[str, Any]]:
        if url.endswith("/chat/completions"):
            return 200, {"choices": [{"message": {"content": "OK"}}]}
        if url.endswith("/embeddings"):
            return 200, {"data": [{"embedding": [0.01] * 768}]}
        return 404, {"error": {"message": "not found"}}

    async def fake_ws_probe(ws_url: str, api_key: str, rt_model: str) -> tuple[bool, float, str]:
        return True, 42.0, f"握手成功 ({rt_model})"

    monkeypatch.setattr(launcher, "_http_json_post", fake_http_post)
    monkeypatch.setattr(launcher, "_probe_realtime_ws", fake_ws_probe)

    report = await launcher.verify_configuration(
        {
            "OMNI_API_BASE": "https://maas.qianwenaiapi.com/compatible-mode/v1",
            "OMNI_API_KEY": "sk-dummy",
            "OMNI_MODEL": "qwen3.8-omni-flash",
            "MEMORY_EMBEDDING_MODEL": "text-embedding-v4",
        },
        include_hardware=False,
    )
    assert report.core_api_ok is True
    assert report.detected_embedding_dims == 768
    assert all(c.ok for c in report.checks)


@pytest.mark.asyncio
async def test_verify_configuration_reports_invalid_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_http_401(url: str, api_key: str, payload: dict[str, Any], timeout: float = 15.0) -> tuple[int, dict[str, Any]]:
        return 401, {"error": {"message": "Invalid API-key provided"}}

    async def fake_ws_fail(ws_url: str, api_key: str, rt_model: str) -> tuple[bool, float, str]:
        return False, 15.0, "HTTP 401 Unauthorized"

    monkeypatch.setattr(launcher, "_http_json_post", fake_http_401)
    monkeypatch.setattr(launcher, "_probe_realtime_ws", fake_ws_fail)

    report = await launcher.verify_configuration(
        {
            "OMNI_API_BASE": "https://maas.qianwenaiapi.com/compatible-mode/v1",
            "OMNI_API_KEY": "sk-wrong",
        },
        include_hardware=False,
    )
    assert report.core_api_ok is False
    assert all(not c.ok for c in report.checks)
