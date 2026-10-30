"""
Standard stdio MCP Server shipped as an out-of-the-box example in `config/mcp_servers.json`.

PiKernel has ZERO hardcoded knowledge of this file — it discovers and invokes these tools
purely over the standard Model Context Protocol (`stdio`), proving plug-and-play extensibility.
"""
from __future__ import annotations

import json
import platform
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer

mcp_app = MCPServer("ambient_utilities")

NOTES_FILE = Path(__file__).resolve().parent.parent / "data" / "home_notes.json"


def _load_notes() -> list[dict]:
    if NOTES_FILE.exists():
        try:
            return json.loads(NOTES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_notes(notes: list[dict]) -> None:
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


@mcp_app.tool(description="查询指定城市实时天气与今日最高最低气温（基于免Key的 Open-Meteo 气象接口）")
def get_realtime_weather(city: str = "Hangzhou") -> str:
    try:
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search?name="
            + urllib.parse.quote(city)
            + "&count=1&language=zh&format=json"
        )
        with urllib.request.urlopen(geo_url, timeout=8) as r:
            geo = json.loads(r.read().decode("utf-8"))
        results = geo.get("results") or []
        if not results:
            return f"未找到城市 '{city}' 的坐标信息。"
        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        resolved_name = f"{loc.get('name', city)} ({loc.get('admin1', '')})"

        wx_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            "&timezone=auto&forecast_days=1"
        )
        with urllib.request.urlopen(wx_url, timeout=8) as r:
            wx = json.loads(r.read().decode("utf-8"))
        cur = wx.get("current", {})
        daily = wx.get("daily", {})
        t_max = (daily.get("temperature_2m_max") or [None])[0]
        t_min = (daily.get("temperature_2m_min") or [None])[0]
        rain_prob = (daily.get("precipitation_probability_max") or [None])[0]
        return json.dumps(
            {
                "location": resolved_name,
                "current_temp_c": cur.get("temperature_2m"),
                "feels_like_c": cur.get("apparent_temperature"),
                "humidity_pct": cur.get("relative_humidity_2m"),
                "wind_kmh": cur.get("wind_speed_10m"),
                "today_min_c": t_min,
                "today_max_c": t_max,
                "rain_probability_pct": rain_prob,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return f"天气接口查询异常: {exc}"


@mcp_app.tool(description="读取当前精确本地时间、星期以及主机运行环境状态")
def get_environment_clock() -> str:
    now = datetime.now()
    return json.dumps(
        {
            "iso_datetime": now.isoformat(timespec="seconds"),
            "weekday": now.strftime("%A"),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
        },
        ensure_ascii=False,
    )


@mcp_app.tool(description="向家庭备忘录/待办便签板新增一条便签")
def add_home_note(title: str, content: str) -> str:
    notes = _load_notes()
    entry = {
        "id": len(notes) + 1,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
        "content": content,
    }
    notes.append(entry)
    _save_notes(notes)
    return f"已保存家庭便签 #{entry['id']}: {title}"


@mcp_app.tool(description="列出家庭备忘录中的所有便签")
def list_home_notes() -> str:
    notes = _load_notes()
    if not notes:
        return "当前家庭备忘录为空。"
    return json.dumps(notes[-20:], ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp_app.run(transport="stdio")
