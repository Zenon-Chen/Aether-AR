from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable


@dataclass
class DialogueTurn:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class HotWorkingContext:
    """
    L0 Zero-Latency In-Memory Hot Context Block (< 800 tokens).

    Maintains atomic slots that are read synchronously in 0ms by both:
      1. The Full-Duplex Realtime Session (via `session.update` hot-swapping)
      2. The Multimodal Perceptual Pulse Observer

    All heavy operations (Hy-Memory 2.0 retrieval, rolling dialogue summarization,
    cron briefing compilation) update this block asynchronously in the background.
    """

    def __init__(
        self,
        assistant_name: str = "Jarvis",
        persona: str = "敏锐、温暖、克制的全模态房间智能管家，说话自然简练，像熟悉主人生活习惯的老朋友。",
        max_chars: int = 2400,
        max_prefetched: int = 6,
        compaction_turn_threshold: int = 8,
    ) -> None:
        self.assistant_name = assistant_name
        self.persona = persona.strip()
        self.max_chars = max_chars
        self.max_prefetched = max_prefetched
        self.compaction_turn_threshold = compaction_turn_threshold

        self.user_profile: str = "尚在熟悉主人的日常习惯与偏好。"
        self.scene_and_spatial_state: str = "系统刚启动，正在静默观察房间环境。"
        self.spatial_observations: list[str] = []
        self.active_intentions: list[str] = []
        self.briefing_cache: dict[str, str] = {}
        self.rolling_dialogue_summary: str = ""
        self.recent_turns: list[DialogueTurn] = []
        self.prefetched_memories: list[str] = []

        self._version: int = 0
        self._lock = asyncio.Lock()
        self._listeners: list[Callable[[str, int], Awaitable[None]]] = []

    @property
    def version(self) -> int:
        return self._version

    def on_hot_swap(self, callback: Callable[[str, int], Awaitable[None]]) -> None:
        """Register listener invoked asynchronously whenever hot instructions change."""
        self._listeners.append(callback)

    async def _bump_and_notify(self, trigger_hot_swap: bool = True) -> int:
        self._version += 1
        ver = self._version
        if trigger_hot_swap and self._listeners:
            instructions = self.render_realtime_instructions()
            for cb in list(self._listeners):
                asyncio.create_task(self._safe_notify(cb, instructions, ver))
        return ver

    @staticmethod
    async def _safe_notify(cb: Callable[[str, int], Awaitable[None]], instructions: str, ver: int) -> None:
        try:
            await cb(instructions, ver)
        except Exception:
            pass

    async def update_user_profile(self, profile_text: str) -> None:
        async with self._lock:
            cleaned = profile_text.strip()
            if cleaned and cleaned != self.user_profile:
                self.user_profile = cleaned[:400]
                await self._bump_and_notify(trigger_hot_swap=True)

    async def update_scene_state(self, scene_summary: str, notify_realtime: bool = False) -> None:
        async with self._lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self.scene_and_spatial_state = f"[{ts}] {scene_summary.strip()[:240]}"
            await self._bump_and_notify(trigger_hot_swap=notify_realtime)

    async def add_spatial_observation(self, observation: str, notify_realtime: bool = True) -> None:
        async with self._lock:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = f"[{ts}] {observation.strip()[:160]}"
            if not self.spatial_observations or self.spatial_observations[-1] != entry:
                self.spatial_observations.append(entry)
                if len(self.spatial_observations) > 8:
                    self.spatial_observations = self.spatial_observations[-8:]
                await self._bump_and_notify(trigger_hot_swap=notify_realtime)

    async def set_prefetched_memories(self, memories: list[str]) -> bool:
        """Atomically update associative memories retrieved from Hy-Memory 2.0. Returns True if changed."""
        async with self._lock:
            deduped: list[str] = []
            seen: set[str] = set()
            for m in memories:
                s = m.strip()
                if s and s not in seen:
                    seen.add(s)
                    deduped.append(s[:200])
                if len(deduped) >= self.max_prefetched:
                    break
            if deduped == self.prefetched_memories:
                return False
            self.prefetched_memories = deduped
            await self._bump_and_notify(trigger_hot_swap=True)
            return True

    async def set_active_intentions(self, intentions: list[str]) -> None:
        async with self._lock:
            self.active_intentions = [i.strip()[:160] for i in intentions if i.strip()][:8]
            await self._bump_and_notify(trigger_hot_swap=True)

    async def set_briefing_cache(self, key: str, content: str) -> None:
        async with self._lock:
            self.briefing_cache[key] = f"[{datetime.now().strftime('%H:%M')}] {content.strip()[:360]}"
            await self._bump_and_notify(trigger_hot_swap=True)

    async def record_turn(self, role: str, content: str) -> list[DialogueTurn] | None:
        """
        Append a completed turn. If recent_turns exceeds compaction threshold,
        slices and returns older turns so the Sidecar Compactor can summarize them asynchronously.
        """
        text = content.strip()
        if not text:
            return None
        async with self._lock:
            self.recent_turns.append(DialogueTurn(role=role, content=text[:600]))
            self._version += 1
            if len(self.recent_turns) >= self.compaction_turn_threshold:
                to_compact = self.recent_turns[:-3]
                self.recent_turns = self.recent_turns[-3:]
                return to_compact
            return None

    async def apply_compacted_summary(self, new_summary: str) -> None:
        async with self._lock:
            self.rolling_dialogue_summary = new_summary.strip()[:450]
            await self._bump_and_notify(trigger_hot_swap=True)

    def render_realtime_instructions(self) -> str:
        """
        Compile L0 Hot Context into compact system instructions for Realtime full-duplex voice.
        Uses priority-preserving budgeting so critical sections (persona, prefetched memories,
        rolling summary, recent turns, and tool guidelines) are never truncated from the tail.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        header_parts = [
            f"你是 {self.assistant_name}。{self.persona}",
            f"【当前时间】{now_str}",
            f"【主人核心画像】{self.user_profile[:260]}",
            f"【当前房间与视觉环境】{self.scene_and_spatial_state[:200]}",
        ]

        footer_parts: list[str] = []
        if self.prefetched_memories:
            mem_text = "\n".join(f"- {m[:160]}" for m in self.prefetched_memories[: self.max_prefetched])
            footer_parts.append(f"【关联长期记忆(Hy-Memory 2.0)】\n{mem_text[:520]}")
        if self.rolling_dialogue_summary:
            footer_parts.append(f"【近期对话背景摘要】{self.rolling_dialogue_summary[:360]}")
        if self.recent_turns:
            turns_text = "\n".join(
                f"[{t.timestamp}] {t.role}: {t.content[:140]}" for t in self.recent_turns[-4:]
            )
            footer_parts.append(f"【最近几句对话】\n{turns_text[:380]}")

        guidelines = (
            "【交互与工具准则】\n"
            "1. 口语自然简短，像真人对话，不要念列表符号或Markdown符号。\n"
            "2. 涉及精确数学计算、查看高清历史画面、联网/文件/MCP工具操作、或设置定时/视觉提醒时，"
            "先自然回应半句（如'好，我帮你算一下/看一下'），同时调用 `delegate_to_kernel` 交给后端内核执行。\n"
            "3. 若主人询问久远细节且上方关联记忆未包含，调用 `recall_deep_memory` 检索。\n"
            "4. 若主人明确表示结束对话（如'先忙了/去睡了/没事了'），简短道别后调用 `enter_ambient_watch`。"
        )
        footer_parts.append(guidelines)

        header_str = "\n".join(header_parts)
        footer_str = "\n".join(footer_parts)

        # Optional middle slots fit within remaining character budget
        middle_candidates: list[str] = []
        if self.briefing_cache:
            brief_text = " | ".join(f"{k}: {v[:220]}" for k, v in list(self.briefing_cache.items())[-2:])
            middle_candidates.append(f"【后台预备简报缓存(可直接零延迟口播)】{brief_text[:340]}")
        if self.active_intentions:
            intent_text = "；".join(self.active_intentions[:4])
            middle_candidates.append(f"【待关注主动意图】{intent_text[:260]}")
        if self.spatial_observations:
            obs_text = "；".join(self.spatial_observations[-4:])
            middle_candidates.append(f"【近期空间与动作观察】{obs_text[:260]}")

        reserved = len(header_str) + len(footer_str) + 2
        remaining = max(0, self.max_chars - reserved)
        kept_middle: list[str] = []
        for cand in middle_candidates:
            if len(cand) + 1 <= remaining:
                kept_middle.append(cand)
                remaining -= len(cand) + 1
            elif remaining > 60:
                kept_middle.append(cand[: remaining - 1])
                remaining = 0
                break

        all_sections = [header_str, *kept_middle, footer_str]
        compiled = "\n".join(s for s in all_sections if s)
        if len(compiled) <= self.max_chars:
            return compiled
        # Final safeguard if max_chars was configured unusually small (e.g., in unit test < 1200)
        return compiled[: self.max_chars]

    def snapshot_dict(self) -> dict:
        return {
            "version": self._version,
            "user_profile": self.user_profile,
            "scene_and_spatial_state": self.scene_and_spatial_state,
            "spatial_observations": list(self.spatial_observations),
            "active_intentions": list(self.active_intentions),
            "briefing_cache": dict(self.briefing_cache),
            "rolling_dialogue_summary": self.rolling_dialogue_summary,
            "recent_turns": [t.__dict__ for t in self.recent_turns],
            "prefetched_memories": list(self.prefetched_memories),
        }
