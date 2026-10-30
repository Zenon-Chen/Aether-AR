from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Protocol

from openai import AsyncOpenAI

from config.loader import AppConfig
from .hot_context import DialogueTurn, HotWorkingContext
from .outbox import DurableOutboxAndEpisodeStore

logger = logging.getLogger("ambient.memory")


class MemoryBackendProtocol(Protocol):
    async def async_add(self, data: Any, *, user_id: str = "", **kwargs: Any) -> dict[str, Any]: ...
    async def async_search(self, query: str | None = None, *, user_ids: list[str] | None = None, **kwargs: Any) -> dict[str, Any]: ...
    async def async_get_user_profile(self, *, user_id: str) -> dict[str, Any]: ...
    def close(self) -> None: ...


class TTLLRUCache:
    """Sub-millisecond in-memory TTL LRU cache for speculative memory prefetching."""

    def __init__(self, max_size: int = 128, ttl_sec: float = 180.0) -> None:
        self.max_size = max_size
        self.ttl_sec = ttl_sec
        self._store: OrderedDict[str, tuple[float, list[str]]] = OrderedDict()

    def get(self, key: str) -> list[str] | None:
        norm = " ".join(key.strip().lower().split())
        if norm not in self._store:
            return None
        ts, val = self._store[norm]
        if (time.monotonic() - ts) > self.ttl_sec:
            self._store.pop(norm, None)
            return None
        self._store.move_to_end(norm)
        return list(val)

    def put(self, key: str, value: list[str]) -> None:
        norm = " ".join(key.strip().lower().split())
        self._store[norm] = (time.monotonic(), list(value))
        self._store.move_to_end(norm)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()


class HyMemorySidecarBus:
    """
    Zero-Blocking Memory Bus bridging L0 HotWorkingContext, Durable SQLite Outbox/Episodes,
    and Tencent Hunyuan Hy-Memory 2.0 (`HyMemoryClient`).
    """

    def __init__(
        self,
        config: AppConfig,
        hot_context: HotWorkingContext,
        store: DurableOutboxAndEpisodeStore,
        hy_client: MemoryBackendProtocol | None = None,
        llm_client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = config
        self.hot_context = hot_context
        self.store = store
        self.user_id = config.memory_user_id

        mem_cfg = config.settings.get("memory", {})
        self.flush_interval_sec = float(mem_cfg.get("outbox_flush_interval_sec", 3.0))
        self._cache = TTLLRUCache(
            max_size=128,
            ttl_sec=float(mem_cfg.get("prefetch_cache_ttl_sec", 180.0)),
        )

        self._hy_client: MemoryBackendProtocol | None = hy_client
        self._llm = llm_client or AsyncOpenAI(
            api_key=config.omni_api_key or "EMPTY",
            base_url=config.omni_api_base,
        )

        self._wake_outbox = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        self._prefetch_lock = asyncio.Semaphore(2)
        self._running = False

    def _ensure_hy_client(self) -> MemoryBackendProtocol:
        if self._hy_client is not None:
            return self._hy_client

        from hy_memory import HyMemoryClient

        chroma_dir = str((self.config.data_dir / "hy_memory_chroma").resolve())
        cfg_dict = {
            "mode": self.config.memory_mode,
            "llm": {
                "model": self.config.memory_llm_model,
                "api_key": self.config.memory_llm_api_key,
                "base_url": self.config.memory_llm_base_url,
            },
            "embedder": {
                "model": self.config.memory_embedding_model,
                "api_key": self.config.omni_api_key,
                "base_url": self.config.omni_api_base,
                "embedding_dims": self.config.memory_embedding_dims,
            },
            "vector_store": {
                "provider": "chroma",
                "path": chroma_dir,
                "embedding_dims": self.config.memory_embedding_dims,
            },
        }
        logger.info("Initializing Tencent Hy-Memory 2.0 engine (mode=%s, chroma=%s)", self.config.memory_mode, chroma_dir)
        self._hy_client = HyMemoryClient.from_config(cfg_dict)
        return self._hy_client

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Warm up HyMemoryClient in background thread so startup is non-blocking
        if self._hy_client is None and self.config.omni_api_key:
            await asyncio.to_thread(self._ensure_hy_client)
            asyncio.create_task(self._refresh_profile_quietly())
        self._worker_task = asyncio.create_task(self._outbox_drain_loop(), name="hy_memory_outbox_worker")

    async def stop(self) -> None:
        self._running = False
        self._wake_outbox.set()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._hy_client is not None:
            await asyncio.sleep(0.25)
            try:
                self._hy_client.close()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Zero-Blocking Write APIs (< 1ms on caller thread)
    # -------------------------------------------------------------------------
    async def record_dialogue_turn(self, role: str, content: str) -> None:
        """
        Record a single spoken or typed turn in <1ms:
          - Updates L0 HotWorkingContext immediately.
          - Triggers sidecar rolling summary compaction if turn window overflows.
          - On assistant turn completion, enqueues latest user+assistant exchange to Outbox for Hy-Memory 2.0.
        """
        overflow = await self.hot_context.record_turn(role, content)
        if overflow:
            asyncio.create_task(self._compact_turns_sidecar(overflow))

        if role == "assistant":
            turns = self.hot_context.recent_turns[-2:]
            payload = [{"role": t.role, "content": t.content} for t in turns]
            if payload:
                self.store.enqueue_outbox("dialogue", payload)
                self._wake_outbox.set()
        elif role == "user":
            # Immediately kick off speculative associative memory prefetch!
            self.trigger_speculative_prefetch(content)

    async def record_visual_observation(
        self,
        summary: str,
        importance: int = 1,
        jpeg_b64: str | None = None,
    ) -> int:
        """
        Record a visual/spatial scene event in <1ms:
          - Immediately updates L0 HotWorkingContext spatial slot.
          - Persists to SQLite FTS5 visual_episodes (with optional JPEG snapshot).
          - Enqueues to Hy-Memory 2.0 outbox if importance >= 1.
        """
        await self.hot_context.add_spatial_observation(summary, notify_realtime=True)
        ep_id = self.store.record_visual_episode(summary=summary, importance=importance, jpeg_b64=jpeg_b64)
        if importance >= 1:
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
            self.store.enqueue_outbox(
                "observation",
                f"[视觉空间记忆 {ts_str}] {summary}",
            )
            self._wake_outbox.set()
        self._cache.clear()
        return ep_id

    async def pin_explicit_fact(self, fact: str) -> None:
        """Explicitly pin a high-priority user fact or preference."""
        cleaned = fact.strip()
        if not cleaned:
            return
        await self.hot_context.add_spatial_observation(f"重要记录: {cleaned}", notify_realtime=True)
        self.store.enqueue_outbox("fact", cleaned)
        self._cache.clear()
        self._wake_outbox.set()

    # -------------------------------------------------------------------------
    # Zero-Blocking Speculative Prefetch & Explicit Deep Recall
    # -------------------------------------------------------------------------
    def trigger_speculative_prefetch(self, query_hint: str) -> asyncio.Task[bool] | None:
        """Fire-and-forget associative memory prefetch (0ms blocking on voice path)."""
        cleaned = query_hint.strip()
        if len(cleaned) < 2:
            return None
        return asyncio.create_task(self._run_prefetch(cleaned))

    async def _run_prefetch(self, query: str) -> bool:
        cached = self._cache.get(query)
        if cached is not None:
            return await self.hot_context.set_prefetched_memories(cached)

        async with self._prefetch_lock:
            merged = await self.deep_recall(query, limit=self.hot_context.max_prefetched)
            self._cache.put(query, merged)
            changed = await self.hot_context.set_prefetched_memories(merged)
            if changed:
                logger.info("Hot-swapped %d associative memories into L0 context for query=%r", len(merged), query[:30])
            return changed

    async def deep_recall(self, query: str, limit: int = 6) -> list[str]:
        """
        Hybrid multi-channel recall combining:
          1. Local SQLite FTS5 / CJK spatio-temporal visual episodes (< 2ms, catches events from 1s ago!)
          2. Tencent Hy-Memory 2.0 multi-channel semantic + BM25 search (`proactive` + `schema` + `normal`)
        """
        results: list[str] = []
        seen: set[str] = set()

        # 1. Fast local spatio-temporal episodes (catches newly placed objects before Hy-Memory finishes extracting)
        episodes = self.store.search_visual_episodes(query, limit=3)
        for ep in episodes:
            line = f"[空间观察 {ep['iso_time']}] {ep['summary']}"
            if line not in seen:
                seen.add(line)
                results.append(line)

        # 2. Tencent Hy-Memory 2.0 multi-channel cognitive search
        if self._hy_client is not None or self.config.omni_api_key:
            try:
                client = self._ensure_hy_client()
                resp = await client.async_search(
                    query=query,
                    user_ids=[self.user_id],
                    limit=limit,
                    min_score=0.35,
                )
                channels = resp.get("memories", {}) if isinstance(resp, dict) else {}
                for ch_name in ("proactive", "schema", "normal"):
                    items = channels.get(ch_name, [])
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        content = str(item.get("content", "")).strip()
                        # Clean compact single-line representation
                        compact = " | ".join(part.strip() for part in content.splitlines() if part.strip())
                        if compact and compact not in seen:
                            seen.add(compact)
                            results.append(compact)
                            if len(results) >= limit:
                                break
                    if len(results) >= limit:
                        break
            except Exception as exc:
                logger.warning("Hy-Memory async_search degraded gracefully: %s", exc)

        return results[:limit]

    # -------------------------------------------------------------------------
    # Sidecar Compactor & Outbox Background Worker
    # -------------------------------------------------------------------------
    async def _compact_turns_sidecar(self, older_turns: list[DialogueTurn]) -> None:
        """Summarize older dialogue turns off the critical path so realtime context stays crisp."""
        turns_str = "\n".join(f"[{t.timestamp}] {t.role}: {t.content}" for t in older_turns)
        prev_summary = self.hot_context.rolling_dialogue_summary
        prompt = (
            "请将以下较早的对话片段与已有背景摘要合并压缩为一段不超过200字的高密度中文摘要，"
            "保留关键事实、承诺、上下文指代与当前话题状态，不要输出多余前缀。\n\n"
            f"【已有摘要】\n{prev_summary or '无'}\n\n"
            f"【新增对话片段】\n{turns_str}"
        )
        try:
            resp = await self._llm.chat.completions.create(
                model=self.config.omni_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=280,
                extra_body={"enable_thinking": False},
            )
            summary = (resp.choices[0].message.content or "").strip()
            if summary:
                await self.hot_context.apply_compacted_summary(summary)
        except Exception as exc:
            logger.warning("Sidecar rolling compaction fallback: %s", exc)
            fallback = (prev_summary + " " + " / ".join(f"{t.role}:{t.content[:40]}" for t in older_turns)).strip()
            await self.hot_context.apply_compacted_summary(fallback[-450:])

    async def _outbox_drain_loop(self) -> None:
        while self._running:
            try:
                await asyncio.wait_for(self._wake_outbox.wait(), timeout=self.flush_interval_sec)
            except asyncio.TimeoutError:
                pass
            self._wake_outbox.clear()

            if not self._running:
                break

            pending = self.store.fetch_pending_outbox(limit=4)
            if not pending:
                continue

            for item in pending:
                if not self._running:
                    break
                item_id = item["id"]
                payload = item["payload"]
                try:
                    client = self._ensure_hy_client()
                    res = await client.async_add(
                        payload,
                        user_id=self.user_id,
                        source=item["kind"],
                    )
                    if isinstance(res, dict) and res.get("success") is False:
                        raise RuntimeError(res.get("error_message") or "HyMemory async_add returned success=False")
                    self.store.mark_outbox_done(item_id)
                    self._cache.clear()
                    logger.info("Flushed outbox item #%d (%s) to Hy-Memory 2.0", item_id, item["kind"])
                except Exception as exc:
                    logger.warning("Outbox item #%d write error: %s", item_id, exc)
                    self.store.mark_outbox_failed(item_id, str(exc))

            asyncio.create_task(self._refresh_profile_quietly())

    async def _refresh_profile_quietly(self) -> None:
        if self._hy_client is None:
            return
        try:
            prof = await self._hy_client.async_get_user_profile(user_id=self.user_id)
            if isinstance(prof, dict):
                # Extract non-empty profile summary fields if available
                parts = []
                for k in ("summary", "profile", "preferences", "basic_info"):
                    val = prof.get(k)
                    if val:
                        parts.append(str(val).strip())
                if parts:
                    await self.hot_context.update_user_profile("；".join(parts))
        except Exception:
            pass
