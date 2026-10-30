from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Awaitable, Callable

from croniter import croniter

from memory.hot_context import HotWorkingContext
from memory.outbox import DurableOutboxAndEpisodeStore
from .agent_loop import PiKernel

logger = logging.getLogger("ambient.scheduler")


class AutonomousIntentScheduler:
    """
    Executes Agent-managed `cron` tasks in the background (e.g., pre-fetching morning briefings
    into HotWorkingContext before the user wakes up) and keeps `visual_watch` intentions
    synchronized into L0 HotWorkingContext so PerceptualPulse naturally honors them.
    """

    def __init__(
        self,
        store: DurableOutboxAndEpisodeStore,
        hot_context: HotWorkingContext,
        kernel: PiKernel,
        on_cron_result: Callable[[str, str], Awaitable[None]] | None = None,
        poll_interval_sec: float = 15.0,
    ) -> None:
        self.store = store
        self.hot_context = hot_context
        self.kernel = kernel
        self.on_cron_result = on_cron_result
        self.poll_interval_sec = poll_interval_sec
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def sync_visual_watches_to_hot_context(self) -> None:
        watches = self.store.list_intents(trigger_type="visual_watch", only_enabled=True)
        formatted = [
            f"[{w['id']}] 条件: {w['spec']} -> 动作: {w['goal']}" + (" (一次性)" if w["one_shot"] else "")
            for w in watches
        ]
        await self.hot_context.set_active_intentions(formatted)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self.sync_visual_watches_to_hot_context()
        self._task = asyncio.create_task(self._loop(), name="autonomous_intent_scheduler")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._check_due_crons()
            except Exception as exc:
                logger.warning("Scheduler tick error: %s", exc)
            await asyncio.sleep(self.poll_interval_sec)

    async def _check_due_crons(self) -> None:
        now = time.time()
        crons = self.store.list_intents(trigger_type="cron", only_enabled=True)
        for item in crons:
            spec = item["spec"]
            last_ts = item["last_triggered_at"] or item["created_at"]
            try:
                base_dt = datetime.fromtimestamp(last_ts).astimezone()
                itr = croniter(spec, base_dt)
                next_ts = itr.get_next(datetime).timestamp()
            except Exception as exc:
                logger.warning("Invalid cron spec '%s' on intent %s: %s", spec, item["id"], exc)
                self.store.mark_intent_triggered(item["id"], disable=True)
                continue

            if now >= next_ts:
                self.store.mark_intent_triggered(item["id"], disable=bool(item["one_shot"]))
                asyncio.create_task(self._run_cron_intent(item["id"], item["goal"]))

    async def _run_cron_intent(self, intent_id: str, goal: str) -> None:
        logger.info("Executing scheduled autonomous cron intent [%s]: %s", intent_id, goal)
        augmented_goal = (
            f"【后台定时任务触发 ({intent_id})】{goal}\n"
            "如果该任务产出了供稍后播报给主人的信息（例如晨间简报、天气日程汇总），"
            "请调用 `memory_ops(action='stage_briefing', key='scheduled_briefing', content=...)` 存入零延迟缓存。"
        )
        result = await self.kernel.run(augmented_goal)
        if self.on_cron_result:
            await self.on_cron_result(intent_id, result)
