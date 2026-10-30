from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from PIL import Image, ImageDraw

from memory.hot_context import HotWorkingContext
from memory.outbox import FrameRecord

logger = logging.getLogger("ambient.pulse")


@dataclass
class PulseDecision:
    scene_state: str
    action: str  # "noop" | "record_observation" | "proactive_engage" | "dispatch_kernel"
    observation: str = ""
    importance: int = 1
    proactive_prompt: str = ""
    kernel_goal: str = ""
    fulfilled_intent_id: str = ""


def compose_temporal_contact_sheet(
    frames: list[FrameRecord],
    tile_width: int = 384,
    tile_height: int = 288,
) -> str:
    """
    Stitch 1..N consecutive keyframes horizontally into a labeled Temporal Contact Sheet JPEG (base64).
    Enables `qwen3.8-omni-flash` to perceive continuous motion, posture transitions (e.g., sleeping -> waking up,
    entering through door, placing an item down) in a single compact multimodal pass.
    """
    if not frames:
        raise ValueError("Cannot compose contact sheet from empty frame list")

    banner_h = 26
    canvas_w = tile_width * len(frames)
    canvas_h = tile_height + banner_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 20, 26))
    draw = ImageDraw.Draw(canvas)

    for idx, rec in enumerate(frames):
        x0 = idx * tile_width
        raw = base64.b64decode(rec.jpeg_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = img.resize((tile_width, tile_height), Image.Resampling.BILINEAR)
        canvas.paste(img, (x0, banner_h))

        # Draw separator and temporal header label
        if idx > 0:
            draw.line([(x0, 0), (x0, canvas_h)], fill=(60, 70, 90), width=2)
        tag = "LATEST" if idx == len(frames) - 1 else f"T-{len(frames) - 1 - idx}"
        time_part = rec.iso_time.split(" ")[-1] if " " in rec.iso_time else rec.iso_time
        label = f"#{idx + 1} [{tag} {time_part} | d={rec.motion_score:.2f}]"
        draw.text((x0 + 8, 6), label, fill=(130, 230, 255) if tag == "LATEST" else (200, 205, 215))

    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=76)
    return base64.b64encode(out.getvalue()).decode("ascii")


class PerceptualPulseObserver:
    """
    Zero-Hardcode Native Multimodal Environment Observer (`qwen3.8-omni-flash`).

    Instead of brittle rule-based CV detectors (`if pose == waking_up`), this observer
    evaluates Temporal Contact Sheets against the live `HotWorkingContext` (time of day,
    previous room state, recent conversation, active intentions, cached briefings)
    and autonomously decides whether to stay silent, log a spatial memory, run a background
    kernel task, or initiate natural proactive speech.
    """

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        model: str,
        hot_context: HotWorkingContext,
        min_interval_sec: float = 4.0,
        proactive_cooldown_sec: float = 20.0,
    ) -> None:
        self.llm = llm_client
        self.model = model
        self.hot_context = hot_context
        self.min_interval_sec = min_interval_sec
        self.proactive_cooldown_sec = proactive_cooldown_sec

        self._last_eval_ts: float = 0.0
        self._last_proactive_ts: float = 0.0

    def can_evaluate_now(self) -> bool:
        return (time.monotonic() - self._last_eval_ts) >= self.min_interval_sec

    async def evaluate_frames(
        self,
        frames: list[FrameRecord],
        bypass_rate_limit: bool = False,
        override_now_str: str | None = None,
    ) -> PulseDecision | None:
        if not frames:
            return None
        if not bypass_rate_limit and not self.can_evaluate_now():
            return None

        self._last_eval_ts = time.monotonic()
        contact_sheet_b64 = compose_temporal_contact_sheet(frames)

        now_str = override_now_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")
        proactive_allowed = bypass_rate_limit or (
            (time.monotonic() - self._last_proactive_ts) >= self.proactive_cooldown_sec
        )

        ctx_snap = self.hot_context.snapshot_dict()
        recent_turns_text = "\n".join(
            f"[{t['timestamp']}] {t['role']}: {t['content']}" for t in ctx_snap["recent_turns"][-3:]
        ) or "暂无近期对话"
        intents_text = "\n".join(f"- {i}" for i in ctx_snap["active_intentions"]) or "无特定守望意图"
        briefings_text = (
            " | ".join(f"{k}: {v}" for k, v in ctx_snap["briefing_cache"].items()) or "无预备简报缓存"
        )
        obs_text = "；".join(ctx_snap["spatial_observations"][-4:]) or "暂无"

        system_prompt = (
            "你是房间全模态视觉感知与社交直觉中枢。你会收到按时间从左到右排列的连续摄像头关键帧拼图（最右一格 LATEST 为当前画面）。\n"
            "请结合【当前时间】、【上一刻房间状态】、【近期对话】和【待关注主动意图】，完全依靠人类管家级的常识判断该如何行动，不要机械死板：\n"
            "1. `proactive_engage`（主动开口说话）：仅在真正自然且有价值的时刻触发！例如：\n"
            "   - 原本房间无人，看到主人刚推门走进房间（欢迎回家/问好）；\n"
            "   - 早晨时段看到原本躺着熟睡的主人醒来坐起/伸懒腰（轻声问早，若缓存或意图中有早报/日程则顺带播报）；\n"
            "   - 满足了【待关注主动意图】中的视觉触发条件；\n"
            "   - 主人明显正对着摄像头挥手示意、展示手中物品求助、或发生明显异常危险。\n"
            "   注意：如果上一刻状态或近期对话显示刚刚才打过招呼，或者主人正戴着耳机/低头专注工作学习且无新事件，绝不要打扰（选 `noop`）！\n"
            "2. `record_observation`（静默记录空间与动作记忆，不开口）：当看到值得日后回忆的位置/状态变化（例如主人把钥匙/眼镜/药瓶/包裹放在某处，或开始某项日常活动），静默记下准确位置与特征，供后续空间问答检索。\n"
            "3. `dispatch_kernel`（后台静默触发内核任务）：需要后台先查信息或执行工具、暂不打断主人时使用。\n"
            "4. `noop`（静默保持）：绝大多数普通画面微动或无事发生时选择此项，仅更新 `scene_state`。\n\n"
            "必须严格只输出一个合法 JSON 对象，字段如下：\n"
            "{\n"
            '  "scene_state": "一句话描述当前画面中人物与房间状态",\n'
            '  "action": "noop | record_observation | proactive_engage | dispatch_kernel",\n'
            '  "observation": "可选：若有值得记住的物品摆放位置或关键动作变化则填写，否则空字符串",\n'
            '  "importance": 1,\n'
            '  "proactive_prompt": "当 action=proactive_engage 时必填：指导实时语音模型对主人说的话与语气要点",\n'
            '  "kernel_goal": "当 action=dispatch_kernel 时必填：交给后端内核执行的目标",\n'
            '  "fulfilled_intent_id": "若本次触发完成了某条一次性待关注意图，填其方括号内的ID，否则空字符串"\n'
            "}"
        )

        user_prompt = (
            f"【当前时间】{now_str}\n"
            f"【主人画像】{ctx_snap['user_profile']}\n"
            f"【上一刻房间状态】{ctx_snap['scene_and_spatial_state']}\n"
            f"【近期空间观察】{obs_text}\n"
            f"【近期对话记录】\n{recent_turns_text}\n"
            f"【待关注主动意图】\n{intents_text}\n"
            f"【后台预备简报缓存】{briefings_text}\n"
            f"【当前是否允许主动开口】{'是' if proactive_allowed else '否（冷却期内，若无紧急状况请勿选 proactive_engage）'}\n"
            "请观察上方时序画面并输出决策 JSON："
        )

        resp = await self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{contact_sheet_b64}"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            max_tokens=360,
            extra_body={"enable_thinking": False},
        )
        raw_text = (resp.choices[0].message.content or "").strip()
        decision = self._parse_decision(raw_text)

        if decision.action == "proactive_engage":
            if not proactive_allowed:
                decision.action = "record_observation" if decision.observation else "noop"
            else:
                self._last_proactive_ts = time.monotonic()

        return decision

    @staticmethod
    def _parse_decision(raw_text: str) -> PulseDecision:
        cleaned = raw_text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if fence:
            cleaned = fence.group(1)
        else:
            brace_start = cleaned.find("{")
            brace_end = cleaned.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                cleaned = cleaned[brace_start : brace_end + 1]

        try:
            data: dict[str, Any] = json.loads(cleaned)
        except Exception as exc:
            logger.warning("Failed to parse PerceptualPulse JSON (%s): %r", exc, raw_text[:200])
            return PulseDecision(scene_state="画面已更新", action="noop")

        action = str(data.get("action", "noop")).strip()
        if action not in ("noop", "record_observation", "proactive_engage", "dispatch_kernel"):
            action = "noop"

        return PulseDecision(
            scene_state=str(data.get("scene_state", "")).strip() or "正在观察环境",
            action=action,
            observation=str(data.get("observation", "")).strip(),
            importance=int(data.get("importance", 1) or 1),
            proactive_prompt=str(data.get("proactive_prompt", "")).strip(),
            kernel_goal=str(data.get("kernel_goal", "")).strip(),
            fulfilled_intent_id=str(data.get("fulfilled_intent_id", "")).strip(),
        )
