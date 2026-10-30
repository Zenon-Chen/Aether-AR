from __future__ import annotations

import base64
import json
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class FrameRecord:
    timestamp: float
    iso_time: str
    jpeg_b64: str
    motion_score: float = 0.0
    is_hd: bool = False


class DurableOutboxAndEpisodeStore:
    """
    Crash-safe local SQLite WAL store + in-memory visual keyframe ring buffer.

    Responsibilities:
      1. `memory_outbox`: O(1) non-blocking write queue drained asynchronously into Hy-Memory 2.0.
      2. `visual_episodes` + FTS5: Spatio-temporal visual observation log linked to saved JPEG snapshots.
      3. `autonomous_intents`: Persistent storage for Agent-created `cron` & `visual_watch` triggers.
      4. `FrameRecord` ring buffer: Instant access to recent frames for Temporal Contact Sheets & HD inspection.
    """

    def __init__(self, data_dir: Path, max_ring_frames: int = 30) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.data_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "ambient_state.db"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        self._recent_frames: deque[FrameRecord] = deque(maxlen=max_ring_frames)
        self._latest_hd_frame: FrameRecord | None = None

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS memory_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    error_msg TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_outbox_status ON memory_outbox(status, id);

                CREATE TABLE IF NOT EXISTS visual_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    iso_time TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 1,
                    frame_rel_path TEXT
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS visual_episodes_fts USING fts5(
                    summary,
                    iso_time,
                    content='visual_episodes',
                    content_rowid='id',
                    tokenize='unicode61'
                );

                CREATE TRIGGER IF NOT EXISTS trg_visual_episodes_ai AFTER INSERT ON visual_episodes BEGIN
                    INSERT INTO visual_episodes_fts(rowid, summary, iso_time)
                    VALUES (new.id, new.summary, new.iso_time);
                END;

                CREATE TABLE IF NOT EXISTS autonomous_intents (
                    id TEXT PRIMARY KEY,
                    trigger_type TEXT NOT NULL,
                    spec TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    one_shot INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    last_triggered_at REAL
                );
                """
            )
            self._conn.commit()

    # -------------------------------------------------------------------------
    # Visual Ring Buffer
    # -------------------------------------------------------------------------
    def push_frame(self, jpeg_b64: str, motion_score: float = 0.0, is_hd: bool = False) -> FrameRecord:
        now = time.time()
        rec = FrameRecord(
            timestamp=now,
            iso_time=datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            jpeg_b64=jpeg_b64,
            motion_score=motion_score,
            is_hd=is_hd,
        )
        if is_hd:
            self._latest_hd_frame = rec
        else:
            self._recent_frames.append(rec)
        return rec

    def get_recent_frames(self, count: int = 3) -> list[FrameRecord]:
        frames = list(self._recent_frames)
        if not frames:
            return []
        if len(frames) <= count:
            return frames
        # Evenly sample across recent buffer up to latest frame for clear temporal progression
        if count == 1:
            return [frames[-1]]
        window = frames[-min(len(frames), count * 2) :]
        indices = [round(i * (len(window) - 1) / (count - 1)) for i in range(count)]
        seen: list[FrameRecord] = []
        used_idx: set[int] = set()
        for idx in indices:
            if idx not in used_idx:
                used_idx.add(idx)
                seen.append(window[idx])
        return seen

    def get_latest_frame(self, prefer_hd: bool = False) -> FrameRecord | None:
        if prefer_hd and self._latest_hd_frame and (time.time() - self._latest_hd_frame.timestamp) < 15.0:
            return self._latest_hd_frame
        return self._recent_frames[-1] if self._recent_frames else self._latest_hd_frame

    # -------------------------------------------------------------------------
    # Durable Memory Outbox (Crash-safe queue -> Hy-Memory 2.0)
    # -------------------------------------------------------------------------
    def enqueue_outbox(self, kind: str, payload: Any) -> int:
        payload_json = json.dumps(payload, ensure_ascii=False)
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memory_outbox (kind, payload_json, status, created_at) VALUES (?, ?, 'pending', ?)",
                (kind, payload_json, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def fetch_pending_outbox(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, kind, payload_json, attempts
                FROM memory_outbox
                WHERE status IN ('pending', 'failed') AND attempts < 5
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            items: list[dict[str, Any]] = []
            for r in rows:
                items.append({
                    "id": r["id"],
                    "kind": r["kind"],
                    "payload": json.loads(r["payload_json"]),
                    "attempts": r["attempts"],
                })
            return items

    def mark_outbox_done(self, item_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE memory_outbox SET status = 'done' WHERE id = ?", (item_id,))
            self._conn.commit()

    def mark_outbox_failed(self, item_id: int, error_msg: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memory_outbox SET status = 'failed', attempts = attempts + 1, error_msg = ? WHERE id = ?",
                (error_msg[:400], item_id),
            )
            self._conn.commit()

    def pending_outbox_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM memory_outbox WHERE status IN ('pending', 'failed') AND attempts < 5"
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def search_dialogue_and_facts(self, query: str, limit: int = 4) -> list[str]:
        """
        Fast local lexical / CJK bigram search over durable outbox history (`fact`, `dialogue`, `observation`).
        Ensures 0ms immediate recall of turns in ultra-long contexts even before background vector indexing finishes,
        and provides exact-match recall for specific numbers, codes, or identifiers.
        """
        cleaned = query.strip()
        if not cleaned:
            return []
        tokens = [t for t in cleaned.replace('"', " ").split() if len(t) >= 2]
        cjk_chars = [c for c in cleaned if "\u4e00" <= c <= "\u9fff"]
        stop_bigrams = {"什么", "在哪", "哪里", "多少", "怎么", "请问", "记得", "告诉", "我的", "主人", "当时", "之前"}
        sub_terms: list[str] = []
        for t in tokens:
            if t not in stop_bigrams and t not in sub_terms:
                sub_terms.append(t)
        if len(cjk_chars) >= 2:
            for i in range(len(cjk_chars) - 1):
                bg = cjk_chars[i] + cjk_chars[i + 1]
                if bg not in stop_bigrams and bg not in sub_terms:
                    sub_terms.append(bg)

        if not sub_terms:
            return []

        clauses = ["payload_json LIKE ?" for _ in sub_terms[:8]]
        score_expr = " + ".join(
            f"(CASE WHEN payload_json LIKE ? THEN 1 ELSE 0 END)" for _ in sub_terms[:8]
        )
        params: list[Any] = [f"%{t}%" for t in sub_terms[:8]] + [f"%{t}%" for t in sub_terms[:8]] + [limit * 2]

        sql = (
            f"SELECT kind, payload_json, ({score_expr}) AS match_score "
            f"FROM memory_outbox WHERE {' OR '.join(clauses)} "
            f"ORDER BY match_score DESC, id DESC LIMIT ?"
        )
        hits: list[str] = []
        seen: set[str] = set()
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            for r in rows:
                try:
                    data = json.loads(r["payload_json"])
                    if isinstance(data, list):
                        text = " | ".join(
                            f"{m.get('role', '')}: {m.get('content', '')}"
                            for m in data
                            if isinstance(m, dict)
                        )
                    else:
                        text = str(data)
                except Exception:
                    text = str(r["payload_json"])
                compact = " ".join(text.split())[:220]
                if compact and compact not in seen:
                    seen.add(compact)
                    hits.append(compact)
                    if len(hits) >= limit:
                        break
        return hits

    # -------------------------------------------------------------------------
    # Visual & Spatio-Temporal Episodes
    # -------------------------------------------------------------------------
    def record_visual_episode(
        self,
        summary: str,
        importance: int = 1,
        jpeg_b64: str | None = None,
    ) -> int:
        now = time.time()
        dt = datetime.fromtimestamp(now)
        iso_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        rel_path: str | None = None

        if jpeg_b64:
            fname = f"ep_{dt.strftime('%Y%m%d_%H%M%S')}_{int((now % 1) * 1000):03d}.jpg"
            fpath = self.frames_dir / fname
            try:
                raw_bytes = base64.b64decode(jpeg_b64)
                fpath.write_bytes(raw_bytes)
                rel_path = f"frames/{fname}"
            except Exception:
                rel_path = None

        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO visual_episodes (created_at, iso_time, summary, importance, frame_rel_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now, iso_time, summary.strip(), importance, rel_path),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def search_visual_episodes(self, query: str = "", limit: int = 6) -> list[dict[str, Any]]:
        cleaned = query.strip()
        with self._lock:
            if not cleaned:
                rows = self._conn.execute(
                    "SELECT id, iso_time, summary, importance, frame_rel_path FROM visual_episodes ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]

            # Build safe FTS5 OR token query with LIKE fallback for Chinese substrings
            tokens = [t for t in cleaned.replace('"', " ").split() if t]
            results: list[dict[str, Any]] = []
            seen_ids: set[int] = set()

            if tokens:
                fts_q = " OR ".join(f'"{t}"' for t in tokens[:6])
                try:
                    fts_rows = self._conn.execute(
                        """
                        SELECT v.id, v.iso_time, v.summary, v.importance, v.frame_rel_path
                        FROM visual_episodes_fts f
                        JOIN visual_episodes v ON v.id = f.rowid
                        WHERE visual_episodes_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_q, limit),
                    ).fetchall()
                    for r in fts_rows:
                        d = dict(r)
                        seen_ids.add(d["id"])
                        results.append(d)
                except sqlite3.OperationalError:
                    pass

            if len(results) < limit:
                # Substring fallback for unsegmented CJK keywords (e.g. "钥匙", "眼镜", "水杯")
                like_clauses = []
                params: list[Any] = []
                # Split 2-char bigrams for CJK recall if query is a phrase
                cjk_chars = [c for c in cleaned if "\u4e00" <= c <= "\u9fff"]
                sub_terms = tokens[:]
                if len(cjk_chars) >= 2:
                    for i in range(len(cjk_chars) - 1):
                        bg = cjk_chars[i] + cjk_chars[i + 1]
                        if bg not in sub_terms:
                            sub_terms.append(bg)
                elif len(cjk_chars) == 1:
                    sub_terms.append(cjk_chars[0])

                for term in sub_terms[:6]:
                    like_clauses.append("summary LIKE ?")
                    params.append(f"%{term}%")

                if like_clauses:
                    sql = (
                        f"SELECT id, iso_time, summary, importance, frame_rel_path "
                        f"FROM visual_episodes WHERE {' OR '.join(like_clauses)} "
                        f"ORDER BY id DESC LIMIT ?"
                    )
                    params.append(limit)
                    for r in self._conn.execute(sql, params).fetchall():
                        d = dict(r)
                        if d["id"] not in seen_ids:
                            seen_ids.add(d["id"])
                            results.append(d)
                        if len(results) >= limit:
                            break

            return results[:limit]

    def load_episode_frame_b64(self, frame_rel_path: str | None) -> str | None:
        if not frame_rel_path:
            return None
        full_path = self.data_dir / frame_rel_path
        if full_path.exists():
            return base64.b64encode(full_path.read_bytes()).decode("ascii")
        return None

    # -------------------------------------------------------------------------
    # Autonomous Intents (Cron & Dynamic Visual Watch Persistence)
    # -------------------------------------------------------------------------
    def upsert_intent(
        self,
        intent_id: str,
        trigger_type: str,
        spec: str,
        goal: str,
        one_shot: bool = False,
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO autonomous_intents (id, trigger_type, spec, goal, one_shot, enabled, created_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(id) DO UPDATE SET
                    trigger_type=excluded.trigger_type,
                    spec=excluded.spec,
                    goal=excluded.goal,
                    one_shot=excluded.one_shot,
                    enabled=1
                """,
                (intent_id, trigger_type, spec, goal, 1 if one_shot else 0, now),
            )
            self._conn.commit()

    def remove_intent(self, intent_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM autonomous_intents WHERE id = ?", (intent_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def list_intents(self, trigger_type: str | None = None, only_enabled: bool = True) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if only_enabled:
            clauses.append("enabled = 1")
        if trigger_type:
            clauses.append("trigger_type = ?")
            params.append(trigger_type)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM autonomous_intents {where_sql} ORDER BY created_at ASC",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_intent_triggered(self, intent_id: str, disable: bool = False) -> None:
        now = time.time()
        with self._lock:
            if disable:
                self._conn.execute(
                    "UPDATE autonomous_intents SET last_triggered_at = ?, enabled = 0 WHERE id = ?",
                    (now, intent_id),
                )
            else:
                self._conn.execute(
                    "UPDATE autonomous_intents SET last_triggered_at = ? WHERE id = ?",
                    (now, intent_id),
                )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
