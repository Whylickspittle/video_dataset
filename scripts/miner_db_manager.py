#!/usr/bin/env python3
"""
矿工数据库管理类。

提供对 miner.db 的 CRUD 操作，包括：
- 视频源管理（含三方链接）
- 切片管理
- 分类统计
- 去重检查
- 导出最终数据集
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class SourceInfo:
    """视频源信息结构。"""
    video_id: str
    url: str
    title: str = ""
    channel: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    tbr: int = 0
    duration: float = 0.0
    vcodec: str = ""
    category: str = "nature"
    subcategory: str = ""
    third_party_url: str = ""
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ClipInfo:
    """切片信息结构。"""
    clip_id: str
    source_id: int
    start_sec: float
    duration_sec: float = 5.04
    caption: str = ""
    quality_score: float = 0.5
    third_party_url: str = ""
    tags: list[str] = None
    notes: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MinerDatabase:
    """矿工切片数据库管理器。"""

    def __init__(self, db_path: str = "miner.db"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ── Sources ──

    def add_source(self, info: SourceInfo) -> int:
        """添加视频源，返回 source_id。"""
        cursor = self.conn.execute(
            """INSERT OR IGNORE INTO sources
               (source_video_id, source_video_url, title, channel,
                src_width, src_height, src_fps, src_tbr, src_duration, src_vcodec,
                category, subcategory, third_party_url, tags,
                total_segments)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (info.video_id, info.url, info.title, info.channel,
             info.width, info.height, info.fps, info.tbr, info.duration, info.vcodec,
             info.category, info.subcategory, info.third_party_url, json.dumps(info.tags),
             int(info.duration // 5.04) if info.duration > 0 else 0)
        )
        self.conn.commit()
        if cursor.lastrowid:
            return cursor.lastrowid
        # 如果已存在，返回已有记录的 id
        row = self.conn.execute(
            "SELECT id FROM sources WHERE source_video_id = ?",
            (info.video_id,)
        ).fetchone()
        return row[0] if row else 0

    def get_source(self, source_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()

    def get_source_by_video_id(self, video_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM sources WHERE source_video_id = ?", (video_id,)
        ).fetchone()

    def update_source_downloaded(self, source_id: int, local_path: str,
                                  status: str = "downloaded"):
        """标记视频源已下载。"""
        self.conn.execute(
            """UPDATE sources
               SET local_path = ?, download_status = ?, downloaded_at = datetime('now')
               WHERE id = ?""",
            (local_path, status, source_id)
        )
        self.conn.commit()

    def list_sources(self, status: str = None, category: str = None,
                     limit: int = 100) -> list[sqlite3.Row]:
        """列出视频源。"""
        query = "SELECT * FROM sources WHERE 1=1"
        params = []
        if status:
            query += " AND download_status = ?"
            params.append(status)
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY discovered_at DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(query, params).fetchall()

    def search_sources(self, keyword: str) -> list[sqlite3.Row]:
        """搜索视频源。"""
        pattern = f"%{keyword}%"
        return self.conn.execute(
            """SELECT * FROM sources
               WHERE title LIKE ? OR channel LIKE ? OR tags LIKE ?
               ORDER BY visual_quality DESC""",
            (pattern, pattern, pattern)
        ).fetchall()

    # ── Clips ──

    def add_clip(self, info: ClipInfo,
                 clip_sha256: str = "", frame_sha256: str = "",
                 clip_path: str = "", frame_path: str = "",
                 actual_width: int = 1280, actual_height: int = 704,
                 actual_fps: float = 24.0, actual_frames: int = 121) -> int:
        """添加切片记录。"""
        cursor = self.conn.execute(
            """INSERT INTO clips
               (clip_id, source_id, start_sec, duration_sec,
                clip_local_path, frame_local_path,
                clip_sha256, frame_sha256,
                caption, self_quality_score, third_party_url, tags, notes,
                actual_width, actual_height, actual_fps, actual_num_frames)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (info.clip_id, info.source_id, info.start_sec, info.duration_sec,
             clip_path, frame_path,
             clip_sha256, frame_sha256,
             info.caption, info.quality_score, info.third_party_url,
             json.dumps(info.tags), info.notes,
             actual_width, actual_height, actual_fps, actual_frames)
        )
        # 更新 source 的使用计数
        self.conn.execute(
            "UPDATE sources SET used_segments = used_segments + 1 WHERE id = ?",
            (info.source_id,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_clip(self, clip_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM clips WHERE clip_id = ?", (clip_id,)
        ).fetchone()

    def update_clip_status(self, clip_id: str, status: str,
                           interval_id: int = None):
        """更新切片提交状态。"""
        if interval_id:
            self.conn.execute(
                """UPDATE clips SET submission_status = ?, interval_id = ?
                   WHERE clip_id = ?""",
                (status, interval_id, clip_id)
            )
        else:
            self.conn.execute(
                "UPDATE clips SET submission_status = ? WHERE clip_id = ?",
                (status, clip_id)
            )
        self.conn.commit()

    def list_clips(self, source_id: int = None, status: str = None,
                   min_score: float = 0.0, limit: int = 100) -> list[sqlite3.Row]:
        """列出切片。"""
        query = "SELECT * FROM clips WHERE 1=1"
        params = []
        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)
        if status:
            query += " AND submission_status = ?"
            params.append(status)
        if min_score > 0:
            query += " AND self_quality_score >= ?"
            params.append(min_score)
        query += " ORDER BY self_quality_score DESC, created_at DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(query, params).fetchall()

    def select_clips_for_interval(self, count: int = 400,
                                   min_score: float = 0.3,
                                   category_balance: bool = True) -> list[sqlite3.Row]:
        """为 interval 选择切片。

        如果 category_balance=True，会尽量按分类平衡选择。
        """
        if not category_balance:
            return self.conn.execute(
                """SELECT * FROM clips
                   WHERE submission_status = 'draft'
                   AND (self_quality_score IS NULL OR self_quality_score >= ?)
                   ORDER BY self_quality_score DESC, created_at ASC
                   LIMIT ?""",
                (min_score, count)
            ).fetchall()

        # 平衡模式：每个分类按比例选取
        categories = self.conn.execute(
            "SELECT name, target_ratio FROM categories ORDER BY priority"
        ).fetchall()

        selected = []
        for cat in categories:
            cat_count = max(1, int(count * cat["target_ratio"]))
            rows = self.conn.execute(
                """SELECT c.* FROM clips c
                   JOIN sources s ON c.source_id = s.id
                   WHERE c.submission_status = 'draft'
                   AND s.category = ?
                   AND (c.self_quality_score IS NULL OR c.self_quality_score >= ?)
                   ORDER BY c.self_quality_score DESC
                   LIMIT ?""",
                (cat["name"], min_score, cat_count)
            ).fetchall()
            selected.extend(rows)

        # 如果不够 400，从剩余 draft 中补
        if len(selected) < count:
            existing_ids = [r["clip_id"] for r in selected]
            placeholders = ",".join("?" * len(existing_ids)) if existing_ids else "''"
            remaining = self.conn.execute(
                f"""SELECT * FROM clips
                    WHERE submission_status = 'draft'
                    AND clip_id NOT IN ({placeholders})
                    ORDER BY self_quality_score DESC
                    LIMIT ?""",
                (*existing_ids, count - len(selected))
            ).fetchall()
            selected.extend(remaining)

        return selected[:count]

    # ── Overlap Check ──

    def check_overlap(self, video_id: str, start_sec: float,
                      window: float = 4.5) -> bool:
        """检查是否与已有切片重叠（< 4.5s）。"""
        # 检查 clips 表（当前周期草稿）
        result = self.conn.execute(
            """SELECT 1 FROM clips c
               JOIN sources s ON c.source_id = s.id
               WHERE s.source_video_id = ?
               AND ABS(c.start_sec - ?) < ?
               LIMIT 1""",
            (video_id, start_sec, window)
        ).fetchone()
        if result:
            return True

        # 检查 global_index（历史提交）
        result = self.conn.execute(
            """SELECT 1 FROM global_index
               WHERE source_video_id = ?
               AND ABS(start_sec - ?) < ?
               LIMIT 1""",
            (video_id, start_sec, window)
        ).fetchone()
        return result is not None

    def get_available_segments(self, source_id: int) -> list[float]:
        """获取某视频源尚未使用的起始时间点。"""
        source = self.get_source(source_id)
        if not source:
            return []

        duration = source["src_duration"]
        total = int(duration // 5.04) if duration > 0 else 0

        used = self.conn.execute(
            "SELECT start_sec FROM clips WHERE source_id = ?",
            (source_id,)
        ).fetchall()
        used_set = {row[0] for row in used}

        available = []
        for i in range(total):
            start = round(i * 5.04, 3)
            if start not in used_set:
                available.append(start)
        return available

    # ── Global Index ──

    def add_to_global_index(self, video_id: str, url: str,
                            start_sec: float, interval_id: int,
                            clip_id: str = ""):
        """添加到全局去重索引。"""
        self.conn.execute(
            """INSERT OR IGNORE INTO global_index
               (source_video_id, source_video_url, start_sec, interval_id, clip_id)
               VALUES (?, ?, ?, ?, ?)""",
            (video_id, url, start_sec, interval_id, clip_id)
        )
        self.conn.commit()

    def sync_clips_to_global_index(self, interval_id: int):
        """将某 interval 的所有已提交切片同步到全局索引。"""
        clips = self.conn.execute(
            """SELECT c.clip_id, c.start_sec, s.source_video_id, s.source_video_url
               FROM clips c
               JOIN sources s ON c.source_id = s.id
               WHERE c.interval_id = ?""",
            (interval_id,)
        ).fetchall()
        for clip in clips:
            self.add_to_global_index(
                clip["source_video_id"], clip["source_video_url"],
                clip["start_sec"], interval_id, clip["clip_id"]
            )

    # ── Statistics ──

    def get_stats(self) -> dict[str, Any]:
        """获取数据库统计信息。"""
        stats = {}

        # 总体统计
        row = self.conn.execute(
            """SELECT
                COUNT(*) as total_sources,
                SUM(CASE WHEN download_status = 'downloaded' THEN 1 ELSE 0 END) as downloaded,
                SUM(total_segments) as total_segments,
                SUM(used_segments) as used_segments
               FROM sources"""
        ).fetchone()
        stats["sources"] = dict(row)

        # 切片统计
        row = self.conn.execute(
            """SELECT
                COUNT(*) as total_clips,
                SUM(CASE WHEN submission_status = 'draft' THEN 1 ELSE 0 END) as draft,
                SUM(CASE WHEN submission_status = 'submitted' THEN 1 ELSE 0 END) as submitted,
                AVG(self_quality_score) as avg_quality
               FROM clips"""
        ).fetchone()
        stats["clips"] = dict(row)

        # 分类分布
        rows = self.conn.execute(
            """SELECT s.category, COUNT(*) as count
               FROM clips c
               JOIN sources s ON c.source_id = s.id
               WHERE c.submission_status = 'draft'
               GROUP BY s.category
               ORDER BY count DESC"""
        ).fetchall()
        stats["category_distribution"] = {r["category"]: r["count"] for r in rows}

        # Interval 统计
        row = self.conn.execute(
            "SELECT COUNT(*), AVG(validator_score) FROM intervals"
        ).fetchone()
        stats["intervals"] = {"count": row[0], "avg_score": row[1]}

        return stats

    def get_category_balance_report(self) -> list[sqlite3.Row]:
        """获取分类平衡报告，显示当前 draft 切片 vs 目标比例。"""
        return self.conn.execute(
            """SELECT
                c.name,
                c.target_ratio,
                COALESCE(d.actual_count, 0) as actual_count,
                COALESCE(d.actual_ratio, 0) as actual_ratio,
                c.target_ratio - COALESCE(d.actual_ratio, 0) as gap
            FROM categories c
            LEFT JOIN (
                SELECT s.category, COUNT(*) as actual_count,
                       CAST(COUNT(*) AS REAL) / NULLIF(
                           (SELECT COUNT(*) FROM clips WHERE submission_status = 'draft'), 0
                       ) as actual_ratio
                FROM clips cl
                JOIN sources s ON cl.source_id = s.id
                WHERE cl.submission_status = 'draft'
                GROUP BY s.category
            ) d ON c.name = d.category
            ORDER BY gap DESC"""
        ).fetchall()

    # ── Export ──

    def export_to_parquet_records(self, clip_ids: list[str]) -> list[dict]:
        """导出为 dataset.parquet 所需的字典列表。"""
        placeholders = ",".join("?" * len(clip_ids))
        rows = self.conn.execute(
            f"""SELECT
                c.clip_id,
                c.clip_local_path,
                c.clip_sha256,
                c.frame_local_path,
                c.frame_sha256,
                s.source_video_id,
                c.start_sec as clip_start_sec,
                c.duration_sec,
                c.caption,
                s.source_video_url,
                c.third_party_url
            FROM clips c
            JOIN sources s ON c.source_id = s.id
            WHERE c.clip_id IN ({placeholders})""",
            clip_ids
        ).fetchall()

        records = []
        for row in rows:
            record = {
                "clip_id": row["clip_id"],
                "clip_uri": f"clips/{Path(row['clip_local_path']).name}" if row["clip_local_path"] else "",
                "clip_sha256": row["clip_sha256"] or "",
                "first_frame_uri": f"frames/{Path(row['frame_local_path']).name}" if row["frame_local_path"] else "",
                "first_frame_sha256": row["frame_sha256"] or "",
                "source_video_id": row["source_video_id"],
                "clip_start_sec": row["clip_start_sec"],
                "duration_sec": row["duration_sec"],
                "width": 1280,
                "height": 704,
                "fps": 24.0,
                "num_frames": 121,
                "source_video_url": row["source_video_url"],
                "caption": row["caption"] or "",
                # 三方链接（Validator 会忽略，但保留在 parquet 中）
                "third_party_url": row["third_party_url"] or "",
            }
            records.append(record)
        return records

    def log_operation(self, operation: str, entity_type: str = "",
                      entity_id: str = "", status: str = "",
                      details: dict = None):
        """记录操作日志。"""
        self.conn.execute(
            """INSERT INTO operation_logs (operation, entity_type, entity_id, status, details)
               VALUES (?, ?, ?, ?, ?)""",
            (operation, entity_type, entity_id, status, json.dumps(details or {}))
        )
        self.conn.commit()


# ── 使用示例 ──

if __name__ == "__main__":
    import sys

    db_file = sys.argv[1] if len(sys.argv) > 1 else "miner.db"

    with MinerDatabase(db_file) as db:
        # 1. 添加视频源（含三方链接）
        source = SourceInfo(
            video_id="LXb3EKWsInQ",
            url="https://www.youtube.com/watch?v=LXb3EKWsInQ",
            title="4K Forest Waterfall",
            channel="NatureRelaxation",
            width=3840, height=2160, fps=60, tbr=25000,
            duration=3600, vcodec="avc1",
            category="nature", subcategory="waterfall",
            third_party_url="https://backup.example.com/forest_waterfall.mp4",
            tags=["4k", "waterfall", "forest", "drone"]
        )
        source_id = db.add_source(source)
        print(f"Added source: {source_id}")

        # 2. 检查可用切片位置
        available = db.get_available_segments(source_id)
        print(f"Available segments: {len(available)}")

        # 3. 检查去重
        if not db.check_overlap("LXb3EKWsInQ", 0.0):
            print("0.0s position is available")

        # 4. 添加切片
        clip = ClipInfo(
            clip_id="clip_001_abc123",
            source_id=source_id,
            start_sec=0.0,
            caption="A waterfall cascading through a lush green forest",
            quality_score=0.85,
            third_party_url="https://backup.example.com/clip_001.mp4",
            tags=["waterfall", "forest"]
        )
        db.add_clip(clip, clip_sha256="a" * 64, frame_sha256="b" * 64,
                    clip_path=".nexis/clips/clip_001.mp4",
                    frame_path=".nexis/frames/clip_001.jpg")
        print(f"Added clip: {clip.clip_id}")

        # 5. 查看统计
        stats = db.get_stats()
        print(f"\nStats: {json.dumps(stats, indent=2, default=str)}")

        # 6. 分类平衡报告
        print("\nCategory Balance:")
        for row in db.get_category_balance_report():
            print(f"  {row['name']:15s}: target={row['target_ratio']:.2f}, "
                  f"actual={row['actual_ratio']:.2f}, gap={row['gap']:+.2f}")

        # 7. 导出为 parquet 记录
        records = db.export_to_parquet_records(["clip_001_abc123"])
        print(f"\nExported {len(records)} records")
        print(json.dumps(records[0], indent=2))
