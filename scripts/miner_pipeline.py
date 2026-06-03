#!/usr/bin/env python3
"""
端到端矿工 Pipeline：流式切片 + Caption + 数据库 + 导出。

用法：
    # 方式1: 从 sources.txt 自动切
    python3 miner_pipeline.py --sources sources.txt \
        --interval-id 1 --hotkey YOUR_HOTKEY \
        --openai-key sk-xxx --output-dir ./interval_1

    # 方式2: 指定单个视频和切点
    python3 miner_pipeline.py --url "URL" --cuts "0,15,30,45,60" \
        --interval-id 1 --hotkey YOUR_HOTKEY \
        --openai-key sk-xxx

输出目录结构：
    ./interval_1/
    ├── dataset.parquet
    ├── manifest.json
    ├── clips/
    │   ├── clip_xxx.mp4
    │   └── ...
    ├── frames/
    │   ├── clip_xxx.jpg
    │   └── ...
    └── miner.db                  # 数据库（自动创建）
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("miner-pipeline")

# ── 协议常量 ──
CLIP_DURATION = 5.04
TARGET_WIDTH = 1280
TARGET_HEIGHT = 704
TARGET_FPS = 24
TARGET_FRAMES = 121
BUFFER_SECONDS = 8


# ── 数据结构 ──

@dataclass
class SourceSpec:
    """视频源规格。"""
    url: str
    video_id: str
    title: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    tbr: int = 0
    vcodec: str = ""
    # 用户指定的切点（空则自动均匀分布）
    cuts: list[float] = field(default_factory=list)


@dataclass
class SliceResult:
    """单次切片结果。"""
    clip_id: str
    source_id: str
    start_sec: float
    clip_path: Path
    frame_path: Path
    clip_sha256: str
    frame_sha256: str
    caption: str = ""


# ── 工具函数 ──

def sha256_file(path: Path) -> str:
    """计算文件 SHA256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_video_id(url: str) -> str:
    """从 URL 提取视频 ID。"""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        return parsed.path.strip("/")
    if host == "youtube.com" or host.endswith(".youtube.com"):
        query = parse_qs(parsed.query)
        vals = query.get("v", [])
        if vals and vals[0].strip():
            return vals[0].strip()
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "v"}:
            return parts[1]
    # 非 YouTube：用 URL 哈希
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def deterministic_clip_id(video_id: str, start_sec: float) -> str:
    """确定性 clip_id。"""
    data = f"{video_id}:{start_sec:.3f}:{CLIP_DURATION:.3f}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def run_cmd(cmd: list[str], timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess:
    """运行子进程命令。"""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


# ── Stage 1: 预检视频源 ──

def inspect_source(url: str) -> SourceSpec:
    """用 yt-dlp 获取视频元数据，不下载。"""
    logger.info("[INSPECT] %s", url[:60])
    try:
        result = run_cmd([
            "yt-dlp", "--no-download",
            "--print", "%(id)s|%(title)s|%(duration)s|%(width)s|%(height)s|%(fps)s|%(tbr)s|%(vcodec)s",
            url,
        ], timeout=60)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error("yt-dlp inspect failed: %s", exc)
        raise RuntimeError(f"Cannot inspect {url}") from exc

    parts = result.stdout.strip().split("|")
    if len(parts) < 8:
        raise RuntimeError(f"Unexpected yt-dlp output: {result.stdout}")

    return SourceSpec(
        url=url,
        video_id=parts[0],
        title=parts[1][:100],
        duration=float(parts[2] or 0),
        width=int(parts[3] or 0),
        height=int(parts[4] or 0),
        fps=float(parts[5] or 0),
        tbr=int(float(parts[6] or 0)),
        vcodec=parts[7],
    )


# ── Stage 2: 流式切片 ──

def get_stream_url(url: str, min_height: int = 1080) -> str:
    """获取视频直接流 URL。"""
    result = run_cmd([
        "yt-dlp",
        "-f", f"bestvideo[height>={min_height}]+bestaudio/best",
        "--no-check-certificates",
        "-g",
        url,
    ], timeout=30)
    # yt-dlp -g 可能返回两行（视频流 + 音频流），取第一行
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        raise RuntimeError("No stream URL returned")
    return lines[0]


def stream_slice(video_url: str, start_sec: float,
                 output_clip: Path, output_frame: Path) -> None:
    """
    从视频流切出精确片段。
    流程: 下载缓冲段 → 精确切 → 提取首帧。
    """
    stream_url = get_stream_url(video_url)
    logger.debug("Stream URL: %s...", stream_url[:80])

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Step 1: 下载缓冲段（多下 BUFFER_SECONDS 秒）
        buf_dur = CLIP_DURATION + BUFFER_SECONDS
        logger.debug("Downloading buffer: %.1fs ~ %.1fs", start_sec, start_sec + buf_dur)
        run_cmd([
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-ss", str(start_sec),
            "-t", str(buf_dur),
            "-i", stream_url,
            "-c:v", "copy",
            "-an",
            "-avoid_negative_ts", "make_zero",
            str(tmp_path),
        ], timeout=120)

        # Step 2: 从缓冲段精确切
        vf = (
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_WIDTH}:{TARGET_HEIGHT},fps={TARGET_FPS}"
        )
        run_cmd([
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-ss", "0",
            "-t", str(CLIP_DURATION),
            "-i", str(tmp_path),
            "-frames:v", str(TARGET_FRAMES),
            "-vf", vf,
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "20",
            str(output_clip),
        ], timeout=120)

        # Step 3: 提取首帧
        run_cmd([
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-i", str(output_clip),
            "-vf", "select=eq(n\\,0)",
            "-frames:v", "1",
            str(output_frame),
        ], timeout=60)

    finally:
        tmp_path.unlink(missing_ok=True)


# ── Stage 3: Caption ──

def generate_caption(frame_path: Path, api_key: str, model: str = "gpt-4o-mini") -> str:
    """用 Vision API 生成 caption。"""
    if not api_key or not frame_path.exists():
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        b64 = base64.b64encode(frame_path.read_bytes()).decode()
        data_url = f"data:image/jpeg;base64,{b64}"
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Describe this video frame in one short sentence (≤ 20 words) "
                        "that would work as a text-to-video generation prompt. "
                        "Focus on subject, setting, and motion cues. Do not add commentary."
                    )},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=80,
        )
        return (resp.choices[0].message.content or "").strip()[:300]
    except Exception as exc:
        logger.warning("Caption failed: %s", exc)
        return ""


# ── Stage 4: 数据库 ──

class PipelineDB:
    """轻量级 SQLite 管理。"""

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT UNIQUE NOT NULL,
                url TEXT NOT NULL,
                title TEXT,
                duration REAL,
                width INTEGER,
                height INTEGER,
                fps REAL,
                tbr INTEGER,
                vcodec TEXT,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id TEXT UNIQUE NOT NULL,
                source_video_id TEXT NOT NULL,
                start_sec REAL NOT NULL,
                clip_path TEXT,
                frame_path TEXT,
                clip_sha256 TEXT,
                frame_sha256 TEXT,
                caption TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_clips_source ON clips(source_video_id);
        """)
        self.conn.commit()

    def add_source(self, spec: SourceSpec) -> int:
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO sources
               (video_id, url, title, duration, width, height, fps, tbr, vcodec)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (spec.video_id, spec.url, spec.title, spec.duration,
             spec.width, spec.height, spec.fps, spec.tbr, spec.vcodec)
        )
        self.conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = self.conn.execute(
            "SELECT id FROM sources WHERE video_id=?", (spec.video_id,)
        ).fetchone()
        return row[0] if row else 0

    def add_clip(self, result: SliceResult):
        self.conn.execute(
            """INSERT INTO clips
               (clip_id, source_video_id, start_sec, clip_path, frame_path,
                clip_sha256, frame_sha256, caption)
               VALUES (?,?,?,?,?,?,?,?)""",
            (result.clip_id, result.source_id, result.start_sec,
             str(result.clip_path), str(result.frame_path),
             result.clip_sha256, result.frame_sha256, result.caption)
        )
        self.conn.commit()

    def count_clips(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM clips").fetchone()
        return row[0] if row else 0

    def get_all_clips(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM clips ORDER BY clip_id").fetchall()

    def close(self):
        self.conn.close()


# ── Stage 5: 导出 ──

def export_interval_package(
    clips_dir: Path,
    frames_dir: Path,
    db: PipelineDB,
    interval_id: int,
    miner_hotkey: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """导出 dataset.parquet + manifest.json。"""
    # 确保目标目录结构
    out_clips = output_dir / "clips"
    out_frames = output_dir / "frames"
    out_clips.mkdir(parents=True, exist_ok=True)
    out_frames.mkdir(parents=True, exist_ok=True)

    rows = db.get_all_clips()
    if len(rows) != 400:
        logger.warning("Clip count is %d, expected 400", len(rows))

    # 构建 parquet 记录
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for row in rows:
        src_path = Path(row["clip_path"])
        frame_path = Path(row["frame_path"])

        # 复制到输出目录（使用 clip_id 命名）
        out_clip = out_clips / f"{row['clip_id']}.mp4"
        out_frame = out_frames / f"{row['clip_id']}.jpg"
        if src_path.exists():
            import shutil
            shutil.copy2(src_path, out_clip)
        if frame_path.exists():
            shutil.copy2(frame_path, out_frame)

        # 计算输出文件的 SHA256
        clip_sha = sha256_file(out_clip) if out_clip.exists() else row["clip_sha256"]
        frame_sha = sha256_file(out_frame) if out_frame.exists() else row["frame_sha256"]

        records.append({
            "clip_id": row["clip_id"],
            "clip_uri": f"clips/{row['clip_id']}.mp4",
            "clip_sha256": clip_sha,
            "first_frame_uri": f"frames/{row['clip_id']}.jpg",
            "first_frame_sha256": frame_sha,
            "source_video_id": row["source_video_id"],
            "clip_start_sec": row["start_sec"],
            "duration_sec": CLIP_DURATION,
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "fps": float(TARGET_FPS),
            "num_frames": TARGET_FRAMES,
            "source_video_url": f"https://www.youtube.com/watch?v={row['source_video_id']}",
            "caption": row["caption"] or "",
        })

    # 写入 parquet
    dataset_path = output_dir / "dataset.parquet"
    table = pa.Table.from_pylist(records)
    pq.write_table(table, dataset_path)

    # 写入 manifest
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "protocol_version": "2.0.0",
        "schema_version": "2.0",
        "spec_id": "video_v1",
        "netuid": 70,
        "miner_hotkey": miner_hotkey,
        "interval_id": interval_id,
        "record_count": len(records),
        "dataset_sha256": sha256_file(dataset_path),
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Exported: %s", output_dir)
    logger.info("  dataset.parquet: %d records", len(records))
    logger.info("  manifest.json: interval=%d", interval_id)
    return dataset_path, manifest_path


# ── 主流程 ──

def parse_sources_file(path: Path) -> list[SourceSpec]:
    """解析 sources.txt，格式支持：
    - URL
    - URL|cut1,cut2,cut3  （指定切点）
    """
    specs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 1)
        url = parts[0].strip()
        spec = inspect_source(url)
        if len(parts) > 1:
            # 用户指定了切点
            spec.cuts = [float(x.strip()) for x in parts[1].split(",") if x.strip()]
        specs.append(spec)
    return specs


def compute_auto_cuts(duration: float, count: int) -> list[float]:
    """在视频时长内均匀分布切点，确保间隔 >= 4.5s。"""
    max_segments = int(math.floor(duration / CLIP_DURATION))
    if max_segments <= 0:
        return []
    if count > max_segments:
        count = max_segments
    step = duration / count
    cuts = [round(i * step, 3) for i in range(count)]
    return cuts


def run_pipeline(args: argparse.Namespace) -> None:
    """执行完整 pipeline。"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    frames_dir = output_dir / "frames"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "miner.db"
    db = PipelineDB(db_path)

    # ── 解析视频源 ──
    if args.url:
        spec = inspect_source(args.url)
        if args.cuts:
            spec.cuts = [float(x) for x in args.cuts.split(",")]
        specs = [spec]
    elif args.sources:
        logger.info("Parsing sources file: %s", args.sources)
        specs = parse_sources_file(Path(args.sources))
    else:
        logger.error("No input source. Use --url or --sources")
        sys.exit(1)

    total_needed = args.count
    collected = 0

    for spec in specs:
        if collected >= total_needed:
            break

        logger.info("\n[VIDEO] %s | %s | %.0fs | %dx%d",
                    spec.video_id, spec.title, spec.duration, spec.width, spec.height)

        # 质量预检
        if spec.width < 1920 or spec.height < 1080:
            logger.warning("Skipping: resolution too low (%dx%d)", spec.width, spec.height)
            continue
        if spec.fps < 24:
            logger.warning("Skipping: fps too low (%.1f)", spec.fps)
            continue
        if spec.duration < 30:
            logger.warning("Skipping: duration too short (%.1f)", spec.duration)
            continue

        # 确定切点
        if spec.cuts:
            cuts = spec.cuts
        else:
            need_from_this = min(total_needed - collected, int(spec.duration // CLIP_DURATION))
            cuts = compute_auto_cuts(spec.duration, need_from_this)

        logger.info("Will cut %d segments from this video", len(cuts))

        db.add_source(spec)

        for start_sec in cuts:
            if collected >= total_needed:
                break

            clip_id = deterministic_clip_id(spec.video_id, start_sec)
            clip_path = clips_dir / f"{clip_id}.mp4"
            frame_path = frames_dir / f"{clip_id}.jpg"

            # 去重：检查是否已存在
            existing = db.conn.execute(
                "SELECT 1 FROM clips WHERE source_video_id=? AND ABS(start_sec - ?) < 4.5",
                (spec.video_id, start_sec)
            ).fetchone()
            if existing:
                logger.debug("Overlap skip: %s @ %.1fs", spec.video_id, start_sec)
                continue

            logger.info("[%d/%d] Slicing %s @ %.1fs → %s",
                        collected + 1, total_needed, spec.video_id, start_sec, clip_id)

            try:
                stream_slice(spec.url, start_sec, clip_path, frame_path)
            except Exception as exc:
                logger.error("Slice failed: %s", exc)
                continue

            # SHA256
            clip_sha = sha256_file(clip_path)
            frame_sha = sha256_file(frame_path)

            # Caption
            caption = ""
            if args.openai_key:
                caption = generate_caption(frame_path, args.openai_key, args.caption_model)
                logger.debug("Caption: %s", caption[:60])

            result = SliceResult(
                clip_id=clip_id,
                source_id=spec.video_id,
                start_sec=start_sec,
                clip_path=clip_path,
                frame_path=frame_path,
                clip_sha256=clip_sha,
                frame_sha256=frame_sha,
                caption=caption,
            )
            db.add_clip(result)
            collected += 1

    logger.info("\n=== Collection complete: %d clips ===", collected)

    if collected == 0:
        logger.error("No clips collected. Check sources and network.")
        sys.exit(1)

    # ── 导出 ──
    logger.info("Exporting interval package...")
    export_interval_package(
        clips_dir, frames_dir, db,
        interval_id=args.interval_id,
        miner_hotkey=args.hotkey,
        output_dir=output_dir,
    )

    db.close()
    logger.info("All done. Output: %s", output_dir)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="End-to-end miner pipeline")
    parser.add_argument("--sources", help="Path to sources.txt")
    parser.add_argument("--url", help="Single video URL")
    parser.add_argument("--cuts", help="Comma-separated cut times (e.g. 0,15,30)")
    parser.add_argument("--interval-id", type=int, default=1, help="Interval ID")
    parser.add_argument("--hotkey", default="test_hotkey", help="Miner hotkey SS58")
    parser.add_argument("--count", type=int, default=400, help="Target clip count")
    parser.add_argument("--openai-key", default="", help="OpenAI API key for caption")
    parser.add_argument("--caption-model", default="gpt-4o-mini", help="Caption model")
    parser.add_argument("--output-dir", default="./interval_out", help="Output directory")
    args = parser.parse_args()

    if not args.sources and not args.url:
        parser.print_help()
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
