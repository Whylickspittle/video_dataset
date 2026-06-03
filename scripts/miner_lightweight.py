#!/usr/bin/env python3
"""
轻量级独立挖矿脚本（1C1G VPS 专用）

无需 git clone 整个仓库，仅需此文件 + sources.txt 即可运行。

依赖：
    pip install pyarrow openai
    # 系统需安装: yt-dlp, ffmpeg, ffprobe

用法：
    # 1. 下载脚本
    wget https://raw.githubusercontent.com/.../miner_lightweight.py

    # 2. 创建 sources.txt（每行一个 YouTube URL）
    cat > sources.txt << 'EOF'
    https://www.youtube.com/watch?v=LXb3EKWsInQ
    https://www.youtube.com/watch?v=aqz-KE-bpKQ
    EOF

    # 3. 执行
    python3 miner_lightweight.py \
        --sources sources.txt \
        --hotkey YOUR_HOTKEY \
        --openai-key sk-xxx \
        --interval-id 1 \
        --count 400
"""

import argparse
import base64
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

# ── 配置 ──
CLIP_DURATION = 5.04
TARGET_WIDTH = 1280
TARGET_HEIGHT = 704
TARGET_FPS = 24
TARGET_FRAMES = 121
BUFFER_SECONDS = 8

# 1C1G 优化：用 ultrafast 减少 CPU 压力
FFMPEG_PRESET = "ultrafast"
FFMPEG_CRF = "22"

# 并行度（1C1G 只能串行）
MAX_WORKERS = 1


def run_cmd(cmd: list[str], timeout: int = 180) -> str:
    """运行命令，返回 stdout。"""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=True
    )
    return result.stdout.strip()


def get_stream_url(video_url: str, min_height: int = 1080) -> str:
    """用 yt-dlp 获取视频直接流 URL。"""
    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height>={min_height}][ext=mp4]/bestvideo[height>={min_height}]",
        "--no-check-certificates",
        "-g",
        video_url,
    ]
    return run_cmd(cmd, timeout=30)


def get_video_info(video_url: str) -> dict:
    """获取视频元数据。"""
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-check-certificates",
        video_url,
    ]
    out = run_cmd(cmd, timeout=30)
    return json.loads(out)


def sha256_file(path: Path) -> str:
    """计算文件 SHA256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_segment(stream_url: str, start: float, duration: float,
                     output: Path) -> None:
    """从流下载指定时间段。"""
    total_duration = duration + BUFFER_SECONDS
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(total_duration),
        "-i", stream_url,
        "-c:v", "copy",
        "-an",
        "-avoid_negative_ts", "make_zero",
        str(output),
    ]
    subprocess.run(cmd, capture_output=True, timeout=120, check=True)


def precise_cut(input_path: Path, output_path: Path) -> None:
    """精确切出目标片段。"""
    vf = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_WIDTH}:{TARGET_HEIGHT},"
        f"fps={TARGET_FPS}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-t", str(CLIP_DURATION),
        "-i", str(input_path),
        "-frames:v", str(TARGET_FRAMES),
        "-vf", vf,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", FFMPEG_PRESET,
        "-crf", FFMPEG_CRF,
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=180, check=True)


def extract_first_frame(video_path: Path, output_path: Path) -> None:
    """提取首帧。"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", "select=eq(n\\,0)",
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=60, check=True)


def generate_caption(frame_path: Path, api_key: str, model: str = "gpt-4o-mini") -> str:
    """用 Vision API 生成 caption。"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        data_url = f"data:image/jpeg;base64,{b64}"
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Describe this video frame in one short sentence (≤ 20 words) "
                        "that would work as a text-to-video generation prompt. "
                        "Focus on subject, setting, and motion cues."
                    )},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=80,
        )
        return (resp.choices[0].message.content or "").strip()[:300]
    except Exception as exc:
        print(f"[WARN] Caption failed: {exc}", file=sys.stderr)
        return ""


def deterministic_clip_id(source_id: str, start_sec: float) -> str:
    """生成确定性 clip_id。"""
    data = f"{source_id}:{start_sec:.3f}:{CLIP_DURATION:.3f}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def extract_source_id(url: str) -> str:
    """从 URL 提取 video ID。"""
    from urllib.parse import parse_qs, urlparse
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        return parsed.path.strip("/")
    elif "youtube.com" in host:
        query = parse_qs(parsed.query)
        vals = query.get("v", [])
        return vals[0].strip() if vals else parsed.path.strip("/").split("/")[-1]
    else:
        return hashlib.sha256(url.encode()).hexdigest()[:16]


class SimpleDB:
    """极简 SQLite 数据库，用于去重和进度跟踪。"""

    def __init__(self, db_path: str = "miner.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS clips (
                clip_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_url TEXT,
                start_sec REAL NOT NULL,
                duration_sec REAL DEFAULT 5.04,
                caption TEXT,
                clip_sha256 TEXT,
                frame_sha256 TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_source_start ON clips(source_id, start_sec);
        """)
        self.conn.commit()

    def is_overlap(self, source_id: str, start_sec: float, window: float = 4.5) -> bool:
        """检查是否与已有切片重叠。"""
        row = self.conn.execute(
            "SELECT 1 FROM clips WHERE source_id = ? AND ABS(start_sec - ?) < ? LIMIT 1",
            (source_id, start_sec, window)
        ).fetchone()
        return row is not None

    def add_clip(self, clip_id: str, source_id: str, source_url: str,
                 start_sec: float, caption: str,
                 clip_sha256: str, frame_sha256: str):
        self.conn.execute(
            """INSERT OR IGNORE INTO clips
               (clip_id, source_id, source_url, start_sec, caption, clip_sha256, frame_sha256)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (clip_id, source_id, source_url, start_sec, caption, clip_sha256, frame_sha256)
        )
        self.conn.commit()

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM clips").fetchone()
        return row[0]

    def get_all_records(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM clips").fetchall()
        records = []
        for r in rows:
            records.append({
                "clip_id": r["clip_id"],
                "clip_uri": f"clips/{r['clip_id']}.mp4",
                "clip_sha256": r["clip_sha256"] or "",
                "first_frame_uri": f"frames/{r['clip_id']}.jpg",
                "first_frame_sha256": r["frame_sha256"] or "",
                "source_video_id": r["source_id"],
                "clip_start_sec": r["start_sec"],
                "duration_sec": CLIP_DURATION,
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
                "fps": float(TARGET_FPS),
                "num_frames": TARGET_FRAMES,
                "source_video_url": r["source_url"] or "",
                "caption": r["caption"] or "",
                "third_party_url": "",
            })
        return records

    def close(self):
        self.conn.close()


def process_single_clip(url: str, start_sec: float, output_dir: Path,
                        db: SimpleDB, args) -> Optional[dict]:
    """
    处理单个切片。返回 metadata dict 或 None（失败/重复）。
    """
    source_id = extract_source_id(url)

    # 去重检查
    if db.is_overlap(source_id, start_sec):
        print(f"  [SKIP] Overlap detected: {source_id} @ {start_sec}s")
        return None

    clip_id = deterministic_clip_id(source_id, start_sec)
    clip_dir = output_dir / "clips"
    frame_dir = output_dir / "frames"
    clip_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    clip_path = clip_dir / f"{clip_id}.mp4"
    frame_path = frame_dir / f"{clip_id}.jpg"

    if clip_path.exists() and frame_path.exists():
        print(f"  [SKIP] Already exists: {clip_id}")
        # 重新计算 caption（如果没有）
        return None

    # Step 1: 获取流 URL
    print(f"  [1/5] Fetching stream URL...")
    try:
        stream_url = get_stream_url(url)
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] yt-dlp failed: {exc.stderr}")
        return None

    # Step 2: 下载缓冲段
    print(f"  [2/5] Downloading buffer from {start_sec}s...")
    tmp_path = Path(tempfile.mktemp(suffix=".mp4"))
    try:
        download_segment(stream_url, start_sec, CLIP_DURATION, tmp_path)
        print(f"  Buffer: {tmp_path.stat().st_size / 1024 / 1024:.1f} MB")
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] Download failed: {exc.stderr}")
        tmp_path.unlink(missing_ok=True)
        return None

    # Step 3: 精确切
    print(f"  [3/5] Cutting precise clip...")
    try:
        precise_cut(tmp_path, clip_path)
        print(f"  Clip: {clip_path.stat().st_size / 1024:.1f} KB")
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] Cut failed: {exc.stderr}")
        tmp_path.unlink(missing_ok=True)
        return None
    finally:
        tmp_path.unlink(missing_ok=True)

    # Step 4: 提取首帧
    print(f"  [4/5] Extracting first frame...")
    try:
        extract_first_frame(clip_path, frame_path)
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] Frame extraction failed: {exc.stderr}")
        return None

    # Step 5: Caption
    caption = ""
    if args.openai_key:
        print(f"  [5/5] Generating caption...")
        caption = generate_caption(frame_path, args.openai_key, args.caption_model)
        print(f"  Caption: {caption[:60]}..." if len(caption) > 60 else f"  Caption: {caption}")
    else:
        print(f"  [5/5] Skipping caption (no API key)")

    # 计算 SHA256
    clip_sha256 = sha256_file(clip_path)
    frame_sha256 = sha256_file(frame_path)

    # 写入数据库
    db.add_clip(clip_id, source_id, url, start_sec, caption, clip_sha256, frame_sha256)

    return {
        "clip_id": clip_id,
        "clip_uri": f"clips/{clip_id}.mp4",
        "clip_sha256": clip_sha256,
        "first_frame_uri": f"frames/{clip_id}.jpg",
        "first_frame_sha256": frame_sha256,
        "source_video_id": source_id,
        "clip_start_sec": start_sec,
        "duration_sec": CLIP_DURATION,
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "fps": float(TARGET_FPS),
        "num_frames": TARGET_FRAMES,
        "source_video_url": url,
        "caption": caption,
        "third_party_url": "",
    }


def generate_manifest(records: list[dict], args) -> dict:
    """生成 manifest.json。"""
    total_size = 0
    for r in records:
        clip_path = Path(args.output_dir) / "clips" / f"{r['clip_id']}.mp4"
        if clip_path.exists():
            total_size += clip_path.stat().st_size

    return {
        "version": "1.0.0",
        "interval_id": args.interval_id,
        "miner_hotkey": args.hotkey,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_clips": len(records),
        "total_size_bytes": total_size,
        "expected_specs": {
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "fps": TARGET_FPS,
            "num_frames": TARGET_FRAMES,
            "duration_sec": CLIP_DURATION,
        },
    }


def export_parquet(records: list[dict], output_path: Path):
    """导出 dataset.parquet。"""
    if not records:
        print("[WARN] No records to export")
        return

    df_data = {key: [r[key] for r in records] for key in records[0].keys()}
    table = pa.table(df_data)
    pq.write_table(table, str(output_path))
    print(f"[INFO] Exported {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Lightweight Nexis Miner (1C1G VPS)")
    parser.add_argument("--sources", required=True, help="sources.txt path")
    parser.add_argument("--interval-id", type=int, required=True)
    parser.add_argument("--hotkey", required=True, help="Your SS58 hotkey")
    parser.add_argument("--openai-key", default="", help="OpenAI API key")
    parser.add_argument("--caption-model", default="gpt-4o-mini")
    parser.add_argument("--count", type=int, default=400, help="Target clip count")
    parser.add_argument("--output-dir", default="./interval_out")
    parser.add_argument("--db", default="miner.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取 sources
    if not os.path.exists(args.sources):
        print(f"[ERROR] sources.txt not found: {args.sources}")
        sys.exit(1)

    urls = []
    with open(args.sources) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    print(f"[INFO] Loaded {len(urls)} video sources")
    print(f"[INFO] Target: {args.count} clips")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] 1C1G Mode: ffmpeg preset={FFMPEG_PRESET}, single-threaded")

    if args.dry_run:
        print("[DRY RUN] Would process the following:")
        for url in urls[:5]:
            print(f"  - {url}")
        sys.exit(0)

    db = SimpleDB(args.db)
    records = []
    total_attempts = 0

    try:
        while len(records) < args.count and total_attempts < args.count * 3:
            url = random.choice(urls)

            # 获取视频时长以确定随机范围
            print(f"\n[{len(records)+1}/{args.count}] Processing: {url}")
            try:
                info = get_video_info(url)
                duration = info.get("duration", 300)
                source_id = info.get("id", extract_source_id(url))
            except Exception as exc:
                print(f"  [WARN] Failed to get info: {exc}")
                duration = 300
                source_id = extract_source_id(url)

            if duration < CLIP_DURATION * 2:
                print(f"  [SKIP] Video too short: {duration}s")
                total_attempts += 1
                continue

            # 随机选取起始时间（避开首尾）
            max_start = duration - CLIP_DURATION - 1
            start_sec = round(random.uniform(5.0, max(6.0, max_start)), 3)

            # 去重预检
            if db.is_overlap(source_id, start_sec):
                print(f"  [SKIP] Overlap: {source_id} @ {start_sec}s")
                total_attempts += 1
                continue

            # 处理切片
            record = process_single_clip(url, start_sec, output_dir, db, args)
            if record:
                records.append(record)

            total_attempts += 1

            # 1C1G 优化：每处理 10 个休息 2 秒，避免 CPU 过热
            if len(records) % 10 == 0:
                time.sleep(2)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        print(f"\n[INFO] Generated {len(records)} clips ({db.count()} total in DB)")

        # 导出
        if records:
            export_parquet(records, output_dir / "dataset.parquet")

            manifest = generate_manifest(records, args)
            with open(output_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            print(f"[INFO] Output files:")
            print(f"  Clips:  {output_dir}/clips/ ({len(list((output_dir/'clips').glob('*.mp4')))} files)")
            print(f"  Frames: {output_dir}/frames/ ({len(list((output_dir/'frames').glob('*.jpg')))} files)")
            print(f"  Parquet: {output_dir}/dataset.parquet")
            print(f"  Manifest: {output_dir}/manifest.json")

        db.close()


if __name__ == "__main__":
    main()
