#!/usr/bin/env python3
"""
流式切片工具：不下载完整视频，直接从 YouTube 流切出指定片段。

用法：
    python3 stream_slice.py "URL" 0.0 --output-dir ./clips
    python3 stream_slice.py "URL" 300.5 --output-dir ./clips --caption-key sk-xxx

输出：
    ./clips/{clip_id}.mp4      # 5.04s 切片
    ./clips/{clip_id}.jpg      # 首帧
    ./clips/{clip_id}.json     # 元数据（含 caption、sha256）
"""

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# ── 配置 ──
CLIP_DURATION = 5.04          # 秒
TARGET_WIDTH = 1280
TARGET_HEIGHT = 704
TARGET_FPS = 24
TARGET_FRAMES = 121
BUFFER_SECONDS = 8            # 多下载几秒作为缓冲，确保精确切


def run_cmd(cmd: list[str], timeout: int = 120) -> str:
    """运行命令，返回 stdout。"""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=True
    )
    return result.stdout.strip()


def get_stream_url(video_url: str, min_height: int = 1080) -> str:
    """用 yt-dlp 获取视频直接流 URL。"""
    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height>={min_height}]",
        "--no-check-certificates",
        "-g",  # 只打印 URL
        video_url,
    ]
    return run_cmd(cmd, timeout=30)


def sha256_file(path: Path) -> str:
    """计算文件 SHA256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_segment(stream_url: str, start: float, duration: float,
                     output: Path) -> None:
    """从流下载指定时间段（比目标多 BUFFER_SECONDS 缓冲）。"""
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


def precise_cut(input_path: Path, output_path: Path,
                duration: float = CLIP_DURATION) -> None:
    """从本地缓存文件精确切出目标片段。"""
    vf = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_WIDTH}:{TARGET_HEIGHT},"
        f"fps={TARGET_FPS}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-t", str(duration),
        "-i", str(input_path),
        "-frames:v", str(TARGET_FRAMES),
        "-vf", vf,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "20",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=120, check=True)


def extract_first_frame(video_path: Path, output_path: Path) -> None:
    """提取首帧。"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", "select=eq(n\\,0)",
        "-frames:v", "1",
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


def main():
    parser = argparse.ArgumentParser(description="Stream-based clip slicer")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("start", type=float, help="Start time in seconds")
    parser.add_argument("--output-dir", default="./clips", help="Output directory")
    parser.add_argument("--caption-key", default="", help="OpenAI API key for caption")
    parser.add_argument("--caption-model", default="gpt-4o-mini", help="Caption model")
    parser.add_argument("--min-height", type=int, default=1080, help="Minimum source height")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取 source_video_id（简化处理）
    from urllib.parse import parse_qs, urlparse
    parsed = urlparse(args.url)
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        source_id = parsed.path.strip("/")
    elif "youtube.com" in host:
        query = parse_qs(parsed.query)
        vals = query.get("v", [])
        source_id = vals[0].strip() if vals else parsed.path.strip("/").split("/")[-1]
    else:
        source_id = hashlib.sha256(args.url.encode()).hexdigest()[:16]

    clip_id = deterministic_clip_id(source_id, args.start)
    clip_path = output_dir / f"{clip_id}.mp4"
    frame_path = output_dir / f"{clip_id}.jpg"
    meta_path = output_dir / f"{clip_id}.json"

    if args.dry_run:
        print(f"[DRY RUN] Would slice {args.url} at {args.start}s")
        print(f"  clip: {clip_path}")
        print(f"  frame: {frame_path}")
        return

    # ── Step 1: 获取流 URL ──
    print(f"[1/5] Fetching stream URL...")
    try:
        stream_url = get_stream_url(args.url, min_height=args.min_height)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Failed to get stream URL: {exc.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"  Stream URL obtained")

    # ── Step 2: 下载缓冲段 ──
    print(f"[2/5] Downloading {CLIP_DURATION + BUFFER_SECONDS}s buffer from {args.start}s...")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        download_segment(stream_url, args.start, CLIP_DURATION, tmp_path)
        tmp_size = tmp_path.stat().st_size
        print(f"  Buffer downloaded: {tmp_size / 1024 / 1024:.1f} MB")
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Download failed: {exc.stderr}", file=sys.stderr)
        tmp_path.unlink(missing_ok=True)
        sys.exit(1)

    # ── Step 3: 精确切 ──
    print(f"[3/5] Cutting precise {CLIP_DURATION}s clip...")
    try:
        precise_cut(tmp_path, clip_path)
        clip_size = clip_path.stat().st_size
        print(f"  Clip created: {clip_size / 1024:.1f} KB")
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Cut failed: {exc.stderr}", file=sys.stderr)
        tmp_path.unlink(missing_ok=True)
        sys.exit(1)
    finally:
        tmp_path.unlink(missing_ok=True)

    # ── Step 4: 提取首帧 ──
    print(f"[4/5] Extracting first frame...")
    try:
        extract_first_frame(clip_path, frame_path)
        print(f"  Frame extracted: {frame_path.stat().st_size / 1024:.1f} KB")
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Frame extraction failed: {exc.stderr}", file=sys.stderr)
        sys.exit(1)

    # ── Step 5: 生成 caption ──
    caption = ""
    if args.caption_key:
        print(f"[5/5] Generating caption...")
        caption = generate_caption(frame_path, args.caption_key, args.caption_model)
        print(f"  Caption: {caption}")
    else:
        print(f"[5/5] Skipping caption (no API key)")

    # ── 写入元数据 ──
    metadata = {
        "clip_id": clip_id,
        "source_video_id": source_id,
        "source_video_url": args.url,
        "clip_start_sec": args.start,
        "duration_sec": CLIP_DURATION,
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "fps": TARGET_FPS,
        "num_frames": TARGET_FRAMES,
        "caption": caption,
        "clip_sha256": sha256_file(clip_path),
        "first_frame_sha256": sha256_file(frame_path),
        "clip_uri": f"clips/{clip_id}.mp4",
        "first_frame_uri": f"frames/{clip_id}.jpg",
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\nDone!")
    print(f"  Clip:  {clip_path}")
    print(f"  Frame: {frame_path}")
    print(f"  Meta:  {meta_path}")


if __name__ == "__main__":
    main()
