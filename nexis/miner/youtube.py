"""Source acquisition and clip extraction utilities for miners.

Despite the historical name, this module supports any URL handled by yt-dlp
(YouTube, Vimeo, Twitch, etc.) and re-encodes clips to the strict v2 spec
(1280x704, 24fps, 121 frames).
"""
"""
矿工视频源获取与切片工具。

虽然文件名是 youtube.py，但实际上支持 yt-dlp 能处理的任何平台
（YouTube、Vimeo、Twitch 等）。核心功能：
1. download_source_video: 下载完整视频
2. probe_video: ffprobe 探测视频信息
3. create_clip: ffmpeg 切出指定规格片段
4. extract_first_frame: 提取首帧图片
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from ..protocol import (
    CLIP_DURATION_SEC,
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_NUM_FRAMES,
    TARGET_WIDTH,
)

# ── 超时设置 ──
YT_DLP_DOWNLOAD_TIMEOUT_SECONDS = 600  # yt-dlp 下载超时（秒）
FFPROBE_TIMEOUT_SEC = 30               # ffprobe 探测超时
FFMPEG_TIMEOUT_SEC = 240               # ffmpeg 切片/提取超时
YTDLP_RETRIES = 2                      # yt-dlp 下载失败重试次数
logger = logging.getLogger(__name__)


def _build_yt_dlp_cmd(args: list[str]) -> list[str]:
    """构造 yt-dlp 命令，统一加上 yt-dlp 前缀。"""
    return ["yt-dlp", *args]


def _run_command(
    cmd: list[str],
    *,
    timeout_sec: int,
    capture_output: bool = False,
    text: bool = False,
) -> subprocess.CompletedProcess:
    """运行外部命令，失败时抛出异常（check=True）。"""
    return subprocess.run(
        cmd,
        check=True,
        timeout=timeout_sec,
        capture_output=capture_output,
        text=text,
    )


def _run_subprocess(
    cmd: list[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess:
    """运行外部命令，不抛出异常（check=False），返回 CompletedProcess。"""
    return subprocess.run(
        cmd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


def read_sources(path: Path) -> list[str]:
    """读取 sources.txt，返回非空且非注释的行列表。"""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def download_source_video(url: str, output_dir: Path) -> Path:
    """Download a video from any yt-dlp supported URL."""
    """
    从 yt-dlp 支持的 URL 下载完整视频。

    流程：
    1. yt-dlp 下载最佳画质视频（≥704p）
    2. 自动合并音频并转为 mp4 格式
    3. 返回本地文件路径

    注意：这会下载整个视频，即使只需要其中几秒片段。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(extractor)s_%(id)s.%(ext)s")
    # 源视频分辨率应不低于目标分辨率（704p），后续 ffmpeg 会统一缩放到 1280x704
    height_floor = TARGET_HEIGHT
    cmd = _build_yt_dlp_cmd(
        [
            "--no-simulate",
            "-f",
            (
                f"bestvideo[height>={height_floor}]+bestaudio/"
                f"bestvideo+bestaudio/best"
            ),
            "--merge-output-format",
            "mp4",
            "--recode-video",
            "mp4",
            "-o",
            output_template,
            "--no-playlist",
            "--no-overwrites",
            "--print",
            "after_move:filepath",
            url,
        ]
    )
    last_error: Exception | None = None
    for attempt in range(1, YTDLP_RETRIES + 1):
        try:
            logger.info("yt-dlp download start url=%s attempt=%d/%d", url, attempt, YTDLP_RETRIES)
            result = _run_subprocess(
                cmd,
                timeout=YT_DLP_DOWNLOAD_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                logger.warning(
                    "yt-dlp download failed attempt=%d/%d rc=%d err=%s",
                    attempt,
                    YTDLP_RETRIES,
                    result.returncode,
                    (result.stderr or "")[:200],
                )
                last_error = RuntimeError(
                    f"yt-dlp download failed: {(result.stderr or '')[:200]}"
                )
                continue

            for line in reversed(result.stdout.splitlines()):
                candidate = line.strip()
                if candidate and Path(candidate).exists():
                    logger.info("yt-dlp download complete url=%s path=%s", url, candidate)
                    return Path(candidate)

            mp4_candidates = sorted(
                output_dir.glob("*.mp4"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if mp4_candidates:
                logger.info(
                    "yt-dlp download complete (fallback) url=%s path=%s",
                    url,
                    mp4_candidates[0],
                )
                return mp4_candidates[0]
            last_error = RuntimeError("yt-dlp completed but output file was not found")
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("yt-dlp download exception attempt=%d/%d: %s", attempt, YTDLP_RETRIES, exc)
            last_error = exc
    raise RuntimeError(f"failed to download source video for url={url}") from last_error


# Back-compat alias retained for any external callers.
download_youtube_video = download_source_video


def probe_video(path: Path) -> dict:
    """使用 ffprobe 探测视频的元数据（流信息 + 格式信息）。"""
    logger.debug("ffprobe start path=%s", path)
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    proc = _run_command(
        cmd,
        timeout_sec=FFPROBE_TIMEOUT_SEC,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    logger.debug("ffprobe complete path=%s", path)
    return payload


def create_clip(src: Path, dst: Path, start_sec: float) -> None:
    """Create a clip re-encoded to the strict spec (1280x704, 24fps, 121 frames, no audio)."""
    """
    从源视频中切出一段严格符合协议的片段。

    ffmpeg 参数说明：
    - -ss {start_sec}: 从指定时间开始
    - -frames:v 121: 只取 121 帧（约 5.04 秒）
    - scale=1280:704:force_original_aspect_ratio=increase: 先放大以覆盖目标尺寸
    - crop=1280:704: 再裁剪为精确 1280x704
    - fps=24: 重采样为 24fps
    - -an: 去掉音频
    - libx264 + yuv420p: H.264 编码，确保兼容性
    - crf=20: 高质量压缩（值越小质量越高，默认 23）
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 先放大到能覆盖目标尺寸，再裁剪为精确 1280x704，然后统一为 24fps
    vf = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_WIDTH}:{TARGET_HEIGHT},"
        f"fps={TARGET_FPS}"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(src),
        "-frames:v",
        str(TARGET_NUM_FRAMES),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        str(dst),
    ]
    logger.debug(
        "ffmpeg create clip src=%s dst=%s start=%.3f frames=%d",
        src,
        dst,
        start_sec,
        TARGET_NUM_FRAMES,
    )
    _run_command(cmd, timeout_sec=FFMPEG_TIMEOUT_SEC)


def extract_first_frame(src: Path, dst: Path) -> None:
    """提取视频的第 0 帧（首帧）作为 JPG 图片。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        "select=eq(n\\,0)",
        "-frames:v",
        "1",
        str(dst),
    ]
    logger.debug("ffmpeg extract first frame src=%s dst=%s", src, dst)
    _run_command(cmd, timeout_sec=FFMPEG_TIMEOUT_SEC)


def get_video_duration_sec(path: Path) -> float:
    """获取视频总时长（秒）。"""
    info = probe_video(path)
    duration_str = info.get("format", {}).get("duration")
    try:
        return float(duration_str)
    except (TypeError, ValueError):
        return 0.0


_ = (CLIP_DURATION_SEC, os)  # quiet unused import warnings; CLIP_DURATION_SEC kept for callers
