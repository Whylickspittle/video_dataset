#!/usr/bin/env python3
"""根据 sources.txt（多时间点格式）生成完整的数据集。

sources.txt 格式：
    # 注释
    URL | 时间1, 时间2, 时间3, ...
    URL ｜时间1，时间2，时间3，...

时间格式支持：
    - 纯秒数: 30, 120
    - 分:秒: 1:23, 12:34
    - 时:分:秒: 1:23:20

示例：
    https://www.youtube.com/watch?v=xxx | 12, 21, 32, 1:34, 2:20
    https://youtu.be/xxx ｜ 8:20, 15:30, 1:23:40

输出：
    workdir/out/{interval_id}/
        dataset.parquet       # 数据集
        manifest.json         # 清单
        clips/                # 视频片段
        frames/               # 首帧图片
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from nexis.hash_utils import deterministic_clip_id, sha256_file
from nexis.miner.captioner import Captioner
from nexis.miner.pipeline import _canonical_url
from nexis.miner.youtube import (
    create_clip,
    download_source_video,
    extract_first_frame,
    probe_video,
)
from nexis.models import ClipRecord, IntervalManifest
from nexis.protocol import (
    CLIP_DURATION_SEC,
    FPS_TOLERANCE,
    OVERLAP_WINDOW_SEC,
    PROTOCOL_VERSION,
    SAMPLE_COUNT,
    SCHEMA_VERSION,
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_NUM_FRAMES,
    TARGET_WIDTH,
)
from nexis.serialization import write_dataset_parquet, write_manifest
from nexis.specs import DEFAULT_SPEC_ID

logger = logging.getLogger(__name__)


def _parse_time_offset(time_str: str) -> float:
    """Parse time string to seconds. Supports: 30, 1:23, 1:23:20."""
    time_str = time_str.strip().replace("：", ":").replace("；", ";").replace("。", ".")
    parts = time_str.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid time format: {time_str!r}")


def parse_sources_with_timestamps(path: Path) -> list[tuple[str, list[float]]]:
    """Parse sources.txt with multiple timestamps per line.

    Returns: [(url, [time1, time2, ...]), ...]
    """
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    results: list[tuple[str, list[float]]] = []

    for line in lines:
        if not line or line.startswith("#"):
            continue

        # Split by either full-width or half-width separator
        # Try full-width first (｜), then half-width (|)
        if "｜" in line:
            url_part, times_part = line.split("｜", 1)
        elif "|" in line:
            url_part, times_part = line.split("|", 1)
        else:
            # No timestamps provided, use default (start from 0)
            url_part = line
            times_part = "0"

        url = url_part.strip()
        if not url:
            continue

        # Parse timestamps - split by comma or Chinese comma
        raw_times = re.split(r"[,，]", times_part)
        timestamps: list[float] = []
        for t in raw_times:
            t = t.strip()
            if not t:
                continue
            try:
                timestamps.append(_parse_time_offset(t))
            except ValueError as exc:
                logger.warning("skip invalid timestamp '%s' in line: %s", t, line[:60])
                continue

        if timestamps:
            results.append((url, timestamps))
        else:
            # No valid timestamps, default to 0
            results.append((url, [0.0]))

    return results


def _video_stream(info: dict[str, Any]) -> dict[str, Any]:
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise ValueError("no video stream")


def _source_video_id(url: str) -> str:
    """Extract a source ID from URL for clip naming."""
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        return parsed.path.strip("/") or url
    if host == "youtube.com" or host.endswith(".youtube.com"):
        query = parse_qs(parsed.query)
        values = query.get("v", [])
        if values and values[0].strip():
            return values[0].strip()
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "v"}:
            return parts[1]
    # Fallback: use last part of URL path
    path_parts = [p for p in parsed.path.split("/") if p]
    if path_parts:
        return path_parts[-1]
    return url


async def generate_dataset(
    sources_file: Path,
    workdir: Path,
    interval_id: int = 1,
    netuid: int = 70,
    miner_hotkey: str = "test_miner",
    captioner: Captioner | None = None,
) -> tuple[Path, Path]:
    """Generate dataset from sources.txt with timestamps."""
    logger.info("reading sources from %s", sources_file)
    sources = parse_sources_with_timestamps(sources_file)
    total_requested = sum(len(ts) for _, ts in sources)
    logger.info("found %d videos, %d total timestamps", len(sources), total_requested)

    if total_requested < SAMPLE_COUNT:
        logger.warning(
            "only %d timestamps requested, need %d for a full dataset",
            total_requested,
            SAMPLE_COUNT,
        )

    raw_dir = workdir / "raw"
    clips_dir = workdir / "clips"
    frames_dir = workdir / "frames"
    out_dir = workdir / "out" / str(interval_id)
    for d in (raw_dir, clips_dir, frames_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    records: list[ClipRecord] = []
    seen_positions: dict[str, list[float]] = {}  # dedup protection

    for url, timestamps in sources:
        if len(records) >= SAMPLE_COUNT:
            break

        canonical = _canonical_url(url)
        source_id = _source_video_id(url)
        logger.info("processing source_id=%s url=%s timestamps=%d", source_id, url, len(timestamps))

        # Download video (with offset = earliest timestamp to minimize download)
        earliest = min(timestamps) if timestamps else 0.0
        # Download a bit before earliest to have margin
        download_start = max(0.0, earliest - 10.0)
        try:
            raw_path = download_source_video(
                url,
                raw_dir,
                start_sec=download_start,
                max_download_sec=max(timestamps) - download_start + 20.0 if timestamps else 600.0,
            )
            logger.info("downloaded %s", raw_path)
        except Exception as exc:
            logger.warning("download failed url=%s err=%s", url, exc)
            continue

        # Probe
        try:
            probe = probe_video(raw_path)
            full_duration = float(probe.get("format", {}).get("duration") or 0.0)
            stream = _video_stream(probe)
            src_width = int(stream.get("width", 0) or 0)
            src_height = int(stream.get("height", 0) or 0)
            if src_width < TARGET_WIDTH or src_height < TARGET_HEIGHT:
                logger.warning(
                    "resolution too low: %dx%d < %dx%d",
                    src_width,
                    src_height,
                    TARGET_WIDTH,
                    TARGET_HEIGHT,
                )
                continue
        except Exception as exc:
            logger.warning("probe failed path=%s err=%s", raw_path, exc)
            continue

        # Process each timestamp
        for start_sec in timestamps:
            if len(records) >= SAMPLE_COUNT:
                break

            # Adjust start_sec relative to downloaded segment
            adjusted_start = start_sec - download_start
            if adjusted_start < 0:
                adjusted_start = 0.0

            # Check if enough duration remains
            available = full_duration - start_sec
            if available < CLIP_DURATION_SEC - 0.15:
                logger.warning(
                    "skip timestamp %s: only %.1fs remains (need %.1fs)",
                    start_sec,
                    available,
                    CLIP_DURATION_SEC,
                )
                continue

            # Deduplication: same video, start times too close
            positions = seen_positions.setdefault(canonical, [])
            if any(abs(start_sec - prev) < OVERLAP_WINDOW_SEC for prev in positions):
                logger.debug("skip duplicate timestamp %s (too close)", start_sec)
                continue

            clip_id = f"{source_id}_{int(start_sec):04d}"
            clip_path = clips_dir / f"{clip_id}.mp4"
            frame_path = frames_dir / f"{clip_id}.jpg"

            try:
                create_clip(raw_path, clip_path, adjusted_start)
                extract_first_frame(clip_path, frame_path)
            except Exception as exc:
                logger.warning("clip extraction failed start=%.1f err=%s", start_sec, exc)
                continue

            positions.append(start_sec)

            # Caption
            caption = ""
            if captioner and captioner.enabled:
                try:
                    caption = captioner.caption_frame(frame_path)
                except Exception as exc:
                    logger.warning("caption failed clip=%s err=%s", clip_id, exc)

            record = ClipRecord(
                clip_id=clip_id,
                clip_uri=f"clips/{clip_path.name}",
                clip_sha256=sha256_file(clip_path),
                first_frame_uri=f"frames/{frame_path.name}",
                first_frame_sha256=sha256_file(frame_path),
                source_video_id=source_id,
                clip_start_sec=start_sec,
                duration_sec=CLIP_DURATION_SEC,
                caption=caption,
                width=TARGET_WIDTH,
                height=TARGET_HEIGHT,
                fps=float(TARGET_FPS),
                num_frames=TARGET_NUM_FRAMES,
                source_video_url=canonical,
            )
            records.append(record)
            logger.info(
                "created clip %s start=%.1fs caption=%r",
                clip_id,
                start_sec,
                caption[:40],
            )

    if len(records) != SAMPLE_COUNT:
        logger.warning(
            "only produced %d/%d clips",
            len(records),
            SAMPLE_COUNT,
        )

    # Write dataset.parquet
    dataset_path = out_dir / "dataset.parquet"
    write_dataset_parquet(records, dataset_path)
    logger.info("wrote dataset %s records=%d", dataset_path, len(records))

    # Write manifest.json
    manifest = IntervalManifest(
        protocol_version=PROTOCOL_VERSION,
        schema_version=SCHEMA_VERSION,
        spec_id=DEFAULT_SPEC_ID,
        netuid=netuid,
        miner_hotkey=miner_hotkey,
        interval_id=interval_id,
        record_count=len(records),
        dataset_sha256=sha256_file(dataset_path),
    )
    manifest_path = out_dir / "manifest.json"
    write_manifest(manifest, manifest_path)
    logger.info("wrote manifest %s", manifest_path)

    return dataset_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset from sources.txt with timestamps")
    parser.add_argument("--sources", default="sources.txt", help="Path to sources.txt")
    parser.add_argument("--workdir", default=".nexis_dataset", help="Working directory")
    parser.add_argument("--interval-id", type=int, default=1, help="Interval ID")
    parser.add_argument("--netuid", type=int, default=70, help="NetUID")
    parser.add_argument("--miner-hotkey", default="test_miner", help="Miner hotkey")
    parser.add_argument("--no-caption", action="store_true", help="Skip caption generation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    workdir = Path(args.workdir).absolute()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    captioner = None if args.no_caption else Captioner()

    dataset_path, manifest_path = asyncio.run(
        generate_dataset(
            sources_file=Path(args.sources),
            workdir=workdir,
            interval_id=args.interval_id,
            netuid=args.netuid,
            miner_hotkey=args.miner_hotkey,
            captioner=captioner,
        )
    )

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print(f"  Dataset:  {dataset_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Workdir:  {workdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
