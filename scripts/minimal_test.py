#!/usr/bin/env python3
"""最小单元化数据集测试：只测1个视频 → 切3~5条 → 快速验证。

用法:
    # 测试 sources.txt 中的第 1 个视频（默认切 5 条）
    python3 scripts/minimal_test.py

    # 测试第 3 个视频
    python3 scripts/minimal_test.py --index 2

    # 只切 3 条（更快）
    python3 scripts/minimal_test.py --clips 3

    # 测试指定的单个 URL
    python3 scripts/minimal_test.py --url "https://www.youtube.com/watch?v=xxx" --offset 120

    # 跳过下载（本地已有视频）
    python3 scripts/minimal_test.py --local-video /path/to/video.mp4

输出:
    每个步骤的详细日志 + 最终验证报告
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# 把 repo 根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexis.hash_utils import sha256_file
from nexis.miner.captioner import Captioner
from nexis.miner.pipeline import _canonical_url, _video_stream
from nexis.miner.providers import GenericSourceProvider
from nexis.miner.youtube import (
    _parse_time_offset,
    create_clip,
    download_source_video,
    extract_first_frame,
    probe_video,
    read_sources,
)
from nexis.models import ClipRecord
from nexis.protocol import (
    CLIP_DURATION_SEC,
    FPS_TOLERANCE,
    OVERLAP_WINDOW_SEC,
    SAMPLE_COUNT,
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_NUM_FRAMES,
    TARGET_WIDTH,
)
from nexis.validator.dataset_check import (
    _check_record_specs,
    _count_global_overlap,
    _ffprobe_metadata,
    _within_dataset_overlap,
    canonical_source_key,
)


def log_step(step: str, msg: str, *, ok: bool | None = None) -> None:
    """打印格式化的步骤日志。"""
    icon = "✅" if ok is True else "❌" if ok is False else "⏳"
    print(f"{icon}  [{step:12s}] {msg}")


def check_video_specs(path: Path) -> dict[str, Any]:
    """用 ffprobe 检查单个 clip 的实际规格。"""
    try:
        info = probe_video(path)
    except Exception as exc:
        return {"ok": False, "error": f"ffprobe failed: {exc}"}

    stream: dict[str, Any] | None = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            stream = s
            break
    if stream is None:
        return {"ok": False, "error": "no video stream"}

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    r_frame_rate = stream.get("r_frame_rate", "0/1")
    try:
        num, den = r_frame_rate.split("/")
        fps = float(num) / max(float(den), 1.0)
    except Exception:
        fps = 0.0
    nb_frames = stream.get("nb_frames")
    try:
        num_frames = int(nb_frames) if nb_frames is not None else 0
    except Exception:
        num_frames = 0
    if num_frames <= 0:
        duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
        num_frames = int(round(duration * fps)) if duration > 0 else 0

    # 验证各规格
    issues: list[str] = []
    if width != TARGET_WIDTH:
        issues.append(f"width={width} (expected {TARGET_WIDTH})")
    if height != TARGET_HEIGHT:
        issues.append(f"height={height} (expected {TARGET_HEIGHT})")
    if abs(fps - TARGET_FPS) > FPS_TOLERANCE:
        issues.append(f"fps={fps:.3f} (expected {TARGET_FPS}±{FPS_TOLERANCE})")
    if num_frames != TARGET_NUM_FRAMES and abs(num_frames - TARGET_NUM_FRAMES) > 1:
        issues.append(f"num_frames={num_frames} (expected {TARGET_NUM_FRAMES})")

    return {
        "ok": len(issues) == 0,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "num_frames": num_frames,
        "issues": issues,
    }


def build_test_record(clip_path: Path, frame_path: Path, url: str, start_sec: float) -> ClipRecord:
    """从本地文件构建一个 ClipRecord（用于验证）。"""
    # 简单生成 clip_id（实际 pipeline 用 deterministic_clip_id）
    source_id = GenericSourceProvider().source_video_id(url)
    clip_id = hashlib.sha256(f"{source_id}:{start_sec}".encode()).hexdigest()[:16]

    # 读取 metadata
    info = probe_video(clip_path)
    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or 0.0)

    return ClipRecord(
        clip_id=clip_id,
        clip_uri=f"clips/{clip_path.name}",
        clip_sha256=sha256_file(clip_path),
        first_frame_uri=f"frames/{frame_path.name}",
        first_frame_sha256=sha256_file(frame_path),
        source_video_id=source_id,
        clip_start_sec=start_sec,
        duration_sec=duration,
        caption="test caption",
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
        fps=float(TARGET_FPS),
        num_frames=TARGET_NUM_FRAMES,
        source_video_url=_canonical_url(url),
    )


def run_minimal_test(
    url: str,
    time_offset: float,
    max_clips: int,
    workdir: Path,
    captioner: Captioner | None = None,
    *,
    raw_path: Path | None = None,
) -> dict[str, Any]:
    """运行最小单元测试，返回完整报告。

    如果提供了 ``raw_path``，则跳过下载步骤，直接使用该本地文件。
    """
    report: dict[str, Any] = {
        "url": url,
        "time_offset": time_offset,
        "max_clips": max_clips,
        "steps": [],
        "records": [],
        "final": {"ok": False, "issues": []},
    }

    raw_dir = workdir / "raw"
    clips_dir = workdir / "clips"
    frames_dir = workdir / "frames"
    for d in (raw_dir, clips_dir, frames_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 下载（或直接使用本地文件）─
    if raw_path is not None:
        log_step("Download", f"using local file: {raw_path}", ok=True)
        report["steps"].append({"name": "download", "ok": True, "path": str(raw_path), "local": True})
    else:
        log_step("Download", f"url={url[:60]}... offset={time_offset:.1f}s")
        try:
            raw_path = download_source_video(url, raw_dir, start_sec=time_offset)
            log_step("Download", f"saved to {raw_path}", ok=True)
            report["steps"].append({"name": "download", "ok": True, "path": str(raw_path)})
        except Exception as exc:
            log_step("Download", f"failed: {exc}", ok=False)
            report["steps"].append({"name": "download", "ok": False, "error": str(exc)})
            return report

    # ── Step 2: Probe ──
    log_step("Probe", f"path={raw_path.name}")
    try:
        probe = probe_video(raw_path)
        full_duration = float(probe.get("format", {}).get("duration") or 0.0)
        stream = _video_stream(probe)
        src_width = int(stream.get("width", 0) or 0)
        src_height = int(stream.get("height", 0) or 0)
        log_step(
            "Probe",
            f"duration={full_duration:.1f}s resolution={src_width}x{src_height}",
            ok=True,
        )
        report["steps"].append({
            "name": "probe",
            "ok": True,
            "duration": full_duration,
            "width": src_width,
            "height": src_height,
        })

        # 分辨率预检
        if src_width < TARGET_WIDTH or src_height < TARGET_HEIGHT:
            log_step(
                "Resolution",
                f"source ({src_width}x{src_height}) < target ({TARGET_WIDTH}x{TARGET_HEIGHT})",
                ok=False,
            )
            report["steps"].append({"name": "resolution_check", "ok": False})
            report["final"]["issues"].append(f"source resolution too low: {src_width}x{src_height}")
            return report
        else:
            log_step("Resolution", f"{src_width}x{src_height} >= {TARGET_WIDTH}x{TARGET_HEIGHT}", ok=True)
            report["steps"].append({"name": "resolution_check", "ok": True})
    except Exception as exc:
        log_step("Probe", f"failed: {exc}", ok=False)
        report["steps"].append({"name": "probe", "ok": False, "error": str(exc)})
        return report

    # ── Step 3: 计算可切段数 ──
    available_duration = max(0.0, full_duration - time_offset)
    total_segments = int(math.floor(available_duration / CLIP_DURATION_SEC))
    log_step("Segments", f"available={available_duration:.1f}s total_segments={total_segments}")
    if total_segments <= 0:
        log_step("Segments", "no usable segments", ok=False)
        report["final"]["issues"].append("no usable segments after offset")
        return report

    clips_to_make = min(max_clips, total_segments)
    log_step("Plan", f"will create {clips_to_make} clips (requested {max_clips})")

    # ── Step 4: 切片 ──
    records: list[ClipRecord] = []
    seen_positions: list[float] = []
    for idx in range(total_segments):
        if len(records) >= clips_to_make:
            break
        start = time_offset + float(idx) * CLIP_DURATION_SEC

        # 去重保护
        if any(abs(start - prev) < OVERLAP_WINDOW_SEC for prev in seen_positions):
            continue

        source_id = GenericSourceProvider().source_video_id(url)
        clip_id = f"{source_id}_{idx:04d}"
        clip_path = clips_dir / f"{clip_id}.mp4"
        frame_path = frames_dir / f"{clip_id}.jpg"

        try:
            create_clip(raw_path, clip_path, start)
            extract_first_frame(clip_path, frame_path)
        except Exception as exc:
            log_step("Slice", f"idx={idx} start={start:.1f}s failed: {exc}", ok=False)
            continue

        seen_positions.append(start)

        # 生成 caption（可选）
        caption = ""
        if captioner and captioner.enabled:
            try:
                caption = captioner.caption_frame(frame_path)
            except Exception as exc:
                log_step("Caption", f"failed: {exc}")

        record = ClipRecord(
            clip_id=clip_id,
            clip_uri=f"clips/{clip_path.name}",
            clip_sha256=sha256_file(clip_path),
            first_frame_uri=f"frames/{frame_path.name}",
            first_frame_sha256=sha256_file(frame_path),
            source_video_id=source_id,
            clip_start_sec=start,
            duration_sec=CLIP_DURATION_SEC,
            caption=caption,
            width=TARGET_WIDTH,
            height=TARGET_HEIGHT,
            fps=float(TARGET_FPS),
            num_frames=TARGET_NUM_FRAMES,
            source_video_url=_canonical_url(url),
        )
        records.append(record)
        log_step("Slice", f"{clip_id} start={start:.1f}s caption={caption[:30]!r}...", ok=True)

    report["records"] = [{"clip_id": r.clip_id, "start": r.clip_start_sec} for r in records]

    if not records:
        report["final"]["issues"].append("no clips were successfully created")
        return report

    # ── Step 5: 逐条验证（模拟 validator 的检查）─
    log_step("Validate", f"checking {len(records)} clips...")
    all_ok = True
    for rec in records:
        # clip_uri 形如 "clips/xxx.mp4"，取 basename 后在 clips_dir 下查找
        clip_path = clips_dir / Path(rec.clip_uri).name
        frame_path = frames_dir / Path(rec.first_frame_uri).name

        # 5a: SHA256
        if sha256_file(clip_path) != rec.clip_sha256:
            log_step("SHA256", f"{rec.clip_id} mismatch", ok=False)
            all_ok = False

        # 5b: ffprobe 实际规格
        specs = check_video_specs(clip_path)
        if not specs["ok"]:
            log_step("Specs", f"{rec.clip_id}: {', '.join(specs['issues'])}", ok=False)
            all_ok = False
        else:
            log_step(
                "Specs",
                f"{rec.clip_id}: {specs['width']}x{specs['height']} {specs['fps']}fps {specs['num_frames']}frames",
                ok=True,
            )

        # 5c: 首帧存在
        if not frame_path.exists():
            log_step("Frame", f"{rec.clip_id}: first frame missing", ok=False)
            all_ok = False

        # 5d: Caption 非空（仅当配置了 captioner 时检查）
        if captioner and captioner.enabled and not rec.caption.strip():
            log_step("Caption", f"{rec.clip_id}: empty caption", ok=False)
            all_ok = False

    # ── Step 6: 数据集级检查 ──
    # 6a: 内部重叠
    overlap_reason = _within_dataset_overlap(records)
    if overlap_reason:
        log_step("Overlap", f"within-dataset overlap detected: {overlap_reason}", ok=False)
        report["final"]["issues"].append(overlap_reason)
        all_ok = False
    else:
        log_step("Overlap", "no within-dataset overlap", ok=True)

    # 6b: record specs（metadata 级别）
    for rec in records:
        reason = _check_record_specs(rec)
        if reason:
            log_step("MetaSpecs", f"{rec.clip_id}: {reason}", ok=False)
            report["final"]["issues"].append(reason)
            all_ok = False

    report["final"]["ok"] = all_ok and len(report["final"]["issues"]) == 0
    return report


def test_local_video(
    video_path: Path,
    max_clips: int,
    workdir: Path,
    captioner: Captioner | None = None,
) -> dict[str, Any]:
    """测试本地已有的视频文件。"""
    log_step("Local", f"testing local video: {video_path}")

    raw_dir = workdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 复制到 raw_dir
    dest = raw_dir / video_path.name
    shutil.copy2(video_path, dest)

    # 伪造一个干净的 URL（用于 canonical_source_key / source_video_id）
    fake_url = f"https://example.com/video/{video_path.stem}"

    return run_minimal_test(fake_url, 0.0, max_clips, workdir, captioner=captioner, raw_path=dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal dataset unit test")
    parser.add_argument("--sources", default="sources.txt", help="Path to sources.txt")
    parser.add_argument("--index", type=int, default=0, help="Index of source to test (0-based)")
    parser.add_argument("--clips", type=int, default=5, help="Max clips to create")
    parser.add_argument("--url", default="", help="Direct URL to test (overrides --sources)")
    parser.add_argument("--offset", type=str, default="0", help="Start offset (e.g. 30, 1:23, 1:23:20)")
    parser.add_argument("--local-video", default="", help="Test a local video file instead of downloading")
    parser.add_argument("--workdir", default=".nexis_test", help="Working directory")
    parser.add_argument("--keep", action="store_true", help="Keep workdir after test")
    parser.add_argument("--no-caption", action="store_true", help="Skip caption generation")
    args = parser.parse_args()

    workdir = Path(args.workdir).absolute()

    # 清理旧的工作目录（除非 --keep）
    if workdir.exists() and not args.keep:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Minimal Dataset Unit Test")
    print("=" * 70)
    print(f"workdir: {workdir}")
    print()

    # 确定要测试的 URL 和偏移
    captioner = None if args.no_caption else Captioner()

    if args.local_video:
        report = test_local_video(Path(args.local_video), args.clips, workdir, captioner=captioner)
    elif args.url:
        url = args.url
        offset = _parse_time_offset(args.offset)
        report = run_minimal_test(url, offset, args.clips, workdir, captioner=captioner)
    else:
        sources = read_sources(Path(args.sources))
        if not sources:
            print(f"ERROR: no sources found in {args.sources}")
            sys.exit(1)
        if args.index >= len(sources):
            print(f"ERROR: index {args.index} out of range (total {len(sources)})")
            sys.exit(1)
        url, offset = sources[args.index]
        report = run_minimal_test(url, offset, args.clips, workdir, captioner=captioner)

    # 打印最终报告
    print()
    print("=" * 70)
    print("  Final Report")
    print("=" * 70)
    print(f"URL:        {report['url']}")
    print(f"Offset:     {report['time_offset']:.1f}s")
    print(f"Max clips:  {report['max_clips']}")
    print(f"Created:    {len(report['records'])}")
    print()

    if report["final"]["ok"]:
        print("✅  ALL CHECKS PASSED")
        print("    Your video source is compatible with the validator requirements.")
    else:
        print("❌  CHECKS FAILED")
        for issue in report["final"]["issues"]:
            print(f"    - {issue}")

    # 写入 JSON 报告
    report_path = workdir / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nDetailed report saved to: {report_path}")

    if not args.keep:
        shutil.rmtree(workdir)
        print(f"Cleaned up: {workdir}")
    else:
        print(f"Kept workdir: {workdir}")

    sys.exit(0 if report["final"]["ok"] else 1)


if __name__ == "__main__":
    main()
