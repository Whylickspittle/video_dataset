"""Miner interval pipeline: build a 400-sample dataset and upload it."""
"""
矿工核心流水线：从 sources.txt 读取视频源，生成 400 段切片数据集并上传。

整个流程：
1. 读取 sources.txt 获取视频 URL 列表
2. 逐个下载视频 → 探测时长/分辨率
3. 按 5.04 秒步长逐段切片（同时做去重保护）
4. 每段提取首帧、生成 caption、计算 SHA256
5. 写入 dataset.parquet + manifest.json
6. 上传到 R2/S3（parquet + clips + frames + manifest）
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..hash_utils import deterministic_clip_id, sha256_file
from ..models import ClipRecord, IntervalManifest
from ..protocol import (
    CLIP_DURATION_SEC,
    OVERLAP_WINDOW_SEC,
    PROTOCOL_VERSION,
    SAMPLE_COUNT,
    SCHEMA_VERSION,
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_NUM_FRAMES,
    TARGET_WIDTH,
)
from ..serialization import write_dataset_parquet, write_manifest
from ..specs import DEFAULT_SPEC_ID
from .captioner import Captioner
from .providers import GenericSourceProvider, SourceProvider

logger = logging.getLogger(__name__)


def _video_stream(info: dict[str, Any]) -> dict[str, Any]:
    """从 ffprobe 返回的 JSON 中提取视频流信息。"""
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise ValueError("video stream not found")


def _canonical_url(url: str) -> str:
    """
    将 YouTube URL 归一化为标准格式。

    支持的输入格式：
    - https://youtu.be/ABC123
    - https://www.youtube.com/watch?v=ABC123
    - https://www.youtube.com/shorts/ABC123
    - https://www.youtube.com/embed/ABC123

    输出统一为：
    - https://www.youtube.com/watch?v=ABC123

    非 YouTube URL 原样返回（但验证者可能拒绝非 YouTube 来源）。
    """
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        video_id = parsed.path.strip("/")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    if host == "youtube.com" or host.endswith(".youtube.com"):
        query = parse_qs(parsed.query)
        values = query.get("v", [])
        if values and values[0].strip():
            return f"https://www.youtube.com/watch?v={values[0].strip()}"
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "v"} and parts[1].strip():
            return f"https://www.youtube.com/watch?v={parts[1].strip()}"
    return url.strip()


class MinerPipeline:
    """Build a strict-spec 400-sample dataset from public video sources."""
    """
    矿工核心流水线。

    负责从公开视频源构建符合严格规格（1280x704, 24fps, 121帧）的 400 段数据集，
    并上传到存储。
    """

    def __init__(
        self,
        store: Any,                              # R2/S3 存储客户端
        source_provider: SourceProvider | None = None,  # 视频源提供者（默认 yt-dlp）
        spec_id: str = DEFAULT_SPEC_ID,          # 数据规格 ID
        sample_count: int = SAMPLE_COUNT,        # 目标样本数（默认 400）
        captioner: Captioner | None = None,      # Caption 生成器（OpenAI/Gemini）
    ):
        self.store = store
        self.source_provider = source_provider or GenericSourceProvider()
        self.spec_id = spec_id
        self.sample_count = sample_count
        # 如果没有提供 captioner 或没有 API Key，则使用空 captioner（生成空字符串）。
        # 此时 Trainer 会回退到默认 prompt。
        self.captioner = captioner or Captioner()

    async def run_interval(
        self,
        *,
        sources_file: Path,          # sources.txt 路径
        netuid: int,                 # Bittensor 网络 ID
        miner_hotkey: str,           # 矿工 SS58 地址
        interval_id: int,            # Interval 编号（从 1 开始）
        workdir: Path,               # 本地工作目录
    ) -> tuple[Path, Path]:
        """
        执行一次完整的 interval 生成流程。

        返回 (dataset_path, manifest_path)。
        """
        if interval_id < 1:
            raise ValueError("interval_id must be >= 1")
        logger.info(
            "miner pipeline start interval_id=%d hotkey=%s sample_count=%d",
            interval_id,
            miner_hotkey,
            self.sample_count,
        )
        workdir.mkdir(parents=True, exist_ok=True)
        raw_dir = workdir / "raw"           # 下载的完整视频存放处
        clips_dir = workdir / "clips"       # 切片后的 mp4 存放处
        frames_dir = workdir / "frames"     # 首帧 jpg 存放处
        out_dir = workdir / "out" / str(interval_id)  # 输出目录（dataset + manifest）
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        records: list[ClipRecord] = []
        assets_to_upload: dict[str, Path] = {}
        # 数据集内去重保护：记录每个 canonical URL 已使用的起始时间点
        seen_positions: dict[str, list[float]] = {}

        # 读取 sources.txt
        urls = list(self.source_provider.read_sources(sources_file))
        if not urls:
            raise RuntimeError(f"no sources defined in {sources_file}")

        # 逐个处理视频源
        for url in urls:
            if len(records) >= self.sample_count:
                break
            canonical = _canonical_url(url)
            source_id = self.source_provider.source_video_id(url)
            logger.info("processing source source_id=%s url=%s", source_id, url)
            try:
                raw_path = self.source_provider.download(url, raw_dir)
            except Exception as exc:
                logger.warning("source download failed url=%s err=%s", url, exc)
                continue

            try:
                probe = self.source_provider.probe(raw_path)
            except Exception as exc:
                logger.warning("source probe failed path=%s err=%s", raw_path, exc)
                continue

            # 计算视频可切出的段数
            duration = float(probe.get("format", {}).get("duration") or 0.0)
            total_segments = int(math.floor(duration / CLIP_DURATION_SEC))
            if total_segments <= 0:
                logger.warning("source has no usable segments source_id=%s", source_id)
                continue

            # 检查分辨率是否达标（>= 1280x704）
            stream = _video_stream(probe)
            src_width = int(stream.get("width", 0) or 0)
            src_height = int(stream.get("height", 0) or 0)
            if src_width < TARGET_WIDTH or src_height < TARGET_HEIGHT:
                logger.warning(
                    "source resolution below target (got %dx%d, need >= %dx%d) source_id=%s",
                    src_width,
                    src_height,
                    TARGET_WIDTH,
                    TARGET_HEIGHT,
                    source_id,
                )
                continue

            # 逐段切片
            for idx in range(total_segments):
                if len(records) >= self.sample_count:
                    break
                start = float(idx) * CLIP_DURATION_SEC

                # 数据集内去重保护：同一视频内，起始时间差 < 4.5 秒则跳过
                positions = seen_positions.setdefault(canonical, [])
                if any(abs(start - prev) < OVERLAP_WINDOW_SEC for prev in positions):
                    continue

                clip_id = deterministic_clip_id(source_id, start, CLIP_DURATION_SEC)
                clip_path = clips_dir / f"{clip_id}.mp4"
                frame_path = frames_dir / f"{clip_id}.jpg"
                try:
                    self.source_provider.create_clip(raw_path, clip_path, start)
                    self.source_provider.extract_first_frame(clip_path, frame_path)
                except Exception as exc:
                    logger.warning(
                        "clip extraction failed source_id=%s start=%.3f err=%s",
                        source_id,
                        start,
                        exc,
                    )
                    continue

                positions.append(start)
                # 生成 caption（如果 captioner 启用）
                caption = self.captioner.caption_frame(frame_path) if self.captioner.enabled else ""
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
                    source_video_url=canonical,
                )
                records.append(record)
                assets_to_upload[record.clip_uri] = clip_path
                assets_to_upload[record.first_frame_uri] = frame_path

        # 凑不满 400 条则报错
        if len(records) != self.sample_count:
            raise RuntimeError(
                f"failed to produce {self.sample_count} samples from sources "
                f"(got {len(records)}); add more URLs to sources.txt"
            )

        # 写入 dataset.parquet
        dataset_path = out_dir / "dataset.parquet"
        write_dataset_parquet(records, dataset_path)
        logger.info("dataset written interval=%d records=%d", interval_id, len(records))

        # 写入 manifest.json
        manifest = IntervalManifest(
            protocol_version=PROTOCOL_VERSION,
            schema_version=SCHEMA_VERSION,
            spec_id=self.spec_id,
            netuid=netuid,
            miner_hotkey=miner_hotkey,
            interval_id=interval_id,
            record_count=len(records),
            dataset_sha256=sha256_file(dataset_path),
        )
        manifest_path = out_dir / "manifest.json"
        write_manifest(manifest, manifest_path)

        # ── 上传到 R2/S3 ──
        base_key = f"{interval_id}"
        await self.store.upload_file(f"{base_key}/dataset.parquet", dataset_path, use_write=True)
        for relative_uri, local_path in assets_to_upload.items():
            await self.store.upload_file(
                f"{base_key}/{relative_uri.lstrip('/')}",
                local_path,
                use_write=True,
            )
        # Manifest 最后上传：作为"上传完成"的信号
        await self.store.upload_file(f"{base_key}/manifest.json", manifest_path, use_write=True)
        logger.info("uploaded interval package hotkey=%s interval=%d", miner_hotkey, interval_id)
        return dataset_path, manifest_path
