"""Strict dataset validation for v2 protocol (spec + global overlap)."""
"""
严格的数据集验证模块（v2 协议）。

本模块是验证者最核心的逻辑，负责对矿工提交的数据集进行逐项检查：
1. Manifest 完整性校验
2. Parquet 数据解析与规格校验
3. 数据集内部去重
4. Caption 非空检查
5. 资产下载与 SHA256 + ffprobe 验证
6. 全局去重（与历史 Top-5 数据比对）

任何一项硬性检查失败，整个数据集立即被拒绝。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..hash_utils import sha256_file
from ..miner.youtube import probe_video
from ..models import ClipRecord, IntervalManifest
from ..protocol import (
    CLIP_DURATION_SEC,
    CLIP_DURATION_TOLERANCE_SEC,
    FPS_TOLERANCE,
    GLOBAL_OVERLAP_REJECT_THRESHOLD,
    OVERLAP_WINDOW_SEC,
    SAMPLE_COUNT,
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_NUM_FRAMES,
    TARGET_WIDTH,
)
from ..serialization import read_dataset_parquet, read_manifest

logger = logging.getLogger(__name__)


@dataclass
class DatasetCheckOutcome:
    """单次数据集验证的结果。"""
    accepted: bool                # 是否被接受
    miner_hotkey: str             # 矿工地址
    interval_id: int              # Interval 编号
    record_count: int = 0         # 有效记录数
    global_overlap_count: int = 0 # 全局重叠数
    failures: list[str] = field(default_factory=list)  # 失败原因列表
    notes: dict[str, Any] = field(default_factory=dict)  # 额外备注


def canonical_source_key(url: str) -> str:
    """
    将 URL 归一化为 canonical 形式，用于去重比对。

    YouTube 的各种短链接、embed、shorts 都会统一为标准 watch?v= 格式。
    """
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    # 处理 youtu.be 短链接：从 path 中提取 video_id
    if host == "youtu.be":
        video_id = parsed.path.strip("/")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    # 处理 youtube.com 及其子域名
    if host == "youtube.com" or host.endswith(".youtube.com"):
        # 优先从查询参数 v 中提取视频 ID
        query = parse_qs(parsed.query)
        values = query.get("v", [])
        if values and values[0].strip():
            return f"https://www.youtube.com/watch?v={values[0].strip()}"
        # 处理 /shorts/、/embed/、/v/ 路径格式
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "v"} and parts[1].strip():
            return f"https://www.youtube.com/watch?v={parts[1].strip()}"
    # 非 YouTube 链接保持原样
    return url.strip()


def _check_record_specs(record: ClipRecord) -> str | None:
    """
    检查单条 ClipRecord 是否符合协议规格。

    返回 None 表示通过，返回字符串表示失败原因。
    """
    # 视频宽度必须严格等于目标宽度
    if record.width != TARGET_WIDTH:
        return f"width:{record.width}!={TARGET_WIDTH}"
    # 视频高度必须严格等于目标高度
    if record.height != TARGET_HEIGHT:
        return f"height:{record.height}!={TARGET_HEIGHT}"
    # 帧率必须在容差范围内
    if abs(record.fps - TARGET_FPS) > FPS_TOLERANCE:
        return f"fps:{record.fps}!={TARGET_FPS}"
    # 帧数必须严格匹配
    if record.num_frames != TARGET_NUM_FRAMES:
        return f"num_frames:{record.num_frames}!={TARGET_NUM_FRAMES}"
    # 时长必须在容差范围内
    if abs(record.duration_sec - CLIP_DURATION_SEC) > CLIP_DURATION_TOLERANCE_SEC:
        return f"duration_sec:{record.duration_sec:.3f}"
    return None


def _within_dataset_overlap(records: list[ClipRecord]) -> str | None:
    """
    数据集内部去重检查。

    规则：同一 canonical_url 下，任意两条记录的起始时间差 < 4.5 秒即视为重叠。
    返回失败原因字符串，None 表示通过。
    """
    seen: dict[str, list[float]] = {}
    # 遍历所有记录，按 canonical_url 分组检查
    for row in records:
        key = canonical_source_key(row.source_video_url)
        positions = seen.setdefault(key, [])
        # 若同一视频下已有时间差小于 4.5s 的记录，判定为内部重叠
        if any(abs(row.clip_start_sec - prev) < OVERLAP_WINDOW_SEC for prev in positions):
            return f"within_dataset_overlap:{row.clip_id}"
        positions.append(row.clip_start_sec)
    return None


def _count_global_overlap(
    records: list[ClipRecord],
    global_record_index: dict[str, list[float]],
) -> int:
    """
    计算矿工数据集与全局历史索引的重叠数量。

    global_record_index 来自 record_info.json，只包含历史 Top-5 矿工的数据。
    """
    if not global_record_index:
        return 0
    count = 0
    for row in records:
        key = canonical_source_key(row.source_video_url)
        positions = global_record_index.get(key)
        if not positions:
            continue
        if any(abs(row.clip_start_sec - prev) < OVERLAP_WINDOW_SEC for prev in positions):
            count += 1
    return count


def build_overlap_index(records: list[ClipRecord]) -> dict[str, list[float]]:
    """
    从记录列表构建重叠索引：{canonical_url: [start_sec, ...]}。

    用于 cross-miner 去重比对。
    """
    index: dict[str, list[float]] = {}
    for row in records:
        index.setdefault(canonical_source_key(row.source_video_url), []).append(
            row.clip_start_sec
        )
    return index


def count_index_overlap(
    a: dict[str, list[float]],
    b: dict[str, list[float]],
) -> int:
    """
    计算两个重叠索引之间的重叠条数。

    规则：同一 canonical_url 下，任意两个 start_sec 差 < 4.5 秒即算一条重叠。
    为了效率，总是遍历较小的那一侧。
    """
    # 任一侧为空则不可能重叠
    if not a or not b:
        return 0
    # 遍历较小的一侧以降低计算量
    if sum(len(v) for v in a.values()) > sum(len(v) for v in b.values()):
        a, b = b, a
    count = 0
    # 遍历较小索引的每条记录
    for key, positions in a.items():
        other = b.get(key)
        if not other:
            continue
        for p in positions:
            if any(abs(p - q) < OVERLAP_WINDOW_SEC for q in other):
                count += 1
    return count


def _ffprobe_metadata(path: Path) -> tuple[int, int, float, int]:
    """
    用 ffprobe 探测视频文件，返回 (width, height, fps, num_frames)。

    如果 nb_frames 不可用，则通过 duration * fps 反推。
    """
    info = probe_video(path)
    stream: dict[str, Any] | None = None
    # 在所有流中查找第一个视频流
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            stream = s
            break
    if stream is None:
        raise ValueError("no video stream")
    # 提取视频分辨率
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    # 提取并解析帧率字符串（格式通常为 "num/den"）
    r_frame_rate = stream.get("r_frame_rate", "0/1")
    try:
        num, den = r_frame_rate.split("/")
        fps = float(num) / max(float(den), 1.0)
    except Exception:
        fps = 0.0
    # 尝试提取总帧数
    nb_frames = stream.get("nb_frames")
    try:
        num_frames = int(nb_frames) if nb_frames is not None else 0
    except Exception:
        num_frames = 0
    # 如果未获取到有效帧数，通过时长和帧率估算
    if num_frames <= 0:
        # 兜底：通过时长 × 帧率反推帧数
        duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
        num_frames = int(round(duration * fps)) if duration > 0 else 0
    return width, height, fps, num_frames


# ── 下载重试逻辑 ──
_DOWNLOAD_BACKOFF_BASE_SEC = 1.0


async def _download_with_retry(
    miner_store: Any,
    key: str,
    dst: Path,
    *,
    max_attempts: int = 3,
) -> bool:
    """
    带指数退避的下载重试。

    R2/S3 下载可能因网络波动失败，重试可以恢复大部分临时故障。
    注意：无法区分"文件不存在"和"网络故障"，所以对真实 404 也会浪费几次重试。
    """
    attempts = max(1, int(max_attempts))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if await miner_store.download_file(key, dst):
                if attempt > 1:
                    logger.info(
                        "download recovered key=%s on attempt %d/%d",
                        key,
                        attempt,
                        attempts,
                    )
                return True
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "download exception key=%s attempt=%d/%d err=%s",
                key,
                attempt,
                attempts,
                exc,
            )
        if attempt < attempts:
            backoff = _DOWNLOAD_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)
    if last_exc is None:
        logger.warning("download failed key=%s after %d attempts", key, attempts)
    return False


async def _download_with_sem(
    sem: asyncio.Semaphore,
    miner_store: Any,
    key: str,
    dst: Path,
    *,
    max_attempts: int = 3,
) -> bool:
    """在信号量限制下执行下载（控制并发数）。"""
    async with sem:
        return await _download_with_retry(
            miner_store, key, dst, max_attempts=max_attempts
        )


async def validate_miner_dataset(
    *,
    miner_hotkey: str,
    interval_id: int,
    miner_store: Any,
    workdir: Path,
    global_record_index: dict[str, list[float]] | None = None,
    download_concurrency: int = 16,
    download_retry_attempts: int = 3,
) -> DatasetCheckOutcome:
    """
    验证矿工数据集的主函数。

    检查流程（严格顺序，任何一步失败立即返回）：
    1. Manifest 下载与解析
    2. dataset.parquet 下载与 SHA256 校验
    3. 逐行规格检查
    4. 数据集内部去重
    5. Caption 非空检查
    6. 资产并发下载 + SHA256 + ffprobe 验证
    7. 全局去重
    """
    out = DatasetCheckOutcome(
        accepted=False,
        miner_hotkey=miner_hotkey,
        interval_id=interval_id,
    )
    miner_dir = workdir / miner_hotkey / str(interval_id)
    miner_dir.mkdir(parents=True, exist_ok=True)
    base_key = f"{interval_id}"

    # ── Step 1: Manifest ──
    manifest_local = miner_dir / "manifest.json"
    ok = await _download_with_retry(
        miner_store,
        f"{base_key}/manifest.json",
        manifest_local,
        max_attempts=download_retry_attempts,
    )
    if not ok or not manifest_local.exists():
        out.failures.append("manifest_missing")
        return out
    try:
        manifest = read_manifest(manifest_local)
    except Exception as exc:
        out.failures.append(f"manifest_parse_error:{exc}")
        return out
    if manifest.miner_hotkey.strip() != miner_hotkey:
        out.failures.append("manifest_hotkey_mismatch")
        return out
    if manifest.interval_id != interval_id:
        out.failures.append("manifest_interval_mismatch")
        return out
    if manifest.record_count != SAMPLE_COUNT:
        out.failures.append(f"record_count:{manifest.record_count}!={SAMPLE_COUNT}")
        return out

    # ── Step 2: dataset.parquet ──
    dataset_local = miner_dir / "dataset.parquet"
    ok = await _download_with_retry(
        miner_store,
        f"{base_key}/dataset.parquet",
        dataset_local,
        max_attempts=download_retry_attempts,
    )
    if not ok or not dataset_local.exists():
        out.failures.append("dataset_missing")
        return out
    if sha256_file(dataset_local) != manifest.dataset_sha256:
        out.failures.append("dataset_sha256_mismatch")
        return out
    try:
        records = read_dataset_parquet(dataset_local)
    except Exception as exc:
        out.failures.append(f"dataset_parse_error:{exc}")
        return out
    if len(records) != SAMPLE_COUNT:
        out.failures.append(f"records_len:{len(records)}!={SAMPLE_COUNT}")
        return out

    # ── Step 3: 逐行规格检查 ──
    for row in records:
        reason = _check_record_specs(row)
        if reason is not None:
            out.failures.append(f"spec:{reason}")
            return out

    # ── Step 4: 数据集内部去重 ──
    overlap_reason = _within_dataset_overlap(records)
    if overlap_reason is not None:
        out.failures.append(overlap_reason)
        return out

    # ── Step 5: Caption 非空检查 ──
    # Trainer 需要每段都有一个 prompt；空 caption 会导致训练失败。
    for row in records:
        if not (getattr(row, "caption", "") or "").strip():
            out.failures.append(f"caption_missing:{row.clip_id}")
            return out

    # ── Step 6: 并发下载全部 800 个资产（400 clips + 400 frames），然后逐条验证 ──
    # 用信号量限制最大并发下载数，防止网络拥塞
    sem = asyncio.Semaphore(max(int(download_concurrency), 1))
    download_specs: list[tuple[ClipRecord, Path, Path]] = []
    download_tasks: list[asyncio.Task[bool]] = []
    # 为每条记录创建 clip 和 frame 的下载任务
    for row in records:
        clip_uri = row.clip_uri.lstrip("/")
        frame_uri = row.first_frame_uri.lstrip("/")
        clip_local = miner_dir / clip_uri
        frame_local = miner_dir / frame_uri
        download_specs.append((row, clip_local, frame_local))
        download_tasks.append(
            asyncio.create_task(
                _download_with_sem(
                    sem,
                    miner_store,
                    f"{base_key}/{clip_uri}",
                    clip_local,
                    max_attempts=download_retry_attempts,
                )
            )
        )
        download_tasks.append(
            asyncio.create_task(
                _download_with_sem(
                    sem,
                    miner_store,
                    f"{base_key}/{frame_uri}",
                    frame_local,
                    max_attempts=download_retry_attempts,
                )
            )
        )
    download_results = await asyncio.gather(*download_tasks)

    # 按顺序提取每条记录的 clip 和 frame 下载结果，逐条验证
    for idx, (row, clip_local, frame_local) in enumerate(download_specs):
        clip_ok = download_results[idx * 2]
        frame_ok = download_results[idx * 2 + 1]
        if not clip_ok or not clip_local.exists():
            out.failures.append(f"clip_missing:{row.clip_id}")
            return out
        if sha256_file(clip_local) != row.clip_sha256:
            out.failures.append(f"clip_sha256_mismatch:{row.clip_id}")
            return out
        try:
            width, height, fps, num_frames = _ffprobe_metadata(clip_local)
        except Exception as exc:
            out.failures.append(f"clip_probe_error:{row.clip_id}:{exc}")
            return out
        if width != TARGET_WIDTH or height != TARGET_HEIGHT:
            out.failures.append(f"clip_resolution:{row.clip_id}:{width}x{height}")
            return out
        if abs(fps - TARGET_FPS) > FPS_TOLERANCE:
            out.failures.append(f"clip_fps:{row.clip_id}:{fps:.3f}")
            return out
        if num_frames and abs(num_frames - TARGET_NUM_FRAMES) > 1:
            # 允许 ±1 帧的容差，因为容器元数据可能偏差一帧
            out.failures.append(f"clip_num_frames:{row.clip_id}:{num_frames}")
            return out
        if not frame_ok or not frame_local.exists():
            out.failures.append(f"frame_missing:{row.clip_id}")
            return out
        if sha256_file(frame_local) != row.first_frame_sha256:
            out.failures.append(f"frame_sha256_mismatch:{row.clip_id}")
            return out

    # ── Step 7: 全局去重 ──
    # 将当前数据集与历史 Top-5 的全局记录索引比对，统计重叠条数
    global_overlap = _count_global_overlap(records, global_record_index or {})
    out.global_overlap_count = global_overlap
    # 若重叠数超过阈值，直接拒绝整个数据集
    if global_overlap > GLOBAL_OVERLAP_REJECT_THRESHOLD:
        out.failures.append(
            f"global_overlap_exceeded:{global_overlap}>{GLOBAL_OVERLAP_REJECT_THRESHOLD}"
        )
        return out

    out.record_count = len(records)
    out.accepted = True
    out.notes = {
        "manifest_protocol": manifest.protocol_version,
        "manifest_schema": manifest.schema_version,
    }
    return out


async def list_miner_interval_ids(miner_store: Any) -> list[int]:
    """列出矿工 bucket 中所有 interval_id（整数前缀）。"""
    keys = await miner_store.list_prefix("")
    seen: set[int] = set()
    for key in keys:
        head = key.split("/", 1)[0]
        if head.isdigit():
            seen.add(int(head))
    return sorted(seen)


async def latest_complete_interval_id(miner_store: Any) -> int | None:
    """
    返回矿工最新的、同时包含 manifest.json 和 dataset.parquet 的 interval_id。

    Validator 只训练最新的完整 interval。
    """
    candidates = await list_miner_interval_ids(miner_store)
    for interval_id in reversed(candidates):
        if await miner_store.object_exists(f"{interval_id}/manifest.json") and \
           await miner_store.object_exists(f"{interval_id}/dataset.parquet"):
            return interval_id
    return None


def manifest_for_interval(local_manifest: Path) -> IntervalManifest:
    """读取本地 manifest 文件。"""
    return read_manifest(local_manifest)
