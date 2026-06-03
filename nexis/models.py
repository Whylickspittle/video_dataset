"""Core schema models for Nexisgen miner submissions."""
"""
Nexisgen 核心数据模型。

本文件定义了矿工提交数据的 Pydantic 模型，包括：
- ClipRecord: 单个视频切片的元数据记录
- IntervalManifest: 每个 interval 的提交清单
- ValidationDecision: 验证者对该矿工的最终判定
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .protocol import (
    CLIP_DURATION_SEC,
    CLIP_DURATION_TOLERANCE_SEC,
    FPS_TOLERANCE,
    SAMPLE_COUNT,
    SCHEMA_VERSION,
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_NUM_FRAMES,
    TARGET_WIDTH,
)

_DEFAULT_SPEC_ID = "video_v1"


class ClipRecord(BaseModel):
    """Single training clip row submitted by a miner."""
    """
    单个视频切片记录。

    这是矿工数据集（dataset.parquet）中的每一行，
    描述了一段 5.04 秒的视频切片及其元数据。
    """

    clip_id: str = Field(min_length=1)              # 切片唯一标识（基于 source_id + start_sec 的确定性哈希）
    clip_uri: str = Field(min_length=1)             # 切片文件的相对路径，如 "clips/xxx.mp4"
    clip_sha256: str = Field(min_length=64, max_length=64)  # 切片文件的 SHA256 哈希（验证者用于校验完整性）
    first_frame_uri: str = Field(min_length=1)      # 首帧图片的相对路径，如 "frames/xxx.jpg"
    first_frame_sha256: str = Field(min_length=64, max_length=64)  # 首帧文件的 SHA256 哈希
    source_video_id: str = Field(min_length=1)      # 源视频的唯一标识（YouTube 视频 ID 等）
    clip_start_sec: float = Field(ge=0.0)           # 切片在源视频中的起始时间（秒）
    duration_sec: float = Field(gt=0.0)             # 切片时长（秒），必须 ≈ 5.0417 ± 0.15
    width: int = Field(gt=0)                        # 视频宽度，必须 = 1280
    height: int = Field(gt=0)                       # 视频高度，必须 = 704
    fps: float = Field(gt=0.0)                      # 帧率，必须 ≈ 24 ± 0.05
    num_frames: int = Field(gt=0)                   # 总帧数，必须 = 121
    source_video_url: str = Field(min_length=1)     # 源视频的完整 URL（用于去重检查）
    # 每段视频对应的文本提示（prompt），供 Trainer 训练 LoRA 使用。
    # 如果为空，Trainer 会回退到默认提示。验证者不对 caption 做 LLM 评分。
    caption: str = Field(default="")

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("duration_sec")
    @classmethod
    def validate_duration(cls, value: float) -> float:
        """验证时长是否在允许范围内。"""
        lower = CLIP_DURATION_SEC - CLIP_DURATION_TOLERANCE_SEC
        upper = CLIP_DURATION_SEC + CLIP_DURATION_TOLERANCE_SEC
        if value < lower or value > upper:
            raise ValueError(
                f"duration_sec must be within ±{CLIP_DURATION_TOLERANCE_SEC}s of "
                f"{CLIP_DURATION_SEC:.4f}"
            )
        return value

    @field_validator("width")
    @classmethod
    def validate_width(cls, value: int) -> int:
        """验证宽度是否严格等于 1280。"""
        if value != TARGET_WIDTH:
            raise ValueError(f"width must be exactly {TARGET_WIDTH}")
        return value

    @field_validator("height")
    @classmethod
    def validate_height(cls, value: int) -> int:
        """验证高度是否严格等于 704。"""
        if value != TARGET_HEIGHT:
            raise ValueError(f"height must be exactly {TARGET_HEIGHT}")
        return value

    @field_validator("fps")
    @classmethod
    def validate_fps(cls, value: float) -> float:
        """验证帧率是否在 24 ± 0.05 范围内。"""
        if abs(value - TARGET_FPS) > FPS_TOLERANCE:
            raise ValueError(f"fps must be within ±{FPS_TOLERANCE} of {TARGET_FPS}")
        return value

    @field_validator("num_frames")
    @classmethod
    def validate_num_frames(cls, value: int) -> int:
        """验证帧数是否严格等于 121。"""
        if value != TARGET_NUM_FRAMES:
            raise ValueError(f"num_frames must be exactly {TARGET_NUM_FRAMES}")
        return value


class IntervalManifest(BaseModel):
    """Interval-level metadata for miner submission package."""
    """
    Interval 级别的提交清单。

    每个 interval 上传时必须包含 manifest.json，
    作为验证者识别和校验数据包的入口文件。
    """

    protocol_version: str = Field(default="2.0.0")  # 协议版本
    schema_version: str = Field(default=SCHEMA_VERSION)  # 数据格式版本
    spec_id: str = Field(default=_DEFAULT_SPEC_ID, min_length=1)  # 数据规格 ID，默认 "video_v1"
    netuid: int = Field(ge=0)                       # Bittensor 网络 ID（默认 70）
    miner_hotkey: str = Field(min_length=1)         # 矿工的 SS58 地址
    interval_id: int = Field(ge=1)                  # Interval 编号（从 1 开始递增）
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # 创建时间（UTC）
    record_count: int = Field(ge=0)                 # 数据集行数，必须 = SAMPLE_COUNT (400)
    dataset_sha256: str = Field(min_length=64, max_length=64)  # dataset.parquet 的 SHA256 哈希

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("record_count")
    @classmethod
    def validate_record_count(cls, value: int) -> int:
        """验证记录数是否严格等于 400。"""
        if value != SAMPLE_COUNT:
            raise ValueError(f"record_count must be exactly {SAMPLE_COUNT}")
        return value

    @model_validator(mode="before")
    @classmethod
    def _normalize_spec_metadata(cls, payload: Any) -> Any:
        """规范化 spec_id：如果未提供则使用默认值。"""
        if not isinstance(payload, dict):
            return payload
        data = dict(payload)
        spec_id = str(data.get("spec_id", "")).strip() or _DEFAULT_SPEC_ID
        data["spec_id"] = spec_id
        return data


class ValidationDecision(BaseModel):
    """Per-miner validator decision for one training cycle."""
    """
    验证者对一个矿工数据集的最终判定结果。

    包含是否接受、失败原因、重叠统计等信息。
    """

    miner_hotkey: str         # 矿工 SS58 地址
    interval_id: int          # 被验证的 interval 编号
    accepted: bool            # 是否被接受（True=通过，False=拒绝）
    failures: list[str] = Field(default_factory=list)  # 失败原因列表
    record_count: int = 0     # 实际检查的记录数
    global_overlap_count: int = 0  # 与全局索引重叠的数量
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # 检查时间
    notes: dict[str, Any] = Field(default_factory=dict)  # 额外备注信息
