from __future__ import annotations

"""Protocol-level constants and policy decisions for Nexisgen.

Nexisgen 协议级常量与策略定义。

本文件定义了整个子网（Subnet 70）的核心规则：
- 数据集规格（分辨率、帧率、样本数等）
- 去重策略（三层去重的阈值）
- 链上权重提交的周期与衰减规则
"""

from dataclasses import dataclass


# ── 协议版本 ──
PROTOCOL_VERSION = "2.0.0"   # 矿工-验证者之间的通信协议版本
SCHEMA_VERSION = "2.0.0"     # 数据格式（dataset.parquet / manifest.json）的版本

# ── 数据集规格（v2 训练/验证共用，不可随意更改） ──
# 这些常量直接决定了 Trainer Docker 容器对输入数据的期望格式。
SAMPLE_COUNT = 400           # 每个数据集必须包含的样本数量（段数）
TARGET_WIDTH = 1280          # 视频宽度（像素）
TARGET_HEIGHT = 704          # 视频高度（像素）
TARGET_FPS = 24              # 帧率（帧/秒）
TARGET_NUM_FRAMES = 121      # 每段视频的总帧数（121 帧 / 24 fps ≈ 5.04 秒）
CLIP_DURATION_SEC = TARGET_NUM_FRAMES / TARGET_FPS  # 每段理论时长 ≈ 5.0417 秒
CLIP_DURATION_TOLERANCE_SEC = 0.15  # 时长允许的误差范围（±0.15 秒）
FPS_TOLERANCE = 0.05                # 帧率允许的误差范围（±0.05 fps）

# ── 去重策略 ──
# OVERLAP_WINDOW_SEC: 同一段源视频中，两个切片的起始时间如果相差小于 4.5 秒，
#                    就被认为是重叠（因为每段时长约 5.04 秒）。
OVERLAP_WINDOW_SEC = 4.5

# GLOBAL_OVERLAP_REJECT_THRESHOLD: 全局去重阈值。
#   矿工数据集与历史 Top-5 全局索引重叠超过 100 条 → 整个数据集被拒绝。
GLOBAL_OVERLAP_REJECT_THRESHOLD = 100  # > threshold -> reject

# CROSS_MINER_OVERLAP_REJECT_THRESHOLD: 跨矿工去重阈值。
#   同一周期内，两个被接受的矿工数据集之间重叠超过 100 条 → 后上传者被拒绝。
CROSS_MINER_OVERLAP_REJECT_THRESHOLD = 100

# ── 链上权重规则 ──
WEIGHT_SUBMISSION_INTERVAL_BLOCKS = 300  # 每 300 个区块（约 1 小时）提交一次权重
WEIGHT_TOP_K = 5                         # 每次只给前 5 名矿工分配权重
WEIGHT_DECAY_BASE = 0.5                  # 权重衰减基数：第1名 1.0, 第2名 0.5, 第3名 0.25...


@dataclass(frozen=True)
class HardFailurePolicy:
    """Hard checks reject the interval immediately for that miner."""
    """
    硬失败策略：一旦检测到任何硬性违规，立即拒绝该矿工的整个数据集，
    不再继续后续检查。
    """
    reject_on_first_violation: bool = True  # True: 第一个违规即拒绝；False: 可继续检查
