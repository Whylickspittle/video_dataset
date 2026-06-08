from __future__ import annotations

"""Owner-trainer orchestrator for the `nexis train` command.

Trainer container expectations (matching `rendixnetwork/train:latest`):

  Internal (container-side) paths the image expects:
    - /workspace/training/Wan2.2DatasetAnalsis/h100_dataset_training/models
    - /workspace/training/Wan2.2DatasetAnalsis/h100_dataset_training/runs
    - /workspace/training/Wan2.2DatasetAnalsis/h100_dataset_training/config.json
    - /workspace/training/<dataset_name>          (read-only, ${DATASET_MANIFEST} points inside)
    - /workspace/eval_data                        (read-only)
    - /workspace/outputs                          (writable; final eval output dir)

The host paths are configurable via NEXIS_TRAINER_* env vars. Per-cycle
miner-specific dirs (runs, outputs, dataset) live inside the cycle workdir.

Phase ordering:
  1. Validate datasets, gather candidates.
  2. Train ALL accepted miners (8-GPU pool by default).
  3. After every training container exits, upload all miners' outputs to
     `nexis_miner/{cycle_id}/{miner_hotkey}/...` sequentially.
  4. Persist training_state.json and clean up the cycle scratch dir.
"""
"""
Owner 训练编排器。

本模块只在验证者 hotkey 等于 NEXIS_OWNER_VALIDATOR_HOTKEY 时执行，
负责一个完整的训练周期的全流程：
1. 筛选候选矿工
2. 并行验证数据集
3. 并行 GPU 训练
4. 上传训练结果
5. 清理临时目录

关键概念：
- cycle: 一个训练/评分周期（包含多个矿工的训练和评测）
- candidate: 通过验证的矿工候选
- training_state: 记录每个矿工上次被训练的 interval_id
"""

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..protocol import CROSS_MINER_OVERLAP_REJECT_THRESHOLD
from ..serialization import read_dataset_parquet, read_manifest
from ..storage.r2 import R2S3Store
from ..storage.shared_bucket import NexisMinerBucket
from .dataset_check import (
    DatasetCheckOutcome,
    build_overlap_index,
    count_index_overlap,
    latest_complete_interval_id,
    validate_miner_dataset,
)
from .dataset_convert import convert_to_trainer_manifest
from .docker_runner import DockerGPUPool, DockerRunResult

logger = logging.getLogger(__name__)


# ── Trainer 容器内部固定路径（与 Docker 镜像硬编码对应，不可随意修改） ──
TRAIN_CONTAINER_MODELS_DIR = "/workspace/training/Wan2.2DatasetAnalsis/h100_dataset_training/models"
TRAIN_CONTAINER_RUNS_DIR = "/workspace/training/Wan2.2DatasetAnalsis/h100_dataset_training/runs"
TRAIN_CONTAINER_CONFIG_JSON = "/workspace/training/Wan2.2DatasetAnalsis/h100_dataset_training/config.json"
TRAIN_CONTAINER_DATASET_BASE = "/workspace/training"
TRAIN_CONTAINER_EVAL_DATA = "/workspace/eval_data"
TRAIN_CONTAINER_OUTPUTS = "/workspace/outputs"


@dataclass
class TrainingCandidate:
    """通过验证的矿工候选。"""
    miner_hotkey: str      # 矿工地址
    interval_id: int       # 被选中训练的 interval 编号
    miner_dir: Path        # 数据集本地目录


@dataclass
class TrainedMiner:
    """训练完成的矿工。"""
    miner_hotkey: str      # 矿工地址
    interval_id: int       # 实际训练的 interval 编号
    outputs_dir: Path      # 训练输出目录（包含生成视频）
    miner_dir: Path        # 数据集目录（用于生成 dataset_index.json）


@dataclass
class TrainingCycleResult:
    """一个训练周期的完整结果统计。"""
    cycle_id: int
    accepted: list[str] = field(default_factory=list)       # 被接受的矿工
    rejected: list[str] = field(default_factory=list)       # 被拒绝的矿工
    trained: list[str] = field(default_factory=list)        # 训练成功的矿工
    failed_training: list[str] = field(default_factory=list)  # 训练失败的矿工
    uploaded: list[str] = field(default_factory=list)       # 上传成功的矿工
    failed_upload: list[str] = field(default_factory=list)  # 上传失败的矿工


def _train_state_path(workdir: Path) -> Path:
    """training_state.json 的本地路径。"""
    return workdir / "training_state.json"


def load_training_state(workdir: Path) -> dict[str, int]:
    """加载训练状态：{miner_hotkey: 上次训练的 interval_id}。"""
    path = _train_state_path(workdir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in raw.items() if isinstance(v, int)}


def save_training_state(workdir: Path, state: dict[str, int]) -> None:
    """持久化训练状态到本地 JSON。"""
    path = _train_state_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


async def select_eligible_hotkeys(
    *,
    candidate_hotkeys: list[str],
    invalid_hotkeys: set[str],
    blacklist_hotkeys: set[str],
    last_winners: set[str],
) -> list[str]:
    """
    筛选本轮有资格被验证的矿工。

    资格规则：(上一轮 Top-5 获胜者 OR 不在 invalid 列表中) AND 不在黑名单中

    说明：
    - blacklist: 永久排除，无条件
    - invalid: 上一轮已被选中的矿工（无论是接受还是拒绝），本轮不再重复验证，
               除非他们进入了上一轮 Top-5（last_winners 可覆盖 invalid）
    """
    eligible: list[str] = []
    for hotkey in candidate_hotkeys:
        if hotkey in blacklist_hotkeys:
            continue
        if hotkey in last_winners or hotkey not in invalid_hotkeys:
            eligible.append(hotkey)
    return eligible


def parse_last_winners(total_score_payload: dict[str, Any] | None, top_k: int = 5) -> set[str]:
    """
    从 total_score.json 中提取上一轮 Top-K 获胜者。

    按 aggregate 分数降序排列，取前 K 名。
    """
    if not total_score_payload:
        return set()
    scores = total_score_payload.get("scores")
    if not isinstance(scores, dict):
        return set()
    flat: list[tuple[str, float]] = []
    for hotkey, entry in scores.items():
        if isinstance(entry, dict):
            value = entry.get("aggregate", entry.get("score"))
        else:
            value = entry
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        flat.append((str(hotkey), score))
    flat.sort(key=lambda pair: (-pair[1], pair[0]))
    return {hotkey for hotkey, _ in flat[:top_k]}


def build_train_volumes(
    *,
    settings: Settings,
    miner_dir: Path,
    miner_hotkey: str,
    runs_dir: Path,
    outputs_dir: Path,
    eval_data_dir: Path,
    config_json: Path | None = None,
) -> list[tuple[Path | str, Path | str, str]]:
    """
    构建 Trainer Docker 容器的 -v 挂载列表。

    容器内路径是硬编码的（镜像内部写死），宿主机路径来自 settings + workdir。
    注意：宿主机路径必须解析为绝对路径，因为 docker 把相对路径当作 named volume。
    """
    dataset_container_path = f"{TRAIN_CONTAINER_DATASET_BASE}/{miner_hotkey}"
    return [
        (Path(settings.trainer_models_dir).resolve(), TRAIN_CONTAINER_MODELS_DIR, ""),
        (Path(runs_dir).resolve(), TRAIN_CONTAINER_RUNS_DIR, ""),
        (Path(settings.trainer_config_json).resolve(), TRAIN_CONTAINER_CONFIG_JSON, ""),
        (Path(miner_dir).resolve(), dataset_container_path, "ro"),
        (Path(eval_data_dir).resolve(), TRAIN_CONTAINER_EVAL_DATA, "ro"),
        (Path(outputs_dir).resolve(), TRAIN_CONTAINER_OUTPUTS, ""),
    ]


def trainer_command() -> list[str]:
    """
    Trainer 容器内部执行的命令。

    先运行 02_train_dataset.py 训练 LoRA，
    再运行 05_eval_with_images.py 在 eval_data 上生成视频。
    """
    return [
        "bash",
        "-c",
        (
            "python 02_train_dataset.py && "
            "python 05_eval_with_images.py "
            f"--eval_manifest {TRAIN_CONTAINER_EVAL_DATA}/manifest.jsonl "
            f"--output_dir {TRAIN_CONTAINER_OUTPUTS}"
        ),
    ]


async def run_train_container(
    *,
    settings: Settings,
    candidate: TrainingCandidate,
    pool: DockerGPUPool,
    cycle_id: int,
    workdir: Path,
    eval_data_dir: Path,
) -> Path | None:
    """
    在 GPU 上训练单个矿工的数据集。

    返回本地 outputs_dir 路径（成功）或 None（失败）。
    注意：本函数只负责训练，不上传结果。上传延后到所有容器结束后统一进行。
    """
    miner_dir = candidate.miner_dir
    miner_hotkey = candidate.miner_hotkey

    # 将矿工的 parquet 数据集转换为 Trainer 需要的 manifest.jsonl 格式
    # manifest 中的路径必须是容器内路径（因为 Trainer 在容器里读取文件）
    container_dataset_dir = f"{TRAIN_CONTAINER_DATASET_BASE}/{miner_hotkey}"
    convert_to_trainer_manifest(
        miner_dir=miner_dir,
        container_dataset_dir=container_dataset_dir,
    )

    runs_dir = workdir / "runs" / miner_hotkey
    outputs_dir = workdir / "outputs" / miner_hotkey
    runs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 每个矿工需要独立的 config.json 副本。
    # 原因：Trainer 启动时会改写 config.json（写入 manifest 路径），
    # 如果多个容器共享同一个文件，会导致竞态（矿工 A 写入的路径被矿工 B 读取）。
    config_src = Path(settings.trainer_config_json).resolve()
    miner_config_path = workdir / "configs" / miner_hotkey / "config.json"
    miner_config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_src, miner_config_path)

    volumes = build_train_volumes(
        settings=settings,
        miner_dir=miner_dir,
        miner_hotkey=miner_hotkey,
        runs_dir=runs_dir,
        outputs_dir=outputs_dir,
        eval_data_dir=eval_data_dir,
        config_json=miner_config_path,
    )
    env = {
        "DATASET_MANIFEST": f"{TRAIN_CONTAINER_DATASET_BASE}/{miner_hotkey}/manifest.jsonl",
    }
    result: DockerRunResult = await pool.run(
        image=settings.trainer_docker_image,
        command=trainer_command(),
        volumes=volumes,
        env=env,
        shm_size=settings.trainer_shm_size,
        timeout_sec=settings.trainer_timeout_sec,
    )
    if not result.success:
        logger.error(
            "training failed miner=%s cycle=%d rc=%d\nstderr:\n%s\nstdout:\n%s",
            miner_hotkey,
            cycle_id,
            result.returncode,
            result.stderr,
            result.stdout[-2000:],
        )
        return None

    if not any(outputs_dir.rglob("*")):
        logger.error(
            "training produced no outputs miner=%s cycle=%d outputs_dir=%s",
            miner_hotkey,
            cycle_id,
            outputs_dir,
        )
        return None

    logger.info(
        "training complete miner=%s cycle=%d outputs_dir=%s",
        miner_hotkey,
        cycle_id,
        outputs_dir,
    )
    return outputs_dir


async def upload_miner_outputs(
    *,
    nexis_miner: NexisMinerBucket,
    trained: TrainedMiner,
    cycle_id: int,
    workdir: Path,
    upload_concurrency: int = 8,
) -> bool:
    """
    上传单个矿工的训练结果到共享 bucket。

    上传内容包括：
    - 训练生成的所有视频文件
    - _done.json（标记训练完成）
    - dataset_index.json（用于全局去重索引更新）
    """
    miner_hotkey = trained.miner_hotkey
    outputs_dir = trained.outputs_dir
    files = sorted(p for p in outputs_dir.rglob("*") if p.is_file())
    sem = asyncio.Semaphore(max(int(upload_concurrency), 1))

    async def _upload_one(path: Path) -> bool:
        rel = path.relative_to(outputs_dir)
        key = f"{cycle_id}/{miner_hotkey}/{rel.as_posix()}"
        async with sem:
            try:
                await nexis_miner.upload_path(key, path)
                return True
            except Exception as exc:
                logger.warning(
                    "upload failed miner=%s key=%s err=%s",
                    miner_hotkey,
                    key,
                    exc,
                )
                return False

    results = await asyncio.gather(*[_upload_one(p) for p in files])
    uploaded = sum(1 for ok in results if ok)
    if uploaded != len(files):
        logger.warning(
            "partial upload miner=%s cycle=%d uploaded=%d of=%d",
            miner_hotkey,
            cycle_id,
            uploaded,
            len(files),
        )
        return False
    if uploaded == 0:
        logger.warning(
            "no files to upload miner=%s cycle=%d outputs_dir=%s",
            miner_hotkey,
            cycle_id,
            outputs_dir,
        )
        return False
    done_marker = workdir / "done" / miner_hotkey / "_done.json"
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(
        json.dumps(
            {
                "miner_hotkey": miner_hotkey,
                "cycle_id": cycle_id,
                "miner_interval_id": trained.interval_id,
                "uploaded_files": uploaded,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    await nexis_miner.upload_path(f"{cycle_id}/{miner_hotkey}/_done.json", done_marker)

    # dataset_index.json: 列出本次训练数据集中所有 (source_url, clip_start_sec)。
    # Owner POST 分数时，API 用它来更新 record_info.json（全局去重索引）。
    parquet_path = trained.miner_dir / "dataset.parquet"
    if parquet_path.exists():
        try:
            records = read_dataset_parquet(parquet_path)
        except Exception as exc:
            logger.warning(
                "dataset_index skipped miner=%s cycle=%d err=%s",
                miner_hotkey,
                cycle_id,
                exc,
            )
        else:
            index_payload = [
                {
                    "source_url": row.source_video_url,
                    "clip_start_sec": float(row.clip_start_sec),
                }
                for row in records
            ]
            index_local = workdir / "index" / f"{miner_hotkey}_dataset_index.json"
            index_local.parent.mkdir(parents=True, exist_ok=True)
            index_local.write_text(
                json.dumps(index_payload, ensure_ascii=True),
                encoding="utf-8",
            )
            await nexis_miner.upload_path(
                f"{cycle_id}/{miner_hotkey}/dataset_index.json",
                index_local,
            )
            logger.info(
                "dataset_index uploaded miner=%s cycle=%d rows=%d",
                miner_hotkey,
                cycle_id,
                len(index_payload),
            )

    logger.info(
        "uploaded miner=%s cycle=%d files=%d",
        miner_hotkey,
        cycle_id,
        uploaded,
    )
    return True


async def _miner_upload_time(
    store: R2S3Store, interval_id: int, manifest_path: Path
) -> Any:
    """
    获取矿工数据集最可信的上传时间戳。

    优先使用 R2 上 dataset.parquet 的 LastModified（由 R2 在 PUT 时设置，矿工无法伪造）；
    如果获取失败，回退到 manifest.json 的 created_at（矿工自己上报，可信度较低）。
    """
    try:
        ts = await store.get_object_last_modified(f"{interval_id}/dataset.parquet")
        if ts is not None:
            return ts
    except Exception as exc:
        logger.warning(
            "last-modified lookup failed interval=%d err=%s", interval_id, exc
        )
    try:
        return read_manifest(manifest_path).created_at
    except Exception:
        return None


async def _filter_cross_miner_overlap(
    candidates: list[TrainingCandidate],
    store_for_hotkey: Callable[[str], R2S3Store],
) -> tuple[list[TrainingCandidate], list[DatasetCheckOutcome]]:
    """
    跨矿工去重：拒绝与先上传矿工重叠超过 100 条的后上传者。

    判定依据：
    - 重叠 = 同一 canonical_url + start_sec 差 < 4.5 秒
    - 时间判定：R2 LastModified（不可伪造）> manifest created_at（回退）
    - 先上传者保留，后上传者如果与任何保留者重叠 > 100 条则被拒
    """
    if len(candidates) < 2:
        return candidates, []

    enriched: list[tuple[TrainingCandidate, Any, dict[str, list[float]], int]] = []
    for cand in candidates:
        parquet_path = cand.miner_dir / "dataset.parquet"
        manifest_path = cand.miner_dir / "manifest.json"
        try:
            records = read_dataset_parquet(parquet_path)
        except Exception as exc:
            logger.warning(
                "cross-miner: parquet re-read failed hotkey=%s err=%s",
                cand.miner_hotkey,
                exc,
            )
            continue
        try:
            store = store_for_hotkey(cand.miner_hotkey)
            upload_time = await _miner_upload_time(
                store, cand.interval_id, manifest_path
            )
        except Exception:
            upload_time = None
        enriched.append((cand, upload_time, build_overlap_index(records), len(records)))

    # 按上传时间升序排列；时间相同则按 hotkey 排序以保证确定性
    # 无法解析时间戳的排在最后（不能抢占别人）
    enriched.sort(
        key=lambda t: (t[1] is None, t[1], t[0].miner_hotkey)
    )

    kept: list[tuple[TrainingCandidate, Any, dict[str, list[float]], int]] = []
    rejections: list[DatasetCheckOutcome] = []
    for cand, upload_time, index, record_count in enriched:
        rejected_by: tuple[str, int] | None = None
        for kept_cand, _, kept_index, _ in kept:
            count = count_index_overlap(index, kept_index)
            if count > CROSS_MINER_OVERLAP_REJECT_THRESHOLD:
                rejected_by = (kept_cand.miner_hotkey, count)
                break
        if rejected_by is not None:
            other_hk, count = rejected_by
            logger.warning(
                "cross-miner overlap reject hotkey=%s vs earlier=%s count=%d > %d",
                cand.miner_hotkey,
                other_hk,
                count,
                CROSS_MINER_OVERLAP_REJECT_THRESHOLD,
            )
            rejections.append(
                DatasetCheckOutcome(
                    accepted=False,
                    miner_hotkey=cand.miner_hotkey,
                    interval_id=cand.interval_id,
                    record_count=record_count,
                    failures=[
                        f"cross_miner_overlap:{other_hk}:{count}"
                    ],
                )
            )
        else:
            kept.append((cand, upload_time, index, record_count))

    return [t[0] for t in kept], rejections


async def gather_candidates(
    *,
    eligible_hotkeys: list[str],
    store_for_hotkey: Callable[[str], R2S3Store],
    workdir: Path,
    cycle_id: int,
    training_state: dict[str, int],
    global_record_index: dict[str, list[float]],
    miner_concurrency: int = 4,
    download_concurrency: int = 16,
) -> tuple[list[TrainingCandidate], list[DatasetCheckOutcome]]:
    """
    并行验证所有候选矿工的数据集。

    并发控制：
    - miner_concurrency: 同时验证多少个矿工（默认 4）
    - download_concurrency: 单个矿工内部同时下载多少个文件（默认 16）
    - 实际并发 GET 数 ≈ miner_concurrency × download_concurrency
    """
    cycle_workdir = workdir / "cycle" / str(cycle_id)
    miner_sem = asyncio.Semaphore(max(int(miner_concurrency), 1))

    async def _process(
        hotkey: str,
    ) -> tuple[str, TrainingCandidate | None, DatasetCheckOutcome | None]:
        async with miner_sem:
            try:
                miner_store = store_for_hotkey(hotkey)
            except Exception as exc:
                logger.warning("store unavailable for hotkey=%s err=%s", hotkey, exc)
                return hotkey, None, None
            try:
                interval_id = await latest_complete_interval_id(miner_store)
            except Exception as exc:
                logger.warning("interval lookup failed hotkey=%s err=%s", hotkey, exc)
                return hotkey, None, None
            if interval_id is None:
                logger.info("hotkey=%s has no uploaded interval; skipping", hotkey)
                return hotkey, None, None
            # 如果该矿工的最新 interval 已经被训练过，跳过
            last_seen = training_state.get(hotkey)
            if last_seen is not None and interval_id <= last_seen:
                logger.info(
                    "hotkey=%s latest interval %d already trained at cycle %d; skipping",
                    hotkey,
                    interval_id,
                    last_seen,
                )
                return hotkey, None, None
            outcome = await validate_miner_dataset(
                miner_hotkey=hotkey,
                interval_id=interval_id,
                miner_store=miner_store,
                workdir=cycle_workdir,
                global_record_index=global_record_index,
                download_concurrency=download_concurrency,
            )
            if not outcome.accepted:
                logger.warning(
                    "dataset rejected hotkey=%s interval=%d failures=%s",
                    hotkey,
                    interval_id,
                    outcome.failures,
                )
                return hotkey, None, outcome
            return (
                hotkey,
                TrainingCandidate(
                    miner_hotkey=hotkey,
                    interval_id=interval_id,
                    miner_dir=cycle_workdir / hotkey / str(interval_id),
                ),
                None,
            )

    results = await asyncio.gather(*[_process(hk) for hk in eligible_hotkeys])
    candidates: list[TrainingCandidate] = []
    rejections: list[DatasetCheckOutcome] = []
    for _, cand, outcome in results:
        if cand is not None:
            candidates.append(cand)
        elif outcome is not None:
            rejections.append(outcome)

    # 同周期跨矿工去重
    candidates, cross_miner_rejections = await _filter_cross_miner_overlap(
        candidates, store_for_hotkey
    )
    rejections.extend(cross_miner_rejections)
    return candidates, rejections


async def determine_next_cycle_id(nexis_miner: NexisMinerBucket) -> int | None:
    """
    确定下一个应该训练的 cycle_id。

    规则：
    - 如果没有任何 cycle → 返回 1
    - 如果最新的 cycle 还没有 total_score → 返回 None（等待评分完成）
    - 否则返回 latest + 1
    """
    latest = await nexis_miner.latest_cycle_id()
    if latest is None:
        return 1
    if not await nexis_miner.has_total_score(latest):
        return None
    return latest + 1


async def cleanup_workdir(path: Path) -> None:
    """清理临时工作目录。"""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except Exception as exc:
        logger.warning("workdir cleanup failed path=%s err=%s", path, exc)


async def run_training_cycle(
    *,
    settings: Settings,
    candidate_hotkeys: list[str],
    invalid_hotkeys: set[str],
    blacklist_hotkeys: set[str],
    last_total_score: dict[str, Any] | None,
    store_for_hotkey: Callable[[str], R2S3Store],
    nexis_miner: NexisMinerBucket,
    pool: DockerGPUPool,
    cycle_id: int,
    workdir: Path,
    global_record_index: dict[str, list[float]],
    eval_data_dir: Path,
    on_select: Callable[[list[str], int], Any] | None = None,
) -> TrainingCycleResult:
    """
    执行一个完整的训练周期。

    四阶段流程：
    1. 筛选候选矿工（eligible）
    2. 并行验证数据集，产出 candidates
    3. GPU 池并行训练所有通过的矿工
    4. 上传训练结果到共享 bucket

    同时更新 invalid_hotkeys：被接受和被拒绝的矿工都标记为 invalid，
    避免下一轮重复验证，除非他们进入 Top-5。
    """
    last_winners = parse_last_winners(last_total_score)
    eligible = await select_eligible_hotkeys(
        candidate_hotkeys=candidate_hotkeys,
        invalid_hotkeys=invalid_hotkeys,
        blacklist_hotkeys=blacklist_hotkeys,
        last_winners=last_winners,
    )
    logger.info(
        "training cycle=%d candidates=%d eligible=%d invalid=%d blacklist=%d "
        "last_winners=%d",
        cycle_id,
        len(candidate_hotkeys),
        len(eligible),
        len(invalid_hotkeys),
        len(blacklist_hotkeys),
        len(last_winners),
    )

    training_state = load_training_state(workdir)
    candidates, rejections = await gather_candidates(
        eligible_hotkeys=eligible,
        store_for_hotkey=store_for_hotkey,
        workdir=workdir,
        cycle_id=cycle_id,
        training_state=training_state,
        global_record_index=global_record_index,
        miner_concurrency=getattr(settings, "miner_gather_concurrency", 4),
        download_concurrency=getattr(settings, "download_concurrency", 16),
    )

    selected_hotkeys = [c.miner_hotkey for c in candidates]
    rejected_hotkeys = [
        outcome.miner_hotkey for outcome in rejections if outcome.miner_hotkey
    ]
    # 被接受和被拒绝的矿工都标记为 invalid：
    #   - 被接受 → 已训练，不要重复训练
    #   - 被拒绝 → 数据有问题，不要再浪费时间验证
    # 唯一的重新进入途径：进入上一周期的 Top-5
    hotkeys_to_invalidate = sorted({*selected_hotkeys, *rejected_hotkeys})
    if on_select and hotkeys_to_invalidate:
        maybe = on_select(hotkeys_to_invalidate, cycle_id)
        if asyncio.iscoroutine(maybe):
            await maybe

    cycle_result = TrainingCycleResult(
        cycle_id=cycle_id,
        accepted=selected_hotkeys,
        rejected=rejected_hotkeys,
    )

    cycle_scratch = workdir / "cycle" / str(cycle_id)

    # 阶段 2：并行训练所有通过的矿工（GPU 池）
    async def _train(candidate: TrainingCandidate) -> tuple[TrainingCandidate, Path | None]:
        outputs_dir = await run_train_container(
            settings=settings,
            candidate=candidate,
            pool=pool,
            cycle_id=cycle_id,
            workdir=cycle_scratch,
            eval_data_dir=eval_data_dir,
        )
        return candidate, outputs_dir

    trained_miners: list[TrainedMiner] = []
    if candidates:
        results = await asyncio.gather(*[_train(c) for c in candidates])
        for candidate, outputs_dir in results:
            if outputs_dir is None:
                cycle_result.failed_training.append(candidate.miner_hotkey)
                continue
            cycle_result.trained.append(candidate.miner_hotkey)
            trained_miners.append(
                TrainedMiner(
                    miner_hotkey=candidate.miner_hotkey,
                    interval_id=candidate.interval_id,
                    outputs_dir=outputs_dir,
                    miner_dir=candidate.miner_dir,
                )
            )
        logger.info(
            "training phase complete cycle=%d trained=%d failed=%d",
            cycle_id,
            len(cycle_result.trained),
            len(cycle_result.failed_training),
        )

    # 阶段 3：上传所有成功训练的结果
    upload_conc = max(int(getattr(settings, "upload_concurrency", 8)), 1)

    async def _upload_one(trained: TrainedMiner) -> tuple[str, bool]:
        try:
            ok = await upload_miner_outputs(
                nexis_miner=nexis_miner,
                trained=trained,
                cycle_id=cycle_id,
                workdir=cycle_scratch,
                upload_concurrency=upload_conc,
            )
        except Exception as exc:
            logger.exception(
                "upload exception miner=%s cycle=%d: %s",
                trained.miner_hotkey,
                cycle_id,
                exc,
            )
            ok = False
        return trained.miner_hotkey, ok

    if trained_miners:
        upload_results = await asyncio.gather(*[_upload_one(t) for t in trained_miners])
        for hotkey, ok in upload_results:
            if ok:
                cycle_result.uploaded.append(hotkey)
            else:
                cycle_result.failed_upload.append(hotkey)

    # 只有训练成功且上传成功的矿工才更新 training_state
    for trained in trained_miners:
        if trained.miner_hotkey in cycle_result.uploaded:
            training_state[trained.miner_hotkey] = trained.interval_id
    save_training_state(workdir, training_state)

    # 清理临时目录
    await cleanup_workdir(cycle_scratch)
    return cycle_result
