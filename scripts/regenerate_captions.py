#!/usr/bin/env python3
"""为已生成的 dataset.parquet 补做/重新生成 Caption。

用法:
    # 1. 为已有的 dataset.parquet 生成 caption
    python3 scripts/regenerate_captions.py \
        --dataset .nexis/out/1/dataset.parquet \
        --frames-dir .nexis/out/1/frames

    # 2. 指定输出路径（默认覆盖原文件）
    python3 scripts/regenerate_captions.py \
        --dataset .nexis/out/1/dataset.parquet \
        --frames-dir .nexis/out/1/frames \
        --output .nexis/out/1/dataset_captioned.parquet

    # 3. 只更新 caption 为空的记录
    python3 scripts/regenerate_captions.py \
        --dataset .nexis/out/1/dataset.parquet \
        --frames-dir .nexis/out/1/frames \
        --only-empty

    # 4. 使用 Gemini 而不是 OpenAI
    python3 scripts/regenerate_captions.py ... --model gemini-3.1-flash-lite-preview

流程:
    1. 读取现有的 dataset.parquet
    2. 对每个记录，读取 first_frame 图片
    3. 调用 Captioner (OpenAI/Gemini) 生成 caption
    4. 更新 ClipRecord.caption 字段
    5. 重新写入 dataset.parquet
    6. 重新计算 dataset_sha256
    7. 更新 manifest.json 中的 dataset_sha256
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nexis.hash_utils import sha256_file
from nexis.miner.captioner import Captioner
from nexis.models import ClipRecord
from nexis.protocol import SAMPLE_COUNT
from nexis.serialization import read_dataset_parquet, write_dataset_parquet, read_manifest, write_manifest

logger = logging.getLogger(__name__)


def regenerate_captions(
    dataset_path: Path,
    frames_dir: Path,
    output_path: Path | None = None,
    captioner: Captioner | None = None,
    only_empty: bool = False,
) -> tuple[Path, int, int]:
    """Regenerate captions for dataset.

    Returns:
        (output_dataset_path, updated_count, total_count)
    """
    logger.info("reading dataset from %s", dataset_path)
    records = read_dataset_parquet(dataset_path)
    total = len(records)
    logger.info("loaded %d records", total)

    if total != SAMPLE_COUNT:
        logger.warning("dataset has %d records, expected %d", total, SAMPLE_COUNT)

    if captioner is None or not captioner.enabled:
        logger.error("no captioner available; set OPENAI_API_KEY or GEMINI_API_KEY")
        raise RuntimeError("captioner not configured")

    updated = 0
    skipped = 0
    failed = 0

    for idx, record in enumerate(records):
        # Check if we should skip this record
        if only_empty and record.caption.strip():
            skipped += 1
            continue

        frame_path = frames_dir / record.first_frame_uri.lstrip("/")
        if not frame_path.exists():
            logger.warning("[%d/%d] frame not found: %s", idx + 1, total, frame_path)
            failed += 1
            continue

        try:
            new_caption = captioner.caption_frame(frame_path)
        except Exception as exc:
            logger.warning("[%d/%d] caption failed for %s: %s", idx + 1, total, record.clip_id, exc)
            failed += 1
            continue

        # Update the record
        record.caption = new_caption
        updated += 1

        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info(
                "[%d/%d] caption updated: %s = %r",
                idx + 1,
                total,
                record.clip_id,
                new_caption[:50],
            )

    logger.info(
        "caption regeneration complete: updated=%d, skipped=%d, failed=%d, total=%d",
        updated,
        skipped,
        failed,
        total,
    )

    # Write updated dataset
    out_path = output_path or dataset_path
    write_dataset_parquet(records, out_path)
    logger.info("wrote updated dataset to %s", out_path)

    return out_path, updated, total


def update_manifest(manifest_path: Path, new_dataset_path: Path) -> None:
    """Update manifest.json with new dataset_sha256."""
    logger.info("updating manifest %s", manifest_path)
    manifest = read_manifest(manifest_path)
    old_sha = manifest.dataset_sha256
    new_sha = sha256_file(new_dataset_path)
    manifest.dataset_sha256 = new_sha
    write_manifest(manifest, manifest_path)
    logger.info("manifest updated: dataset_sha256 %s... -> %s...", old_sha[:16], new_sha[:16])


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate captions for dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset.parquet")
    parser.add_argument("--frames-dir", required=True, help="Directory containing frame images")
    parser.add_argument("--output", default="", help="Output path for updated dataset (default: overwrite)")
    parser.add_argument("--only-empty", action="store_true", help="Only update records with empty captions")
    parser.add_argument("--model", default="", help="Caption model (default from env)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset_path = Path(args.dataset)
    frames_dir = Path(args.frames_dir)
    output_path = Path(args.output) if args.output else None

    if not dataset_path.exists():
        logger.error("dataset not found: %s", dataset_path)
        sys.exit(1)
    if not frames_dir.exists():
        logger.error("frames directory not found: %s", frames_dir)
        sys.exit(1)

    captioner = Captioner()
    if not captioner.enabled:
        print("ERROR: Captioner not configured. Set OPENAI_API_KEY or GEMINI_API_KEY in .env")
        sys.exit(1)

    out_path, updated, total = regenerate_captions(
        dataset_path=dataset_path,
        frames_dir=frames_dir,
        output_path=output_path,
        captioner=captioner,
        only_empty=args.only_empty,
    )

    # Update manifest if it exists next to the dataset
    manifest_path = dataset_path.parent / "manifest.json"
    if manifest_path.exists():
        update_manifest(manifest_path, out_path)
    else:
        logger.warning("manifest.json not found at %s", manifest_path)

    print("\n" + "=" * 60)
    print("Caption regeneration complete!")
    print(f"  Updated: {updated}/{total} records")
    print(f"  Output:  {out_path}")
    if manifest_path.exists():
        print(f"  Manifest updated: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
