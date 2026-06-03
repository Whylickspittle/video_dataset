#!/usr/bin/env python3
"""
数据集管理工具：合并 Parquet、上传 HuggingFace、迭代更新。

用法：
    # 合并多个 interval 的 parquet
    python3 dataset_manager.py merge \
        --inputs ./interval_1/dataset.parquet ./interval_2/dataset.parquet \
        --output combined.parquet

    # 上传到 HuggingFace（需要 hf_token）
    python3 dataset_manager.py upload \
        --parquet combined.parquet \
        --repo-id your-username/nexis-dataset \
        --token hf_xxx

    # 从 HF 下载并合并到本地
    python3 dataset_manager.py pull \
        --repo-id your-username/nexis-dataset \
        --output ./local_dataset/

    # 检查数据集统计
    python3 dataset_manager.py stats --parquet combined.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa


def merge_parquets(inputs: list[str], output: str, dedup_by: str = "clip_id"):
    """合并多个 parquet 文件，自动去重。"""
    tables = []
    seen_ids = set()
    total_rows = 0
    deduped = 0

    for path in inputs:
        if not Path(path).exists():
            print(f"[WARN] File not found, skipping: {path}")
            continue

        table = pq.read_table(path)
        total_rows += table.num_rows

        if dedup_by in table.column_names:
            # 去重：只保留未出现过的 clip_id
            ids = table.column(dedup_by).to_pylist()
            mask = [i not in seen_ids for i in ids]
            table = table.filter(pa.array(mask))
            deduped += sum(not m for m in mask)
            seen_ids.update(ids)

        tables.append(table)
        print(f"  Loaded {path}: {table.num_rows} rows")

    if not tables:
        print("[ERROR] No valid input files")
        sys.exit(1)

    combined = pa.concat_tables(tables)
    pq.write_table(combined, output)

    print(f"\n[INFO] Merge complete:")
    print(f"  Input rows: {total_rows}")
    print(f"  Duplicates removed: {deduped}")
    print(f"  Output rows: {combined.num_rows}")
    print(f"  Saved to: {output}")


def check_stats(parquet_path: str):
    """检查数据集统计信息。"""
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    print(f"[INFO] Dataset stats: {parquet_path}")
    print(f"  Total clips: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    if "source_video_url" in df.columns:
        unique_sources = df["source_video_url"].nunique()
        print(f"  Unique sources: {unique_sources}")

    if "caption" in df.columns:
        has_caption = df["caption"].astype(bool).sum()
        print(f"  With caption: {has_caption} / {len(df)} ({100*has_caption/len(df):.1f}%)")

    if "duration_sec" in df.columns:
        print(f"  Duration range: {df['duration_sec'].min():.2f}s - {df['duration_sec'].max():.2f}s")

    if "width" in df.columns and "height" in df.columns:
        print(f"  Resolution: {df['width'].iloc[0]}x{df['height'].iloc[0]}")

    # 按来源统计
    if "source_video_id" in df.columns:
        print(f"\n  Top sources by clip count:")
        top = df["source_video_id"].value_counts().head(5)
        for vid, count in top.items():
            print(f"    {vid}: {count} clips")


def upload_to_hf(parquet_path: str, repo_id: str, token: str, split: str = "train"):
    """上传到 HuggingFace Dataset。"""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"[INFO] Uploading {parquet_path} to {repo_id}...")

    # 如果仓库不存在则创建
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as exc:
        print(f"[WARN] Repo creation check: {exc}")

    # 上传文件
    api.upload_file(
        path_or_fileobj=parquet_path,
        path_in_repo=f"data/{split}.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"[INFO] Uploaded to https://huggingface.co/datasets/{repo_id}")


def pull_from_hf(repo_id: str, output_dir: str, token: str = None):
    """从 HuggingFace 下载数据集。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] datasets not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"[INFO] Downloading {repo_id}...")

    ds = load_dataset(repo_id, token=token)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_ds in ds.items():
        parquet_file = output_path / f"{split_name}.parquet"
        split_ds.to_parquet(str(parquet_file))
        print(f"  {split_name}: {len(split_ds)} rows -> {parquet_file}")

    print(f"[INFO] Downloaded to {output_dir}")


def create_hf_repo(repo_id: str, token: str):
    """创建新的 HF Dataset 仓库。"""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
        print(f"[INFO] Created: https://huggingface.co/datasets/{repo_id}")
    except Exception as exc:
        print(f"[ERROR] {exc}")


def main():
    parser = argparse.ArgumentParser(description="Dataset management tool")
    subparsers = parser.add_subparsers(dest="command")

    # merge
    p_merge = subparsers.add_parser("merge", help="Merge multiple parquet files")
    p_merge.add_argument("--inputs", nargs="+", required=True, help="Input parquet files")
    p_merge.add_argument("--output", default="combined.parquet")
    p_merge.add_argument("--dedup-by", default="clip_id")

    # stats
    p_stats = subparsers.add_parser("stats", help="Show dataset statistics")
    p_stats.add_argument("--parquet", required=True)

    # upload
    p_upload = subparsers.add_parser("upload", help="Upload to HuggingFace")
    p_upload.add_argument("--parquet", required=True)
    p_upload.add_argument("--repo-id", required=True)
    p_upload.add_argument("--token", required=True)
    p_upload.add_argument("--split", default="train")

    # pull
    p_pull = subparsers.add_parser("pull", help="Download from HuggingFace")
    p_pull.add_argument("--repo-id", required=True)
    p_pull.add_argument("--output", default="./hf_dataset")
    p_pull.add_argument("--token", default=None)

    # create-repo
    p_create = subparsers.add_parser("create-repo", help="Create HF dataset repo")
    p_create.add_argument("--repo-id", required=True)
    p_create.add_argument("--token", required=True)

    args = parser.parse_args()

    if args.command == "merge":
        merge_parquets(args.inputs, args.output, args.dedup_by)
    elif args.command == "stats":
        check_stats(args.parquet)
    elif args.command == "upload":
        upload_to_hf(args.parquet, args.repo_id, args.token, args.split)
    elif args.command == "pull":
        pull_from_hf(args.repo_id, args.output, args.token)
    elif args.command == "create-repo":
        create_hf_repo(args.repo_id, args.token)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
