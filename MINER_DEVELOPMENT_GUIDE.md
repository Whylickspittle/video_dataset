# SN70 Miner 开发实战指南

本文档是一份从零开始的完整开发手册，覆盖环境搭建、数据收集、本地测试、离线自测到正式上传的全流程。

---

## 目录

1. [核心概念速览](#一核心概念速览)
2. [环境准备](#二环境准备)
3. [Step 1: 数据源收集策略](#三step-1-数据源收集策略)
4. [Step 2: 视频筛选与下载](#四step-2-视频筛选与下载)
5. [Step 3: 本地切片与数据集生成](#五step-3-本地切片与数据集生成)
6. [Step 4: 数据集质量自检](#六step-4-数据集质量自检)
7. [Step 5: 离线 LoRA 训练自测](#七step-5-离线-lora-训练自测)
8. [Step 6: 离线 VBench 打分](#八step-6-离线-vbench-打分)
9. [Step 7: 正式上传与监控](#九step-7-正式上传与监控)
10. [迭代优化策略](#十迭代优化策略)
11. [常见问题排查](#十一常见问题排查)
12. [完整工作流速查表](#十二完整工作流速查表)

---

## 一、核心概念速览

| 概念 | 说明 |
|------|------|
| **Interval** | 50 个 Bittensor 区块为一个周期，每周期提交一次数据集 |
| **Dataset** | 400 条视频切片（clips）+ 400 张首帧（frames）+ manifest + parquet |
| **Clip 规格** | 1280x704, 24fps, 121 帧, ~5.04 秒时长 |
| **Top-K=5** | 每轮只有前 5 名获得权重，且权重按 `[1, 0.5, 0.25, 0.125, 0.0625]` 衰减 |
| **去重三层** | ① 数据集内去重（同视频 <4.5s）② 全局去重（vs 历史 Top-5）③ Cross-miner 去重（同周期先上传者赢） |
| **评分方式** | Validator 用你的 400 clips 训练 LoRA -> 在 eval 数据上生成视频 -> VBench 8 维度打分 -> 取均值排名 |

**关键结论**：数据质量是唯一决定因素。训练、生成、打分流程完全自动化且对所有矿工公平。

---

## 二、环境准备

### 2.1 硬件要求

| 用途 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **数据下载/切片** | 任意机器 | Mac Studio / Linux 服务器 |
| **本地 LoRA 自测** | RTX 4090 24GB | RTX 6000 Ada 48GB / A100 40GB |
| **本地 VBench 自测** | RTX 4090 24GB | A100 40GB / H100 80GB |
| **磁盘空间** | 500GB | 2TB+ SSD |

> **注意**：正式挖矿只需做数据下载/切片（任意机器即可），训练和打分由 Validator 完成。本地自测 LoRA/VBench 是可选的质量验证手段。

### 2.2 软件安装

```bash
# 1. Python 3.10+
python3 --version

# 2. 克隆仓库
git clone https://github.com/rendixnetwork/nexisgen.git
cd nexisgen
pip install -e . --break-system-packages  # 或先用 venv

# 3. 安装 yt-dlp
pip install yt-dlp --break-system-packages
# 验证
yt-dlp --version

# 4. 安装 ffmpeg + ffprobe
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt-get install ffmpeg
# 验证
ffmpeg -version
ffprobe -version

# 5. 安装 Docker（用于本地 VBench 自测）
# macOS: https://docs.docker.com/desktop/install/mac-install/
# Linux: https://docs.docker.com/engine/install/
```

### 2.3 环境变量配置

```bash
cp .env.example .env
```

编辑 `.env`，Miner 至少填写：

```env
# Bittensor 钱包（正式挖矿需要，本地测试可先用假值）
BT_WALLET_NAME=default
BT_WALLET_HOTKEY=default

# R2 存储（每个矿工一个 bucket，bucket 名 = 小写 hotkey）
R2_ACCOUNT_ID=你的账户ID
R2_REGION=auto
R2_READ_ACCESS_KEY=xxx
R2_READ_SECRET_KEY=xxx
R2_WRITE_ACCESS_KEY=xxx
R2_WRITE_SECRET_KEY=xxx

# Caption 生成（强烈建议配置，空 caption 会降低训练效果）
OPENAI_API_KEY=sk-xxx
NEXIS_CAPTION_MODEL=gpt-4o-mini

# 工作目录
NEXIS_WORKDIR=.nexis
NEXIS_SOURCES_FILE=sources.txt
```

---

## 三、Step 1: 数据源收集策略

### 3.1 YouTube 视频筛选硬指标

| 指标 | 最低要求 | 理想值 | 筛选命令片段 |
|------|---------|--------|------------|
| 分辨率 | >= 1920x1080 | 3840x2160 (4K) | `[height>=1080]` |
| 帧率 | >= 24fps | 60fps | `[fps>=24]` |
| 码率 | >= 8000 kbps | >= 20000 kbps | `[tbr>=8000]` |
| 时长 | >= 30 秒 | >= 60 秒 | `--match-filter "duration>=30"` |
| 编码 | H.264 (avc1) 优先 | H.264 | `[vcodec^=avc1]` |
| 内容类型 | nature/landscape/scenery | 自然风光、航拍 | 人工筛选 |

### 3.2 搜索渠道

1. **YouTube 搜索关键词**（英文效果最佳）：
   - `4K nature documentary`
   - `8K aerial landscape`
   - `cinematic nature b-roll`
   - `wildlife slow motion`
   - `drone footage mountains`
   - `timelapse forest clouds`

2. **高质量频道推荐**（需人工确认版权）：
   - Nature relaxation channels
   - 4K/8K demo channels
   - 国家公园官方频道

3. **Playlist 批量获取**：
   ```bash
   # 提取 playlist 中所有视频 URL
   yt-dlp --flat-playlist --print "%(webpage_url)s" "PLAYLIST_URL" > urls.txt
   ```

### 3.3 建立自己的 sources.txt

```bash
# 创建 sources.txt，每行一个 YouTube URL
# 建议至少准备 50-100 个候选 URL（最终只需要从中切出 400 段）

touch sources.txt
```

**示例内容**：
```
https://www.youtube.com/watch?v=LXb3EKWsInQ
https://www.youtube.com/watch?v=abc123DEF45
https://youtu.be/xyz789ABC12
```

---

## 四、Step 2: 视频筛选与下载

### 4.1 批量预检脚本（不下载，只看指标）

创建 `scripts/check_sources.py`：

```python
#!/usr/bin/env python3
"""批量检查 sources.txt 中所有视频的质量指标，输出 CSV 报告。"""

import csv
import subprocess
import sys
from pathlib import Path


def get_video_info(url: str) -> dict:
    """用 yt-dlp 获取视频元数据，不下载视频。"""
    cmd = [
        "yt-dlp",
        "--no-download",
        "--print", "%(id)s|%(width)s|%(height)s|%(fps)s|%(vcodec)s|%(tbr)s|%(duration)s|%(title)s",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return {"error": result.stderr.strip()}
        parts = result.stdout.strip().split("|")
        if len(parts) < 8:
            return {"error": "parse failed"}
        return {
            "id": parts[0],
            "width": int(parts[1] or 0),
            "height": int(parts[2] or 0),
            "fps": float(parts[3] or 0),
            "vcodec": parts[4],
            "tbr": float(parts[5] or 0),
            "duration": float(parts[6] or 0),
            "title": parts[7],
        }
    except Exception as exc:
        return {"error": str(exc)}


def check_quality(info: dict) -> tuple[bool, list[str]]:
    """检查视频是否符合 SN70 素材要求。"""
    if "error" in info:
        return False, [info["error"]]

    issues = []
    if info["width"] < 1920 or info["height"] < 1080:
        issues.append(f"resolution too low ({info['width']}x{info['height']})")
    if info["fps"] < 24:
        issues.append(f"fps too low ({info['fps']})")
    if info["tbr"] < 8000:
        issues.append(f"bitrate too low ({info['tbr']}kbps)")
    if info["duration"] < 30:
        issues.append(f"duration too short ({info['duration']}s)")
    # H.264 优先，VP9 也可接受但处理更慢
    if not ("avc" in info["vcodec"] or "vp09" in info["vcodec"]):
        issues.append(f"codec not preferred ({info['vcodec']})")

    return len(issues) == 0, issues


def main() -> None:
    sources_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sources.txt")
    if not sources_file.exists():
        print(f"sources file not found: {sources_file}")
        sys.exit(1)

    urls = [line.strip() for line in sources_file.read_text().splitlines() if line.strip()]

    print(f"Checking {len(urls)} URLs...")
    results = []

    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] {url[:60]}...", end=" ")
        info = get_video_info(url)
        ok, issues = check_quality(info)
        status = "PASS" if ok else "FAIL"
        issue_str = "; ".join(issues) if issues else ""
        print(f"{status} {issue_str}")

        results.append({
            "url": url,
            "status": status,
            "id": info.get("id", ""),
            "title": info.get("title", ""),
            "resolution": f"{info.get('width',0)}x{info.get('height',0)}",
            "fps": info.get("fps", 0),
            "tbr": info.get("tbr", 0),
            "duration": info.get("duration", 0),
            "codec": info.get("vcodec", ""),
            "issues": issue_str,
        })

    # 写入 CSV 报告
    report_path = Path("source_report.csv")
    with report_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\nReport saved to {report_path}")
    print(f"Total: {len(results)}, Pass: {passed}, Fail: {len(results) - passed}")


if __name__ == "__main__":
    main()
```

运行：

```bash
chmod +x scripts/check_sources.py
python3 scripts/check_sources.py sources.txt
```

输出示例：

```
Checking 100 URLs...
[1/100] https://www.youtube.com/watch?v=LXb3EKWsInQ... PASS
[2/100] https://www.youtube.com/watch?v=abc123... FAIL resolution too low (1280x720); fps too low (30.0)
...
Report saved to source_report.csv
Total: 100, Pass: 73, Fail: 27
```

### 4.2 批量下载脚本

创建 `scripts/download_sources.py`：

```python
#!/usr/bin/env python3
"""从 sources.txt 下载通过预检的视频，使用 yt-dlp 格式筛选。"""

import csv
import subprocess
import sys
from pathlib import Path


def download_video(url: str, output_dir: Path) -> bool:
    """下载单个视频，使用 SN70 友好的格式选择。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "%(id)s_%(height)sp.%(ext)s")

    cmd = [
        "yt-dlp",
        # 格式选择：最佳 H.264 视频 + 最佳音频
        "-f", "bestvideo[height>=1080][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        # 时长过滤
        "--match-filter", "duration >= 30",
        # 输出模板
        "-o", template,
        # 断点续传
        "--continue",
        # 限制重试
        "--retries", "5",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  timeout: {url}")
        return False


def main() -> None:
    report_path = Path("source_report.csv")
    if not report_path.exists():
        print("Run check_sources.py first to generate source_report.csv")
        sys.exit(1)

    with report_path.open() as f:
        rows = list(csv.DictReader(f))

    # 只下载通过预检的
    pass_rows = [r for r in rows if r["status"] == "PASS"]
    download_dir = Path("downloads")

    print(f"Downloading {len(pass_rows)} videos to {download_dir}/")

    for idx, row in enumerate(pass_rows, 1):
        url = row["url"]
        print(f"[{idx}/{len(pass_rows)}] {row['title'][:50]}...")
        ok = download_video(url, download_dir)
        print(f"  {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 scripts/download_sources.py
```

### 4.3 下载后的二次筛选（ffprobe）

有些视频 yt-dlp 报告的元数据与实际文件可能不一致，下载后需用 ffprobe 再次确认：

```bash
#!/bin/bash
# scripts/probe_downloads.sh

for f in downloads/*; do
    echo "=== $f ==="
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,r_frame_rate,nb_frames,duration,bit_rate \
        -of csv=s=x:p=0 "$f"
done
```

---

## 五、Step 3: 本地切片与数据集生成

### 5.1 理解 pipeline.py 的工作流程

Miner 的核心流水线在 `nexis/miner/pipeline.py`：

1. 读取 `sources.txt`
2. 下载每个视频到 `workdir/raw/`
3. 用 ffprobe 探测时长
4. 按 `5.04s` 步长逐段切片（同时做去重保护）
5. 每段提取首帧、生成 caption、计算 SHA256
6. 写入 `dataset.parquet` + `manifest.json`
7. 上传到 R2/S3

### 5.2 本地运行 miner pipeline（不上传）

```python
#!/usr/bin/env python3
"""本地运行 miner pipeline，只生成数据集不上传。"""

import asyncio
from pathlib import Path
from nexis.miner.pipeline import MinerPipeline
from nexis.miner.captioner import Captioner
from nexis.config import load_settings


async def main():
    settings = load_settings()

    # 使用假 store（不上传）
    class FakeStore:
        async def upload_file(self, key, path, use_write=False):
            print(f"[FAKE UPLOAD] {key} -> {path}")

    store = FakeStore()

    captioner = Captioner(
        api_key=settings.openai_api_key,
        model=settings.caption_model,
        timeout_sec=settings.caption_timeout_sec,
    )

    pipeline = MinerPipeline(
        store=store,
        captioner=captioner,
    )

    dataset_path, manifest_path = await pipeline.run_interval(
        sources_file=Path(settings.sources_file),
        netuid=70,
        miner_hotkey="test_hotkey_local",
        interval_id=1,
        workdir=Path(settings.workdir),
    )

    print(f"\nDataset: {dataset_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
```

保存为 `scripts/local_pipeline.py` 并运行：

```bash
python3 scripts/local_pipeline.py
```

输出目录结构：

```
.nexis/
├── raw/                    # 下载的完整视频
├── clips/                  # 5.04s 切片
│   ├── clip_xxx.mp4
│   └── ...
├── frames/                 # 首帧 jpg
│   ├── clip_xxx.jpg
│   └── ...
└── out/
    └── 1/
        ├── dataset.parquet  # 400 条记录
        └── manifest.json    # 元数据
```

### 5.3 手动 FFmpeg 切片（调试/理解用）

如果你想手动理解切片逻辑：

```bash
# 假设有一个 60 秒的视频，切出第 0-5.04 秒
ffmpeg -ss 0 -t 5.04 -i input.mp4 \
    -vf "scale=1280:704,fps=24" \
    -c:v libx264 -preset fast -crf 18 \
    -frames:v 121 \
    clip_001.mp4

# 提取首帧
ffmpeg -i clip_001.mp4 -vf "select=eq(n\,0)" -vframes 1 frame_001.jpg

# 验证帧数
ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 clip_001.mp4
```

---

## 六、Step 4: 数据集质量自检

### 6.1 运行 dataset_check.py 离线验证

这是最关键的一步。`dataset_check.py` 是 Validator 使用的同一套验证逻辑，你可以本地提前跑通。

```python
#!/usr/bin/env python3
"""离线运行 dataset_check.py 验证本地生成的数据集。"""

import asyncio
from pathlib import Path
from nexis.validator.dataset_check import validate_miner_dataset


class FakeMinerStore:
    """假 store：从本地目录读取文件，不连接 R2。"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    async def download_file(self, key: str, dst: Path) -> bool:
        src = self.base_dir / key
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
            return True
        return False

    async def list_prefix(self, prefix: str):
        return []

    async def object_exists(self, key: str):
        return (self.base_dir / key).exists()


async def main():
    workdir = Path(".nexis")
    miner_dir = workdir / "test_hotkey_local" / "1"

    # 需要把 out/1/ 的内容复制到 validator 期望的结构
    # validate_miner_dataset 期望从 store 下载文件，我们直接用本地文件

    store = FakeMinerStore(base_dir=workdir / "out")

    # 创建一个空的 global_record_index（本地测试无历史数据）
    global_index = {}

    result = await validate_miner_dataset(
        miner_hotkey="test_hotkey_local",
        interval_id=1,
        miner_store=store,
        workdir=workdir,
        global_record_index=global_index,
        download_concurrency=16,
        download_retry_attempts=3,
    )

    print(f"Accepted: {result.accepted}")
    print(f"Record count: {result.record_count}")
    print(f"Global overlap: {result.global_overlap_count}")
    if result.failures:
        print(f"Failures: {result.failures}")
    print(f"Notes: {result.notes}")


if __name__ == "__main__":
    asyncio.run(main())
```

保存为 `scripts/validate_local.py`：

```bash
python3 scripts/validate_local.py
```

### 6.2 验证失败常见原因与修复

| 失败原因 | 含义 | 修复方法 |
|---------|------|---------|
| `manifest_missing` | manifest.json 不存在 | 检查 pipeline 输出 |
| `manifest_hotkey_mismatch` | manifest 中的 hotkey 不符 | 确保传入的 hotkey 一致 |
| `manifest_interval_mismatch` | interval_id 不符 | 传入正确的 interval_id |
| `record_count:xx!=400` | 记录数不是 400 | 增加 sources.txt 中的 URL |
| `spec:width:xxx!=1280` | 宽度不对 | FFmpeg scale=1280:704 |
| `spec:height:xxx!=704` | 高度不对 | FFmpeg scale=1280:704 |
| `spec:fps:xxx!=24` | 帧率不对 | FFmpeg fps=24 |
| `spec:num_frames:xxx!=121` | 帧数不对 | FFmpeg -frames:v 121 |
| `within_dataset_overlap` | 同一视频内切片重叠 <4.5s | 增大步长或去掉重复源 |
| `caption_missing` | caption 为空 | 配置 OPENAI_API_KEY |
| `clip_sha256_mismatch` | clip 文件被篡改 | 重新生成 dataset |
| `clip_probe_error` | ffprobe 无法解析 | 检查 FFmpeg 输出 |
| `global_overlap_exceeded` | 与历史数据重叠 >100 | 换新的视频源 |

### 6.3 快速验证单个 clip 的脚本

```bash
#!/bin/bash
# scripts/verify_clip.sh

CLIP="$1"

echo "=== File info ==="
ls -lh "$CLIP"

echo "=== SHA256 ==="
sha256sum "$CLIP"

echo "=== Video specs ==="
ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
    -of json "$CLIP" | python3 -m json.tool

echo "=== Frame count (actual) ==="
ffprobe -v error -select_streams v:0 -count_packets \
    -show_entries stream=nb_read_packets -of csv=p=0 "$CLIP"
```

运行：

```bash
chmod +x scripts/verify_clip.sh
./scripts/verify_clip.sh .nexis/clips/clip_xxx.mp4
```

---

## 七、Step 5: 离线 LoRA 训练自测

> **注意**：这一步需要 GPU，且耗时较长。如果你没有 GPU 或时间紧张，可以跳过到 Step 6 使用预训练模型，或者直接相信 dataset_check.py 通过即可。

### 7.1 下载基础模型

```bash
# 使用 huggingface-cli 下载 Wan2.2-TI2V-5B-Diffusers
pip install huggingface-hub
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --local-dir ./models/Wan2.2-TI2V-5B-Diffusers \
    --local-dir-use-symlinks False
```

模型约 10GB（fp16），下载时间较长。

### 7.2 准备 trainer manifest

Trainer 需要的输入格式是 `manifest.jsonl`，每行一个 JSON：

```json
{"video": "/path/to/clip_001.mp4", "prompt": "a forest waterfall in spring", "image": "/path/to/frame_001.jpg", "id": "clip_001"}
```

转换脚本：

```python
#!/usr/bin/env python3
"""将本地 dataset.parquet 转换为 trainer manifest.jsonl。"""

import json
from pathlib import Path
from nexis.serialization import read_dataset_parquet


def convert(miner_dir: Path, output_manifest: Path):
    records = read_dataset_parquet(miner_dir / "dataset.parquet")
    with output_manifest.open("w") as f:
        for row in records:
            clip_path = miner_dir / row.clip_uri.lstrip("/")
            frame_path = miner_dir / row.first_frame_uri.lstrip("/")
            entry = {
                "video": str(clip_path.absolute()),
                "prompt": row.caption or "a video",
                "image": str(frame_path.absolute()),
                "id": row.clip_id,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} entries to {output_manifest}")


if __name__ == "__main__":
    import sys
    miner_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".nexis/out/1")
    convert(miner_dir, Path("local_manifest.jsonl"))
```

### 7.3 修改 config.json 用于本地训练

复制 `config.json` 为 `config.local.json`，修改以下字段：

```json
{
  "active_dataset": "dataset_a",
  "datasets": {
    "dataset_a": {
      "manifest": "/absolute/path/to/local_manifest.jsonl",
      "notes": "local test"
    }
  },
  "training": {
    "max_train_steps": 200,
    "early_stop_warmup_steps": 100,
    "sample_every": 50,
    "validation_every": 50
  },
  "device": "cuda:0"
}
```

> 本地测试把 `max_train_steps` 降到 200 以节省时间。正式环境是 1000。

### 7.4 运行训练容器

```bash
# 拉取训练镜像
docker pull rendixnetwork/train:latest

# 运行训练
docker run --rm --gpus all \
    -v $(pwd)/models:/workspace/nexisgen/models \
    -v $(pwd)/local_manifest.jsonl:/workspace/training/manifest.jsonl \
    -v $(pwd)/config.local.json:/workspace/nexisgen/config.json \
    -v $(pwd)/local_runs:/workspace/nexisgen/runs \
    --shm-size=16g \
    rendixnetwork/train:latest \
    python 02_train_dataset.py
```

训练日志会输出到 `local_runs/`。

---

## 八、Step 6: 离线 VBench 打分

### 8.1 生成 eval 视频

训练完成后，用训练好的 LoRA 生成 eval 视频：

```bash
docker run --rm --gpus all \
    -v $(pwd)/models:/workspace/nexisgen/models \
    -v $(pwd)/local_runs:/workspace/nexisgen/runs \
    -v $(pwd)/eval_data:/workspace/eval_data \
    -v $(pwd)/local_outputs:/workspace/outputs \
    -v $(pwd)/config.local.json:/workspace/nexisgen/config.json \
    --shm-size=16g \
    rendixnetwork/train:latest \
    python 05_eval_with_images.py \
    --eval_manifest /workspace/eval_data/manifest.jsonl \
    --output_dir /workspace/outputs
```

> `eval_data/manifest.jsonl` 可以从 Validator 的 eval bucket 下载，或者自己准备 8-30 条测试数据。

### 8.2 运行 VBench 打分

```bash
# 拉取 VBench 镜像
docker pull rendixnetwork/vbench:latest

# 运行打分
docker run --rm --gpus all \
    -v $(pwd)/local_outputs:/workspace/videos \
    -v $(pwd)/vbench_results:/workspace/VBench/results \
    rendixnetwork/vbench:latest \
    python evaluate.py \
    --videos_path /workspace/videos \
    --output_path /workspace/VBench/results \
    --dimension i2v_subject,i2v_background,subject_consistency,background_consistency,motion_smoothness,dynamic_degree,aesthetic_quality,imaging_quality
```

### 8.3 解析分数

```python
#!/usr/bin/env python3
import json
from pathlib import Path

results = json.loads(Path("vbench_results/results.json").read_text())
for dim, score in results.items():
    print(f"{dim:30s}: {score:.4f}")
print(f"{'Average':30s}: {sum(results.values()) / len(results):.4f}")
```

**目标参考值**（基于社区经验，非官方）：

| 维度 | 较差 | 一般 | 优秀 |
|------|------|------|------|
| i2v_subject | <0.40 | 0.40-0.55 | >0.55 |
| i2v_background | <0.40 | 0.40-0.55 | >0.55 |
| subject_consistency | <0.60 | 0.60-0.75 | >0.75 |
| background_consistency | <0.60 | 0.60-0.75 | >0.75 |
| motion_smoothness | <0.60 | 0.60-0.75 | >0.75 |
| dynamic_degree | <0.40 | 0.40-0.55 | >0.55 |
| aesthetic_quality | <0.50 | 0.50-0.65 | >0.65 |
| imaging_quality | <0.50 | 0.50-0.65 | >0.65 |
| **Average** | <0.50 | 0.50-0.60 | >0.60 |

---

## 九、Step 7: 正式上传与监控

### 9.1 配置正式环境

确保 `.env` 中的 R2 凭据正确，且 bucket 名等于你的小写 hotkey。

### 9.2 运行正式 miner

```bash
# 提交凭据到链上（只需一次）
nexis commit-credentials

# 持续运行 miner
nexis mine
```

### 9.3 检查上传结果

```bash
# 使用 rclone 或 AWS CLI 检查 bucket 内容
aws s3 ls s3://你的小写hotkey/ --endpoint-url https://你的accountid.r2.cloudflarestorage.com
```

### 9.4 监控 Validator 反馈

1. **查看 invalid hotkeys**：
   ```bash
   curl -s "https://api.nexisgen.ai/v1/invalid-hotkeys" | jq .
   ```

2. **查看训练分数**（如果你进入 Top-5）：
   ```bash
   curl -s "https://api.nexisgen.ai/v1/training-scores" | jq '.[] | select(.hotkey=="你的hotkey")'
   ```

3. **本地日志**：
   ```bash
   tail -f .nexis/logs/miner.log
   ```

---

## 十、迭代优化策略

### 10.1 数据质量 > 数据数量

Validator 只取 400 条，多无用。关键是每条都要高质量：

1. **视觉清晰度**：4K > 1080p，避免压缩严重的视频
2. **内容多样性**：森林、海洋、山脉、城市、动物等混合
3. **运动特性**：有适度运动（流水、云动、树叶摇摆），避免完全静态
4. **光照稳定**：避免快速闪烁、过曝、欠曝
5. **Caption 质量**：描述要具体、准确，包含主体、场景、动作

### 10.2 Caption 优化

Caption 直接影响 LoRA 训练效果。建议：

- 避免模糊词："beautiful", "nice", "good"
- 使用具体描述："aerial view of tropical coastline with turquoise waves"
- 包含主体 + 场景 + 动作："red fox walking through snowy pine forest at sunset"
- 长度 10-20 个词最佳

### 10.3 去重策略

1. **同一视频内**：起始时间差 >= 4.5 秒（pipeline 自动处理）
2. **不同视频间**：避免使用同一个热门 YouTube 视频（会被全局索引命中）
3. **跨周期**：已用过的视频源在后续周期会触发全局去重，需持续寻找新源

### 10.4 A/B 测试框架

如果你有多组数据想对比，可以：

```python
# 生成两组 dataset，分别运行本地 VBench，比较平均分
# group_a: 纯自然风光
# group_b: 自然风光 + 城市航拍

# 分别生成 manifest.jsonl -> 分别训练 -> 分别打分 -> 比较 average
```

---

## 十一、常见问题排查

### Q1: yt-dlp 下载失败 / 被封 IP

- 添加 `--cookies-from-browser chrome` 使用登录态
- 使用代理：`--proxy socks5://127.0.0.1:1080`
- 降低并发，避免触发限流

### Q2: Caption 生成失败

- 检查 `OPENAI_API_KEY` 是否有效
- 检查网络是否能访问 OpenAI API
- 备用：配置 `GEMINI_API_KEY`

### Q3: dataset_check.py 报 `clip_num_frames` 错误

- 某些编码器 nb_frames 元数据不准，属于正常情况
- 确保 FFmpeg 切片时 `-frames:v 121` 严格执行
- 可容忍 +-1 帧误差

### Q4: 训练时 CUDA OOM

- 降低 batch_size（但 config.json 已固定为 1）
- 换用更大显存的 GPU
- 确认没有其它进程占用显存：`nvidia-smi`

### Q5: VBench 打分特别低

- 检查 eval 视频是否正常播放（不是黑屏/花屏）
- 检查首帧条件图是否正确传入
- 检查 caption 是否与视频内容匹配

---

## 十二、完整工作流速查表

```
Day 1: 环境搭建
  [ ] git clone + pip install -e .
  [ ] 安装 yt-dlp, ffmpeg, docker
  [ ] cp .env.example .env，填写 R2 和 OpenAI Key

Day 1-2: 数据源收集
  [ ] 用 YouTube 搜索收集 100+ 候选 URL
  [ ] 写入 sources.txt
  [ ] 运行 check_sources.py 预检
  [ ] 运行 download_sources.py 下载通过的视频

Day 2-3: 本地生成与验证
  [ ] 运行 local_pipeline.py 生成本地数据集
  [ ] 运行 validate_local.py 通过 dataset_check.py
  [ ] 抽查 clips 的 ffprobe 参数
  [ ] 检查 captions 质量

Day 3-5: 离线自测（可选但强烈推荐）
  [ ] 下载 Wan2.2-TI2V-5B-Diffusers 基础模型
  [ ] 转换 manifest.jsonl
  [ ] 运行 02_train_dataset.py（200 steps 快速测试）
  [ ] 运行 05_eval_with_images.py 生成 eval 视频
  [ ] 运行 VBench 打分
  [ ] 平均分 > 0.55 再进入正式流程

Day 5+: 正式挖矿
  [ ] nexis commit-credentials
  [ ] nexis mine
  [ ] 监控上传状态
  [ ] 监控 invalid hotkeys API
  [ ] 每周期更换 30-50% 视频源以避免全局去重
```

---

## 附录：关键文件路径速查

| 文件 | 路径 | 用途 |
|------|------|------|
| 协议常量 | `nexis/protocol.py` | CLIP_DURATION_SEC, TARGET_WIDTH 等 |
| 数据集验证 | `nexis/validator/dataset_check.py` | validate_miner_dataset() |
| Miner 流水线 | `nexis/miner/pipeline.py` | MinerPipeline.run_interval() |
| Caption 生成 | `nexis/miner/captioner.py` | Captioner.caption_frame() |
| 模型定义 | `nexis/models.py` | ClipRecord, IntervalManifest |
| 序列化 | `nexis/serialization.py` | read/write parquet, manifest |
| 训练配置 | `config.json` | LoRA 超参数 |
| 环境配置 | `.env` | API Key, R2 凭据 |

---

*本文档基于 Nexisgen v1.0.1 代码编写，后续版本可能有更新，请以实际代码为准。*
