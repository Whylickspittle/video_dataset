#!/bin/bash
# =============================================================================
# 一站式矿工执行脚本
#
# 功能：
#   1. 检查环境依赖
#   2. 读取 sources.txt
#   3. 流式切片 + 生成 caption
#   4. 导出 dataset.parquet + manifest.json
#   5. 上传到 R2
#
# 使用方法：
#   1. 修改下方 "用户配置区" 的变量
#   2. chmod +x scripts/run_full_miner.sh
#   3. ./scripts/run_full_miner.sh
# =============================================================================

set -euo pipefail  # 严格模式：遇错即停

# ═══════════════════════════════════════════════════════════════════════════
# 用户配置区（必须修改）
# ═══════════════════════════════════════════════════════════════════════════

# Bittensor 钱包
HOTKEY="YOUR_HOTKEY_HERE"                    # 你的 SS58 地址

# OpenAI API（用于生成 caption）
OPENAI_KEY="sk-YOUR_OPENAI_KEY_HERE"         # OpenAI API Key
CAPTION_MODEL="gpt-4o-mini"                  # 模型名称

# R2 存储配置
R2_ACCOUNT_ID="YOUR_ACCOUNT_ID"              # Cloudflare R2 Account ID
R2_BUCKET="YOUR_BUCKET_NAME"                 # bucket 名（小写 hotkey）
R2_ACCESS_KEY="YOUR_ACCESS_KEY"              # R2 Access Key
R2_SECRET_KEY="YOUR_SECRET_KEY"              # R2 Secret Key
R2_REGION="auto"                             # 区域

# 任务配置
INTERVAL_ID=1                                # interval 编号
CLIP_COUNT=400                               # 目标切片数
SOURCES_FILE="sources.txt"                   # 视频源列表
OUTPUT_DIR="./interval_out"                  # 输出目录

# ═══════════════════════════════════════════════════════════════════════════
# 以下内容通常不需要修改
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/miner_${TIMESTAMP}.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# ── Step 0: 检查配置 ──
log_info "=== Step 0: 检查配置 ==="

if [[ "$HOTKEY" == "YOUR_HOTKEY_HERE" ]]; then
    log_error "请修改脚本中的 HOTKEY 变量"
    exit 1
fi

if [[ "$OPENAI_KEY" == sk-YOUR_* ]]; then
    log_warn "OPENAI_KEY 未配置，将生成空 caption（可能被 Validator 拒绝）"
fi

if [[ "$R2_ACCOUNT_ID" == "YOUR_ACCOUNT_ID" ]]; then
    log_warn "R2 未配置，将跳过上传步骤"
    SKIP_UPLOAD=1
else
    SKIP_UPLOAD=0
fi

# ── Step 1: 检查依赖 ──
log_info "=== Step 1: 检查依赖 ==="

check_cmd() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 未安装，请先安装"
        exit 1
    fi
    log_info "✓ $1 已安装"
}

check_cmd python3
check_cmd yt-dlp
check_cmd ffmpeg
check_cmd ffprobe

# 检查 Python 依赖
if ! python3 -c "import pyarrow" 2>/dev/null; then
    log_info "安装 pyarrow..."
    pip install pyarrow
fi

if ! python3 -c "import openai" 2>/dev/null; then
    log_info "安装 openai..."
    pip install openai
fi

# ── Step 2: 检查 sources.txt ──
log_info "=== Step 2: 检查视频源 ==="

if [[ ! -f "$SOURCES_FILE" ]]; then
    log_error "sources.txt 不存在: $SOURCES_FILE"
    log_info "请创建 sources.txt，每行一个 YouTube URL"
    exit 1
fi

URL_COUNT=$(grep -c '^https*://' "$SOURCES_FILE" || true)
log_info "找到 $URL_COUNT 个视频源"

if [[ "$URL_COUNT" -lt 5 ]]; then
    log_warn "视频源较少，建议准备 20+ 个 URL"
fi

# ── Step 3: 运行 Pipeline ──
log_info "=== Step 3: 开始切片 + Caption ==="
log_info "输出目录: $OUTPUT_DIR"
log_info "目标数量: $CLIP_COUNT"

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

python3 scripts/miner_pipeline.py \
    --sources "$SOURCES_FILE" \
    --interval-id "$INTERVAL_ID" \
    --hotkey "$HOTKEY" \
    --openai-key "$OPENAI_KEY" \
    --caption-model "$CAPTION_MODEL" \
    --count "$CLIP_COUNT" \
    --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

# ── Step 4: 验证输出 ──
log_info "=== Step 4: 验证输出 ==="

CLIP_COUNT=$(ls -1 "$OUTPUT_DIR/clips/" 2>/dev/null | wc -l)
FRAME_COUNT=$(ls -1 "$OUTPUT_DIR/frames/" 2>/dev/null | wc -l)

log_info "切片文件: $CLIP_COUNT"
log_info "首帧文件: $FRAME_COUNT"

if [[ ! -f "$OUTPUT_DIR/dataset.parquet" ]]; then
    log_error "dataset.parquet 未生成"
    exit 1
fi

if [[ ! -f "$OUTPUT_DIR/manifest.json" ]]; then
    log_error "manifest.json 未生成"
    exit 1
fi

if [[ "$CLIP_COUNT" -ne "$CLIP_COUNT" ]]; then
    log_warn "切片数量不足: $CLIP_COUNT / $CLIP_COUNT"
fi

log_info "✓ 文件验证通过"

# ── Step 5: 上传到 R2 ──
if [[ "$SKIP_UPLOAD" -eq 1 ]]; then
    log_info "=== Step 5: 跳过上传（未配置 R2）==="
    log_info "请手动上传: $OUTPUT_DIR"
    exit 0
fi

log_info "=== Step 5: 上传到 R2 ==="

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# 配置 AWS CLI（临时）
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_KEY"

# 上传 clips
log_info "上传 clips..."
aws s3 sync "$OUTPUT_DIR/clips/" "s3://${R2_BUCKET}/${INTERVAL_ID}/clips/" \
    --endpoint-url "$R2_ENDPOINT" \
    --region "$R2_REGION" 2>&1 | tee -a "$LOG_FILE"

# 上传 frames
log_info "上传 frames..."
aws s3 sync "$OUTPUT_DIR/frames/" "s3://${R2_BUCKET}/${INTERVAL_ID}/frames/" \
    --endpoint-url "$R2_ENDPOINT" \
    --region "$R2_REGION" 2>&1 | tee -a "$LOG_FILE"

# 上传 dataset.parquet
log_info "上传 dataset.parquet..."
aws s3 cp "$OUTPUT_DIR/dataset.parquet" "s3://${R2_BUCKET}/${INTERVAL_ID}/" \
    --endpoint-url "$R2_ENDPOINT" \
    --region "$R2_REGION" 2>&1 | tee -a "$LOG_FILE"

# 上传 manifest.json（最后上传，作为完成信号）
log_info "上传 manifest.json..."
aws s3 cp "$OUTPUT_DIR/manifest.json" "s3://${R2_BUCKET}/${INTERVAL_ID}/" \
    --endpoint-url "$R2_ENDPOINT" \
    --region "$R2_REGION" 2>&1 | tee -a "$LOG_FILE"

log_info "✓ 上传完成"

# ── 完成 ──
log_info "=== 全部完成 ==="
log_info "本地输出: $OUTPUT_DIR"
log_info "日志文件: $LOG_FILE"
log_info "R2 路径: s3://${R2_BUCKET}/${INTERVAL_ID}/"
