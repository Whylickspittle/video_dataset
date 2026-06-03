#!/bin/bash
# =============================================================================
# 1C1G VPS 轻量级安装脚本（无需 git clone）
#
# 安装内容：
#   - yt-dlp (pip)
#   - ffmpeg (静态二进制)
#   - Python 3 + pyarrow + openai
#
# 使用方法：
#   wget https://raw.githubusercontent.com/YOUR_REPO/main/scripts/vps_install.sh
#   chmod +x vps_install.sh
#   ./vps_install.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

FFMPEG_VERSION="7.0.2"
ARCH=$(uname -m)

# 检测架构
detect_arch() {
    case "$ARCH" in
        x86_64)
            FFMPEG_ARCH="amd64"
            ;;
        aarch64)
            FFMPEG_ARCH="arm64"
            ;;
        *)
            log_error "不支持的架构: $ARCH"
            exit 1
            ;;
    esac
    log_info "检测到架构: $ARCH"
}

# 检查系统资源
check_resources() {
    local mem_mb=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}' || echo "0")
    local cpu_cores=$(nproc 2>/dev/null || echo "1")

    log_info "内存: ${mem_mb}MB, CPU: ${cpu_cores}核"

    if [[ "$mem_mb" -lt 512 ]]; then
        log_error "内存不足 512MB，无法运行"
        exit 1
    fi

    if [[ "$mem_mb" -lt 1024 ]]; then
        log_warn "内存仅 1GB，ffmpeg 编码会很慢，建议分批处理"
    fi
}

# 安装系统基础依赖
install_system_deps() {
    log_info "=== 安装系统基础依赖 ==="

    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        apt-get update -qq
        apt-get install -y -qq \
            python3 python3-pip python3-venv \
            wget curl ca-certificates \
            libgl1-mesa-glx 2>/dev/null || true
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL/Alibaba Cloud Linux
        yum install -y -q \
            python3 python3-pip \
            wget curl ca-certificates \
            libglvnd-glx 2>/dev/null || true
    elif command -v apk &> /dev/null; then
        # Alpine
        apk add --no-cache \
            python3 py3-pip \
            wget curl ca-certificates
    else
        log_error "未知的包管理器，请手动安装 python3 和 wget"
        exit 1
    fi

    log_info "✓ 系统依赖安装完成"
}

# 安装 ffmpeg 静态构建
install_ffmpeg() {
    log_info "=== 安装 ffmpeg 静态构建 ==="

    if command -v ffmpeg &> /dev/null && command -v ffprobe &> /dev/null; then
        log_info "ffmpeg 已安装: $(ffmpeg -version | head -1)"
        return 0
    fi

    local ffmpeg_url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"

    if [[ "$FFMPEG_ARCH" == "arm64" ]]; then
        # ARM64 使用 johnvansickle 的构建
        ffmpeg_url="https://johnvansickle.com/ffmpeg/releases/ffmpeg-${FFMPEG_VERSION}-arm64-static.tar.xz"
    fi

    log_info "下载 ffmpeg..."
    cd /tmp
    wget -q --show-progress "$ffmpeg_url" -O ffmpeg.tar.xz || {
        log_error "ffmpeg 下载失败"
        exit 1
    }

    log_info "解压 ffmpeg..."
    tar -xf ffmpeg.tar.xz

    # 查找解压后的目录
    local ffmpeg_dir=$(find /tmp -maxdepth 1 -type d -name "ffmpeg*" | head -1)

    if [[ -z "$ffmpeg_dir" ]]; then
        log_error "解压后找不到 ffmpeg 目录"
        exit 1
    fi

    # 复制到 /usr/local/bin
    find "$ffmpeg_dir" -name "ffmpeg" -type f -executable -exec cp {} /usr/local/bin/ffmpeg \;
    find "$ffmpeg_dir" -name "ffprobe" -type f -executable -exec cp {} /usr/local/bin/ffprobe \;

    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

    # 清理
    rm -rf ffmpeg.tar.xz "$ffmpeg_dir"

    log_info "✓ ffmpeg 安装完成: $(ffmpeg -version | head -1)"
}

# 安装 yt-dlp
install_ytdlp() {
    log_info "=== 安装 yt-dlp ==="

    if command -v yt-dlp &> /dev/null; then
        log_info "yt-dlp 已安装: $(yt-dlp --version)"
        return 0
    fi

    # 方法1: pip 安装（推荐，自动更新方便）
    pip3 install -q --user yt-dlp 2>/dev/null || {
        log_warn "pip 安装失败，尝试直接下载二进制..."
        # 方法2: 直接下载二进制
        wget -q https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -O /usr/local/bin/yt-dlp
        chmod +x /usr/local/bin/yt-dlp
    }

    log_info "✓ yt-dlp 安装完成: $(yt-dlp --version)"
}

# 安装 Python 依赖
install_python_deps() {
    log_info "=== 安装 Python 依赖 ==="

    # 创建虚拟环境（避免系统包冲突）
    local venv_dir="$HOME/.nexis_venv"

    if [[ ! -d "$venv_dir" ]]; then
        python3 -m venv "$venv_dir"
        log_info "创建虚拟环境: $venv_dir"
    fi

    source "$venv_dir/bin/activate"

    # 升级 pip
    pip install -q --upgrade pip

    # 安装依赖（使用预编译 wheel，避免源码编译）
    log_info "安装 pyarrow, openai..."
    pip install -q pyarrow openai pillow requests tqdm

    log_info "✓ Python 依赖安装完成"
    log_info "虚拟环境路径: $venv_dir"
    log_info "激活命令: source $venv_dir/bin/activate"
}

# 创建最小化工作目录
setup_workspace() {
    log_info "=== 创建工作目录 ==="

    local work_dir="$HOME/nexis_miner"
    mkdir -p "$work_dir"/{clips,frames,logs}

    log_info "工作目录: $work_dir"
}

# 主函数
main() {
    log_info "开始安装 Nexis Miner 依赖..."

    detect_arch
    check_resources
    install_system_deps
    install_ffmpeg
    install_ytdlp
    install_python_deps
    setup_workspace

    log_info "=== 全部安装完成 ==="
    echo ""
    echo "使用方式:"
    echo "  1. 激活环境: source ~/.nexis_venv/bin/activate"
    echo "  2. 创建工作目录: cd ~/nexis_miner"
    echo "  3. 下载脚本: wget ..."
    echo "  4. 执行挖矿"
    echo ""
    echo "1C1G 优化建议:"
    echo "  - ffmpeg 使用 ultrafast preset（已内置）"
    echo "  - 每次只处理 1 个视频"
    echo "  - 避免并行下载+编码"
}

main "$@"
