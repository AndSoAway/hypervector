#!/bin/bash
# =============================================================================
# hypervector 跨平台一键编译脚本
#
# 用法：cd 到项目目录，执行 ./build.sh
#
# 自动检测：CPU 架构(ARM/x86)、编译器、数学库
# 自动补齐：缺少的工具通过 Miniconda 安装（不需要 root 权限）
# 编译产物：libhypervec.a + HNSW/IVF demo
#
# 支持平台：飞腾(ARM) / 鲲鹏(ARM) / 海光(x86) / Intel(x86)
# =============================================================================

# 遇到错误立即退出，防止出错后继续执行
set -e

# ---------- 终端彩色输出 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'       # 恢复默认颜色

# 四种日志函数：info(蓝色)、ok(绿色)、warn(黄色)、err(红色并退出)
info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ---------- 全局变量 ----------
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"   # 项目根目录
BUILD_DIR="$PROJECT_DIR/build"                   # 编译输出目录
CONDA_DIR=""                                     # Miniconda 安装路径（用的时候才赋）

CPU_ARCH=$(uname -m)                             # 获取 CPU 架构，如 aarch64、x86_64

echo ""
echo "================================================"
echo "  hypervector 跨平台自动编译脚本"
echo "================================================"
echo ""
info "检测到 CPU 架构: $CPU_ARCH"

# ========== 步骤 1：识别 CPU 架构 ==========
# ARM 架构 → 飞腾 / 鲲鹏
# x86 架构 → 海光 / Intel
case "$CPU_ARCH" in
    aarch64|arm64|armv8*)
        ARCH_TYPE="arm"
        ok "ARM 架构 - 兼容飞腾/鲲鹏"
        ;;
    x86_64|amd64)
        ARCH_TYPE="x86"
        ok "x86 架构 - 兼容海光/Intel"
        ;;
    *)
        warn "未知架构: $CPU_ARCH，尝试继续编译"
        ARCH_TYPE="unknown"
        ;;
esac

# conda 的包名和 Linux uname 不一样：
# x86_64 → conda 叫 "64"     （gxx_linux-64）
# aarch64 → conda 叫 "aarch64"（gxx_linux-aarch64）
case "$CPU_ARCH" in
    x86_64|amd64)  CONDA_ARCH="64" ;;
    aarch64|arm64) CONDA_ARCH="aarch64" ;;
    *)             CONDA_ARCH="$CPU_ARCH" ;;
esac

# ---------- 工具函数 ----------

# 检查某个命令是否存在
has_cmd() { command -v "$1" &>/dev/null; }

# 检查并报告某个编译工具是否存在
check_have() {
    local name=$1 cmd=$2 pkg=$3
    if has_cmd "$cmd"; then
        ok "$name: $(command -v "$cmd")"
        return 0
    else
        warn "$name: 未找到"
        return 1
    fi
}

# ---------- 安装 Miniconda 编译环境 ----------
# 当系统缺少 g++/cmake/OpenBLAS 时调用
# 全部装到用户 HOME 下，不需要 root 权限
install_conda() {
    info "开始通过 Miniconda 安装编译环境..."
    CONDA_DIR="$HOME/miniconda3"

    # 如果已经装过 miniconda，就跳过下载步骤
    if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
        ok "Miniconda 已存在于 $CONDA_DIR"
    else
        # 根据 CPU 架构下载对应版本的 miniconda
        info "下载 Miniconda..."
        local url
        case "$CPU_ARCH" in
            aarch64) url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
            x86_64)  url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
            *)       err "不支持的 CPU 架构: $CPU_ARCH" ;;
        esac
        wget -q "$url" -O /tmp/miniconda_install.sh || err "Miniconda 下载失败"
        # -b: 静默安装  -p: 指定安装路径
        bash /tmp/miniconda_install.sh -b -p "$CONDA_DIR" || err "Miniconda 安装失败"
        rm -f /tmp/miniconda_install.sh
        ok "Miniconda 安装完成"
    fi

    # 接受 conda 服务条款（不然后续 install 会卡住）
    "$CONDA_DIR/bin/conda" tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main \
        --channel https://repo.anaconda.com/pkgs/r &>/dev/null || true

    # 通过 conda-forge 安装：C++ 编译器、CMake、Make、OpenBLAS、LAPACK
    info "安装编译工具..."
    "$CONDA_DIR/bin/conda" install -y -c conda-forge \
        gxx_linux-${CONDA_ARCH} cmake make openblas lapack || \
        err "conda 安装工具失败"
    ok "conda 工具安装完成"

    # 创建 gcc/g++ 软链接，方便直接使用
    ln -sf "$CONDA_DIR/bin/${CPU_ARCH}-conda-linux-gnu-gcc" "$CONDA_DIR/bin/gcc" 2>/dev/null || true
    ln -sf "$CONDA_DIR/bin/${CPU_ARCH}-conda-linux-gnu-g++" "$CONDA_DIR/bin/g++" 2>/dev/null || true

    # 设置环境变量，指向 conda 的编译器
    export PATH="$CONDA_DIR/bin:$PATH"
    export CC="$CONDA_DIR/bin/gcc"
    export CXX="$CONDA_DIR/bin/g++"
}

# ========== 步骤 2：检查编译工具 ==========
# 如果 g++、cmake、make 任意一个缺了，就需要用 conda
NEED_CONDA=false

echo ""
info "--- 检查编译器 ---"
check_have "g++"      "g++"      "gcc-c++" || NEED_CONDA=true
check_have "cmake"    "cmake"    "cmake"   || NEED_CONDA=true
check_have "make"     "make"     "make"    || NEED_CONDA=true

# ========== 步骤 3：检查数学库 ==========
# 项目需要 BLAS/LAPACK（矩阵运算），OpenBLAS 是最常用的开源实现
echo ""
info "--- 检查数学库 ---"

HAVE_BLAS=false
# 逐个位置查找 libopenblas.so
if ldconfig -p 2>/dev/null | grep -q libopenblas; then
    ok "OpenBLAS: 系统已安装"
    HAVE_BLAS=true
elif [ -d "/usr/lib64" ] && ls /usr/lib64/libopenblas* &>/dev/null; then
    ok "OpenBLAS: /usr/lib64 已安装"
    HAVE_BLAS=true
elif [ -d "/usr/lib" ] && ls /usr/lib/libopenblas* &>/dev/null; then
    ok "OpenBLAS: /usr/lib 已安装"
    HAVE_BLAS=true
else
    warn "OpenBLAS: 未找到"
    HAVE_BLAS=false
fi

# 如果编译器都在但缺数学库，也走 conda 方案
if ! $HAVE_BLAS && ! $NEED_CONDA; then
    warn "缺少数学库，切换到 conda 方案"
    NEED_CONDA=true
fi

echo ""

# ========== 步骤 4：决定编译方式 ==========
if $NEED_CONDA; then
    # 环境不完整 → 走 conda（自动装 g++ cmake openblas lapack）
    info "环境不完整，通过 conda 补齐"
    install_conda
else
    # 环境完整 → 直接用系统自带的编译器
    ok "系统环境完整，直接使用系统工具编译"
    export CC=gcc
    export CXX=g++
    CONDA_DIR=""
fi

# ========== 步骤 5：CMake 配置 ==========
echo ""
echo "================================================"
info "开始 CMake 配置..."

# 清空旧的编译缓存，避免路径冲突
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# -DCMAKE_BUILD_TYPE=Release  →  开启编译优化，代码跑得更快
# -DBUILD_TESTING=OFF         →  跳过单元测试下载，加快配置
# -DHYPERVEC_ENABLE_EXTRAS=ON →  编译 demo 示例
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DHYPERVEC_ENABLE_EXTRAS=ON \
    || err "CMake 配置失败"

ok "CMake 配置成功"

# ========== 步骤 6：编译核心库 ==========
echo ""
info "开始编译核心库..."
JOBS=$(nproc 2>/dev/null || echo 4)           # 用全部 CPU 核心并行编译
make -j"$JOBS" hypervec || err "编译失败"       # hypervec 目标 = libhypervec.a
ok "核心库编译成功"

# ========== 步骤 7：编译 Demo 示例 ==========
echo ""
info "开始编译示例 demo..."
make -j"$JOBS" demo_hnsw_indexing demo_ivf_indexing 2>&1 | tail -5
ok "Demo 编译成功"

# ========== 步骤 8：打印结果 ==========
echo ""
echo "================================================"
echo -e "${GREEN}  全部编译完成！${NC}"
echo "================================================"
echo ""
echo "  产物位置:"
echo "    库:    $BUILD_DIR/src/libhypervec.a"
echo "    Demo:  $BUILD_DIR/test/examples/cpp/"
echo ""
echo "  运行 demo:"
echo "    $BUILD_DIR/test/examples/cpp/demo_hnsw_indexing"
echo "    $BUILD_DIR/test/examples/cpp/demo_ivf_indexing"
echo ""
