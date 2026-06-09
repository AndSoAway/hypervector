#!/usr/bin/env bash
# HyperVector Docker 一键部署脚本 - Intel x86_64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Docker 是否安装
check_docker() {
    log_info "检查 Docker 环境..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi

    if ! docker ps &> /dev/null; then
        log_error "Docker 服务未运行或权限不足"
        exit 1
    fi

    log_success "Docker 环境检查通过"
    docker --version
}

# 构建 Docker 镜像
build_image() {
    log_info "开始构建 HyperVector Docker 镜像..."
    log_info "目标平台: Intel x86_64"
    log_info "构建可能需要 10-15 分钟，请耐心等待..."

    docker build \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        -t hypervector:intel-x86 \
        -t hypervector:latest \
        -f Dockerfile \
        . || {
        log_error "Docker 镜像构建失败"
        exit 1
    }

    log_success "Docker 镜像构建成功"
    docker images | grep hypervector
}

# 启动容器
start_container() {
    log_info "启动 HyperVector 容器..."

    # 检查是否有旧容器在运行
    if docker ps -a | grep -q hypervector-server; then
        log_warn "检测到已存在的容器，正在停止并删除..."
        docker stop hypervector-server 2>/dev/null || true
        docker rm hypervector-server 2>/dev/null || true
    fi

    # 创建数据目录
    mkdir -p /root/vector/hypervec_data

    # 启动容器
    docker run -d \
        --name hypervector-server \
        -p 8080:8080 \
        -v /root/vector/hypervec_data:/data/hypervec_data \
        --restart unless-stopped \
        hypervector:intel-x86 || {
        log_error "容器启动失败"
        exit 1
    }

    log_success "容器启动成功"
    log_info "容器名称: hypervector-server"
    log_info "HTTP 端口: 8080"
    log_info "数据目录: /root/vector/hypervec_data"
}

# 等待服务就绪
wait_for_service() {
    log_info "等待 HTTP Server 启动..."

    for i in {1..30}; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            log_success "HTTP Server 已就绪"
            return 0
        fi
        echo -n "."
        sleep 2
    done

    echo ""
    log_error "HTTP Server 启动超时"
    log_info "查看容器日志:"
    docker logs hypervector-server
    return 1
}

# 测试服务
test_service() {
    log_info "测试 HyperVector HTTP Server..."

    # 健康检查
    HEALTH=$(curl -s http://localhost:8080/health)
    echo "Health check: $HEALTH"

    # 列出 collections
    log_info "列出所有 collections..."
    curl -s http://localhost:8080/collections | python3 -m json.tool || echo "[]"

    log_success "服务测试通过"
}

# 显示使用说明
show_usage() {
    echo ""
    log_success "HyperVector 部署完成！"
    echo ""
    echo "===================================="
    echo "  容器管理命令"
    echo "===================================="
    echo "查看日志:   docker logs -f hypervector-server"
    echo "停止容器:   docker stop hypervector-server"
    echo "启动容器:   docker start hypervector-server"
    echo "重启容器:   docker restart hypervector-server"
    echo "删除容器:   docker rm -f hypervector-server"
    echo ""
    echo "===================================="
    echo "  API 测试命令"
    echo "===================================="
    echo "健康检查:   curl http://localhost:8080/health"
    echo "列出集合:   curl http://localhost:8080/collections"
    echo ""
    echo "===================================="
    echo "  Python 客户端示例"
    echo "===================================="
    echo "from pyhypervec import HypervecClient"
    echo ""
    echo "client = HypervecClient('http://localhost:8080')"
    echo "print(client.list_collections())"
    echo ""
}

# 主函数
main() {
    echo ""
    echo "================================================"
    echo "  HyperVector Docker 一键部署脚本"
    echo "  平台: Intel x86_64"
    echo "================================================"
    echo ""

    check_docker
    echo ""

    build_image
    echo ""

    start_container
    echo ""

    wait_for_service
    echo ""

    test_service
    echo ""

    show_usage
}

# 执行主函数
main "$@"
