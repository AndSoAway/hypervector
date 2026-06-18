#!/bin/bash
# HyperVector Wiki 1M 完整测试流程

set -e

echo "============================================================"
echo "  HyperVector Wiki 1M 完整测试流程"
echo "============================================================"
echo ""

# 配置
FBIN_FILE="/data/ljx/test_fbin/wiki_all_1M/base.1M.fbin"
QUERY_FILE="/data/ljx/test_fbin/wiki_all_1M/query.public.10K.fbin"
COLLECTION_NAME="wiki_hnsw_1m"

# 步骤 0：检查环境
echo "步骤 0: 检查环境"
echo "------------------------------------------------------------"

# 检查 HyperVector Server
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ HyperVector Server 正在运行"
else
    echo "❌ HyperVector Server 未运行"
    echo "提示：请先运行 ./deploy_docker.sh"
    exit 1
fi

# 检查数据文件
if [ ! -f "$FBIN_FILE" ]; then
    echo "❌ 数据文件不存在: $FBIN_FILE"
    echo "提示：请确认数据文件路径是否正确"
    exit 1
fi
echo "✅ Wiki 1M 数据文件存在"

if [ ! -f "$QUERY_FILE" ]; then
    echo "⚠️  查询文件不存在: $QUERY_FILE"
    echo "将使用随机生成的查询向量"
fi

echo ""
read -p "按 Enter 继续..."

# 步骤 1：导入数据
echo ""
echo "============================================================"
echo "步骤 1: 导入 Wiki 1M 数据"
echo "============================================================"
echo ""
echo "这一步大约需要 20-30 分钟"
echo "如果数据已导入，可以选择跳过"
echo ""
read -p "是否需要导入数据？(y/N): " IMPORT

if [ "$IMPORT" = "y" ] || [ "$IMPORT" = "Y" ]; then
    python3 import_wiki_1m.py \
        --file "$FBIN_FILE" \
        --collection "$COLLECTION_NAME" \
        --batch-size 10000 \
        --M 32 \
        --ef-construction 200

    if [ $? -ne 0 ]; then
        echo "❌ 数据导入失败"
        exit 1
    fi
else
    echo "⏭️  跳过数据导入"
fi

echo ""
read -p "按 Enter 继续..."

# 步骤 2：运行性能测试
echo ""
echo "============================================================"
echo "步骤 2: 运行性能基准测试"
echo "============================================================"
echo ""
echo "测试配置："
echo "  - 10,000 次查询"
echo "  - 100 个并发用户"
echo "  - 测试 3 个 ef_search 参数 (128, 256, 512)"
echo ""
echo "预计耗时：5-10 分钟"
echo ""
read -p "按 Enter 开始测试..."

python3 benchmark_wiki_1m.py \
    --queries-file "$QUERY_FILE" \
    --collection "$COLLECTION_NAME" \
    --num-queries 10000 \
    --concurrent 100 \
    --ef-search "128,256,512"

if [ $? -ne 0 ]; then
    echo "❌ 性能测试失败"
    exit 1
fi

echo ""
echo "============================================================"
echo "  ✅ 测试流程完成！"
echo "============================================================"
echo ""
