#!/bin/bash
# HyperVector Examples Interface 完整演示脚本

echo "============================================================"
echo "  HyperVector Examples Interface 演示"
echo "============================================================"
echo ""

# 1. 测试 Server 端接口
echo "📡 Server 端接口测试"
echo "------------------------------------------------------------"
echo ""

echo "1️⃣  列出所有支持的索引类型："
curl -s http://localhost:8080/examples | python3 -m json.tool
echo ""
echo ""

echo "2️⃣  获取 HNSW 索引详细示例："
echo "------------------------------------------------------------"
curl -s http://localhost:8080/examples/HNSW | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'索引名称：{data[\"name\"]}')
print(f'完整名称：{data[\"full_name\"]}')
print(f'功能介绍：{data[\"description\"]}')
print(f'使用场景：{data[\"use_case\"]}')
print()
print('参数说明：')
for param, info in data['parameters'].items():
    print(f'  • {param}:')
    print(f'    描述: {info[\"description\"]}')
    print(f'    范围: {info[\"range\"]}')
    print(f'    默认: {info[\"default\"]}')
    print(f'    建议: {info[\"recommendation\"]}')
print()
print('Python 代码示例（创建 Collection）：')
print(data['example_code']['python']['create'])
print()
print('性能建议：')
for i, tip in enumerate(data['performance_tips'], 1):
    print(f'  {i}. {tip}')
"
echo ""
echo ""

echo "3️⃣  获取 Flat 索引示例："
echo "------------------------------------------------------------"
curl -s http://localhost:8080/examples/Flat | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'索引名称：{data[\"name\"]}')
print(f'功能介绍：{data[\"description\"]}')
print(f'使用场景：{data[\"use_case\"]}')
"
echo ""
echo ""

echo "4️⃣  测试错误处理（无效索引类型）："
echo "------------------------------------------------------------"
curl -s http://localhost:8080/examples/InvalidIndex | python3 -m json.tool
echo ""
echo ""

# 2. 测试 Client 端接口
echo "🐍 Client 端接口测试"
echo "------------------------------------------------------------"
echo ""

python3 << 'EOF'
import sys
sys.path.insert(0, '/root/vector/hypervector/pyhypervec')
from pyhypervec import HypervecClient

client = HypervecClient('http://localhost:8080')

print("5️⃣  使用 Python Client 获取示例：")
print()

# 列出所有索引
all_examples = client.get_examples()
print(f"✅ 支持的索引类型：{', '.join(all_examples['supported_indexes'])}")
print()

# 获取 HNSW 示例
hnsw = client.get_examples('HNSW')
print(f"✅ HNSW 索引描述：{hnsw['description']}")
print()

# 获取 IVF 示例
ivf = client.get_examples('IVF')
print(f"✅ IVF 索引描述：{ivf['description']}")
print()

print("✅ Python Client get_examples() 方法工作正常！")
EOF

echo ""
echo "============================================================"
echo "  ✅ 演示完成！"
echo "============================================================"
echo ""
echo "总结："
echo "  • Server 端提供 GET /examples 和 GET /examples/{type}"
echo "  • Client 端提供 get_examples(index_type) 方法"
echo "  • 支持 8 种索引类型的完整文档"
echo "  • 每种索引包含：介绍、参数、代码示例、性能建议"
echo ""
