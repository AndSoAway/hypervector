# -*- coding: utf-8 -*-
"""
HyperVector 索引类型使用示例数据

提供 8 种向量索引的详细文档，包括：
- 算法原理和特点
- 适用场景
- 参数详解
- 完整代码示例
- 性能调优建议
"""

EXAMPLES_DATA = {
    "HNSW": {
        "name": "HNSW",
        "full_name": "Hierarchical Navigable Small World Graph（分层导航小世界图）",

        "description": """HNSW 是一种基于图结构的向量索引算法，通过构建多层导航图实现高效的近似最近邻搜索。

【核心原理】
HNSW 将小世界网络的可导航性与分层结构相结合：
• 底层：包含所有数据点，形成密集连接的图
• 上层：逐层稀疏化，作为"高速公路"加速搜索
• 搜索过程：从顶层开始，逐层下降直到找到最近邻

【技术特点】
• 基于 NSW（Navigable Small World）算法改进
• 对数级搜索复杂度：O(log N)
• 支持动态增删，适合在线服务
• 通过参数灵活平衡速度与精度""",

        "use_case": """【推荐场景】
✓ 大规模向量检索（百万级 - 亿级）
✓ 实时查询响应（<10ms）
✓ 高召回率要求（>95%）
✓ 图像检索、语义搜索、推荐系统

【适用数据规模】
• 最佳：10万 - 1亿条向量
• 可用：1万 - 10亿条向量
• 推荐维度：50 - 2048 维""",

        "advantages": [
            "召回率高：合理参数下可达 99% 以上",
            "查询速度快：毫秒级响应，对数复杂度",
            "可扩展性强：支持动态增删，在线更新",
            "参数灵活：可根据需求调整速度与精度平衡",
            "内存效率：相比 Flat 减少 90% 以上计算量"
        ],

        "limitations": [
            "内存占用：需存储完整图结构，约为原始数据的 1.5-3 倍",
            "构建时间：初始构建较慢，百万数据约需 5-10 分钟",
            "参数敏感：需根据数据特性调整参数以达最佳性能",
            "近似算法：非 100% 精确（但可通过参数接近精确）"
        ],

        "parameters": {
            "M": {
                "name": "M",
                "description": "每个节点在图中的最大连接数（出度）",
                "type": "int",
                "range": "4 - 128（推荐 16-64）",
                "default": 16,
                "impact": "• 增大 M：提高召回率，增加内存占用（平方关系）\n• 减小 M：降低内存，但召回率下降",
                "recommendation": """• 小数据集（<10万）：M=16
• 中等数据（10-100万）：M=32
• 大规模数据（>100万）：M=48-64
• 超大规模（>1000万）：M=64-96

【内存占用参考】
M=16: +50% 内存
M=32: +100% 内存
M=64: +200% 内存""",
                "example": "index_params={'M': 32}"
            },
            "ef_construction": {
                "name": "ef_construction",
                "description": "构建索引时的搜索宽度，控制构建质量",
                "type": "int",
                "range": "50 - 1000（推荐 100-500）",
                "default": 200,
                "impact": "• 增大：索引质量更高，但构建时间显著增加\n• 减小：构建更快，但可能影响后续搜索召回率",
                "recommendation": """• 通用建议：ef_construction = M × 10 ~ 20
• 快速构建：100-200（适合开发测试）
• 平衡模式：200-400（推荐生产环境）
• 高质量：400-800（追求最优召回率）

【构建时间参考】（百万数据，M=32）
ef=100: ~3 分钟
ef=200: ~5 分钟
ef=400: ~8 分钟
ef=800: ~15 分钟""",
                "example": "index_params={'ef_construction': 200}"
            },
            "ef_search": {
                "name": "ef_search",
                "description": "搜索时的候选集大小，运行时可动态调整",
                "type": "int",
                "range": "k - 2000（推荐 limit × 2 ~ 20）",
                "default": 128,
                "impact": "• 增大：召回率提升，但查询变慢（线性关系）\n• 减小：查询更快，召回率下降",
                "recommendation": """• 快速查询：ef = limit × 2（召回率 ~90%）
• 平衡模式：ef = limit × 5（召回率 ~95%）
• 高精度：ef = limit × 10-20（召回率 >99%）

【性能与召回率权衡】（limit=10）
ef=20:  0.5ms, 召回率 88%
ef=50:  0.8ms, 召回率 93%
ef=100: 1.2ms, 召回率 96%
ef=200: 2.0ms, 召回率 98%
ef=500: 4.5ms, 召回率 99.5%

【动态调整示例】
• 实时服务：ef=50-100（速度优先）
• 精准召回：ef=200-500（质量优先）
• A/B 测试：同一查询尝试不同 ef 值""",
                "example": "search_params={'ef_search': 256}"
            }
        },

        "example_code": {
            "python": {
                "step1_create": """# ============================================================
# 步骤 1：创建 HNSW Collection
# ============================================================
from pyhypervec import HypervecClient

# 连接到 HyperVector Server
client = HypervecClient('http://localhost:8080')

# 创建 Collection 并配置 HNSW 索引
response = client.create_collection(
    collection_name='wiki_vectors',      # Collection 名称
    dimension=768,                        # 向量维度（如 BERT embeddings）
    index_type='HNSW',                   # 索引类型
    metric_type='L2',                    # 距离度量（L2/IP/COSINE）
    index_params={
        'M': 32,                         # 每个节点最大连接数
        'ef_construction': 200           # 构建时搜索宽度
    }
)

print(f"✅ Collection 创建成功：{response}")""",

                "step2_insert": """# ============================================================
# 步骤 2：插入向量数据
# ============================================================
import numpy as np

# 方式 1：逐批插入（推荐，避免内存溢出）
batch_size = 1000
for i in range(0, total_vectors, batch_size):
    batch_data = [
        {
            'id': f'doc_{j}',
            'vector': embeddings[j].tolist(),
            # 可选：添加元数据字段
            'title': titles[j],
            'category': categories[j]
        }
        for j in range(i, min(i + batch_size, total_vectors))
    ]

    client.insert(
        collection_name='wiki_vectors',
        data=batch_data
    )
    print(f"已插入 {i + len(batch_data)}/{total_vectors} 条")

# 方式 2：单条插入（适合实时更新）
client.insert(
    collection_name='wiki_vectors',
    data=[{
        'id': 'doc_new',
        'vector': new_embedding.tolist()
    }]
)""",

                "step3_build": """# ============================================================
# 步骤 3：构建索引（Flush）
# ============================================================
import time

print("开始构建 HNSW 索引...")
start_time = time.time()

# 构建索引（必须调用，否则无法搜索）
result = client.flush(collection_name='wiki_vectors')

build_time = time.time() - start_time

print(f"✅ 索引构建完成")
print(f"   - 向量数量：{result['total']}")
print(f"   - 向量维度：{result['dim']}")
print(f"   - 构建耗时：{build_time:.2f} 秒")
print(f"   - 索引大小：{result['index_size_bytes'] / 1024 / 1024:.2f} MB")""",

                "step4_search": """# ============================================================
# 步骤 4：向量搜索
# ============================================================

# 查询向量（例如用户输入的文本经过 embedding）
query_text = "机器学习算法"
query_vector = embed_model.encode(query_text)  # 使用你的 embedding 模型

# 执行搜索
results = client.search(
    collection_name='wiki_vectors',
    data=[query_vector.tolist()],              # 查询向量
    limit=10,                                   # 返回 Top-K 结果
    search_params={'ef_search': 200},          # 搜索参数（可动态调整）
    output_fields=['id', 'title', 'category'] # 返回的字段
)

# 解析结果
print("\\n搜索结果（Top-10）：")
print("=" * 70)
for i, result in enumerate(results[0], 1):
    print(f"{i}. ID: {result['id']}")
    print(f"   标题: {result.get('title', 'N/A')}")
    print(f"   相似度: {result['distance']:.4f}")
    print(f"   类别: {result.get('category', 'N/A')}")
    print("-" * 70)

# 调整 ef_search 进行对比测试
for ef in [50, 100, 200, 500]:
    start = time.time()
    results = client.search(
        collection_name='wiki_vectors',
        data=[query_vector.tolist()],
        limit=10,
        search_params={'ef_search': ef}
    )
    latency = (time.time() - start) * 1000
    print(f"ef_search={ef:3d}: {latency:.2f} ms")"""
            },

            "curl": {
                "step1_create": """# ============================================================
# 步骤 1：创建 HNSW Collection
# ============================================================
curl -X POST http://localhost:8080/collections/wiki_vectors/create \\
  -H 'Content-Type: application/json' \\
  -d '{
    "schema": {
      "auto_id": false,
      "fields": [
        {
          "name": "id",
          "datatype": "VARCHAR",
          "is_primary": true,
          "max_length": 128
        },
        {
          "name": "vector",
          "datatype": "FLOAT_VECTOR",
          "dim": 768
        },
        {
          "name": "title",
          "datatype": "VARCHAR",
          "max_length": 256
        },
        {
          "name": "category",
          "datatype": "VARCHAR",
          "max_length": 64
        }
      ]
    },
    "index_params": {
      "indexes": [{
        "field_name": "vector",
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {
          "M": 32,
          "ef_construction": 200
        }
      }]
    }
  }'""",

                "step2_insert": """# ============================================================
# 步骤 2：插入向量数据
# ============================================================
curl -X POST http://localhost:8080/collections/wiki_vectors/insert \\
  -H 'Content-Type: application/json' \\
  -d '{
    "data": [
      {
        "id": "doc_001",
        "vector": [0.12, 0.34, ..., 0.78],
        "title": "机器学习入门",
        "category": "AI"
      },
      {
        "id": "doc_002",
        "vector": [0.45, 0.67, ..., 0.89],
        "title": "深度学习实战",
        "category": "AI"
      }
    ]
  }'""",

                "step3_build": """# ============================================================
# 步骤 3：构建索引
# ============================================================
curl -X POST http://localhost:8080/collections/wiki_vectors/flush""",

                "step4_search": """# ============================================================
# 步骤 4：向量搜索
# ============================================================
curl -X POST http://localhost:8080/collections/wiki_vectors/search \\
  -H 'Content-Type: application/json' \\
  -d '{
    "data": [[0.23, 0.45, ..., 0.67]],
    "limit": 10,
    "search_params": {"ef_search": 200},
    "output_fields": ["id", "title", "category"]
  }'"""
            }
        },

        "performance_tips": [
            "【参数调优原则】\n  • M 值：优先保证召回率，内存充足时尽量用大值（32-64）\n  • ef_construction：一次性构建，可以设大一些（200-400），换取更好的长期性能\n  • ef_search：根据实际延迟要求动态调整，建议从 limit×5 开始测试",

            "【内存优化】\n  • 使用量化技术：结合 HNSWLVQ 可减少 50-70% 内存\n  • 控制 M 值：M=32 是性价比平衡点\n  • 数据预处理：L2 normalize 可提升性能",

            "【构建速度优化】\n  • 批量插入：每批 1000-10000 条，避免频繁 I/O\n  • 并行构建：多个 Collection 可并行构建\n  • 增量更新：支持动态添加，无需重建全部索引",

            "【查询性能优化】\n  • 预热索引：首次查询较慢，建议预热（执行几次查询）\n  • 批量查询：多个查询合并为一次请求可提升吞吐\n  • 缓存策略：高频查询结果可缓存",

            "【生产环境建议】\n  • 推荐配置：M=32, ef_construction=200, ef_search=100-200\n  • 监控指标：QPS、P99延迟、召回率、内存占用\n  • 降级策略：高负载时降低 ef_search 保证延迟"
        ],

        "real_world_examples": [
            {
                "scenario": "图像检索系统（1000万图片）",
                "config": "M=48, ef_construction=400, ef_search=200",
                "performance": "QPS=500, P99延迟=5ms, 召回率=98.5%",
                "hardware": "64GB 内存，索引占用 45GB"
            },
            {
                "scenario": "语义搜索（500万文档）",
                "config": "M=32, ef_construction=200, ef_search=150",
                "performance": "QPS=800, P99延迟=3ms, 召回率=97.2%",
                "hardware": "32GB 内存，索引占用 20GB"
            },
            {
                "scenario": "实时推荐（2亿商品）",
                "config": "M=64, ef_construction=500, ef_search=100",
                "performance": "QPS=1200, P99延迟=8ms, 召回率=96.8%",
                "hardware": "256GB 内存，索引占用 180GB"
            }
        ]
    },

    "Flat": {
        "name": "Flat",
        "full_name": "Flat Index（暴力搜索 / 精确搜索）",
        "description": """Flat 索引通过暴力计算查询向量与所有数据向量的距离，返回精确的最近邻结果。

【核心原理】
• 遍历所有向量，计算与查询向量的距离
• 选取距离最小的 Top-K 个结果
• 100% 精确，无近似误差

【适用场景】
• 小规模数据（<10万条）
• 需要绝对精确的结果
• 作为其他索引的 baseline 对比""",

        "use_case": "适合小数据集或需要 100% 召回率的场景，常用于算法验证和性能对比基准",
        "advantages": ["100% 精确召回", "无需构建索引", "实现简单"],
        "limitations": ["数据量大时极慢（O(N) 复杂度）", "不可扩展"],
        "parameters": {},
        "example_code": {
            "python": {
                "step1_create": """# ============================================================
# 步骤 1：创建 Flat Collection
# ============================================================
from pyhypervec import HypervecClient

client = HypervecClient('http://localhost:8080')

# 创建 Flat 索引（无需额外参数）
client.create_collection(
    collection_name='my_flat',
    dimension=768,
    index_type='Flat',
    metric_type='L2'  # 或 IP（内积）
)
print("✅ Flat Collection 创建成功")""",

                "step2_insert": """# ============================================================
# 步骤 2：插入向量数据
# ============================================================
import numpy as np

# 生成测试数据
vectors = np.random.randn(10000, 768).astype('float32')

# 批量插入
for i in range(0, len(vectors), 1000):
    batch = [
        {'id': f'vec_{j}', 'vector': vectors[j].tolist()}
        for j in range(i, min(i + 1000, len(vectors)))
    ]
    client.insert(collection_name='my_flat', data=batch)
    print(f"已插入 {min(i + 1000, len(vectors))}/{len(vectors)} 条")

print("✅ 数据插入完成")""",

                "step3_build": """# ============================================================
# 步骤 3：构建索引（Flat 无需构建，但仍需 flush）
# ============================================================
# Flat 是暴力搜索，无需构建索引结构
# 但仍需调用 flush 确保数据持久化
result = client.flush(collection_name='my_flat')
print(f"✅ 数据持久化完成，共 {result['total']} 条向量")""",

                "step4_search": """# ============================================================
# 步骤 4：向量搜索（100% 精确）
# ============================================================
# 查询向量
query_vector = np.random.randn(768).astype('float32')

# 执行搜索
results = client.search(
    collection_name='my_flat',
    data=[query_vector.tolist()],
    limit=10,  # Top-10 结果
    output_fields=['id']
)

# 结果保证 100% 精确
print("搜索结果（Top-10，精确）：")
for i, result in enumerate(results[0], 1):
    print(f"{i}. {result['id']}: 距离 {result['distance']:.4f}")"""
            },
            "curl": {
                "step1_create": """# 创建 Flat Collection
curl -X POST http://localhost:8080/collections/my_flat/create \\
  -H 'Content-Type: application/json' \\
  -d '{
    "schema": {
      "fields": [
        {"name": "id", "datatype": "VARCHAR", "is_primary": true, "max_length": 64},
        {"name": "vector", "datatype": "FLOAT_VECTOR", "dim": 768}
      ]
    },
    "index_params": {
      "indexes": [{
        "field_name": "vector",
        "index_type": "Flat",
        "metric_type": "L2"
      }]
    }
  }'""",

                "step2_insert": """# 插入数据
curl -X POST http://localhost:8080/collections/my_flat/insert \\
  -H 'Content-Type: application/json' \\
  -d '{
    "data": [
      {"id": "vec_1", "vector": [0.1, 0.2, ...]},
      {"id": "vec_2", "vector": [0.5, 0.6, ...]}
    ]
  }'""",

                "step3_build": """# 持久化数据
curl -X POST http://localhost:8080/collections/my_flat/flush""",

                "step4_search": """# 搜索（100% 精确）
curl -X POST http://localhost:8080/collections/my_flat/search \\
  -H 'Content-Type: application/json' \\
  -d '{
    "data": [[0.3, 0.4, ...]],
    "limit": 10,
    "output_fields": ["id"]
  }'"""
            }
        },
        "performance_tips": [
            "仅用于小数据集（<10万）或精确度验证",
            "可作为其他索引的召回率 baseline"
        ]
    },

    "IVF": {
        "name": "IVF",
        "full_name": "Inverted File Index（倒排文件索引）",
        "description": """IVF 通过聚类将向量空间划分为多个区域（桶），搜索时只查询最近的几个桶，大幅减少计算量。

【核心原理】
• 训练阶段：使用 k-means 将数据聚类为 nlist 个中心
• 索引阶段：每个向量分配到最近的聚类中心
• 搜索阶段：先找最近的 nprobe 个中心，再在这些桶内精确搜索

【技术特点】
• 复杂度：O(nprobe × (N/nlist))
• 适合中等规模数据（10万 - 1000万）""",

        "use_case": "适合百万级数据，可接受轻微召回率损失以换取速度提升的场景",
        "advantages": ["速度较快", "内存占用适中", "参数简单"],
        "limitations": ["需要训练阶段", "召回率低于 HNSW"],
        "parameters": {
            "nlist": {
                "description": "聚类中心数量",
                "range": "100-10000",
                "default": 100,
                "recommendation": "推荐设为 sqrt(N)，例如 100万数据用 nlist=1000"
            },
            "nprobe": {
                "description": "搜索时查询的桶数量",
                "range": "1-nlist",
                "default": 10,
                "recommendation": "nprobe=10-50，越大越准但越慢"
            }
        },
        "example_code": {
            "python": {
                "step1_create": """# ============================================================
# 步骤 1：创建 IVF Collection
# ============================================================
from pyhypervec import HypervecClient

client = HypervecClient('http://localhost:8080')

# 创建 IVF 索引
client.create_collection(
    collection_name='my_ivf',
    dimension=768,
    index_type='IVF',
    metric_type='L2',
    index_params={
        'nlist': 100  # 聚类中心数量
    }
)
print("✅ IVF Collection 创建成功")""",

                "step2_insert": """# ============================================================
# 步骤 2：插入训练和索引数据
# ============================================================
import numpy as np

# IVF 需要训练数据（用于 k-means 聚类）
# 训练数据量建议：nlist × 100 以上
train_vectors = np.random.randn(10000, 768).astype('float32')
vectors = np.random.randn(100000, 768).astype('float32')

# 先插入训练数据
for i in range(0, len(train_vectors), 1000):
    batch = [
        {'id': f'train_{j}', 'vector': train_vectors[j].tolist()}
        for j in range(i, min(i + 1000, len(train_vectors)))
    ]
    client.insert(collection_name='my_ivf', data=batch)

# 再插入索引数据
for i in range(0, len(vectors), 1000):
    batch = [
        {'id': f'vec_{j}', 'vector': vectors[j].tolist()}
        for j in range(i, min(i + 1000, len(vectors)))
    ]
    client.insert(collection_name='my_ivf', data=batch)
    print(f"已插入 {min(i + 1000, len(vectors))}/{len(vectors)} 条")

print("✅ 数据插入完成")""",

                "step3_build": """# ============================================================
# 步骤 3：训练并构建 IVF 索引
# ============================================================
# IVF 会自动训练聚类中心并构建索引
result = client.flush(collection_name='my_ivf')
print(f"✅ IVF 索引构建完成")
print(f"   - 向量数量：{result['total']}")
print(f"   - 聚类中心：100 个")""",

                "step4_search": """# ============================================================
# 步骤 4：向量搜索（近似搜索）
# ============================================================
query_vector = np.random.randn(768).astype('float32')

# 执行搜索，通过 nprobe 控制速度与召回率
results = client.search(
    collection_name='my_ivf',
    data=[query_vector.tolist()],
    limit=10,
    search_params={'nprobe': 10},  # 查询 10 个桶
    output_fields=['id']
)

print("搜索结果（Top-10）：")
for i, result in enumerate(results[0], 1):
    print(f"{i}. {result['id']}: 距离 {result['distance']:.4f}")

# 调整 nprobe 对比性能
print("\\n不同 nprobe 值的性能对比：")
for nprobe in [5, 10, 20, 50]:
    import time
    start = time.time()
    client.search(
        collection_name='my_ivf',
        data=[query_vector.tolist()],
        limit=10,
        search_params={'nprobe': nprobe}
    )
    latency = (time.time() - start) * 1000
    print(f"nprobe={nprobe:2d}: {latency:.2f} ms")"""
            },
            "curl": {
                "step1_create": """# 创建 IVF Collection
curl -X POST http://localhost:8080/collections/my_ivf/create \\
  -H 'Content-Type: application/json' \\
  -d '{
    "schema": {
      "fields": [
        {"name": "id", "datatype": "VARCHAR", "is_primary": true, "max_length": 64},
        {"name": "vector", "datatype": "FLOAT_VECTOR", "dim": 768}
      ]
    },
    "index_params": {
      "indexes": [{
        "field_name": "vector",
        "index_type": "IVF",
        "metric_type": "L2",
        "params": {"nlist": 100}
      }]
    }
  }'""",

                "step2_insert": """# 插入数据
curl -X POST http://localhost:8080/collections/my_ivf/insert \\
  -H 'Content-Type: application/json' \\
  -d '{
    "data": [
      {"id": "vec_1", "vector": [0.1, 0.2, ...]},
      {"id": "vec_2", "vector": [0.5, 0.6, ...]}
    ]
  }'""",

                "step3_build": """# 训练并构建索引
curl -X POST http://localhost:8080/collections/my_ivf/flush""",

                "step4_search": """# 搜索（可调整 nprobe）
curl -X POST http://localhost:8080/collections/my_ivf/search \\
  -H 'Content-Type: application/json' \\
  -d '{
    "data": [[0.3, 0.4, ...]],
    "limit": 10,
    "search_params": {"nprobe": 10},
    "output_fields": ["id"]
  }'"""
            }
        },
        "performance_tips": [
            "nlist 建议为 sqrt(数据量)",
            "nprobe 动态调整平衡速度与召回率",
            "需要足够训练数据（至少 nlist × 100 条）"
        ]
    },

    "IVFPQ": {
        "name": "IVFPQ",
        "full_name": "IVF + Product Quantization（IVF + 乘积量化）",
        "description": """IVFPQ 结合 IVF 的快速搜索和 PQ 的内存压缩，适合超大规模数据。

【核心原理】
• IVF：快速缩小搜索范围
• PQ：将向量压缩为短码，大幅减少内存
• 内存占用可降至原始数据的 1/32""",

        "use_case": "适合千万级以上数据，内存受限但仍需较快查询的场景",
        "advantages": ["内存占用极小", "可处理海量数据"],
        "limitations": ["召回率略低于 HNSW", "需要训练"],
        "parameters": {
            "nlist": {"description": "聚类中心数", "recommendation": "同 IVF"},
            "m": {"description": "子向量数量", "recommendation": "通常 8 或 16"},
            "nbits": {"description": "每子向量量化位数", "default": 8}
        },
        "example_code": {
            "python": {
                "step1_create": """# ============================================================
# 步骤 1：创建 IVFPQ Collection
# ============================================================
from pyhypervec import HypervecClient

client = HypervecClient('http://localhost:8080')

# 创建 IVFPQ 索引（IVF + 乘积量化）
client.create_collection(
    collection_name='my_ivfpq',
    dimension=768,
    index_type='IVFPQ',
    metric_type='L2',
    index_params={
        'nlist': 100,  # 聚类中心数
        'm': 8,        # 子向量数（768/8=96维每段）
        'nbits': 8     # 每段量化为 8 位
    }
)
print("✅ IVFPQ Collection 创建成功")""",

                "step2_insert": """# ============================================================
# 步骤 2：插入向量数据
# ============================================================
import numpy as np

# IVFPQ 适合大规模数据
vectors = np.random.randn(1000000, 768).astype('float32')

# 批量插入
for i in range(0, len(vectors), 10000):
    batch = [
        {'id': f'vec_{j}', 'vector': vectors[j].tolist()}
        for j in range(i, min(i + 10000, len(vectors)))
    ]
    client.insert(collection_name='my_ivfpq', data=batch)
    print(f"已插入 {min(i + 10000, len(vectors))}/{len(vectors)} 条")

print("✅ 数据插入完成")""",

                "step3_build": """# ============================================================
# 步骤 3：训练并构建 IVFPQ 索引
# ============================================================
# IVFPQ 会：
# 1. 训练 k-means 聚类
# 2. 训练 PQ 量化器
# 3. 量化所有向量
result = client.flush(collection_name='my_ivfpq')
print(f"✅ IVFPQ 索引构建完成")
print(f"   - 原始向量：{result['total']} 条")
print(f"   - 内存占用：约为 Flat 的 1/32")""",

                "step4_search": """# ============================================================
# 步骤 4：向量搜索（压缩搜索）
# ============================================================
query_vector = np.random.randn(768).astype('float32')

# 在压缩空间中搜索
results = client.search(
    collection_name='my_ivfpq',
    data=[query_vector.tolist()],
    limit=10,
    search_params={'nprobe': 10},
    output_fields=['id']
)

print("搜索结果（Top-10，近似）：")
for i, result in enumerate(results[0], 1):
    print(f"{i}. {result['id']}: 距离 {result['distance']:.4f}")

print("\\n✅ IVFPQ 优势：")
print("   - 内存占用仅为原始数据的 3-5%")
print("   - 搜索速度较快")
print("   - 适合千万级以上数据")"""
            },
            "curl": {
                "step1_create": """# 创建 IVFPQ Collection
curl -X POST http://localhost:8080/collections/my_ivfpq/create \\
  -H 'Content-Type: application/json' \\
  -d '{
    "schema": {"fields": [...]},
    "index_params": {
      "indexes": [{
        "field_name": "vector",
        "index_type": "IVFPQ",
        "metric_type": "L2",
        "params": {"nlist": 100, "m": 8, "nbits": 8}
      }]
    }
  }'""",

                "step2_insert": """# 插入数据
curl -X POST http://localhost:8080/collections/my_ivfpq/insert \\
  -d '{"data": [{"id": "vec_1", "vector": [...]}]}'""",

                "step3_build": """# 构建索引
curl -X POST http://localhost:8080/collections/my_ivfpq/flush""",

                "step4_search": """# 搜索
curl -X POST http://localhost:8080/collections/my_ivfpq/search \\
  -d '{"data": [[...]], "limit": 10, "search_params": {"nprobe": 10}}'"""
            }
        },
        "performance_tips": [
            "可将内存降至 1/32，适合超大规模",
            "m=8 是性价比平衡点"
        ]
    },

    "IVFLVQ": {
        "name": "IVFLVQ",
        "full_name": "IVF + Locally-adaptive Vector Quantization",
        "description": "IVF 结合局部自适应向量量化，比 IVFPQ 召回率更高",
        "use_case": "需要较高召回率的大规模场景",
        "parameters": {
            "nlist": {"description": "聚类中心数"},
            "nlocal": {"description": "局部量化器数量"},
            "nbits": {"description": "量化位数"}
        },
        "example_code": {
            "python": {
                "step1_create": """# 创建 IVFLVQ Collection
from pyhypervec import HypervecClient

client = HypervecClient('http://localhost:8080')
client.create_collection(
    collection_name='my_ivflvq',
    dimension=768,
    index_type='IVFLVQ',
    metric_type='L2',
    index_params={'nlist': 100, 'nlocal': 8, 'nbits': 8}
)""",

                "step2_insert": """# 插入数据
vectors = np.random.randn(500000, 768).astype('float32')
for i in range(0, len(vectors), 5000):
    batch = [{'id': f'vec_{j}', 'vector': vectors[j].tolist()}
             for j in range(i, min(i + 5000, len(vectors)))]
    client.insert(collection_name='my_ivflvq', data=batch)""",

                "step3_build": """# 构建索引
client.flush(collection_name='my_ivflvq')
print("✅ IVFLVQ 索引构建完成，召回率优于 IVFPQ")""",

                "step4_search": """# 搜索
query = np.random.randn(768).astype('float32')
results = client.search(
    collection_name='my_ivflvq',
    data=[query.tolist()],
    limit=10,
    search_params={'nprobe': 10}
)"""
            },
            "curl": {
                "step1_create": """curl -X POST http://localhost:8080/collections/my_ivflvq/create \\
  -d '{"index_params": {"indexes": [{"index_type": "IVFLVQ", "params": {"nlist": 100, "nlocal": 8}}]}}'""",
                "step2_insert": """curl -X POST http://localhost:8080/collections/my_ivflvq/insert \\
  -d '{"data": [{"id": "vec_1", "vector": [...]}]}'""",
                "step3_build": """curl -X POST http://localhost:8080/collections/my_ivflvq/flush""",
                "step4_search": """curl -X POST http://localhost:8080/collections/my_ivflvq/search \\
  -d '{"data": [[...]], "limit": 10, "search_params": {"nprobe": 10}}'"""
            }
        },
        "performance_tips": ["比 IVFPQ 更准确", "内存占用适中"]
    },

    "PQ": {
        "name": "PQ",
        "full_name": "Product Quantization（乘积量化）",
        "description": "纯量化索引，极致压缩内存",
        "use_case": "极大规模数据，内存极度受限",
        "parameters": {
            "m": {"description": "子向量数"},
            "nbits": {"description": "量化位数"}
        },
        "example_code": {
            "python": {
                "step1_create": """# 创建 PQ Collection
from pyhypervec import HypervecClient

client = HypervecClient('http://localhost:8080')
client.create_collection(
    collection_name='my_pq',
    dimension=768,
    index_type='PQ',
    metric_type='L2',
    index_params={'m': 8, 'nbits': 8}  # 768维分8段，每段8位
)""",

                "step2_insert": """# 插入数据
vectors = np.random.randn(10000000, 768).astype('float32')  # 千万级
for i in range(0, len(vectors), 10000):
    batch = [{'id': f'vec_{j}', 'vector': vectors[j].tolist()}
             for j in range(i, min(i + 10000, len(vectors)))]
    client.insert(collection_name='my_pq', data=batch)""",

                "step3_build": """# 构建 PQ 量化索引
client.flush(collection_name='my_pq')
print("✅ PQ 索引构建完成，内存占用极小")""",

                "step4_search": """# 搜索（全量扫描，但使用量化距离）
query = np.random.randn(768).astype('float32')
results = client.search(
    collection_name='my_pq',
    data=[query.tolist()],
    limit=10
)"""
            },
            "curl": {
                "step1_create": """curl -X POST http://localhost:8080/collections/my_pq/create \\
  -d '{"index_params": {"indexes": [{"index_type": "PQ", "params": {"m": 8, "nbits": 8}}]}}'""",
                "step2_insert": """curl -X POST http://localhost:8080/collections/my_pq/insert \\
  -d '{"data": [{"id": "vec_1", "vector": [...]}]}'""",
                "step3_build": """curl -X POST http://localhost:8080/collections/my_pq/flush""",
                "step4_search": """curl -X POST http://localhost:8080/collections/my_pq/search \\
  -d '{"data": [[...]], "limit": 10}'"""
            }
        },
        "performance_tips": ["内存最小但召回率相对较低"]
    },

    "LVQ": {
        "name": "LVQ",
        "full_name": "Locally-adaptive Vector Quantization",
        "description": "局部自适应向量量化，适应数据分布不均的场景",
        "use_case": "数据分布不均匀时效果好",
        "parameters": {
            "nlocal": {"description": "局部量化器数量"},
            "nbits": {"description": "量化位数"}
        },
        "example_code": {
            "python": {
                "step1_create": """# 创建 LVQ Collection
from pyhypervec import HypervecClient

client = HypervecClient('http://localhost:8080')
client.create_collection(
    collection_name='my_lvq',
    dimension=768,
    index_type='LVQ',
    metric_type='L2',
    index_params={'nlocal': 8, 'nbits': 8}  # 局部自适应量化
)""",

                "step2_insert": """# 插入数据（适合非均匀分布）
vectors = np.random.randn(100000, 768).astype('float32')
for i in range(0, len(vectors), 5000):
    batch = [{'id': f'vec_{j}', 'vector': vectors[j].tolist()}
             for j in range(i, min(i + 5000, len(vectors)))]
    client.insert(collection_name='my_lvq', data=batch)""",

                "step3_build": """# 构建 LVQ 量化索引
client.flush(collection_name='my_lvq')
print("✅ LVQ 索引构建完成，对非均匀数据友好")""",

                "step4_search": """# 搜索
query = np.random.randn(768).astype('float32')
results = client.search(
    collection_name='my_lvq',
    data=[query.tolist()],
    limit=10
)"""
            },
            "curl": {
                "step1_create": """curl -X POST http://localhost:8080/collections/my_lvq/create \\
  -d '{"index_params": {"indexes": [{"index_type": "LVQ", "params": {"nlocal": 8, "nbits": 8}}]}}'""",
                "step2_insert": """curl -X POST http://localhost:8080/collections/my_lvq/insert \\
  -d '{"data": [{"id": "vec_1", "vector": [...]}]}'""",
                "step3_build": """curl -X POST http://localhost:8080/collections/my_lvq/flush""",
                "step4_search": """curl -X POST http://localhost:8080/collections/my_lvq/search \\
  -d '{"data": [[...]], "limit": 10}'"""
            }
        },
        "performance_tips": ["对非均匀分布数据友好"]
    },

    "HNSWFlat": {
        "name": "HNSWFlat",
        "full_name": "HNSW + Flat Refinement",
        "description": "HNSW 图结构 + Flat 精确距离计算，兼顾速度与精度",
        "use_case": "需要接近 100% 召回率且速度也重要的场景",
        "parameters": {
            "M": {"description": "同 HNSW"},
            "ef_construction": {"description": "同 HNSW"}
        },
        "example_code": {
            "python": {
                "step1_create": """# 创建 HNSWFlat Collection
from pyhypervec import HypervecClient

client = HypervecClient('http://localhost:8080')
client.create_collection(
    collection_name='my_hnswflat',
    dimension=768,
    index_type='HNSWFlat',
    metric_type='L2',
    index_params={'M': 32, 'ef_construction': 200}
)""",

                "step2_insert": """# 插入数据
vectors = np.random.randn(100000, 768).astype('float32')
for i in range(0, len(vectors), 5000):
    batch = [{'id': f'vec_{j}', 'vector': vectors[j].tolist()}
             for j in range(i, min(i + 5000, len(vectors)))]
    client.insert(collection_name='my_hnswflat', data=batch)""",

                "step3_build": """# 构建 HNSW 图结构
client.flush(collection_name='my_hnswflat')
print("✅ HNSWFlat 索引构建完成，兼顾速度与精度")""",

                "step4_search": """# 搜索（比纯 HNSW 更准确）
query = np.random.randn(768).astype('float32')
results = client.search(
    collection_name='my_hnswflat',
    data=[query.tolist()],
    limit=10,
    search_params={'ef_search': 200}  # 高召回率
)
print("✅ 召回率接近 100%，但内存占用较大")"""
            },
            "curl": {
                "step1_create": """curl -X POST http://localhost:8080/collections/my_hnswflat/create \\
  -d '{"index_params": {"indexes": [{"index_type": "HNSWFlat", "params": {"M": 32, "ef_construction": 200}}]}}'""",
                "step2_insert": """curl -X POST http://localhost:8080/collections/my_hnswflat/insert \\
  -d '{"data": [{"id": "vec_1", "vector": [...]}]}'""",
                "step3_build": """curl -X POST http://localhost:8080/collections/my_hnswflat/flush""",
                "step4_search": """curl -X POST http://localhost:8080/collections/my_hnswflat/search \\
  -d '{"data": [[...]], "limit": 10, "search_params": {"ef_search": 200}}'"""
            }
        },
        "performance_tips": ["比纯 HNSW 更准", "内存占用较大"]
    }
}
