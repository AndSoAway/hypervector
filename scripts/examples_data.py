# -*- coding: utf-8 -*-
"""
Examples data for HyperVector index types
"""

EXAMPLES_DATA = {
    "HNSW": {
        "name": "HNSW",
        "full_name": "Hierarchical Navigable Small World",
        "description": "分层导航小世界图索引，适合大规模向量的快速近似搜索",
        "use_case": "百万级以上数据，需要高召回率（>95%）和低延迟（<10ms）的场景",
        "parameters": {
            "M": {
                "description": "每个节点的最大连接数",
                "type": "int",
                "range": "16-64",
                "default": 16,
                "recommendation": "小数据集（<10万）用16，大数据集（>100万）用32-64"
            },
            "ef_construction": {
                "description": "构建索引时的搜索宽度",
                "type": "int",
                "range": "100-500",
                "default": 200,
                "recommendation": "默认200，越大质量越好但构建越慢，建议设为M的10-20倍"
            },
            "ef_search": {
                "description": "搜索时的宽度（动态参数）",
                "type": "int",
                "range": "10-1000",
                "default": 128,
                "recommendation": "搜索时动态调整，越大越准确但越慢，可根据实时需求平衡速度和准确率"
            }
        },
        "example_code": {
            "python": {
                "create": """from pyhypervec import HypervecClient

client = HypervecClient('http://localhost:8080')

# 创建 HNSW Collection
client.create_collection(
    collection_name='my_hnsw',
    dimension=768,
    index_type='HNSW',
    metric_type='L2',
    index_params={'M': 32, 'ef_construction': 200}
)""",
                "insert": """# 插入向量数据
data = [
    {'id': 'vec_1', 'vector': [0.1, 0.2, ..., 0.768]},
    {'id': 'vec_2', 'vector': [0.5, 0.6, ..., 0.234]}
]
client.insert(collection_name='my_hnsw', data=data)

# 构建索引
client.flush(collection_name='my_hnsw')""",
                "search": """# 搜索向量
query_vector = [0.3, 0.4, ..., 0.567]
results = client.search(
    collection_name='my_hnsw',
    data=[query_vector],
    limit=10,
    search_params={'ef_search': 256}
)

# 结果：Top-10 最相似向量
for result in results[0]:
    print(f"{result['id']}: {result['distance']}")"""
            },
            "curl": {
                "create": """curl -X POST http://localhost:8080/collections/my_hnsw/create \\
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
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 32, "ef_construction": 200}
      }]
    }
  }'""",
                "insert": """curl -X POST http://localhost:8080/collections/my_hnsw/insert \\
  -H 'Content-Type: application/json' \\
  -d '{
    "data": [
      {"id": "vec_1", "vector": [0.1, 0.2, ...]},
      {"id": "vec_2", "vector": [0.5, 0.6, ...]}
    ]
  }'

curl -X POST http://localhost:8080/collections/my_hnsw/flush""",
                "search": """curl -X POST http://localhost:8080/collections/my_hnsw/search \\
  -H 'Content-Type: application/json' \\
  -d '{
    "data": [[0.3, 0.4, ...]],
    "limit": 10,
    "search_params": {"ef_search": 256},
    "output_fields": ["id"]
  }'"""
            }
        },
        "performance_tips": [
            "M值越大，索引质量越好，但占用内存成平方增长",
            "ef_construction建议设为M的10-20倍，过大会显著增加构建时间",
            "搜索时ef_search可动态调整：128（快速）、256（平衡）、512（高精度）",
            "对于亿级数据，建议M=32-48，ef_construction=400-600"
        ]
    },

    "Flat": {
        "name": "Flat",
        "full_name": "Flat Index (Brute Force)",
        "description": "暴力搜索索引，遍历所有向量计算距离，保证100%准确",
        "use_case": "小数据集（<10万条），需要精确结果的场景",
        "parameters": {},
        "example_code": {
            "python": {
                "create": """client.create_collection(
    collection_name='my_flat',
    dimension=768,
    index_type='Flat',
    metric_type='L2'
)"""
            },
            "curl": {
                "create": """curl -X POST http://localhost:8080/collections/my_flat/create \\
  -d '{"schema": {...}, "index_params": {"indexes": [{"index_type": "Flat"}]}}'"""
            }
        },
        "performance_tips": [
            "适合小数据集，数据量大时搜索会很慢",
            "100%准确率，适合作为baseline对比其他索引"
        ]
    },

    "IVF": {
        "name": "IVF",
        "full_name": "Inverted File Index",
        "description": "倒排文件索引，将向量空间划分为多个聚类中心，搜索时只查询最近的几个聚类",
        "use_case": "百万级数据，平衡速度和准确率的场景",
        "parameters": {
            "nlist": {
                "description": "聚类中心数量",
                "range": "100-10000",
                "default": 100,
                "recommendation": "建议为sqrt(N)，N为数据量"
            },
            "nprobe": {
                "description": "搜索时查询的聚类数量",
                "range": "1-nlist",
                "default": 10,
                "recommendation": "越大越准确但越慢"
            }
        },
        "example_code": {
            "python": {
                "create": """client.create_collection(
    collection_name='my_ivf',
    index_type='IVF',
    index_params={'nlist': 100}
)"""
            }
        },
        "performance_tips": ["nlist建议设为sqrt(数据量)", "nprobe可动态调整"]
    },

    "IVFPQ": {
        "name": "IVFPQ",
        "full_name": "IVF + Product Quantization",
        "description": "IVF索引结合乘积量化，通过压缩向量大幅减少内存占用",
        "use_case": "千万级以上数据，内存受限的场景",
        "parameters": {
            "nlist": {"description": "聚类中心数量", "recommendation": "同IVF"},
            "m": {"description": "子向量数量", "recommendation": "通常为8或16"},
            "nbits": {"description": "每个子向量的量化位数", "default": 8}
        },
        "example_code": {
            "python": {
                "create": """client.create_collection(
    index_type='IVFPQ',
    index_params={'nlist': 100, 'm': 8, 'nbits': 8}
)"""
            }
        },
        "performance_tips": ["可将内存占用减少到原来的1/32", "准确率略低于HNSW"]
    },

    "IVFLVQ": {
        "name": "IVFLVQ",
        "full_name": "IVF + Locally-adaptive Vector Quantization",
        "description": "IVF索引结合局部向量量化，兼顾压缩率和准确率",
        "use_case": "大规模数据，需要较高准确率和内存优化",
        "parameters": {
            "nlist": {"description": "聚类中心数量"},
            "nlocal": {"description": "局部量化器数量"},
            "nbits": {"description": "量化位数"}
        },
        "example_code": {
            "python": {
                "create": """client.create_collection(
    index_type='IVFLVQ',
    index_params={'nlist': 100, 'nlocal': 8, 'nbits': 8}
)"""
            }
        },
        "performance_tips": ["比IVFPQ准确率更高", "内存占用适中"]
    },

    "PQ": {
        "name": "PQ",
        "full_name": "Product Quantization",
        "description": "乘积量化索引，将向量分段量化压缩",
        "use_case": "极大规模数据，内存极度受限",
        "parameters": {
            "m": {"description": "子向量数量"},
            "nbits": {"description": "量化位数"}
        },
        "example_code": {
            "python": {
                "create": """client.create_collection(
    index_type='PQ',
    index_params={'m': 8, 'nbits': 8}
)"""
            }
        },
        "performance_tips": ["最大程度压缩内存", "准确率相对较低"]
    },

    "LVQ": {
        "name": "LVQ",
        "full_name": "Locally-adaptive Vector Quantization",
        "description": "局部自适应向量量化，根据数据分布自适应调整量化策略",
        "use_case": "数据分布不均匀的场景",
        "parameters": {
            "nlocal": {"description": "局部量化器数量"},
            "nbits": {"description": "量化位数"}
        },
        "example_code": {
            "python": {
                "create": """client.create_collection(
    index_type='LVQ',
    index_params={'nlocal': 8, 'nbits': 8}
)"""
            }
        },
        "performance_tips": ["对非均匀数据效果好"]
    },

    "HNSWFlat": {
        "name": "HNSWFlat",
        "full_name": "HNSW + Flat",
        "description": "HNSW图结构结合Flat精确距离计算，兼顾速度和精度",
        "use_case": "需要高速度和高准确率的场景",
        "parameters": {
            "M": {"description": "同HNSW"},
            "ef_construction": {"description": "同HNSW"}
        },
        "example_code": {
            "python": {
                "create": """client.create_collection(
    index_type='HNSWFlat',
    index_params={'M': 32, 'ef_construction': 200}
)"""
            }
        },
        "performance_tips": ["比纯HNSW更准确", "内存占用较大"]
    }
}
