# -*- coding: utf-8 -*-
import numpy as np
import requests
import json

SERVER = "http://localhost:8080"

# 1. 创建 Collection
def create_collection():
    url = f"{SERVER}/collections/test_hnsw/create"
    payload = {
        "schema": {
            "auto_id": False,
            "fields": [
                {"name": "id", "datatype": "VARCHAR", "is_primary": True, "max_length": 64},
                {"name": "vector", "datatype": "FLOAT_VECTOR", "dim": 128}  # 128维，方便测试
            ]
        },
        "index_params": {
            "indexes": [{
                "field_name": "vector",
                "index_type": "HNSW",
                "metric_type": "L2",
                "params": {"M": 16, "ef_construction": 200}
            }]
        }
    }

    response = requests.post(url, json=payload)
    print(f"创建 Collection: {response.json()}")

# 2. 生成并插入测试数据
def insert_data(num_vectors=1000):
    url = f"{SERVER}/collections/test_hnsw/insert"

    # 生成随机向量
    vectors = np.random.randn(num_vectors, 128).astype('float32')

    # 批量插入（每次100条）
    batch_size = 100
    for i in range(0, num_vectors, batch_size):
        batch = vectors[i:i+batch_size]
        data = [
            {"id": f"vec_{j}", "vector": vec.tolist()}
            for j, vec in enumerate(batch, start=i)
        ]

        response = requests.post(url, json={"data": data})
        print(f"插入 {i}-{i+len(batch)}: {response.status_code}")

    print(f"✅ 总共插入 {num_vectors} 条数据")

# 3. 构建索引
def flush_collection():
    url = f"{SERVER}/collections/test_hnsw/flush"
    response = requests.post(url, json={})
    print(f"构建索引: {response.json()}")

# 4. 搜索测试
def search_test():
    url = f"{SERVER}/collections/test_hnsw/search"

    # 生成查询向量
    query_vector = np.random.randn(128).astype('float32').tolist()

    payload = {
        "data": [query_vector],
        "limit": 10,
        "output_fields": ["id"]
    }

    response = requests.post(url, json=payload)
    result = response.json()
    print(f"搜索结果: {json.dumps(result, indent=2)}")

# 5. 查看 Collection 信息
def describe_collection():
    url = f"{SERVER}/collections/test_hnsw/describe"
    response = requests.get(url)
    print(f"Collection 信息: {json.dumps(response.json(), indent=2)}")

# 执行测试
if __name__ == "__main__":
    print("=== 1. 创建 Collection ===")
    create_collection()

    print("\n=== 2. 插入测试数据 ===")
    insert_data(1000)

    print("\n=== 3. 构建索引 ===")
    flush_collection()

    print("\n=== 4. 查看 Collection 信息 ===")
    describe_collection()

    print("\n=== 5. 搜索测试 ===")
    search_test()

    print("\n✅ 测试完成！")
