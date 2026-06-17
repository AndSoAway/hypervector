## 快速启动

### 第一步：安装依赖

```bash
# 服务端依赖
pip install fastapi uvicorn grpcio grpcio-tools

# 客户端依赖
pip install -e pyhypervec/
```

### 第二步：启动服务

**只启动 gRPC server（默认端口 50051）：**

```bash
./scripts/start_grpc_server.sh --data-root ./data
```

**只启动 HTTP server（默认端口 8080）：**

```bash
cd src/python
uvicorn hypervec_http_server:app --host 0.0.0.0 --port 8080
```

**同时启动 gRPC + HTTP（共享同一数据目录）：**

```bash
./scripts/start_all_servers.sh --data-root ./data
```

### 第三步：连接并使用

```python
from pyhypervec import HypervecClient
from pyhypervec.schema import CollectionSchema
import numpy as np

# 连接 gRPC（推荐）或 HTTP，接口完全一致
client = HypervecClient("tcp://localhost:50051")   # gRPC
# client = HypervecClient("http://localhost:8080") # HTTP

# 创建集合
schema = CollectionSchema()
schema.add_field("id", "VARCHAR", is_primary=True, max_length=64)
schema.add_field("vector", "FLOAT_VECTOR", dim=128)
client.create_collection("my_collection", schema=schema)

# 写入数据
rows = [{"id": f"v{i}", "vector": np.random.rand(128).tolist()} for i in range(100)]
client.insert("my_collection", rows)
client.flush("my_collection")
client.load_collection("my_collection")

# 搜索
results = client.search(
    collection_name="my_collection",
    data=[np.random.rand(128).tolist()],
    limit=5,
)
print(results)
```

### 第四步：运行测试

```bash
# 运行所有 Python 测试（不依赖编译的 C++ extension）
python -m pytest test/unit_tests/python/ \
  --ignore=test/unit_tests/python/external_module_test.py \
  --ignore=test/unit_tests/python/test_io.py \
  -v

# 只跑 gRPC 相关测试
python -m pytest test/unit_tests/python/test_hypervec_grpc_server.py \
                 test/unit_tests/python/test_grpc_integration.py \
                 test/unit_tests/python/test_uri.py -v
```

预期结果：**54 passed, 1 skipped**（skipped 的 1 个是 `test_hypervec_http_server_sync_routes`，该测试依赖尚未实现的 HTTP `/sync` 路由，在 `main` 分支上原本就是 skip 状态，非本次引入）。

详细说明见 [docs/pyhypervec_grpc_server.md](docs/pyhypervec_grpc_server.md) 和 [docs/pyhypervec_http_server.md](docs/pyhypervec_http_server.md)。

