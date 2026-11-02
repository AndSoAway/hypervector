# 全文检索（BM25）功能说明与测试指南

本文档说明 HyperVector 服务端的全文检索能力（任务 T2）：用户直接传**原始中文文本** query，
服务端自动完成「文本 → 分词 → IDF 稀疏向量 → 倒排 BM25 检索 → 打分排序」，对齐 Milvus 的
`search_by_full_text` 语义。

分支：`feat/t2-full-text-search`。

## 1. 功能概览

- **入库**：写入文本时无需传稀疏向量。`flush` 时服务端对每条文本分词、生成 TF 稀疏向量、
  构建 BM25 倒排索引。
- **检索**：`search_by_full_text(collection_name, query, limit, output_fields)`，`query` 是原始
  中文文本。返回与向量 `search` 对称的结构 `{id, distance, entity}`，其中 `distance` 是 BM25 分。
- **多协议**：HTTP（`POST /collections/{name}/search-text`）与 gRPC（`FullTextSearch`）两条路径
  行为一致，均调用同一个引擎方法。
- **共存**：同一 collection 既可向量 `search`，也可全文 `search_by_full_text`。
- **配置**：`analyzer_params`（透传给中文分词器的 jieba/hmm/user 词典路径）可选，默认用内置词典。

## 2. 架构与数据流

底层 BM25 算法是 C++（T1 交付），经 SWIG 暴露给 Python 服务层调用。

```
原始中文 query
  → CppjiebaAnalyzer.analyze(text)         # 中文分词 → token 列表
  → TextToSparse.query_to_sparse(tokens)   # 查询侧：IDF 稀疏向量
  → BM25InvertedIndex.search(idf, k)       # 倒排 TAAT 检索，返回 (分数, 文档编号)
  → row_ids[文档编号] → 标量行             # 映射回原文
  → {id, distance=BM25分, entity}
```

入库侧（`flush`）：
```
每条文本 → analyze(text) → TextToSparse.add_document(tokens)  # 文档侧：TF 稀疏向量
        → BM25InvertedIndex.add(tf_rows)                      # 建倒排
```

**两种稀疏向量**（BM25 核心约定）：文档侧存 TF（词频），查询侧存 IDF（逆文档频率）。

### 关键文件
| 层 | 文件 | 要点 |
|----|------|------|
| SWIG 绑定 | `src/python/swighypervec.swig` | 暴露 `CppjiebaAnalyzer` / `AnalyzerParams` / `DefaultAnalyzerParams` / `TextToSparse` / `BM25InvertedIndex` |
| 引擎 | `src/python/hypervec_server_engine.py` | `search_by_full_text`、`_build_inverted_index`、`_hydrate_hits`、`_inverted_indexes` 缓存 |
| 协议 | `src/python/hypervec.proto` | `FullTextSearch` RPC + `FullTextSearchRequest`（raw `query` 字符串） |
| gRPC/HTTP | `hypervec_grpc_server.py` / `hypervec_http_server.py` | handler / 路由 |
| 客户端 | `pyhypervec/pyhypervec/{client,_grpc_transport}.py` | `search_by_full_text` 方法 |

### 两个正确性要点（实现时已处理）
1. **doc-id 排序**：倒排索引内部给文档编号 0..n-1，与真实 row_id 不一定一致。`flush` 时用
   `scalar_store.export_rows`（`ORDER BY row_id ASC`）按序建倒排，并保存 `row_ids` 列表，检索时
   用 `row_ids[label]` 映射回原文——不依赖 row_id 恰好连续。
2. **avgdl**：BM25 的文档长度归一化依赖语料平均文档长度 `avgdl`。检索时必须把
   `inverted.avgdl()` 传入打分参数，否则退化为 1.0，分数系统性偏差。

### 索引生命周期
倒排索引随向量索引同生命周期：`flush` 时构建，缓存在 `_inverted_indexes`，在
drop / insert / close / purge 四个逐出点同步清除。仅内存（不持久化），进程重启或
`load_collection` 后首次全文检索会 **lazy 重建**（前提是向量索引文件已 flush 存在）。

## 3. 使用示例（Python 服务端引擎）

```python
import swighypervec as hv
from hypervec_server_engine import HypervecServerEngine

engine = HypervecServerEngine(data_root="/path/to/data", hypervec_module=hv)
schema = {"fields": [
    {"name": "id", "is_primary": True, "dtype": "VARCHAR"},
    {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 1},
    {"name": "contents", "dtype": "VARCHAR"},
    # 可选：{"analyzer_params": {"user_dict_path": "/path/to/user.dict.utf8"}}
]}
engine.create_collection("docs", schema=schema,
    index_params={"indexes": [{"field_name": "vector", "index_type": "HNSW",
                               "metric_type": "L2", "params": {}}]})
engine.insert("docs", [{"id": "1", "vector": [1.0], "contents": "自然语言处理 是 人工智能 的 分支"}])
engine.flush("docs")                       # 此处构建 BM25 倒排
hits = engine.search_by_full_text("docs", query="自然语言处理", limit=3)
# hits: [{"id": "1", "distance": <BM25分>, "entity": {"contents": "...", "id": "1"}}]
```

客户端（HTTP 或 gRPC 透明）：
```python
from pyhypervec import HypervecClient
client = HypervecClient("http://host:8080")   # 或 "grpc://host:port"
client.search_by_full_text(collection_name="docs", query="自然语言处理", limit=3)
```

## 4. 错误约定

| 情况 | 异常 | HTTP / gRPC |
|------|------|-------------|
| 空 query | `ValueError` | 400 / INVALID_ARGUMENT |
| collection 不存在 | `FileNotFoundError` | 404 / NOT_FOUND |
| collection 未 flush（索引文件不存在） | `FileNotFoundError` | 404 / NOT_FOUND |
| data_state 为 invalid | `ConflictError` | 409 / FAILED_PRECONDITION |

## 5. 测试

### 5.1 构建 Python 绑定（前置）

全文检索需要 SWIG 绑定包含 BM25 类。官方构建用 CMake（`-DHYPERVEC_ENABLE_PYTHON=ON`，需
CMake ≥ 3.17）。若环境 CMake 过旧，可手动构建单个模块：

```bash
# 1. 生成 wrapper（务必带 BM25 宏）
swig -python -c++ -DSWIGWORDSIZE64 -DHYPERVEC_ENABLE_BM25_ANALYZER \
  -I src/include -I src -o /tmp/build/swighypervec_wrap.cxx -outdir /tmp/build \
  src/python/swighypervec.swig
# 2. 编译链接（libhypervec.a 需已含 BM25 符号，即 HYPERVEC_ENABLE_BM25_ANALYZER=ON 编出的）
g++ -std=c++2a -O2 -fopenmp -fPIC -shared -DHYPERVEC_ENABLE_BM25_ANALYZER \
  -I src/include -I src -I third_party/cppjieba/include -I third_party/cppjieba/deps \
  -I <python-headers> -I <numpy-headers> \
  /tmp/build/swighypervec_wrap.cxx build/src/libhypervec.a -lopenblas -llapack \
  -o /tmp/build/_swighypervec.so
```

分词词典：设 `HYPERVEC_DICT_DIR=third_party/cppjieba/dict`（或在 `analyzer_params` 里给绝对路径）。

### 5.2 单元 / 集成测试

```bash
export PYTHONPATH=/tmp/build            # 指向上面构建出的 _swighypervec.so + swighypervec.py
export HYPERVEC_DICT_DIR=$PWD/third_party/cppjieba/dict
python3 -m pytest -q \
  test/unit_tests/python/test_hypervec_server_engine.py \
  test/unit_tests/python/test_hypervec_grpc_server.py \
  test/unit_tests/python/test_hypervec_http_server.py \
  test/unit_tests/python/test_pyhypervec_client.py
```

覆盖的全文检索用例：
- `test_hypervec_server_engine.py`
  - `test_engine_full_text_search_chinese_end_to_end` — 中文语料建库→flush→检索，断言相关 doc
    命中、无关 doc 不命中、BM25 降序；空 query / 不存在 collection 报错；与向量 search 共存。
    （需真实 BM25 绑定 + 词典，否则自动 `skip`）
  - `test_engine_full_text_search_not_flushed` — 未 flush 的 collection 全文检索报 `FileNotFoundError`。
- `test_hypervec_grpc_server.py`
  - `test_grpc_full_text_search` — gRPC 路径，原始中文 query 经 proto `string` 字段往返。
  - `test_full_text_http_and_grpc_agree` — 同 query 经 HTTP 与 gRPC 返回**逐字节一致**。
- `test_hypervec_http_server.py`
  - `test_hypervec_http_server_full_text_search` — `POST /search-text` happy path + 空 query(400)
    + 不存在 collection(404)。
- `test_pyhypervec_client.py`
  - `test_pyhypervec_client_search_by_full_text_payload` — 客户端构造正确的 path/body。

### 5.3 真实数据集（可选，验证规模与相关性）

用真实中文维基语料端到端验证（非 CI，手动）：加载语料 → 建库 → `flush` → 对若干中文 query
跑 `search_by_full_text`，人工核对命中相关性与 BM25 排序。

实测参考（64 核）：4.4 万条中文维基，`flush`（建倒排）约 21s，单次检索 0.1–0.4ms；教科书式
查询（如「人工智能 机器学习」→ 机器学习 / 人工神经网络 / 计算机视觉）相关性良好。

> 注：纯 BM25 是词袋模型、无语义。大规模语料中「年份页」等高频词堆叠的页面可能挤占排序，
> 属 BM25 固有局限（可用停用词表 / 页面过滤 / 混合向量检索缓解，后者为 T3 范围）。

## 6. 范围与限制

- 本功能只做**单路全文检索**，不含向量+全文的融合打分（T3 混合检索范围）。
- 倒排索引目前仅内存缓存、flush 重建，未持久化（与向量索引的内存缓存策略一致）。
- `consistency_level` 参数接收但不强制语义（单机强一致），与向量 `search` 一致。
