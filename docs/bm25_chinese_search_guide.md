# BM25 中文全文检索 —— 原理与客户端使用教程

本文面向**使用全文检索功能的开发者**：讲清 BM25 中文检索怎么工作，以及怎么从
客户端调用。只讲用法与必要原理，实现细节与测试见 [full_text_search.md](full_text_search.md)。

## 一、它能做什么

直接传**原始中文文本** query，返回按相关度（BM25 分）排序的命中文档——无需自己分词、
无需传稀疏向量。对齐 Milvus 的 `search_by_full_text`。

```
client.search_by_full_text(collection_name="docs", query="量子力学", limit=5)
→ [{"id": "...", "distance": <BM25分>, "entity": {"contents": "..."}}, ...]
```

## 二、原理（够用就好）

### BM25 是什么
BM25 是经典的**词袋**检索打分算法：把 query 和文档都看成"词的集合"，按**词的匹配程度**
打分。它不理解语义（不知道"汽车"≈"轿车"），只看字面词是否对得上、对得多不多。

一篇文档对 query 的得分，直观上由三部分决定：
- **词频 TF**：query 的词在该文档出现越多，分越高（但有上限，不会无限涨）
- **逆文档频率 IDF**：越稀有的词权重越大。"量子"比"的""是"稀有得多，所以它才是
  决定相关性的关键词，常见停用词几乎不影响排序
- **文档长度归一化**：长文档天然包含更多词，会做归一化拉平，避免长文档总是占优

### 中文为什么要先分词
中文没有空格分词。`"量子力学很有趣"` 要先切成 `["量子力学", "很", "有趣"]` 才能做词匹配。
本功能用 **cppjieba**（jieba 的 C++ 实现）做中文分词，这一步对用户透明——你传整句中文
即可，服务端自动切词。

### 完整流程（透明，了解即可）
```
你传: query="量子力学"
服务端: 分词 → 词的 IDF 权重 → 倒排索引检索 → BM25 打分 → 按分排序 → 返回 top-k
```
`distance` 字段就是 BM25 分，**越大越相关**。

## 三、前置：建库与写入

全文检索前，collection 必须有一个**文本字段**并且已 `flush`（flush 时服务端才构建
倒排索引）。

文本字段名默认约定为 **`contents`**（服务端 `_text_field` 默认找这个名字的字段）。

```python
from pyhypervec import HypervecClient
from pyhypervec.schema import CollectionSchema, IndexParams, DataType

client = HypervecClient("http://127.0.0.1:8099")   # 或 "grpc://host:port"

schema = CollectionSchema(description="docs")
schema.add_field("id", DataType.VARCHAR, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1)   # 向量字段仍需存在
schema.add_field("contents", DataType.VARCHAR)             # 文本字段
ip = IndexParams(); ip.add_index("vector", metric_type="L2", index_type="HNSW", params={})

client.create_collection(collection_name="docs", schema=schema, index_params=ip)
client.insert(collection_name="docs", data=[
    {"id": "1", "vector": [1.0], "contents": "量子力学 是 物理学 的 分支"},
    {"id": "2", "vector": [2.0], "contents": "机器学习 推动 人工智能"},
])
client.flush(collection_name="docs")   # 必须 flush，否则检索报 404(index not flushed)
```

> `flush` 在大 collection 上可能耗时数分钟（服务端建倒排）。客户端默认给 flush 10 分钟
> 超时（`flush_timeout=600`），普通请求仍 30 秒。更大的库可 `HypervecClient(uri,
> flush_timeout=1200)` 调大。

## 四、检索（三种调用方式）

### 1. Python 客户端（推荐）
```python
hits = client.search_by_full_text(
    collection_name="docs",
    query="量子力学",            # 原始中文，不是稀疏向量
    limit=5,
    output_fields=["contents"],  # 返回哪些字段
)
for h in hits:
    print(h["id"], h["distance"], h["entity"]["contents"])
```
HTTP 与 gRPC 透明——`HypervecClient("grpc://...")` 用法完全一样。

### 2. HTTP（curl / 任意语言）
```bash
curl -sS -X POST http://127.0.0.1:8099/collections/docs/search-text \
  -H "Content-Type: application/json" \
  -d '{"query":"量子力学","limit":5,"output_fields":["contents"]}'
```
响应：
```json
{"results":[{"id":"1","distance":1.386,"entity":{"contents":"量子力学 是 物理学 的 分支"}}]}
```
中文直接写进 JSON（UTF-8），响应中文不转义。

### 3. gRPC
RPC `FullTextSearch`，请求 `FullTextSearchRequest{collection_name, query, limit,
output_fields, filter, consistency_level}`，响应复用 `SearchResponse`（`results_json`）。

### 返回结构
```
[{"id": 主键, "distance": BM25分(越大越相关), "entity": {请求的字段}}, ...]
```
与向量 `search` 对称，只是 `distance` 从"向量距离"换成"BM25 分"。单 query 返回一个
扁平列表（向量 search 返回 list-of-lists，因为它支持批量 query）。

## 五、终端快速试用

仓库提供两个脚本：

```bash
# 1. 本机起 HTTP server（已封装环境变量）
bash scripts/start_fulltext_server_local.sh

# 2. 交互式检索：输入中文回车即查
python3 scripts/fulltext_repl.py --collection docs
```
REPL 里：输入中文 query 回车看命中；`:limit N` 改数量；`:quit` 退出。

## 六、可选配置：analyzer_params

默认用内置 jieba 词典。如需自定义分词（如加专业词典让某些词不被切开），在建库 schema
里加 `analyzer_params`：
```python
schema = CollectionSchema(...)
# 在 schema dict 里设: {"analyzer_params": {"user_dict_path": "/path/to/user.dict.utf8"}}
```
支持 `jieba_dict_path` / `hmm_model_path` / `user_dict_path`。留空则用内置词典。

## 七、效果与限制（实事求是）

- **擅长**：关键词匹配。query 的词和文档字面对得上时，命中准、头部排序好。实测
  T2Ranking 基准 MRR@10 0.42（达官方 BM25 水平）。
- **不擅长**：语义理解。不认识同义词、近义词；大规模语料中"年份页"这类高频词堆叠的
  页面可能挤占排序——这是纯 BM25 词袋模型的固有特性，不是缺陷。
- 要语义检索 / 向量+全文融合，是混合检索（T3）的范围，不在本功能内。
- 同一 collection 可同时做向量 `search` 和全文 `search_by_full_text`。

## 八、常见错误

| 现象 | 原因 | 处理 |
|------|------|------|
| 404 `index has not been flushed` | 没 flush | 先 `client.flush(...)` |
| 404 collection not found | collection 名错/不存在 | 检查名字 |
| 400 `query must not be empty` | query 空或全空白 | 传非空中文 |
| flush 超时 | 库太大、超过 flush_timeout | 调大 `HypervecClient(uri, flush_timeout=1200)` |
| 命中为空 | query 分词后无一个词在语料中出现过 | 正常，换词 |
