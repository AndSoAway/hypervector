# HyperVec Collection Data Bundle 接口说明

本文档介绍为 UltraRAG 用户退出场景新增的 collection data bundle 接口体系，包括数据整体下载、上传与清除能力。

## 背景

UltraRAG 客户有明确的数据安全要求：用户数据可以在运行期间暂时落盘，但用户退出后，服务器上不得继续保留原始向量数据和向量索引文件。同时，Collection Metadata 可以保留，用于用户再次登录后识别已有 collection，并通过上传完整数据包恢复使用，避免重新创建 collection。

现有的 `download_index`/`upload_index` 只处理向量索引文件，不含关系表数据，不满足该需求。UltraRAG 设计师确认：用户再次连接时，不会重新上传文本数据或原始向量数据，因此关系表数据和向量索引数据必须作为一个整体下载、上传和恢复。

---

## Collection 数据的三类划分

| 类型 | 存储位置 | 退出后处理 |
|---|---|---|
| Collection Metadata | `collections.json` | **保留** |
| 关系表存储数据（doc_id / vector / text_content / metadata 等） | `scalar.db` 中 `docs_<collection_name>` 表 | 必须删除 |
| 向量索引数据 | `collections/<collection_name>/index.hypervec` | 必须删除 |

---

## 典型调用流程

### 用户退出

```
1. UltraRAG 查出当前用户关联的 collection 列表
2. 对每个 collection 调用 download_collection_bundle，保存到安全位置
3. 保存成功后调用 purge_collection_data
4. 调用 describe_collection 验证 data_state=purged
```

### 用户重新登录

```
1. 调用 describe_collections，查找 data_state=purged 的 collection
2. 找到对应 bundle 文件
3. 调用 upload_collection_bundle 恢复数据
4. 调用 search 验证恢复正常
```

---

## 新增 HTTP 接口

### GET /collections/{collection_name}/bundle

下载 collection 的完整数据包，包含向量索引和所有关系表数据。

**响应：** 二进制 ZIP 文件（`.hypervec-bundle`）。

**响应头：**

| 头部 | 说明 |
|---|---|
| `X-Hypervec-Collection-Version` | collection 版本号 |
| `X-Hypervec-Bundle-Format` | bundle 格式标识，当前为 `hypervector.collection.bundle.v1` |
| `X-Hypervec-Bundle-Checksum` | bundle 文件的 SHA-256 校验和 |
| `X-Hypervec-Bundle-Size` | bundle 文件字节数 |

**错误码：**

| 状态码 | 原因 |
|---|---|
| 404 | collection 不存在 |
| 409 | `data_state=purged`，没有可下载的数据；`data_state` 为 `importing`/`invalid`；索引过期（`index_version != data_version`，需先 flush）；或 index / scalar / metadata 行数、维度不一致 |
| 500 | 打包失败 |

---

### PUT /collections/{collection_name}/bundle

上传并恢复 bundle，还原关系表数据和向量索引。collection metadata 必须已存在（不会自动创建）。

**Query 参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `checksum` | — | 可选，`sha256:...` 格式，server 端验证 bundle 完整性 |
| `mode` | `replace` | 目前仅支持 `replace`，覆盖已有数据 |

**成功响应示例：**

```json
{
  "uploaded": true,
  "collection_name": "hongloumeng",
  "version": 3,
  "total": 1846,
  "dim": 1024,
  "data_state": "ready",
  "index_checksum": "sha256:...",
  "index_size_bytes": 8388608
}
```

**错误码：**

| 状态码 | 原因 |
|---|---|
| 400 | bundle 格式错误、checksum 不匹配、请求体为空、缺少 v1 必填 checksum 字段、ZIP 含非法条目/超限（解压体积或压缩比）、`row_id` 不连续/重复、请求体超过大小上限、`mode != replace` |
| 404 | collection metadata 不存在 |
| 409 | bundle 的 `collection_name` 与 URL 不一致；`dim`、schema、字段名与 collection 不兼容 |
| 500 | 导入失败 |

---

### POST /collections/{collection_name}/purge-data

删除 server 上该 collection 的用户数据（向量索引 + 关系表），保留 Collection Metadata。

> **注意：** 这不是 `drop_collection`。`drop_collection` 会同时删除 metadata；`purge-data` 只删除用户数据，metadata 保留。

**请求体（可选，默认如下）：**

```json
{"require_exported": true}
```

`require_exported=true` 时，server 要求最近一次成功导出**同时覆盖当前的数据快照与索引快照**（`exported_data_version == data_version` 且 `exported_index_version == index_version` 且 `exported_index_checksum == index_checksum`）。若导出后又发生 insert / import / purge 使数据版本前进，或 `upload_index` 替换了索引（数据版本不变但索引变了），或从未导出过，server 拒绝执行 purge，防止新增数据或新索引在未导出的情况下被永久删除。

purge 执行的操作：
- 从内存中移除 index（`_indexes`）
- 删除 `collections/<name>/index.hypervec`
- 清理 `collections/<name>/` 下所有服务端生成的残留：`*.tmp`、`*.hypervec-bundle`、`*.hypervec-bundle.tmp`、`*.import.staging`、`*.pre-import`、`*.import.tmp`、`*.upload.tmp`，以及受控导出临时目录 `.export.tmp/`
- DROP 任何遗留的导入暂存表 `docs_<name>__import`
- DROP `scalar.db` 中的 `docs_<name>` 表
- 执行 SQLite `PRAGMA wal_checkpoint(TRUNCATE)` + `VACUUM` 减少 WAL/SHM 残留
- 保留 `collections.json` 中的 metadata
- 更新 `data_state=purged`，并推进 `data_version`、重置 `index_version=0`

**成功响应示例：**

```json
{
  "purged": true,
  "collection_name": "hongloumeng",
  "metadata_preserved": true,
  "scalar_deleted": true,
  "index_deleted": true,
  "memory_unloaded": true,
  "data_state": "purged",
  "last_known_total": 1846,
  "last_purged_at": 1781162000.0
}
```

**错误码：**

| 状态码 | 原因 |
|---|---|
| 404 | collection 不存在 |
| 409 | `require_exported=true` 但最近导出未覆盖当前数据+索引快照（`exported_data_version`/`exported_index_version`/`exported_index_checksum` 与当前不一致） |
| 500 | 删除失败 |

---

## describe_collection / get_version 新增字段

`describe_collection`、`describe_collections`、`get_version` 在原有字段基础上新增以下字段，不影响已有字段，旧客户端可忽略：

| 字段 | 类型 | 说明 |
|---|---|---|
| `data_state` | string | `"ready"` / `"purged"` / `"importing"` / `"invalid"` |
| `last_known_total` | int \| null | 最近一次 purge 或 export 时的行数 |
| `last_exported_at` | float \| null | 最近一次 bundle 导出的时间戳（Unix 时间戳） |
| `last_purged_at` | float \| null | 最近一次 purge 的时间戳（Unix 时间戳） |
| `bundle_format` | string \| null | 最近一次导出使用的 bundle 格式标识 |
| `data_version` | int | 数据版本，每次 insert / import / purge 递增 |
| `index_version` | int | 索引版本，flush / upload_index / import 时置为当时的 `data_version`；`index_version == data_version` 表示索引与数据一致（新鲜） |
| `exported_data_version` | int \| null | 最近一次成功导出所覆盖的 `data_version`，用于 purge 资格判断 |
| `exported_index_version` | int \| null | 最近一次成功导出所覆盖的 `index_version`，purge 资格同时校验此项 |

---

## 一致性与事务性保障

为避免联调中发现的数据丢失 / 不一致问题，bundle 体系提供以下保障：

- **导出一致性**：导出前校验索引新鲜（`index_version == data_version`）且 `index.n_total == scalar 行数 == metadata.total`、`index.d == dim`；任一不满足返回 409，提示先 `flush`。因此导出的 bundle 三份内容（manifest / index / scalar）始终自洽。
- **导出资格与 purge**：导出成功记录 `exported_data_version`、`exported_index_version` 与 `exported_index_checksum`；purge（`require_exported=true`）要求这三者与当前 `data_version` / `index_version` / `index_checksum` 全部一致，杜绝“导出后又新增数据再 purge”或“导出后 upload_index 换索引再 purge”导致的永久丢失。
- **索引 label 映射**：导入前校验 bundle 的 `row_id` 为唯一整数（排除 bool）且恰好覆盖 `0..total-1`，每行 vector 一维且维度等于 manifest.dim；manifest 记录 `label_strategy`（v1 为 `implicit_sequential`），保证 index label 能正确映射到 scalar row，而非仅数量相等。
- **insert 崩溃安全**：insert 先推进 `data_version`（使旧导出资格失效）再写标量数据；两次持久化之间崩溃时 purge 会被拒绝（fail-safe），不会误删未导出的新数据。
- **导入清理**：导入的 staging 阶段（写 staging 索引 / staging 表 / commit-intent）任一步失败都统一清理 staging 文件与表并恢复 `data_state`；commit 切换阶段崩溃则由启动恢复前滚或回滚。
- **导入事务性**：导入先把索引写入 `index.hypervec.import.staging`、把行写入暂存表 `docs_<name>__import` 并完成校验，全部通过后才在一次提交中原子切换 index 文件、scalar 表与 metadata；期间以 `data_state=importing` + `import_txn` 作为持久化提交意图记录。任一步失败都回滚到导入前状态。
- **崩溃恢复**：服务启动时扫描 `importing` 状态，切换已完成则前滚定稿、否则回滚；无法恢复时标记 `invalid` 并阻断 search/export，直至重新导入有效 bundle。
- **bundle 完整性**：v1 bundle 必须带齐 `index_checksum` / `scalar_checksum` / `schema_checksum`（缺失即拒绝）；读取时校验校验和、ZIP 条目白名单、解压体积与压缩比上限，并校验 `row_id` 连续唯一。导入时还校验 schema、字段名与目标 collection 兼容。

---

## 新增 pyhypervec 客户端接口

### download_collection_bundle

```python
result = client.download_collection_bundle(
    "hongloumeng",
    "/safe/storage/hongloumeng.hypervec-bundle",
)
# {
#   "collection_name": "hongloumeng",
#   "path": "/safe/storage/hongloumeng.hypervec-bundle",
#   "bytes": 12345678,
#   "version": "2",
#   "bundle_format": "hypervector.collection.bundle.v1",
#   "bundle_checksum": "sha256:..."
# }
```

### upload_collection_bundle

```python
result = client.upload_collection_bundle(
    "hongloumeng",
    "/safe/storage/hongloumeng.hypervec-bundle",
    checksum="sha256:...",   # 可选，传入则 server 端验证 bundle 完整性
    mode="replace",
)
# {"uploaded": True, "data_state": "ready", "total": 1846, "dim": 1024, ...}
```

### purge_collection_data

```python
result = client.purge_collection_data("hongloumeng", require_exported=True)
# {
#   "purged": True,
#   "metadata_preserved": True,
#   "data_state": "purged",
#   "last_known_total": 1846,
#   "last_purged_at": 1781162000.0
# }
```

> **v1 说明：** `download_collection_bundle` 和 `upload_collection_bundle` 当前将完整 bundle 加载到内存后传输。对于超大 collection（数 GB 以上），后续版本需要改为流式传输。

---

## Bundle 文件格式

bundle 是一个 ZIP 压缩包，后缀为 `.hypervec-bundle`，包含以下三个文件：

```
manifest.json    — 元数据与 SHA-256 校验和
index.hypervec   — 二进制向量索引文件
scalar.jsonl     — 每行一条记录，包含全部关系表数据
```

### manifest.json 示例

```json
{
  "format": "hypervector.collection.bundle.v1",
  "collection_name": "hongloumeng",
  "version": 2,
  "dim": 1024,
  "total": 1846,
  "id_field": "id",
  "vector_field": "vector",
  "text_field": "contents",
  "index_checksum": "sha256:...",
  "index_size_bytes": 8388608,
  "scalar_checksum": "sha256:...",
  "schema_checksum": "sha256:...",
  "exported_at": 1781161000.0
}
```

### scalar.jsonl 行示例

```json
{"row_id": 0, "doc_id": "chunk_001", "vector": [0.1, 0.2, 0.3], "text_content": "文本片段", "metadata": {"file_name": "hongloumeng.jsonl", "segment_id": 0}, "created_at": 1781160000.0, "updated_at": 1781160000.0}
```

### 一致性保证

导入时必须保证 `index.hypervec` 与 `scalar.jsonl` 在以下维度上一致：

- `dim`（向量维度）
- `total`（行数）
- `row_id` 顺序（index 用 row_id 回查 scalar）

导入时先验证 manifest 中的 SHA-256 校验和，通过后才写入数据，不允许单独上传不匹配的 index 和 scalar 数据。

> **v1 说明：** scalar.jsonl 中每行包含完整的 float 向量，bundle 体积较大。后续版本可优化为 npz/parquet/SQLite dump 等二进制格式。

---

## 安全边界说明

`purge_collection_data` 执行 SQLite DROP + VACUUM + `secure_delete=ON` + WAL checkpoint，可减少普通文件系统层面的数据残留。

**但这不是密码学级别的安全擦除。** 以下情况仍可能在块层面保留数据痕迹：

- SSD wear-levelling（闪存均衡写入）
- 文件系统日志（journal）
- 系统级快照或备份

如有更高安全要求，需在操作系统层面另行处理。

---

## 与现有接口的关系

| 接口 | 行为 | 是否变更 |
|---|---|---|
| `download_index` / `upload_index` | 只处理 `index.hypervec` 文件 | **不变** |
| `drop_collection` | 删除 metadata + scalar + index | **不变** |
| `describe_collection` / `get_version` | 原有字段保持不变，新增 bundle/purge 相关字段 | 仅新增字段 |
| `download_collection_bundle` | 下载 index + scalar rows 整体 bundle | **新增** |
| `upload_collection_bundle` | 上传恢复整体 bundle | **新增** |
| `purge_collection_data` | 删除用户数据，保留 metadata | **新增** |

`download_index`/`upload_index` 继续可用于低层调试或单独索引同步；UltraRAG 安全退出/恢复场景应使用 bundle 接口。
