# HyperVec Index Server 启动指南

---

## 目录

1. [项目结构](#项目结构)
2. [编译链说明](#编译链说明)
3. [前置要求](#前置要求)
4. [第一步：编译 C++ 库与 Python 绑定](#第一步编译-c-库与-python-绑定)
5. [第二步：验证 Python 绑定](#第二步验证-python-绑定)
6. [第三步：安装 Python 依赖](#第三步安装-python-依赖)
7. [第四步：运行集成测试](#第四步运行集成测试)
8. [第五步：启动服务器](#第五步启动服务器)
9. [API 一览](#api-一览)
10. [作为 RAG 后端使用](#作为-rag-后端使用替换-milvus)
11. [常见问题](#常见问题)

---

## 项目结构

```
hypervector/
├── build3/                         # CMake 构建输出目录
│   └── src/
│       ├── libhypervec.a           # 编译产物①：HyperVec 核心静态库
│       └── python/
│           ├── _swighypervec.pyd   # 编译产物②：Python 扩展动态库
│           └── swighypervec.py     # SWIG 自动生成的 Python 桥接层
├── src/
│   ├── python/
│   │   └── swighypervec.swig       # SWIG 接口定义（手写）
│   └── ...                         # HyperVec C++ 源码
└── server/                         # 本目录
    ├── hv.py                       # _swighypervec.pyd 的 Python 包装层
    ├── index_manager.py            # 索引生命周期管理（创建/添加/搜索/持久化）
    ├── scalar_store.py             # SQLite 标量字段存储
    ├── store.py                    # Collection 元数据持久化（JSON 文件）
    ├── main.py                     # FastAPI 路由入口
    ├── hypervec_backend.py         # MilvusIndexBackend 的 drop-in 替代
    ├── test_server.py              # 集成测试（使用真实 .pyd）
    └── indexes/                    # 运行时自动创建，存放序列化的 .index 文件
```

运行时会在 `server/` 下自动生成：

| 文件/目录 | 说明 |
|-----------|------|
| `indexes/*.index` | HyperVec 序列化索引文件（可被 ReadIndex 直接加载） |
| `collections.json` | Collection 元数据（名称、版本号、索引路径、时间戳） |
| `scalar.db` | SQLite 数据库，存储文档文本和标量字段 |

---

## 编译链说明

本项目的 Python 绑定通过 SWIG 桥接 C++ 实现，编译分两个产物：

```
HyperVec C++ 源码（src/）
        │
        ▼ mingw32-make hypervec
  libhypervec.a              ← 产物①：所有 C++ 功能打包成静态库
  （HNSW / Flat / IVF 索引、AVX2 距离计算、序列化等）
        │
        │   swighypervec.swig
        │        │
        │        ▼ SWIG 工具
        │   swighypervecPYTHON_wrap.cxx   ← 自动生成的 C++ 胶水代码
        │        │
        │        ▼ GCC 编译 + 链接 libhypervec.a
        └──────► _swighypervec.pyd        ← 产物②：Python 可直接 import 的扩展库
                        │
                        ▼ Python 调用
                      hv.py               ← 封装为 add/search/ntotal 等高层接口
                        │
                        ▼
               index_manager.py / server
```

**只编译了这两个目标**，其余目标（`swighypervec_avx512`、`swighypervec_sve` 等其他 SIMD 变体）在 CMake 中被标记为 `EXCLUDE_FROM_ALL`，未参与构建。

---

## 前置要求

| 依赖 | 版本要求 | 安装方式 |
|------|----------|----------|
| MSYS2 MinGW64 | 最新 | 安装路径必须为 `C:\msys64` |
| GCC (MinGW64) | ≥ 13 | `pacman -S mingw-w64-x86_64-gcc` |
| CMake | ≥ 3.17 | `pacman -S mingw-w64-x86_64-cmake` |
| SWIG | ≥ 4.0 | `pacman -S mingw-w64-x86_64-swig` |
| Python | ≥ 3.11 | 系统 Python（非 MSYS2 自带的） |

---

## 第一步：编译 C++ 库与 Python 绑定

> 以下命令均在 **MSYS2 MinGW64 Shell** 中执行（不是 cmd 或 PowerShell）。

```bash
cd /d/QIYUAN/hypervector

# 创建构建目录（已存在则跳过）
mkdir -p build3 && cd build3

# 生成 CMake 构建文件
cmake .. \
  -G "MinGW Makefiles" \
  -DCMAKE_BUILD_TYPE=Release

# 1. 编译核心 C++ 库 → 生成 build3/src/libhypervec.a
mingw32-make hypervec -j$(nproc)

# 2. 用 SWIG 将 swighypervec.swig 转译为 C++ 胶水代码（swighypervecPYTHON_wrap.cxx）
mingw32-make swighypervec_swig_compilation

# 3. 编译胶水代码为目标文件
RSP="src/python/CMakeFiles/swighypervec.dir/includes_CXX.rsp"
OBJ="src/python/CMakeFiles/swighypervec.dir/CMakeFiles/swighypervec.dir/swighypervecPYTHON_wrap.cxx.obj"
WRAP="src/python/CMakeFiles/swighypervec.dir/swighypervecPYTHON_wrap.cxx"

c++ -DPy_NO_LINK_LIB -Dswighypervec_EXPORTS \
    -mavx2 -mfma -O3 -DNDEBUG -std=gnu++20 \
    -fvisibility=hidden -fno-keep-inline-dllexport \
    -Wa,-mbig-obj -fopenmp \
    @"$RSP" -o "$OBJ" -c "$WRAP"

# 4. 链接胶水目标文件与 libhypervec.a → 生成 build3/src/python/_swighypervec.pyd
mingw32-make swighypervec -j1
```

编译成功后确认产物存在：

```bash
ls build3/src/libhypervec.a
ls build3/src/python/_swighypervec.pyd
```

---

## 第二步：验证 Python 绑定

在系统 Python（PowerShell 或 cmd）中执行，验证底层 C++ 索引可以被 Python 正常调用：

```python
import os, sys, numpy as np

os.add_dll_directory(r"C:\msys64\mingw64\bin")
sys.path.insert(0, r"D:\QIYUAN\hypervector\build3\src\python")

import swighypervec as hv

idx = hv.IndexFlatL2(4)
vecs = np.eye(4, dtype=np.float32)
idx.Add(4, vecs.ctypes.data)
print("ntotal =", idx.n_total)              # 期望: 4

D = np.empty((1, 2), dtype=np.float32)
I = np.empty((1, 2), dtype=np.int64)
idx.Search(1, vecs[:1].ctypes.data, 2, D.ctypes.data, I.ctypes.data)
print("最近邻 IDs:", I[0])                  # 期望: [0, 1]
print("距离:      ", D[0])                  # 期望: [0. 2.]

hv.WriteIndex(idx, "test.index")
idx2 = hv.ReadIndex("test.index")
print("加载后 ntotal =", idx2.n_total)      # 期望: 4
```

**实际运行输出：**

```
ntotal = 4
最近邻 IDs: [0 1]
距离:       [0. 2.]
加载后 ntotal = 4
```

---

## 第三步：安装 Python 依赖

```bash
pip install fastapi "uvicorn[standard]" numpy httpx pytest
```

---

## 第四步：运行集成测试

测试直接调用真实的 `_swighypervec.pyd`，需第一步编译完成后才能运行：

```bash
cd D:\QIYUAN\hypervector
python -m pytest server/test_server.py -v
```

**实际运行输出：**

```
============================= test session starts =============================
platform win32 -- Python 3.13.13, pytest-9.0.3, pluggy-1.6.0
collected 24 items

server/test_server.py::TestScalarStore::test_count PASSED                [  4%]
server/test_server.py::TestScalarStore::test_create_and_retrieve PASSED  [  8%]
server/test_server.py::TestScalarStore::test_drop_table PASSED           [ 12%]
server/test_server.py::TestScalarStore::test_missing_row_returns_none PASSED [ 16%]
server/test_server.py::TestMetaStore::test_bump_version PASSED           [ 20%]
server/test_server.py::TestMetaStore::test_create_and_get PASSED         [ 25%]
server/test_server.py::TestMetaStore::test_delete PASSED                 [ 29%]
server/test_server.py::TestMetaStore::test_persist_and_reload PASSED     [ 33%]
server/test_server.py::TestIndexManager::test_create_and_add PASSED      [ 37%]
server/test_server.py::TestIndexManager::test_delete_collection PASSED   [ 41%]
server/test_server.py::TestIndexManager::test_ntotal_matches_added PASSED [ 45%]
server/test_server.py::TestIndexManager::test_persist_and_reload PASSED  [ 50%]
server/test_server.py::TestIndexManager::test_rebuild_resets_rows PASSED [ 54%]
server/test_server.py::TestIndexManager::test_search_missing_collection_raises PASSED [ 58%]
server/test_server.py::TestIndexManager::test_search_nearest_is_exact_match PASSED [ 62%]
server/test_server.py::TestIndexManager::test_search_returns_metadata PASSED [ 66%]
server/test_server.py::TestIndexManager::test_version_increments_on_each_add PASSED [ 70%]
server/test_server.py::TestAPI::test_create_and_list PASSED              [ 75%]
server/test_server.py::TestAPI::test_delete PASSED                       [ 79%]
server/test_server.py::TestAPI::test_health PASSED                       [ 83%]
server/test_server.py::TestAPI::test_sync_check_needs_sync PASSED        [ 87%]
server/test_server.py::TestAPI::test_sync_check_no_sync PASSED           [ 91%]
server/test_server.py::TestAPI::test_version_not_found PASSED            [ 95%]
server/test_server.py::TestAPI::test_version_polling PASSED              [100%]

======================== 24 passed, 1 warning in 0.91s ========================
```

**每条测试验证了什么功能：**

### SQLite 标量存储（4 项）

| 测试 | 验证功能 |
|------|---------|
| `test_create_and_retrieve` | 插入文档后能按 row_id 查回原始文本、metadata |
| `test_missing_row_returns_none` | 查询不存在的 row_id 返回 None，不报错 |
| `test_drop_table` | 删除 collection 后对应表被清空 |
| `test_count` | 批量插入后行数统计正确 |

### Collection 元数据（4 项）

| 测试 | 验证功能 |
|------|---------|
| `test_create_and_get` | 创建 collection 后能查到元数据，初始 version=1 |
| `test_bump_version` | 每次更新索引后版本号正确递增 |
| `test_delete` | 删除后 get 返回 None |
| `test_persist_and_reload` | 重启后从 JSON 文件恢复版本号（持久化正确） |

### 索引管理（9 项，调用真实 C++ 索引）

| 测试 | 验证功能 |
|------|---------|
| `test_create_and_add` | 创建 collection、写入向量后 SQLite 行数与版本号正确 |
| `test_ntotal_matches_added` | C++ 索引内部 ntotal 与写入数量一致 |
| `test_search_returns_metadata` | 搜索结果中 doc_id、score、metadata 字段均正确返回 |
| `test_search_nearest_is_exact_match` | 搜索每个向量自身时 score=0，最近邻就是自己（向量精度验证） |
| `test_rebuild_resets_rows` | 重建索引后旧向量清除，ntotal 和 SQLite 行数均更新 |
| `test_persist_and_reload` | WriteIndex 后重新加载，ntotal 正确，索引可继续搜索 |
| `test_version_increments_on_each_add` | 每次 add_vectors 版本号都递增 |
| `test_delete_collection` | 删除后索引和 SQLite 均清除 |
| `test_search_missing_collection_raises` | 查询不存在的 collection 抛 KeyError |

### FastAPI 路由（7 项）

| 测试 | 验证功能 |
|------|---------|
| `test_health` | `/health` 接口返回 `{"status": "ok"}` |
| `test_create_and_list` | POST 创建后 GET 列表中出现该 collection |
| `test_version_polling` | 新建后 version=1，可被前端轮询到 |
| `test_version_not_found` | 查询不存在的 collection 返回 404 |
| `test_sync_check_needs_sync` | 客户端版本落后时 `needs_sync=true` |
| `test_sync_check_no_sync` | 客户端版本与服务端一致时 `needs_sync=false` |
| `test_delete` | DELETE 后再查 version 返回 404 |

---

## 第五步：启动服务器

```bash
cd D:\QIYUAN\hypervector\server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：

| 地址 | 说明 |
|------|------|
| http://localhost:8000/docs | Swagger 交互文档 |
| http://localhost:8000/health | 服务健康检查 |

---

## API 一览

### Collection 管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/collections` | 列出所有 collection |
| `POST` | `/collections` | 创建 collection |
| `DELETE` | `/collections/{id}` | 删除 collection |

**创建请求体：**
```json
{ "name": "my_kb", "dim": 768 }
```

### 版本同步

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/collections/{id}/version` | 轮询版本号（前端定时调用） |
| `POST` | `/collections/{id}/sync-check` | 断线重连检查 |
| `GET` | `/collections/{id}/index` | 下载序列化索引文件 |

**sync-check 请求体：**
```json
{ "client_version": 3 }
```

**sync-check 响应：**
```json
{ "needs_sync": true, "server_version": 5, "client_version": 3 }
```

---

## 作为 RAG 后端使用（替换 Milvus）

`HyperVecIndexBackend` 与 `MilvusIndexBackend` 接口完全兼容：

```python
from hypervec_backend import HyperVecIndexBackend

backend = HyperVecIndexBackend(
    contents=[],
    config={
        "collection_name": "my_collection",
        "collection_display_name": "我的知识库",
    },
    logger=logger,
)

# 构建索引
backend.build_index(
    embeddings=np.array(..., dtype=np.float32),  # shape (N, dim)
    ids=np.arange(N),
    overwrite=False,
    contents=["文档1", "文档2", ...],
    metadatas=[{"source": "doc.pdf"}, ...],
)

# 检索，返回每个查询的 Top-K 命中列表
results = backend.search_with_meta(query_embeddings, top_k=5)
# results[i][j] 含 content / score / rank / doc_id / metadata 字段
```

---

## 常见问题

**Q：启动时报 `DLL load failed`**

`hv.py` 会自动调用 `os.add_dll_directory(r"C:\msys64\mingw64\bin")` 注册运行时库路径。若仍失败，手动将 `C:\msys64\mingw64\bin` 加入系统 PATH，确认其中有 `libgomp-1.dll`、`libstdc++-6.dll`。

**Q：`import swighypervec` 报 `ModuleNotFoundError`**

检查 `build3/src/python/_swighypervec.pyd` 是否存在。不存在则重新执行第一步编译。

**Q：`IndexFlatL2(dim)` 抛 `TypeError`**

使用的是旧版 `.pyd`。需重新编译（含 `%apply long long { hypervec::idx_t }` 的版本）。

**Q：Search 抛 `Wrong number or type of arguments`**

使用的是旧版 `.pyd`。需重新编译（含 `%typemap(typecheck)` 的版本）。
