# HyperVector 脚本使用指南

## 📁 目录结构

```
hypervector/
├── deploy_docker.sh          # ⭐ Docker 一键部署
│
├── benchmark/                # 📊 性能测试脚本
│   ├── fbin_utils.py             # 读取 .fbin 文件工具
│   ├── import_wiki_1m.py         # 导入 Wiki 1M 数据
│   ├── benchmark_wiki_1m.py      # 性能基准测试（100并发）
│   └── run_full_test.sh          # ⭐ 一键运行完整测试
│
└── demo/                     # 🎬 演示脚本
    ├── demo_interactive.sh       # ⭐ 交互式 examples 演示（推荐）
    ├── demo_examples_full.sh     # 完整 examples 演示
    ├── test_hnsw.py              # HNSW 小数据功能测试
    └── demo_examples.py          # Python 演示脚本
```

---

## 🚀 快速开始

### 1. 部署 HyperVector

```bash
./deploy_docker.sh
```

等待几分钟，Docker 镜像构建并启动。

---

### 2. 演示 examples 接口

```bash
cd demo
./demo_interactive.sh
```

**功能**：
- 交互式选择索引类型（1-8）
- 查看完整的使用文档
- 包含参数详解和代码示例

---

### 3. 运行性能测试

```bash
cd benchmark
./run_full_test.sh
```

**功能**：
- 导入 Wiki 1M 数据（100万条向量）
- 运行 100 并发性能测试
- 输出 QPS、延迟等指标

---

## 📖 详细说明

### 演示脚本（demo/）

#### `demo_interactive.sh` ⭐ 推荐

**用途**：给老师演示 examples 接口

**特点**：
- 可选择查看任意索引（HNSW, Flat, IVF...）
- 分步展示：基本信息 → 优势限制 → 参数详解 → 代码示例
- 格式清晰，适合演示

**运行**：
```bash
cd demo
./demo_interactive.sh
```

---

#### `test_hnsw.py`

**用途**：快速验证 HNSW 功能

**功能**：
- 创建 HNSW Collection
- 插入 1000 条测试向量
- 执行搜索测试

**运行**：
```bash
cd demo
python3 test_hnsw.py
```

---

### 性能测试脚本（benchmark/）

#### `run_full_test.sh` ⭐ 推荐

**用途**：一键运行完整的性能测试

**流程**：
1. 检查环境（Server 是否运行，数据文件是否存在）
2. 导入 Wiki 1M 数据（可选）
3. 运行性能测试（10,000 次查询，100 并发）
4. 输出性能报告

**运行**：
```bash
cd benchmark
./run_full_test.sh
```

**时间**：首次运行约 30-50 分钟（含数据导入）

---

#### `import_wiki_1m.py`

**用途**：导入 Wiki 1M 数据到 HyperVector

**参数**：
```bash
--file              # .fbin 文件路径
--collection        # Collection 名称
--batch-size        # 批大小（默认 10000）
--M                 # HNSW M 参数（默认 32）
--ef-construction   # ef_construction（默认 200）
```

**示例**：
```bash
cd benchmark
python3 import_wiki_1m.py \
    --file /data/ljx/test_fbin/wiki_all_1M/base.1M.fbin \
    --collection wiki_hnsw_1m \
    --M 32 \
    --ef-construction 200
```

---

#### `benchmark_wiki_1m.py`

**用途**：运行性能基准测试

**参数**：
```bash
--queries-file      # 查询向量文件
--collection        # Collection 名称
--num-queries       # 查询总数（默认 10000）
--concurrent        # 并发数（默认 100）
--ef-search         # ef_search 参数列表（默认 128,256,512）
```

**示例**：
```bash
cd benchmark
python3 benchmark_wiki_1m.py \
    --collection wiki_hnsw_1m \
    --num-queries 10000 \
    --concurrent 100 \
    --ef-search "128,256,512"
```

---

## 🎯 验收演示流程

### 完整演示（40分钟）

```bash
# 1. 部署系统（5分钟）
./deploy_docker.sh

# 2. 演示 examples 接口（10分钟）
cd demo
./demo_interactive.sh
# 选择 1 (HNSW) 演示完整文档

# 3. 演示基本功能（5分钟）
python3 test_hnsw.py
# 展示创建、插入、搜索功能

# 4. 运行性能测试（20分钟）
cd ../benchmark
./run_full_test.sh
# 如果数据已导入，选择 N 跳过导入步骤
```

---

### 快速演示（15分钟，数据已导入）

```bash
# 1. 检查系统运行
curl http://localhost:8080/health

# 2. 演示 examples 接口（5分钟）
cd demo
./demo_interactive.sh

# 3. 运行性能测试（10分钟）
cd ../benchmark
python3 benchmark_wiki_1m.py --collection wiki_hnsw_1m
```

---

## 📊 预期输出

### examples 接口演示

```
支持的索引类型：
  1. HNSW
  2. Flat
  ...

请选择要查看的索引类型：
[输入 1]

【索引名称】HNSW
【完整名称】Hierarchical Navigable Small World Graph（分层导航小世界图）

【功能描述】
HNSW 是一种基于图结构的向量索引算法...

【优势】
  ✓ 召回率高：合理参数下可达 99% 以上
  ...

【参数详解】
参数名：M
说明：每个节点在图中的最大连接数...
```

---

### 性能测试输出

```
================================================================================
HyperVector Wiki 1M 性能测试报告
================================================================================

数据集信息:
  数据集: Wiki 1M (1,000,000 vectors, 768 dimensions)
  索引类型: HNSW (M=32, ef_construction=200)

测试配置:
  并发用户数: 100
  每轮查询数: 10,000

--------------------------------------------------------------------------------
ef_search    QPS        平均(ms)   P50(ms)    P95(ms)    P99(ms)   
--------------------------------------------------------------------------------
128          1215.07    1.85       1.52       3.21       5.89      
256          850.23     2.34       2.01       4.56       8.45      
512          445.67     4.23       3.89       8.91       15.23     
--------------------------------------------------------------------------------
```

---

## ⚠️ 注意事项

1. **数据文件路径**：确认 `/data/ljx/test_fbin/wiki_all_1M/` 下的文件存在
2. **首次导入时间**：Wiki 1M 数据导入约需 20-30 分钟
3. **内存需求**：建议至少 8GB 可用内存
4. **端口占用**：确保 8080 端口未被占用

---

## 🆘 故障排查

### Server 未运行
```bash
docker ps | grep hypervector
# 如果没有输出，重新部署
./deploy_docker.sh
```

### 数据文件不存在
```bash
ls -lh /data/ljx/test_fbin/wiki_all_1M/
# 确认文件路径是否正确
```

### 导入失败
```bash
# 检查 Collection 是否已存在
curl http://localhost:8080/collections
# 删除旧 Collection
# （在 import_wiki_1m.py 运行时会提示）
```

---

## 📞 联系方式

如有问题，请联系项目维护者。
