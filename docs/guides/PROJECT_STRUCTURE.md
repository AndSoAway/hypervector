# HyperVector 项目结构详细说明

## 📁 根目录概览

```
hypervector/
├── 📁 benchmark/           # 性能测试相关
├── 📁 build/               # CMake 构建输出目录
├── 📁 cmake/               # CMake 配置文件
├── 📁 demo/                # 演示脚本
├── 📁 docker/              # Docker 配置文件
├── 📁 docs/                # 文档目录
├── 📁 pyhypervec/          # Python 客户端 SDK
├── 📁 scripts/             # 构建和工具脚本
├── 📁 src/                 # C++ 源代码
├── 📁 test/                # 测试代码
│
├── 📄 CMakeLists.txt       # CMake 主配置文件
├── 📄 deploy_docker.sh     # ⭐ Docker 一键部署脚本
├── 📄 README.md            # 项目说明文档
├── 📄 CHANGELOG.md         # 更新日志
├── 📄 LICENSE              # 开源协议
└── 📄 .gitignore           # Git 忽略配置
```

---

## 📊 核心目录详解

### 1. 📁 `benchmark/` - 性能测试

**用途**：Wiki 1M 数据集的性能基准测试

```
benchmark/
├── fbin_utils.py           # .fbin 格式文件读取工具
├── import_wiki_1m.py       # Wiki 1M 数据导入脚本
├── benchmark_wiki_1m.py    # 性能基准测试脚本
└── run_full_test.sh        # ⭐ 一键运行完整测试流程
```

### 2. 📁 `demo/` - 演示脚本

**用途**：examples 接口演示和功能测试

```
demo/
├── demo_interactive.sh         # ⭐ 交互式 examples 演示（推荐）
├── demo_examples_full.sh       # 完整 examples 演示
├── demo_examples.py            # Python 演示脚本
└── test_hnsw.py                # HNSW 功能快速测试
```

### 3. 📁 `docker/` - Docker 配置

```
docker/
├── Dockerfile              # Docker 镜像构建文件
├── .dockerignore           # Docker 构建忽略文件
└── docker-compose.yml      # Docker Compose 配置
```

### 4. 📁 `docs/` - 文档

```
docs/
├── guides/                                     # 用户指南
│   ├── README_SCRIPTS.md                          # ⭐ 脚本使用指南
│   ├── PROJECT_STRUCTURE.md                       # 项目结构说明（本文件）
│   └── DEPLOYMENT_STATUS.md                       # 部署状态报告
│
├── arm_linux_pyhypervec_server_deployment.md  # ARM 部署指南
└── pyhypervec_http_server.md                  # HTTP Server 文档
```

### 5. 📁 `pyhypervec/` - Python 客户端

```
pyhypervec/
└── pyhypervec/                 # Python 包源代码
    ├── client.py                   # ⭐ HypervecClient 客户端类
    ├── exceptions.py               # 异常定义
    ├── schema.py                   # Schema 定义
    └── __init__.py                 # 包初始化
```

### 6. 📁 `src/` - C++ 源代码

```
src/
├── index/                      # 索引实现
│   ├── flat/                       # Flat 索引实现
│   ├── hnsw/                       # HNSW 索引实现
│   └── ivf/                        # IVF 索引实现
│
├── quantization/               # 量化实现
│   ├── lvq/                        # LVQ 量化
│   └── pq/                         # PQ 量化
│
├── persistence/                # 持久化实现
│   ├── index_read.cpp
│   └── index_write.cpp
│
└── python/                     # Python 绑定
    └── hypervec/
        ├── hypervec_http_server.py     # ⭐ HTTP Server 主程序
        ├── hypervec_server_engine.py   # Server 引擎
        ├── examples_data.py            # ⭐ examples 数据
        ├── hypervec_meta_store.py      # 元数据存储
        └── ...
```

### 7. 📁 `scripts/` - 构建脚本

```
scripts/
├── build.sh                        # 本地构建脚本
└── build_arm_pyhypervec_server.sh  # ARM 架构构建脚本
```

### 8. 📁 `cmake/` - CMake 配置

```
cmake/
├── FindMKL.cmake               # 查找 Intel MKL 库
├── hypervec-config.cmake.in    # HyperVec 配置模板
└── thirdparty/                 # 第三方依赖
```

### 9. 📁 `build/` - 构建输出

**注意**：此目录自动生成，不提交到 Git

```
build/
├── src/
│   ├── libhypervec.a               # HyperVec 静态库
│   └── python/
│       ├── _swighypervec.so        # Python C++ 扩展
│       └── swighypervec.py         # Python 绑定模块
└── ...
```

### 10. 📁 `test/` - 测试代码

```
test/
└── examples/                   # 示例程序
    └── cpp/
        └── demo_hnsw               # HNSW 性能测试 demo
```

---

## 📄 根目录文件说明

```
📄 deploy_docker.sh             # ⭐ 一键部署脚本
📄 CMakeLists.txt               # CMake 主配置文件
📄 README.md                    # 项目说明文档
📄 CHANGELOG.md                 # 更新日志
📄 LICENSE                      # 开源协议
📄 .gitignore                   # Git 忽略配置
```

---

## 🎯 关键路径速查

| 功能 | 文件路径 |
|------|---------|
| HTTP Server 主程序 | `src/python/hypervec/hypervec_http_server.py` |
| examples 接口数据 | `src/python/hypervec/examples_data.py` |
| Python 客户端 | `pyhypervec/pyhypervec/client.py` |
| HNSW 索引实现 | `src/index/hnsw/index_hnsw.cpp` |
| Docker 镜像 | `docker/Dockerfile` |
| 脚本使用指南 | `docs/guides/README_SCRIPTS.md` |

---

## 🚀 快速导航

**部署系统** → `./deploy_docker.sh`

**演示 examples 接口** → `cd demo && ./demo_interactive.sh`

**运行性能测试** → `cd benchmark && ./run_full_test.sh`

**查看使用文档** → `docs/guides/README_SCRIPTS.md`

---

*最后更新：2026-06-10*
