# HyperVector 用户配置模块

> 文档状态：已实现并通过配置、启动链路和 package 验证
> 适用范围：HyperVector Python HTTP Server 进程启动与日志配置
> 调研输入：`docs/guc_summary.md`、`docs/runtime_parameter_entry_report.md`、`docs/runtime_parameter_inventory_report.md`

## 1. 背景

改造前，HyperVector HTTP Server 在 `src/python/hypervec_http_server.py` 中直接使用 `argparse.Namespace`，没有项目级配置对象，也没有用户启动配置文件。当前实现已由 `src/python/hypervec_config.py` 提供统一配置对象、INI 加载、CLI 合并、校验、日志初始化和 sample 导出能力。

本设计引入一个轻量、类型化的配置模块，将参数流程收敛为：

```text
配置项定义表默认值
    < INI 配置文件
    < 显式 CLI 参数
    -> 类型解析
    -> 统一校验
    -> HypervecConfig
    -> 日志初始化
    -> HTTP Server 启动
```

设计借鉴 openGauss GUC 的表驱动元数据、类型校验和来源优先级，但不引入数据库 session、事务、reload 和权限上下文机制。

## 2. 设计目标

1. 使用一个配置文件完成 HTTP Server 启动。
2. 保留现有 CLI 启动方式和参数名。
3. 支持 bool、int、string 及 optional string/path。
4. 将默认值、解析、校验、sample 和文档建立在同一配置项元数据上。
5. 对文件、类型、范围、未知项和交叉字段错误做启动前校验。
6. 保证只有显式传入的 CLI 值才覆盖配置文件。
7. 使用 Python 标准库，不为配置功能增加第三方依赖。
8. 保持配置核心可在不导入 FastAPI、Uvicorn、Hypercorn 和 SWIG 扩展的情况下单元测试。

## 3. 非目标

本期不实现：

- C++ 算法、索引构建、搜索或 HTTP 请求级参数迁移。
- `HYPERVEC_OPT_LEVEL`、`HYPERVEC_DISABLE_CPU_FEATURES`、`HYPERVEC_SIMD_LEVEL` 等库加载环境变量整合。
- ARM 构建脚本、CMake 选项或工具链变量整合。
- 配置热更新、SIGHUP reload、配置文件 watch 或运行时修改。
- database/user/session 作用域、SQL `SET/SHOW/RESET` 或事务回滚。
- 多配置文件 include、profile 继承、变量插值、加密密文或远程配置中心。
- Pydantic Settings、PyYAML、TOML parser 等新的配置依赖。
- C++ 配置头文件、动态注册和复杂 hook/extra 机制。
- 自动搜索 `/etc`、当前目录或用户目录中的默认配置文件。

## 4. 实现位置与依赖边界

### 4.1 文件布局

新增：

```text
src/python/hypervec_config.py
test/unit_tests/python/test_hypervec_config.py
configs/hypervec.ini.sample
```

修改：

```text
src/python/hypervec_http_server.py
src/python/CMakeLists.txt
src/python/setup.py
test/unit_tests/python/test_hypervec_http_server.py
```

使用 flat Python module 是因为现有 `hypervec` Python package 通过 `setup.py` 显式复制各个 `.py` 文件组装，单文件模块与当前仓库结构一致。

### 4.2 依赖规则

`hypervec_config.py` 只允许依赖 Python 标准库：

```text
argparse (只限 CLI 辅助类型，可选)
configparser
dataclasses
logging
pathlib
sys
typing
```

配置核心不导入：

- `fastapi`
- `uvicorn`
- `hypercorn`
- `hypervec` SWIG 模块
- `numpy`

ASGI server 和 FastAPI 的导入保留在 `hypervec_http_server.py` 中，并尽量延迟到实际启动分支。

### 4.3 构建和打包

- `src/python/CMakeLists.txt` 增加 `configure_file(hypervec_config.py hypervec_config.py COPYONLY)`。
- `src/python/setup.py` 将 `hypervec_config.py` 复制到构建目录下的 `hypervec` package。
- `configs/hypervec.ini.sample` 作为仓库中的运维示例；CLI exporter 从元数据生成同样内容，不要在 package 中再维护第二份默认值。
- 不新增 CMake C++ target 或公开 C++ header。

## 5. 配置文件格式

### 5.1 选择 INI

本期选择 INI，使用标准库 `configparser`。

相对 JSON 的优点：

- INI 支持注释，适合人工编辑的 server 配置。
- `[server]`、`[defaults]` 和 `[logging]` section 可直接对应三个子配置对象。
- `configparser` 是 Python 标准库，无新依赖。
- sample 中可直接展示说明、默认值和可选值。

本期不选 JSON 的主要原因是 JSON 标准不支持注释，使用它会降低 sample config 的可用性。当前数据只有三组扁平 section，不需要 JSON 的嵌套能力。

### 5.2 INI 解析约定

`ConfigParser` 配置：

```python
ConfigParser(
    interpolation=None,
    strict=True,
    allow_no_value=False,
    empty_lines_in_values=False,
)
```

其他约定：

- 关闭 interpolation，保证路径中的 `%` 不被解析。
- section 名只允许小写 `[server]`、`[defaults]` 和 `[logging]`。
- key 只允许元数据表中的小写名称。实现设置 `optionxform = str` 保留大小写，使非法大小写与拼写错误一样被报告。
- 重复 section 和重复 key 报错。
- 未知 section 和未知 key 报错，不静默忽略。
- `#` 和 `;` 只用于整行注释；sample 不生成行尾注释。
- 值不需要引号；引号不会被自定义剔除，因此引号会成为值的一部分。
- 字符串值移除首尾空白，保留内部空白。
- optional string/path 的空值解析为 `None`。
- bool 不区分大小写，允许 `true/false`、`yes/no`、`on/off`、`1/0`。
- int 只使用十进制解析，不接受浮点、单位或数字尾巴。
- enum string 解析后转为小写再校验。

### 5.3 路径语义

- `--config` 路径相对于当前工作目录解析。
- 配置文件中的 `data_root`、`certfile`、`keyfile`、`log_file_path` 如果是相对路径，则相对于配置文件所在目录解析。
- CLI 中的同类路径保留现有语义，相对于当前工作目录解析。
- 所有路径执行 `expanduser()`。
- 配置解析不扩展 `$VAR` 或 `${VAR}`，避免隐式引入环境变量优先级。

### 5.4 Sample

```ini
[server]
data_root =
host = 127.0.0.1
port = 8080
server = hypercorn
enable_http2 = true
certfile =
keyfile =

[defaults]
default_index_type = hnswflat
default_metric_type = l2

[logging]
enable_logging = true
log_level = info
log_to_stderr = true
log_to_file = false
log_file_path =
```

sample 实际内容由元数据表生成，每个配置项前包含描述、默认值、可选值或范围注释。`data_root` 没有内置默认值，因此 sample 输出空值并注明启动前必须填写。

## 6. 配置数据模型

### 6.1 不可变配置对象

配置对象使用 `@dataclass(frozen=True)`。对象代表已合并、已校验的启动快照，不在运行中修改。

```python
@dataclass(frozen=True)
class ServerConfig:
    data_root: str | None
    host: str
    port: int
    server: str
    enable_http2: bool
    certfile: str | None
    keyfile: str | None


@dataclass(frozen=True)
class IndexDefaultsConfig:
    default_index_type: str
    default_metric_type: str


@dataclass(frozen=True)
class LoggingConfig:
    enable_logging: bool
    log_level: str
    log_to_stderr: bool
    log_to_file: bool
    log_file_path: str | None


@dataclass(frozen=True)
class HypervecConfig:
    server: ServerConfig
    defaults: IndexDefaultsConfig
    logging: LoggingConfig
```

dataclass 字段不再独立声明默认值。`default_config()` 从 `CONFIG_OPTIONS` 构造完整默认对象，以避免元数据表和 dataclass 两处维护同一默认值。

### 6.2 `ServerConfig`

| 字段 | 类型 | 默认值 | 职责 |
|---|---|---|---|
| `data_root` | `str \| None` | `None` | collection 元数据、SQLite 和索引文件根目录；合并后启动必填 |
| `host` | `str` | `127.0.0.1` | ASGI server 绑定主机/IP |
| `port` | `int` | `8080` | ASGI server 绑定端口 |
| `server` | `str` | `hypercorn` | ASGI 实现：`hypercorn` 或 `uvicorn` |
| `enable_http2` | `bool` | `true` | Hypercorn 是否通过 ALPN 声明 HTTP/2；Uvicorn 仍仅提供 HTTP/1.1 |
| `certfile` | `str \| None` | `None` | TLS 证书文件 |
| `keyfile` | `str \| None` | `None` | TLS 私钥文件 |

### 6.3 `IndexDefaultsConfig`

| 字段 | 类型 | 默认值 | 职责 |
|---|---|---|---|
| `default_index_type` | `str` | `hnswflat` | 预留的 collection 默认索引类型；本期只完成加载和统一访问 |
| `default_metric_type` | `str` | `l2` | 预留的 collection 默认度量类型；本期只完成加载和统一访问 |

这两个值不会覆盖请求中显式提供的 `index_type` 或 `metric_type`，本期也不修改现有 collection 创建接口。

### 6.4 `LoggingConfig`

| 字段 | 类型 | 默认值 | 职责 |
|---|---|---|---|
| `enable_logging` | `bool` | `true` | HyperVector Python 日志总开关 |
| `log_level` | `str` | `info` | HyperVector 日志级别，同时传给 ASGI server |
| `log_to_stderr` | `bool` | `true` | 是否创建 stderr handler |
| `log_to_file` | `bool` | `false` | 是否创建 file handler |
| `log_file_path` | `str \| None` | `None` | file handler 输出路径 |

### 6.5 `HypervecConfig`

`HypervecConfig` 仅负责聚合 `server`、`defaults` 和 `logging`。它不持有配置文件路径、来源标记、`argparse.Namespace` 或 ASGI server 对象，避免将解析过程状态混入最终业务配置。

## 7. 配置项元数据

### 7.1 `ConfigOption`

```python
ConfigValue = bool | int | str | None


@dataclass(frozen=True)
class ConfigOption:
    section: str
    key: str
    field_path: tuple[str, str]
    value_type: type
    default: ConfigValue
    description: str
    cli_dest: str | None
    choices: tuple[str, ...] = ()
    validator: Callable[[ConfigValue], None] | None = None
    optional: bool = False
    is_path: bool = False
```

字段职责：

| 字段 | 用途 |
|---|---|
| `section/key` | INI 唯一键与 sample 输出名 |
| `field_path` | 映射到 `HypervecConfig.server/defaults/logging` 的目标字段 |
| `value_type` | bool/int/string 通用解析选择 |
| `default` | 唯一默认值来源 |
| `description` | sample 和配置项文档注释 |
| `cli_dest` | 从 `argparse.Namespace` 提取显式 CLI 覆盖 |
| `choices` | enum-like string 可选值 |
| `validator` | 单配置项的范围/格式校验 |
| `optional` | 空值是否转为 `None` |
| `is_path` | 是否执行路径展开和基准目录解析 |

### 7.2 定义表

| section.key | 类型 | 默认值 | CLI destination | 单项校验 |
|---|---|---|---|---|
| `server.data_root` | string/path | `None` | `data_root` | optional 字符串；启动必填属于最终校验 |
| `server.host` | string | `127.0.0.1` | `host` | 非空 |
| `server.port` | int | `8080` | `port` | `1 <= value <= 65535` |
| `server.server` | string | `hypercorn` | `server` | `hypercorn/uvicorn` |
| `server.enable_http2` | bool | `true` | `enable_http2` | bool |
| `server.certfile` | string/path | `None` | `certfile` | optional |
| `server.keyfile` | string/path | `None` | `keyfile` | optional |
| `defaults.default_index_type` | string | `hnswflat` | `default_index_type` | `flat/ivfflat/ivflvq/ivfpq/hnswflat/hnswlvq/hnswpq` |
| `defaults.default_metric_type` | string | `l2` | `default_metric_type` | `l2/ip/cosine` |
| `logging.enable_logging` | bool | `true` | `enable_logging` | bool |
| `logging.log_level` | string | `info` | `log_level` | `debug/info/warning/error/critical` |
| `logging.log_to_stderr` | bool | `true` | `log_to_stderr` | bool |
| `logging.log_to_file` | bool | `false` | `log_to_file` | bool |
| `logging.log_file_path` | string/path | `None` | `log_file_path` | optional |

### 7.3 元数据驱动的能力

`CONFIG_OPTIONS` 必须驱动：

1. `default_config()` 的默认值构造。
2. INI section/key 合法性检查。
3. bool/int/string 类型解析。
4. 单项 choices/validator 校验。
5. CLI destination 到嵌套配置字段的映射，以及 CLI type/choices/help 中可复用的元数据。
6. sample config 的值、排序和注释。
7. 用户配置项清单中的默认值和约束。

dataclass 、CLI parser 和 sample 不允许再写一套业务默认值。CLI help 如果需要显示默认值，必须从定义表读取。
