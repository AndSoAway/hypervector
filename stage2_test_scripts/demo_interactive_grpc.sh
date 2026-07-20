#!/bin/bash
# HyperVector gRPC examples/index 交互式展示脚本
#
# 这个脚本的作用不是单纯打印说明文字，而是先通过 gRPC 客户端连到
# HyperVector 服务，确认服务健康、读取当前 collection 列表，然后再进入
# 一个交互式菜单，让测试人员选择索引类型并查看对应的 example 说明。
#
# 推荐在 Docker 容器里运行本脚本，并设置：
#   PYTHON_BIN=/app/venv/bin/python
# 这样可以使用镜像内已经安装好的 grpcio、protobuf 和 pyhypervec 绑定，
# 避免宿主机 Python 环境缺依赖导致脚本失败。

set -euo pipefail

# gRPC 服务地址。默认访问本机 50052 端口，也可以通过环境变量覆盖：
#   HYPERVECTOR_GRPC_SERVER=tcp://host:port bash demo_interactive_grpc.sh
SERVER="${HYPERVECTOR_GRPC_SERVER:-tcp://localhost:50052}"

# 下面会临时生成一段 Python 脚本来做真正的 gRPC 调用和菜单展示。
# 用临时文件是为了让 shell 负责准备环境，Python 负责调用服务和输出内容。
TMP_SCRIPT="$(mktemp)"
trap 'rm -f "$TMP_SCRIPT"' EXIT

cat > "$TMP_SCRIPT" <<'PY'
import sys

# 优先使用项目里的 pyhypervec 客户端代码。
# 容器运行时该路径会通过 volume 挂载到 /root/vector/hypervector，
# 这样脚本展示内容可以跟当前项目代码保持一致。
sys.path.insert(0, "/root/vector/hypervector/pyhypervec")
from pyhypervec import HypervecClient

# shell 会把 gRPC 服务地址作为第一个参数传进来。
server = sys.argv[1]

# EXAMPLES 是交互菜单的数据源。
# 每个索引都包含简介、实现原理、适用场景、参数说明和调用示例。
EXAMPLES = {
    "IVFFlat": {
        "name": "IVFFlat",
        "full_name": "Inverted File Flat",
        "description": "倒排文件配合原始向量存储的近似最近邻索引。",
        "implementation": "先训练聚类中心，再将向量分配到最近的倒排桶；搜索时通过 nprobe 扫描若干桶，并在桶内用原始 float 向量做精确距离计算。",
        "use_case": ["高召回 ANN 检索", "可接受训练与倒排结构的场景"],
        "advantages": ["召回高", "距离计算精确", "适合做 IVF 基线"],
        "limitations": ["需要训练", "nprobe 过小会影响召回"],
        "parameters": {"nlist": "聚类中心数", "nprobe": "查询探测桶数"},
        "example_code": {"Python": "index_params.add_index(field_name='vector', index_type='IVFFlat', metric_type='L2', params={'nlist': 1024})"},
        "performance_tips": ["正式 benchmark 使用 nprobe=64", "nlist 与数据规模一起调优"],
        "real_world_examples": ["大规模语义召回", "图片向量检索"],
    },
    "IVFLVQ": {
        "name": "IVFLVQ",
        "full_name": "Inverted File LVQ",
        "description": "倒排文件结合 LVQ 量化的压缩索引。",
        "implementation": "倒排桶负责缩小候选范围，桶内使用 LVQ 压缩向量存储并估计距离。",
        "use_case": ["压缩召回", "内存带宽敏感场景"],
        "advantages": ["比原始向量更省内存", "召回在当前 10K 配置下可超过 0.9"],
        "limitations": ["量化会损失部分召回", "参数过低会明显降精度"],
        "parameters": {"nlist": "聚类中心数", "nlocal": "局部分组数", "nbits": "量化位数", "nprobe": "查询探测桶数"},
        "example_code": {"Python": "index_params.add_index(field_name='vector', index_type='IVFLVQ', metric_type='L2', params={'nlist': 1024, 'nlocal': 64, 'nbits': 10})"},
        "performance_tips": ["正式 benchmark 使用 nprobe=64", "nlocal=64、nbits=10 是当前高召回配置"],
        "real_world_examples": ["压缩向量检索", "在线候选召回"],
    },
    "IVFPQ": {
        "name": "IVFPQ",
        "full_name": "Inverted File Product Quantization",
        "description": "倒排文件结合 PQ 的压缩索引。",
        "implementation": "先用 IVF 缩小候选，再用 PQ 对向量残差进行压缩编码和近似距离计算。",
        "use_case": ["大规模压缩检索", "内存受限召回"],
        "advantages": ["存储紧凑", "当前配置召回可超过 0.9"],
        "limitations": ["m 过小会显著伤召回", "需要和向量维度匹配"],
        "parameters": {"nlist": "聚类中心数", "m_pq": "PQ 子空间数量", "nbits": "每个子空间编码位数", "nprobe": "查询探测桶数"},
        "example_code": {"Python": "index_params.add_index(field_name='vector', index_type='IVFPQ', metric_type='L2', params={'nlist': 1024, 'm_pq': 256, 'nbits': 8})"},
        "performance_tips": ["正式 benchmark 使用 nprobe=64", "m_pq=256、nbits=8 是当前高召回配置"],
        "real_world_examples": ["压缩大规模检索"],
    },
    "HNSWFlat": {
        "name": "HNSWFlat",
        "full_name": "HNSW with Flat vectors",
        "description": "HNSW 图结构配合原始向量精确距离计算。",
        "implementation": "图结构用于快速缩小候选集合，底层距离计算使用原始 float 向量。",
        "use_case": ["通用在线 ANN 检索", "高召回低延迟场景"],
        "advantages": ["召回高", "延迟低", "不需要训练"],
        "limitations": ["内存占用高于量化索引"],
        "parameters": {"M": "图连接数", "ef_construction": "构建搜索宽度", "ef_search": "查询搜索宽度"},
        "example_code": {"Python": "index_params.add_index(field_name='vector', index_type='HNSWFlat', metric_type='L2', params={'M': 32, 'ef_construction': 200})"},
        "performance_tips": ["正式 benchmark 使用 ef_search=64"],
        "real_world_examples": ["在线语义检索"],
    },
    "HNSWLVQ": {
        "name": "HNSWLVQ",
        "full_name": "HNSW with LVQ",
        "description": "HNSW 图结构结合 LVQ 压缩表示。",
        "implementation": "先用 HNSW 图缩小候选，再用 LVQ 压缩向量做距离估计。",
        "use_case": ["压缩图索引召回", "内存带宽敏感场景"],
        "advantages": ["压缩友好", "召回稳定"],
        "limitations": ["量化会带来召回损失"],
        "parameters": {"M": "图连接数", "nlocal": "局部分组数", "nbits": "量化位数", "ef_search": "查询搜索宽度"},
        "example_code": {"Python": "index_params.add_index(field_name='vector', index_type='HNSWLVQ', metric_type='L2', params={'M': 32, 'nlocal': 16, 'nbits': 10})"},
        "performance_tips": ["正式 benchmark 使用 ef_search=64"],
        "real_world_examples": ["低内存图索引召回"],
    },
    "HNSWPQ": {
        "name": "HNSWPQ",
        "full_name": "HNSW with PQ",
        "description": "HNSW 图结构结合 PQ 压缩表示。",
        "implementation": "先用 HNSW 图快速定位候选，再用 PQ 压缩表示进行距离估计。",
        "use_case": ["压缩在线检索", "低内存召回"],
        "advantages": ["延迟低", "内存占用低"],
        "limitations": ["PQ 参数过小会影响召回"],
        "parameters": {"M": "图连接数", "m": "PQ 子空间数量", "nbits": "每个子空间编码位数", "ef_search": "查询搜索宽度"},
        "example_code": {"Python": "index_params.add_index(field_name='vector', index_type='HNSWPQ', metric_type='L2', params={'M': 32, 'm': 256, 'nbits': 8})"},
        "performance_tips": ["正式 benchmark 使用 ef_search=64"],
        "real_world_examples": ["压缩图召回"],
    },
}


def print_block(title, value):
    if value is None or value == "":
        return
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if isinstance(value, list):
        for item in value:
            print(f"- {item}")
    elif isinstance(value, dict):
        for key, item in value.items():
            print(f"\n[{key}]")
            if isinstance(item, dict):
                for sub_key, sub_value in item.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(item)
    else:
        print(value)


def print_code_examples(example_code):
    if not example_code:
        return
    print("\n" + "=" * 80)
    print("实现代码 / 调用示例")
    print("=" * 80)
    for lang, steps in example_code.items():
        print(f"\n--- {lang} ---")
        if isinstance(steps, dict):
            for step, code in steps.items():
                print(f"\n# {step}")
                print(code)
        else:
            print(steps)


def show_detail(index_name):
    data = EXAMPLES[index_name]
    print("\n" + "#" * 80)
    print(f"{data.get('name', index_name)} - {data.get('full_name', '')}")
    print("#" * 80)
    print_block("简介", data.get("description"))
    print_block("实现原理", data.get("implementation"))
    print_block("适用场景", data.get("use_case"))
    print_block("优势", data.get("advantages"))
    print_block("限制", data.get("limitations"))
    print_block("参数说明", data.get("parameters"))
    print_code_examples(data.get("example_code"))
    print_block("调优建议", data.get("performance_tips"))
    print_block("真实场景示例", data.get("real_world_examples"))


def main():
    client = HypervecClient(server)
    health = client.health()
    if health.get("status") != "ok":
        raise SystemExit(f"服务状态异常: {health}")

    collections = []
    try:
        collections = client.list_collections()
    except Exception:
        collections = []

    indexes = ["IVFFlat", "IVFLVQ", "IVFPQ", "HNSWFlat", "HNSWLVQ", "HNSWPQ"]

    while True:
        print("\n" + "=" * 80)
        print("HyperVector gRPC Examples 交互式展示")
        print("=" * 80)
        print(f"gRPC 服务地址: {server}")
        print(f"服务状态: {health.get('status')}")
        print("当前 collections: " + (", ".join(collections) if collections else "无或未加载"))
        print("\n请选择要展示的索引类型：")
        for i, name in enumerate(indexes, 1):
            info = EXAMPLES[name]
            print(f"  {i}. {name} - {info.get('full_name', '')}")
        print("  q. 退出")

        try:
            choice = input("请输入序号: ").strip()
        except EOFError:
            print("\n已退出。")
            return

        if choice.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            return
        if not choice.isdigit() or not (1 <= int(choice) <= len(indexes)):
            print("输入无效，请重新选择。")
            continue

        show_detail(indexes[int(choice) - 1])
        try:
            input("\n按回车返回菜单...")
        except EOFError:
            print()
            return


if __name__ == "__main__":
    main()
PY

# 优先使用容器内的 Python。镜像里这个 Python 已经带好 grpcio、protobuf、pyhypervec，
# 所以测试脚本建议在容器里跑，而不是依赖宿主机 Python 环境。
PYTHON_BIN="${PYTHON_BIN:-/app/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

# 执行临时生成的 Python 脚本，并把 gRPC 服务地址传进去。
"$PYTHON_BIN" "$TMP_SCRIPT" "$SERVER"
