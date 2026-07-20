#!/bin/bash
# HyperVector gRPC examples/index 交互式展示脚本。
# 这个脚本不是单纯打印静态文本：它会启动一个临时 Python gRPC 客户端，
# 连接正在运行的 HyperVector gRPC 服务，调用 health() 检查服务状态，
# 再调用 list_collections() 读取当前 collections，然后进入交互式菜单。
# 用户输入序号后，脚本展示对应索引的 example 说明。
# 推荐在 Docker 容器内运行，直接使用镜像内 grpcio/protobuf，避免宿主机 Python 依赖不一致。

set -euo pipefail

SERVER="${HYPERVECTOR_GRPC_SERVER:-tcp://localhost:50052}"
TMP_SCRIPT="$(mktemp)"
trap 'rm -f "$TMP_SCRIPT"' EXIT

cat > "$TMP_SCRIPT" <<'PY'
import sys

sys.path.insert(0, "/root/vector/hypervector/pyhypervec")
from pyhypervec import HypervecClient

server = sys.argv[1]

EXAMPLES = {
    "IVFFlat": {
        "name": "IVFFlat",
        "full_name": "Inverted File Flat Index",
        "description": "倒排聚类索引，通过只搜索部分聚类降低查询开销。",
        "implementation": "先训练 nlist 个聚类中心，插入时把向量写入对应倒排桶；查询时使用 nprobe 控制探测桶数量。为了保证召回率，测试时应优先提高 nprobe，而不是追求低延迟。",
        "use_case": ["大规模向量粗召回", "可接受近似结果的搜索"],
        "advantages": ["查询成本可控", "适合大规模数据"],
        "limitations": ["需要训练", "召回受 nprobe 影响", "如果索引没有 flush 持久化，重启后可能加载不完整导致召回偏低"],
        "parameters": {
            "nlist": "聚类中心数，示例默认 1024",
            "nprobe": "查询探测聚类数，召回不足时优先提高",
        },
        "example_code": {
            "Python": {
                "create": "index_params.add_index(field_name='vector', index_type='IVFFlat', metric_type='L2', params={'nlist': 1024})",
                "search": "client.search(collection_name='demo_ivf_flat', data=[query], limit=10, search_params={'nprobe': 128})",
            }
        },
        "performance_tips": ["提高 nprobe 可提升召回但增加延迟", "测试后确认 collection total 与导入向量数一致", "必须 flush 后再复测，避免持久化不完整影响召回"],
        "metric_types": ["L2", "IP", "COSINE"],
    },
    "IVFLVQ": {
        "name": "IVFLVQ",
        "full_name": "Inverted File with LVQ",
        "description": "倒排索引结合 LVQ 量化，兼顾压缩和查询效率。",
        "implementation": "IVF 负责缩小候选桶，LVQ 对桶内向量做局部自适应量化。召回率同时受 nprobe、nlocal、nbits 和持久化完整性影响。",
        "use_case": ["大规模压缩检索", "内存受限场景"],
        "advantages": ["压缩率高", "适合批量检索"],
        "limitations": ["参数调优复杂", "存在量化误差", "当前平台可能尚未完全支持该索引"],
        "parameters": {
            "nlist": "聚类中心数，示例默认 1024",
            "nlocal": "局部量化参数，示例默认 16",
            "nbits": "量化位数，示例默认 8",
            "nprobe": "查询探测聚类数，召回优先时可提高",
        },
        "example_code": {
            "Python": {
                "create": "index_params.add_index(field_name='vector', index_type='IVFLVQ', metric_type='L2', params={'nlist': 1024, 'nlocal': 16, 'nbits': 8})",
                "search": "client.search(collection_name='demo_ivf_lvq', data=[query], limit=10, search_params={'nprobe': 128})",
            }
        },
        "performance_tips": ["提高 nprobe 可提升召回但增加延迟", "提高 nlocal 和 nbits 会影响压缩率与精度的平衡", "复测前确认索引已写入挂载的数据目录"],
        "metric_types": ["L2"],
    },
    "IVFPQ": {
        "name": "IVFPQ",
        "full_name": "Inverted File with Product Quantization",
        "description": "倒排索引结合乘积量化，降低内存占用。",
        "implementation": "IVF 先通过倒排桶缩小候选范围，PQ 使用 m_pq 个子量化器对向量编码。召回率主要受 nprobe、m_pq、nbits 和索引持久化完整性影响。",
        "use_case": ["超大规模向量检索", "内存敏感场景"],
        "advantages": ["内存占用低", "查询速度快"],
        "limitations": ["量化会损失精度", "需要训练", "当前平台可能尚未完全支持该索引"],
        "parameters": {
            "nlist": "聚类中心数，示例默认 1024",
            "m_pq": "子量化器数量，示例默认 8",
            "nbits": "编码位数，示例默认 8",
            "nprobe": "查询探测聚类数，召回优先时可提高",
        },
        "example_code": {
            "Python": {
                "create": "index_params.add_index(field_name='vector', index_type='IVFPQ', metric_type='L2', params={'nlist': 1024, 'm_pq': 8, 'nbits': 8})",
                "search": "client.search(collection_name='demo_ivf_pq', data=[query], limit=10, search_params={'nprobe': 128})",
            }
        },
        "performance_tips": ["提高 nprobe 可提升召回但增加延迟", "提高 m_pq 会降低单码压缩比并改善重构精度", "若召回异常低，先确认 flush 和重启后的 total 是否正确"],
        "metric_types": ["L2"],
    },
    "HNSWFlat": {
        "name": "HNSWFlat",
        "full_name": "Hierarchical Navigable Small World with Flat Vectors",
        "description": "基于多层小世界图的近似最近邻索引，适合高召回、低延迟向量检索。",
        "implementation": "构建阶段使用 m_hnsw/M 和 ef_construction 控制图质量；查询阶段用 ef_search 控制候选宽度。Flat 向量保留原始精度，召回通常更稳。",
        "use_case": ["百万级以上向量检索", "低延迟在线搜索", "高召回召回阶段"],
        "advantages": ["查询速度快", "召回率高", "无需训练"],
        "limitations": ["索引内存占用较高", "构建耗时随 M 和 ef_construction 增加"],
        "parameters": {
            "m_hnsw/M": "图连接数，示例默认 32",
            "ef_construction": "构建搜索宽度，示例默认 200",
            "ef_search": "查询搜索宽度，召回优先时建议提高到 128 或更高",
        },
        "example_code": {
            "Python": {
                "create": "index_params.add_index(field_name='vector', index_type='HNSWFlat', metric_type='L2', params={'M': 32, 'ef_construction': 200})",
                "search": "client.search(collection_name='wiki_hnswflat_10k', data=[query], limit=10, search_params={'ef_search': 128})",
            }
        },
        "performance_tips": ["提高 ef_search 可提升召回但增加延迟", "提高 M 可提升图质量但增加内存", "重启容器后先确认 collection 已从磁盘加载"],
        "metric_types": ["L2", "IP", "COSINE"],
    },
    "HNSWLVQ": {
        "name": "HNSWLVQ",
        "full_name": "Hierarchical Navigable Small World with LVQ",
        "description": "基于多层小世界图的近似最近邻索引，结合 LVQ 压缩以降低内存占用，适合高召回、较低内存场景。",
        "implementation": "HNSW 图负责候选搜索，LVQ 压缩负责降低向量存储和内存带宽。召回率优先时应提高 ef_search，并确认 flush 后索引文件已持久化。",
        "use_case": ["大规模向量近似检索", "内存受限场景", "高召回检索"],
        "advantages": ["查询速度快", "召回率高", "索引占用低于纯浮点 HNSW"],
        "limitations": ["仅支持 L2", "存在量化误差", "构建耗时随 m_hnsw 增加"],
        "parameters": {
            "nlocal": "局部量化参数，示例默认 16，高召回可提高",
            "nbits": "量化位数，示例默认 8，高召回可提高",
            "m_hnsw/M": "图连接数，示例默认 32",
            "ef_search": "查询搜索宽度，召回优先时建议提高",
        },
        "example_code": {
            "Python": {
                "create": "index_params.add_index(field_name='vector', index_type='HNSWLVQ', metric_type='L2', params={'nlocal': 16, 'nbits': 8, 'm_hnsw': 32})",
                "search": "client.search(collection_name='wiki_hnsw_lvq', data=[query], limit=10, search_params={'ef_search': 128})",
            }
        },
        "performance_tips": ["提高 ef_search 可提升召回但增加延迟", "提高 m_hnsw 可提升图质量但增加内存", "若召回低，先排查 total 是否完整、索引是否从持久化目录加载"],
        "metric_types": ["L2"],
    },
    "HNSWPQ": {
        "name": "HNSWPQ",
        "full_name": "Hierarchical Navigable Small World with Product Quantization",
        "description": "基于多层小世界图的近似最近邻索引，结合 PQ 压缩以降低内存占用，适合超大规模向量检索。",
        "implementation": "HNSW 图负责近邻候选探索，PQ 负责向量压缩。召回优先时提高 ef_search，并关注 m_pq/nbits 是否适合当前维度。",
        "use_case": ["超大规模向量检索", "内存敏感场景", "高召回检索"],
        "advantages": ["内存占用低", "查询速度快", "索引规模可扩展"],
        "limitations": ["仅支持 L2", "量化会损失精度", "要求维度可被 m_pq 整除", "当前平台可能尚未完全支持该索引"],
        "parameters": {
            "m_pq": "子量化器数量，示例默认 8",
            "nbits": "编码位数，示例默认 8",
            "m_hnsw/M": "图连接数，示例默认 32",
            "ef_search": "查询搜索宽度，召回优先时建议提高",
        },
        "example_code": {
            "Python": {
                "create": "index_params.add_index(field_name='vector', index_type='HNSWPQ', metric_type='L2', params={'m_pq': 8, 'nbits': 8, 'm_hnsw': 32})",
                "search": "client.search(collection_name='wiki_hnsw_pq', data=[query], limit=10, search_params={'ef_search': 128})",
            }
        },
        "performance_tips": ["提高 ef_search 可提升召回但增加延迟", "提高 m_hnsw 可提升图质量但增加内存", "召回异常低时先确认索引是否成功构建并持久化"],
        "metric_types": ["L2"],
    },
}

EXAMPLE_ORDER = ["IVFFlat", "IVFLVQ", "IVFPQ", "HNSWFlat", "HNSWLVQ", "HNSWPQ"]


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
    print_block("支持距离", data.get("metric_types"))


def main():
    client = HypervecClient(server)
    health = client.health()
    if health.get("status") != "ok":
        raise SystemExit(f"服务状态异常: {health}")

    try:
        collections = client.list_collections()
    except Exception:
        collections = []

    while True:
        print("\n" + "=" * 80)
        print("HyperVector gRPC Examples 交互式展示")
        print("=" * 80)
        print(f"gRPC 服务地址: {server}")
        print(f"服务状态: {health.get('status')}")
        print("当前 collections: " + (", ".join(collections) if collections else "无或未加载"))
        print("\n请选择要展示的索引类型：")
        for i, name in enumerate(EXAMPLE_ORDER, 1):
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
        if not choice.isdigit() or not (1 <= int(choice) <= len(EXAMPLE_ORDER)):
            print("输入无效，请重新选择。")
            continue

        show_detail(EXAMPLE_ORDER[int(choice) - 1])
        try:
            input("\n按回车返回菜单...")
        except EOFError:
            print()
            return


if __name__ == "__main__":
    main()
PY

PYTHON_BIN="${PYTHON_BIN:-/app/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" "$TMP_SCRIPT" "$SERVER"
