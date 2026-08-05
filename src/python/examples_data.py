# -*- coding: utf-8 -*-
"""Canonical index example data for HTTP and engine example endpoints."""

from __future__ import annotations

from typing import Any


def _create(index_type: str, params: dict[str, Any] | None = None) -> str:
    return (
        "index_params.add_index("
        "field_name='vector', "
        f"index_type='{index_type}', "
        "metric_type='L2', "
        f"params={params or {}})"
    )


def _search(collection_name: str, params: dict[str, Any] | None = None) -> str:
    suffix = f", search_params={params}" if params else ""
    return (
        "client.search("
        f"collection_name='{collection_name}', "
        "data=[query], "
        "limit=10"
        f"{suffix})"
    )


INDEX_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "index_type": "IndexIVFFlat",
        "name": "IVFFlat",
        "aliases": ["IVF", "IndexIVFFlat"],
        "cpp_classes": ["IndexIVFFlat"],
        "cpp_class": "hypervec.IndexIVFFlat",
        "full_name": "Inverted File Flat Index",
        "description": "倒排聚类索引。IVF 是 IVFFlat 的兼容别名，二者使用同一个底层 IndexIVFFlat。",
        "use_case": ["大规模向量粗召回", "可接受近似结果的搜索"],
        "advantages": ["查询成本可控", "适合大规模数据", "支持 L2/IP/COSINE"],
        "limitations": ["需要训练", "召回受搜索参数 nprobe 影响"],
        "parameters": [
            {"name": "nlist", "type": "int", "default": 1024, "required": False, "description": "聚类中心数"},
            {"name": "nprobe", "type": "int", "default": 10, "required": False, "scope": "search", "description": "搜索时探测的聚类数"},
        ],
        "example_code": {
            "Python": {
                "create": _create("IVFFlat", {"nlist": 1024}),
                "search": _search("demo_ivf_flat", {"nprobe": 16}),
            }
        },
        "performance_tips": ["提高 nprobe 可提升召回但增加延迟", "提高 nlist 可提升粗聚类粒度但增加训练和索引开销"],
        "metric_types": ["L2", "IP", "COSINE"],
    },
    {
        "index_type": "IndexIVFLVQ",
        "name": "IVFLVQ",
        "aliases": ["IndexIVFLVQ"],
        "cpp_classes": ["IndexIVFLVQ"],
        "cpp_class": "hypervec.IndexIVFLVQ",
        "full_name": "Inverted File with LVQ",
        "description": "倒排索引结合 LVQ 量化，兼顾压缩和查询效率。",
        "use_case": ["大规模压缩检索", "内存受限场景"],
        "advantages": ["压缩率高", "适合批量检索"],
        "limitations": ["参数调优复杂", "存在量化误差", "召回受搜索参数 nprobe 影响"],
        "parameters": [
            {"name": "nlist", "type": "int", "default": 1024, "required": False, "description": "聚类中心数"},
            {"name": "nlocal", "type": "int", "default": 16, "required": False, "description": "局部量化参数"},
            {"name": "nbits", "type": "int", "default": 8, "required": False, "description": "量化位数"},
            {"name": "nprobe", "type": "int", "default": 10, "required": False, "scope": "search", "description": "搜索时探测的聚类数"},
        ],
        "example_code": {
            "Python": {
                "create": _create("IVFLVQ", {"nlist": 1024, "nlocal": 16, "nbits": 8}),
                "search": _search("demo_ivf_lvq", {"nprobe": 16}),
            }
        },
        "performance_tips": ["提高 nprobe 可提升召回但增加延迟", "提高 nlocal 和 nbits 会影响压缩率与精度的平衡"],
        "metric_types": ["L2"],
    },
    {
        "index_type": "IndexIVFPQ",
        "name": "IVFPQ",
        "aliases": ["IndexIVFPQ"],
        "cpp_classes": ["IndexIVFPQ"],
        "cpp_class": "hypervec.IndexIVFPQ",
        "full_name": "Inverted File with Product Quantization",
        "description": "倒排索引结合乘积量化，降低内存占用。",
        "use_case": ["超大规模向量检索", "内存敏感场景"],
        "advantages": ["内存占用低", "查询速度快"],
        "limitations": ["量化会损失精度", "需要训练", "向量维度必须能被 m_pq 整除"],
        "parameters": [
            {"name": "nlist", "type": "int", "default": 1024, "required": False, "description": "聚类中心数"},
            {"name": "m_pq", "type": "int", "default": 8, "required": False, "description": "子量化器数量"},
            {"name": "nbits", "type": "int", "default": 8, "required": False, "description": "编码位数"},
            {"name": "nprobe", "type": "int", "default": 10, "required": False, "scope": "search", "description": "搜索时探测的聚类数"},
        ],
        "example_code": {
            "Python": {
                "create": _create("IVFPQ", {"nlist": 1024, "m_pq": 8, "nbits": 8}),
                "search": _search("demo_ivf_pq", {"nprobe": 16}),
            }
        },
        "performance_tips": ["提高 nprobe 可提升召回但增加延迟", "提高 m_pq 会降低单码压缩比并改善重构精度"],
        "metric_types": ["L2"],
    },
    {
        "index_type": "IndexHNSWFlat",
        "name": "HNSWFlat",
        "aliases": ["HNSW", "AutoIndex", "IndexHNSWFlat"],
        "cpp_classes": ["IndexHNSWFlat"],
        "cpp_class": "hypervec.IndexHNSWFlat",
        "full_name": "Hierarchical Navigable Small World with Flat Vectors",
        "description": "基于多层小世界图的近似最近邻索引。HNSW、HNSWFlat 和 AutoIndex 是同一底层 IndexHNSWFlat 的别名。",
        "use_case": ["百万级以上向量检索", "低延迟在线搜索", "高召回召回阶段"],
        "advantages": ["查询速度快", "召回率高", "无需训练", "支持 L2/IP/COSINE"],
        "limitations": ["索引内存占用较高", "构建耗时随 m_hnsw 增加"],
        "parameters": [
            {"name": "m_hnsw", "type": "int", "default": 32, "required": False, "description": "图连接数"},
            {"name": "ef_search", "type": "int", "default": 100, "required": False, "scope": "search", "description": "查询搜索宽度"},
        ],
        "example_code": {
            "Python": {
                "create": _create("HNSWFlat", {"m_hnsw": 32}),
                "search": _search("demo_hnsw_flat", {"ef_search": 128}),
            }
        },
        "performance_tips": ["提高 ef_search 可提升召回但增加延迟", "提高 m_hnsw 可提升图质量但增加内存"],
        "metric_types": ["L2", "IP", "COSINE"],
    },
    {
        "index_type": "IndexHNSWLVQ",
        "name": "HNSWLVQ",
        "aliases": ["IndexHNSWLVQ"],
        "cpp_classes": ["IndexHNSWLVQ"],
        "cpp_class": "hypervec.IndexHNSWLVQ",
        "full_name": "Hierarchical Navigable Small World with LVQ",
        "description": "HNSW 图索引结合 LVQ 压缩，适合高召回、较低内存场景。",
        "use_case": ["大规模向量近似检索", "内存受限场景", "高召回检索"],
        "advantages": ["查询速度快", "召回率高", "索引占用低于纯浮点 HNSW"],
        "limitations": ["仅支持 L2", "存在量化误差", "构建耗时随 m_hnsw 增加"],
        "parameters": [
            {"name": "nlocal", "type": "int", "default": 16, "required": False, "description": "局部量化参数"},
            {"name": "nbits", "type": "int", "default": 8, "required": False, "description": "量化位数"},
            {"name": "m_hnsw", "type": "int", "default": 32, "required": False, "description": "图连接数"},
            {"name": "ef_search", "type": "int", "default": 100, "required": False, "scope": "search", "description": "查询搜索宽度"},
        ],
        "example_code": {
            "Python": {
                "create": _create("HNSWLVQ", {"nlocal": 16, "nbits": 8, "m_hnsw": 32}),
                "search": _search("demo_hnsw_lvq", {"ef_search": 128}),
            }
        },
        "performance_tips": ["提高 ef_search 可提升召回但增加延迟", "提高 m_hnsw 可提升图质量但增加内存"],
        "metric_types": ["L2"],
    },
    {
        "index_type": "IndexHNSWPQ",
        "name": "HNSWPQ",
        "aliases": ["IndexHNSWPQ"],
        "cpp_classes": ["IndexHNSWPQ"],
        "cpp_class": "hypervec.IndexHNSWPQ",
        "full_name": "Hierarchical Navigable Small World with Product Quantization",
        "description": "HNSW 图索引结合 PQ 压缩，适合超大规模向量检索。",
        "use_case": ["超大规模向量检索", "内存敏感场景", "高召回检索"],
        "advantages": ["内存占用低", "查询速度快", "索引规模可扩展"],
        "limitations": ["仅支持 L2", "量化会损失精度", "要求维度可被 m_pq 整除"],
        "parameters": [
            {"name": "m_pq", "type": "int", "default": 8, "required": False, "description": "子量化器数量"},
            {"name": "nbits", "type": "int", "default": 8, "required": False, "description": "编码位数"},
            {"name": "m_hnsw", "type": "int", "default": 32, "required": False, "description": "图连接数"},
            {"name": "ef_search", "type": "int", "default": 100, "required": False, "scope": "search", "description": "查询搜索宽度"},
        ],
        "example_code": {
            "Python": {
                "create": _create("HNSWPQ", {"m_pq": 8, "nbits": 8, "m_hnsw": 32}),
                "search": _search("demo_hnsw_pq", {"ef_search": 128}),
            }
        },
        "performance_tips": ["提高 ef_search 可提升召回但增加延迟", "提高 m_hnsw 可提升图质量但增加内存"],
        "metric_types": ["L2"],
    },
)


def is_example_supported(hypervec_module: Any, example: dict[str, Any]) -> bool:
    return any(hasattr(hypervec_module, class_name) for class_name in example["cpp_classes"])


def supported_index_examples(hypervec_module: Any) -> list[dict[str, Any]]:
    return [
        _public_example(example)
        for example in INDEX_EXAMPLES
        if is_example_supported(hypervec_module, example)
    ]


def find_index_example(hypervec_module: Any, index_type: str) -> dict[str, Any] | None:
    requested = str(index_type or "").casefold()
    for example in supported_index_examples(hypervec_module):
        names = [example["name"], example["index_type"], *example.get("aliases", [])]
        if any(str(name).casefold() == requested for name in names):
            return example
    return None


def _public_example(example: dict[str, Any]) -> dict[str, Any]:
    public = dict(example)
    public.pop("cpp_classes", None)
    return public
