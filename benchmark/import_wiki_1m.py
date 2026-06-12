#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入 Wiki 1M 数据到 HyperVector

步骤：
1. 读取 wiki 1M .fbin 文件
2. 创建 HNSW Collection
3. 批量插入向量数据
4. 构建索引
"""

import sys
import time
import argparse
from tqdm import tqdm

# 添加路径
sys.path.insert(0, '/root/vector/hypervector/pyhypervec')
sys.path.insert(0, '/root/vector/hypervector')

from pyhypervec import HypervecClient
from fbin_utils import read_fbin

def import_wiki_1m(
    fbin_path='/data/ljx/test_fbin/wiki_all_1M/base.1M.fbin',
    collection_name='wiki_hnsw_1m',
    batch_size=10000,
    M=32,
    ef_construction=200
):
    """
    导入 Wiki 1M 数据到 HyperVector

    Args:
        fbin_path: .fbin 文件路径
        collection_name: Collection 名称
        batch_size: 每批插入的向量数
        M: HNSW 参数
        ef_construction: HNSW 构建参数
    """
    print("=" * 80)
    print("HyperVector Wiki 1M 数据导入")
    print("=" * 80)
    print()

    # ========================================
    # 步骤 1：读取 fbin 文件
    # ========================================
    print("步骤 1/5：读取 Wiki 1M 数据")
    print("-" * 80)
    try:
        vectors, n, d = read_fbin(fbin_path)
        print(f"✅ 读取成功：{n:,} 条向量，维度 {d}")
    except FileNotFoundError:
        print(f"❌ 文件不存在：{fbin_path}")
        print("提示：请确认数据文件路径是否正确")
        return False
    except Exception as e:
        print(f"❌ 读取失败：{e}")
        return False

    print()

    # ========================================
    # 步骤 2：连接 HyperVector Server
    # ========================================
    print("步骤 2/5：连接 HyperVector Server")
    print("-" * 80)
    try:
        client = HypervecClient('http://localhost:8080')
        # 测试连接
        all_collections = client.list_collections()
        print(f"✅ 连接成功，当前已有 {len(all_collections)} 个 collections")
    except Exception as e:
        print(f"❌ 连接失败：{e}")
        print("提示：请确认 HyperVector Server 是否已启动")
        return False

    print()

    # ========================================
    # 步骤 3：创建 Collection
    # ========================================
    print("步骤 3/5：创建 HNSW Collection")
    print("-" * 80)
    print(f"Collection 名称：{collection_name}")
    print(f"索引类型：HNSW")
    print(f"参数：M={M}, ef_construction={ef_construction}")

    try:
        # 检查是否已存在
        existing = client.list_collections()
        if collection_name in existing:
            print(f"⚠️  Collection '{collection_name}' 已存在")
            response = input("是否删除并重新创建？(y/N): ")
            if response.lower() == 'y':
                print("删除旧 Collection...")
                client.drop_collection(collection_name)
                print("✅ 删除成功")
            else:
                print("❌ 取消导入")
                return False

        # 创建 Collection
        # 构建 schema
        schema = client.create_schema()
        schema.add_field(field_name='id', datatype='VARCHAR', is_primary=True, max_length=128)
        schema.add_field(field_name='vector', datatype='FLOAT_VECTOR', dim=d)

        # 构建索引参数
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name='vector',
            index_type='HNSW',
            metric_type='L2',
            params={'M': M, 'ef_construction': ef_construction}
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        print(f"✅ Collection 创建成功")
    except Exception as e:
        print(f"❌ 创建失败：{e}")
        return False

    print()

    # ========================================
    # 步骤 4：批量插入向量
    # ========================================
    print("步骤 4/5：批量插入向量")
    print("-" * 80)
    print(f"总向量数：{n:,}")
    print(f"批大小：{batch_size:,}")
    print(f"预计批数：{(n + batch_size - 1) // batch_size}")
    print()

    start_time = time.time()
    inserted = 0

    try:
        for i in tqdm(range(0, n, batch_size), desc="插入进度", unit="batch"):
            end = min(i + batch_size, n)
            batch_vectors = vectors[i:end]

            # 构造数据
            batch_data = [
                {
                    'id': f'wiki_{j}',
                    'vector': batch_vectors[j - i].tolist()
                }
                for j in range(i, end)
            ]

            # 插入
            client.insert(
                collection_name=collection_name,
                data=batch_data
            )

            inserted += len(batch_data)

        insert_time = time.time() - start_time
        print(f"\n✅ 插入完成：{inserted:,} 条向量")
        print(f"   - 耗时：{insert_time:.2f} 秒")
        print(f"   - 速度：{inserted / insert_time:.0f} 条/秒")

    except Exception as e:
        print(f"\n❌ 插入失败：{e}")
        return False

    print()

    # ========================================
    # 步骤 5：构建索引
    # ========================================
    print("步骤 5/5：构建 HNSW 索引")
    print("-" * 80)
    print("开始构建索引，这可能需要几分钟...")

    try:
        start_time = time.time()
        result = client.flush(collection_name=collection_name)
        build_time = time.time() - start_time

        print(f"✅ 索引构建完成")
        print(f"   - 向量数量：{result.get('total', result.get('n_total', 0)):,}")
        print(f"   - 构建耗时：{build_time:.1f} 秒 ({build_time / 60:.1f} 分钟)")

        if 'index_size_bytes' in result:
            size_mb = result['index_size_bytes'] / 1024 / 1024
            print(f"   - 索引大小：{size_mb:.2f} MB")

    except Exception as e:
        print(f"❌ 构建失败：{e}")
        return False

    print()
    print("=" * 80)
    print("✅ Wiki 1M 数据导入完成！")
    print("=" * 80)
    print()
    print(f"Collection 名称：{collection_name}")
    print(f"可以开始运行性能测试了")
    print()

    return True

def main():
    parser = argparse.ArgumentParser(description='导入 Wiki 1M 数据到 HyperVector')
    parser.add_argument(
        '--file',
        default='/data/ljx/test_fbin/wiki_all_1M/base.1M.fbin',
        help='Wiki 1M .fbin 文件路径'
    )
    parser.add_argument(
        '--collection',
        default='wiki_hnsw_1m',
        help='Collection 名称'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='批量插入大小'
    )
    parser.add_argument(
        '--M',
        type=int,
        default=32,
        help='HNSW M 参数'
    )
    parser.add_argument(
        '--ef-construction',
        type=int,
        default=200,
        help='HNSW ef_construction 参数'
    )

    args = parser.parse_args()

    success = import_wiki_1m(
        fbin_path=args.file,
        collection_name=args.collection,
        batch_size=args.batch_size,
        M=args.M,
        ef_construction=args.ef_construction
    )

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
