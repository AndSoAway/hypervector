#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for HyperVector examples() interface
"""

from pyhypervec import HypervecClient

def main():
    print("=" * 60)
    print("HyperVector Examples Interface Demo")
    print("=" * 60)
    print()

    # Initialize client
    client = HypervecClient("http://localhost:8080")

    # 1. List all supported index types
    print("1️⃣  List all supported index types")
    print("-" * 60)
    all_examples = client.get_examples()
    print(f"Supported indexes: {', '.join(all_examples['supported_indexes'])}")
    print()

    # 2. Get HNSW examples
    print("2️⃣  Get HNSW Index Examples")
    print("-" * 60)
    hnsw = client.get_examples("HNSW")

    print(f"Name: {hnsw['name']}")
    print(f"Full Name: {hnsw['full_name']}")
    print(f"Description: {hnsw['description']}")
    print(f"Use Case: {hnsw['use_case']}")
    print()

    print("Parameters:")
    for param_name, param_info in hnsw['parameters'].items():
        print(f"  • {param_name}:")
        print(f"    - Description: {param_info['description']}")
        print(f"    - Range: {param_info['range']}")
        print(f"    - Default: {param_info['default']}")
        print(f"    - Recommendation: {param_info['recommendation']}")
    print()

    print("Python Code Example (Create Collection):")
    print("-" * 60)
    print(hnsw['example_code']['python']['create'])
    print()

    print("Python Code Example (Insert Data):")
    print("-" * 60)
    print(hnsw['example_code']['python']['insert'])
    print()

    print("Python Code Example (Search):")
    print("-" * 60)
    print(hnsw['example_code']['python']['search'])
    print()

    print("Performance Tips:")
    for i, tip in enumerate(hnsw['performance_tips'], 1):
        print(f"  {i}. {tip}")
    print()

    # 3. Get Flat examples
    print("3️⃣  Get Flat Index Examples")
    print("-" * 60)
    flat = client.get_examples("Flat")
    print(f"Name: {flat['name']}")
    print(f"Description: {flat['description']}")
    print(f"Use Case: {flat['use_case']}")
    print()

    # 4. Test error handling
    print("4️⃣  Test Error Handling")
    print("-" * 60)
    try:
        client.get_examples("InvalidIndex")
    except Exception as e:
        print(f"✅ Error handled correctly: {e}")
    print()

    print("=" * 60)
    print("✅ Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
