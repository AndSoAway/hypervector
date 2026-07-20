#!/bin/bash
# HyperVector examples 接口交互式展示脚本

set -euo pipefail

SERVER="${HYPERVECTOR_SERVER:-http://localhost:8080}"
TMP_SCRIPT="$(mktemp)"
trap 'rm -f "$TMP_SCRIPT"' EXIT

cat > "$TMP_SCRIPT" <<'PY'
import json
import sys
import urllib.error
import urllib.request

server = sys.argv[1].rstrip("/")


def fetch(path):
    url = server + path
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"无法访问 {url}: {exc}")


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
    print("代码示例")
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
    data = fetch(f"/examples/{index_name}")
    print("\n" + "#" * 80)
    print(f"{data.get('name', index_name)} - {data.get('full_name', '')}")
    print("#" * 80)
    print_block("简介", data.get("description"))
    print_block("适用场景", data.get("use_case"))
    print_block("优势", data.get("advantages"))
    print_block("限制", data.get("limitations"))
    print_block("参数说明", data.get("parameters"))
    print_code_examples(data.get("example_code"))
    print_block("调优建议", data.get("performance_tips"))
    print_block("真实场景示例", data.get("real_world_examples"))


def main():
    health = fetch("/health")
    if health.get("status") != "ok":
        raise SystemExit(f"服务状态异常: {health}")

    examples = fetch("/examples")
    indexes = examples.get("supported_indexes", [])
    if not indexes:
        raise SystemExit("/examples 未返回 supported_indexes")

    while True:
        print("\n" + "=" * 80)
        print("HyperVector Examples 交互式展示")
        print("=" * 80)
        print(f"服务地址: {server}")
        print("请选择要展示的索引类型：")
        for i, name in enumerate(indexes, 1):
            print(f"  {i}. {name}")
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

python3 "$TMP_SCRIPT" "$SERVER"
