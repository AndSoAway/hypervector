#!/bin/bash
# HyperVector Examples 接口交互式演示脚本

clear
echo "============================================================"
echo "  HyperVector Examples 接口演示"
echo "============================================================"
echo ""

# 第 1 步：列出所有索引
echo "1️⃣  获取所有支持的索引类型"
echo "------------------------------------------------------------"
curl -s http://localhost:8080/examples | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('支持的索引类型：')
for i, idx in enumerate(data['supported_indexes'], 1):
    print(f'  {i}. {idx}')
print()
print(f'共 {len(data[\"supported_indexes\"])} 种索引')
"

echo ""
echo "============================================================"
echo "请选择要查看的索引类型："
echo "------------------------------------------------------------"
echo "  1. HNSW"
echo "  2. Flat"
echo "  3. IVF"
echo "  4. IVFPQ"
echo "  5. IVFLVQ"
echo "  6. PQ"
echo "  7. LVQ"
echo "  8. HNSWFlat"
echo "  0. 退出"
echo ""
read -p "请输入数字 (1-8): " choice

case $choice in
    1) INDEX="HNSW" ;;
    2) INDEX="Flat" ;;
    3) INDEX="IVF" ;;
    4) INDEX="IVFPQ" ;;
    5) INDEX="IVFLVQ" ;;
    6) INDEX="PQ" ;;
    7) INDEX="LVQ" ;;
    8) INDEX="HNSWFlat" ;;
    0) echo "退出演示"; exit 0 ;;
    *) echo "无效选择"; exit 1 ;;
esac

clear
echo "============================================================"
echo "  ${INDEX} 索引详细文档"
echo "============================================================"
echo ""

# 第 2 步：显示选中索引的基本信息
echo "2️⃣  基本信息"
echo "------------------------------------------------------------"
curl -s http://localhost:8080/examples/${INDEX} | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('【索引名称】', data['name'])
print('【完整名称】', data.get('full_name', 'N/A'))
print()
print('【功能描述】')
print(data['description'])
print()
print('【适用场景】')
print(data.get('use_case', 'N/A'))
"

echo ""
read -p "按 Enter 查看优势与限制..."

# 第 3 步：优势与限制
clear
echo "============================================================"
echo "  ${INDEX} 索引 - 优势与限制"
echo "============================================================"
echo ""
curl -s http://localhost:8080/examples/${INDEX} | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'advantages' in data and data['advantages']:
    print('【优势】')
    for adv in data['advantages']:
        print(f'  ✓ {adv}')
    print()
if 'limitations' in data and data['limitations']:
    print('【限制】')
    for lim in data['limitations']:
        print(f'  ✗ {lim}')
"

echo ""
read -p "按 Enter 查看参数详解..."

# 第 4 步：参数详解
clear
echo "============================================================"
echo "  ${INDEX} 索引 - 参数详解"
echo "============================================================"
echo ""
curl -s http://localhost:8080/examples/${INDEX} | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'parameters' in data and data['parameters']:
    for param_name, param_info in data['parameters'].items():
        print('┌' + '─' * 68 + '┐')
        print(f'│ 参数：{param_name:60s} │')
        print('└' + '─' * 68 + '┘')
        print(f'说明：{param_info.get(\"description\", \"N/A\")}')
        if 'range' in param_info:
            print(f'范围：{param_info[\"range\"]}')
        if 'default' in param_info:
            print(f'默认：{param_info[\"default\"]}')
        if 'impact' in param_info:
            print(f'影响：')
            print(param_info['impact'])
        if 'recommendation' in param_info:
            print(f'建议：')
            print(param_info['recommendation'])
        print()
else:
    print('该索引无需配置参数')
"

echo ""
read -p "按 Enter 查看代码示例..."

# 第 5 步：代码示例
clear
echo "============================================================"
echo "  ${INDEX} 索引 - Python 代码示例"
echo "============================================================"
echo ""
curl -s http://localhost:8080/examples/${INDEX} | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'example_code' in data and 'python' in data['example_code']:
    for step_name, code in data['example_code']['python'].items():
        print(f'【{step_name}】')
        print(code)
        print()
        print('-' * 70)
        print()
"

echo ""
read -p "按 Enter 查看 curl 示例..."

# 第 6 步：curl 示例
clear
echo "============================================================"
echo "  ${INDEX} 索引 - curl 命令示例"
echo "============================================================"
echo ""
curl -s http://localhost:8080/examples/${INDEX} | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'example_code' in data and 'curl' in data['example_code']:
    for step_name, code in data['example_code']['curl'].items():
        print(f'【{step_name}】')
        print(code)
        print()
        print('-' * 70)
        print()
"

echo ""
read -p "按 Enter 查看性能建议..."

# 第 7 步：性能建议
clear
echo "============================================================"
echo "  ${INDEX} 索引 - 性能调优建议"
echo "============================================================"
echo ""
curl -s http://localhost:8080/examples/${INDEX} | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'performance_tips' in data and data['performance_tips']:
    for i, tip in enumerate(data['performance_tips'], 1):
        print(f'{i}. {tip}')
        print()
"

echo ""
echo "============================================================"
echo "  ✅ ${INDEX} 索引演示完成！"
echo "============================================================"
echo ""
