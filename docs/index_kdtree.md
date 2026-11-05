# IndexKDTree：精确 L2 KD-tree 检索

本文档说明 `IndexKDTree`——一个基于 KD-tree 的**精确** k 近邻索引,支持 L2 距离,
与 `IndexFlat` / `IndexHNSW` / `IndexIVF` 等并列接入 HyperVec 的构建、检索、持久化
和 Python 绑定。

## 背景

库内已有的检索索引分两类:

- **精确**:`IndexFlatL2` —— 暴力穷举,每条 query 跟全部库向量算距离,结果无漏,
  是所有近似索引的精度基线,但没有任何索引结构。
- **近似**:`IndexHNSW` / `IndexIVF` / PQ / LVQ —— 用图或倒排/量化加速,牺牲少量
  召回换取速度。

`IndexKDTree` 补上了"**带空间划分结构的精确检索**"这一档:它用轴对齐超平面递归划分
空间,检索时回溯并剪枝,**保证返回真正的 k 近邻**(与 `IndexFlatL2` 结果一致),在
**低维**数据上通过剪枝跳过大量距离计算。

> 适用范围:KD-tree 的剪枝只在**低维**(经验上 d ≲ 20)有效。维度越高,回溯越难剪掉
> 分支,检索会退化到接近暴力扫描——详见「高维退化」一节。

## 算法

- **存储**:像 `IndexFlat` 一样保存全部原始向量(`xb`,行主序 `n_total * d`),外加
  一棵树。树节点 `KDNode` 存于一个扁平数组 `nodes`,叶子覆盖 `point_ids` 的一段连续
  区间——扁平结构便于整块序列化,无需处理指针。
- **构建**(`BuildTree`):递归划分。每个区间:
  1. 扫全维,选**方差(极差)最大的轴**作为切分轴;
  2. 用 `std::nth_element` 按该轴**中位数**把点划成左右两半(左 `<=` 中位数,右 `>`);
  3. 点数 `<= leaf_size` 时停,建叶子。
  得到一棵近似平衡的二叉树,深度约 `log2(n / leaf_size)`。
- **检索**(`Search`):每条 query 用一个容量 k 的**最大堆**保留当前最近的 k 个,
  显式栈做回溯:
  1. 沿切分面下行到 query 所在叶,叶内逐点算 L2 距离更新堆;
  2. 回溯时,若另一侧子空间到 query 的**下界距离**(累积的超平面距离平方)已经
     大于堆顶(即第 k 近的当前距离),就整枝剪掉,否则进入;
  3. 结束后把堆重排成升序输出,不足 k 个时尾部填 `label=-1` / `distance=FLT_MAX`。
  查询间用 OpenMP 并行(`#pragma omp parallel for`)。

**复杂度**(n 个点、维度 d、叶大小 L):

| 阶段 | 复杂度 | 说明 |
|---|---|---|
| 构建 | O(n · d · log(n/L)) | 每层对区间扫全维选轴 + `nth_element` 划分 |
| 单次查询(低维) | ≈ O(log n · d) | 剪枝有效时只访问少量叶子 |
| 单次查询(高维) | → O(n · d) | 剪枝失效,接近暴力 |

**距离语义**:返回的是 **L2 距离的平方**(与 `IndexFlatL2` 一致),省掉开方;需要真实
距离自行开方。

## C++ 接口

头文件:`index/kdtree/index_kdtree.h`。

```cpp
#include <index/kdtree/index_kdtree.h>

// 构造:d=维度;metric 只支持 kMetricL2;leaf_size 默认 16
hypervec::IndexKDTree kd(d);                           // 用默认 leaf_size
hypervec::IndexKDTree kd(d, hypervec::kMetricL2, 64);  // 显式 leaf_size

// 构建:Add 累积向量,之后必须显式 BuildTree()
kd.Add(n, xb);     // xb 为 n*d 的 row-major float 数组;可多次 Add
kd.BuildTree();    // 不调直接 Search 会抛异常

// 检索:调用方预分配输出,结果 row-major
std::vector<float>           dis(nq * k);   // 距离平方
std::vector<hypervec::idx_t> lab(nq * k);   // 标签(idx_t = int64_t)
kd.Search(nq, xq, k, dis.data(), lab.data());
// query i 的第 j 近邻:lab[i*k + j],距离² 为 dis[i*k + j]
```

其它方法:`Reset()` 清空索引;`Reconstruct(key, recons)` 取回第 `key` 条原始向量。

**约定与限制**:

- 构建是**显式的**——`Add` 之后、`Search` 之前必须调一次 `BuildTree()`。再次 `Add`
  会把 `tree_built` 置回 false,需重新 `BuildTree()`。这与库内 `IndexFlat1D` 要求先调
  `UpdatePermutation()` 的约定一致。
- 只支持 `kMetricL2`;其它 metric 在构造时抛异常。
- 不支持 `SearchParameters::sel`(IDSelector 过滤),传入会抛异常。

## 持久化

走库的通用入口(fourcc 标签 `"IKDt"` 已在 `persistence/index_read.cpp` 注册):

```cpp
hypervec::WriteIndex(&kd, "idx.hypervec");   // 须在 BuildTree 之后;写入 xb + 树

std::unique_ptr<hypervec::Index> p(hypervec::ReadIndex("idx.hypervec"));
auto* kd2 = dynamic_cast<hypervec::IndexKDTree*>(p.get());
// 读回即可直接 Search——树随索引一起序列化,无需重建
```

## Python 接口

SWIG 已暴露 `IndexKDTree`;`.add()` / `.search()` 由基类 `Index` 统一提供。

```python
import hypervec
import numpy as np

idx = hypervec.IndexKDTree(d)          # 或 IndexKDTree(d, hypervec.kMetricL2, 64)
idx.add(xb)                            # xb: (n, d) C-contiguous float32 ndarray
idx.BuildTree()                        # Add 之后必须调
D, I = idx.search(xq, k)               # xq: (nq, d) float32;返回 (D, I),各 (nq, k)

hypervec.write_index(idx, "idx.hypervec")
idx2 = hypervec.read_index("idx.hypervec")
```

> 说明:构建 Python 扩展需要 CMake ≥ 3.17。SWIG 绑定本身已验证可正确生成
> `IndexKDTree` 类(含构造函数、`leaf_size` 属性、`BuildTree()`)。

## 高维退化

KD-tree 的轴对齐超平面剪枝依赖"另一侧子空间离 query 足够远"这一判断。维度升高后,
几乎每个超平面都离 query 很近,回溯时两侧都得进,树结构近乎失效,检索退化到接近
**遍历全部叶子**——此时它的每 query 代价与 `IndexFlatL2` 暴力扫描同阶。

**结果始终精确**(这是 exact 索引,不是近似),退化的只有速度。因此:

- 低维(几维到十几维,如地理坐标、低维 embedding):KD-tree 剪枝有效,比暴力快。
- 高维(几十维以上,如 128/768 维语义向量):不要期待 KD-tree 比暴力快;要高维
  加速应选近似索引 `IndexHNSW` / `IndexIVF`。

### 关于基准数字的一个提醒

在 Wiki 1M(768 维)上实测,`IndexKDTree` 的每 query 耗时反而**低于** `IndexFlatL2`
(约 35 ms vs 466 ms),recall@10 = 1.0。这**不代表 KD-tree 在 768 维剪枝有效**——
真实原因是**并行粒度不同**:`IndexKDTree::Search` 按 query 用 OpenMP 并行,少量 query
就能吃满多核;而 `IndexFlatL2` 这条路径主要在单条 query 内部并行,query 少时并行度低。
要公平比较剪枝能力,应单线程运行,或让 query 数远多于线程数。换句话说,768 维下
KD-tree "每 query 仍要算接近全量距离"的本质没变,只是被多线程摊薄了。

## 文件一览

| 文件 | 内容 |
|---|---|
| `src/include/index/kdtree/index_kdtree.h` | `IndexKDTree` / `KDNode` 声明 |
| `src/index/kdtree/index_kdtree.cpp` | 构建、回溯检索、序列化实现 |
| `src/persistence/index_read.cpp` | `IKDt` 反序列化分支 |
| `src/python/swighypervec.swig` | Python 绑定 |
| `test/unit_tests/cpp/test_index_kdtree.cpp` | 单元测试(与 `IndexFlatL2` 精确对拍) |
