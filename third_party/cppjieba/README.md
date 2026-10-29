# cppjieba（vendored）

本目录是 [cppjieba](https://github.com/yanyiwu/cppjieba) 的仓库内副本，供 BM25
中文分词层（`hypervec::CppjiebaAnalyzer`）离线构建使用。构建时**不执行**上游
CMakeLists.txt，也不访问网络。

## 版本与溯源

| 组件 | 版本 / 提交 |
| --- | --- |
| cppjieba | `v5.4.0`，commit `76c242f46d3bf319906e2dea6cf6260d3a909335` |
| limonp（`deps/limonp` 子模块） | commit `ac32f1f287f65d5ce0ce295010c88026fae060ee` |

## 目录内容

```
include/cppjieba/*.hpp            16 个头文件（完整上游 include 目录）
deps/limonp/include/limonp/*.hpp  11 个头文件（cppjieba 的依赖）
dict/jieba.dict.utf8              主词典，约 5.1 MB
dict/hmm_model.utf8               HMM 模型，约 0.5 MB
dict/user.dict.utf8               用户词典占位（空词典）
LICENSE.cppjieba                  MIT
LICENSE.limonp                    MIT
```

上游 `dict/` 中的 `idf.utf8`（约 6 MB）与 `stop_words.utf8` 仅被
`KeywordExtractor` 使用，本模块只做分词，故不引入。

## 升级步骤

1. 将 `include/cppjieba/` 与 `deps/limonp/include/limonp/` 整体替换为新版本内容；
2. 同步 `dict/jieba.dict.utf8`、`dict/hmm_model.utf8`、`dict/user.dict.utf8`；
3. 更新本文件中的版本号与 commit SHA；
4. 重新构建 `HYPERVEC_ENABLE_BM25_ANALYZER=ON` 配置并运行 `hypervec_test`，
   重点确认 `CppjiebaAnalyzer.*` 用例全部通过。

升级时若上游改动了 `DictTrie` / `HMMModel` / `MixSegment` 的构造签名或线程安全
语义，需要同步检查 `src/utils/algo/bm25/analyzer_cn.cpp`。

## 许可证

cppjieba 与 limonp 均为 MIT 许可，见 `LICENSE.cppjieba` 与 `LICENSE.limonp`。
