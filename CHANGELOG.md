# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

- Added a versioned, checksummed index-file envelope containing normalized
  algorithm configuration. New readers continue to accept legacy payloads,
  expose inferred legacy metadata, validate metadata against the payload, and
  restore HNSW search defaults that the legacy payload did not record. Files
  emitted by the new writer require a reader that understands format version 2.
- Exposed the C++ registry factory and normalized JSON configuration through
  the Python binding. The Python service now uses the same C++ creation and
  per-query search-parameter validation path while retaining legacy direct
  index constructors.
- Added reusable graph storage, search, validation, and neighbor-pruning
  components. Bounded HNSW level-0 traversal now uses the shared graph search
  path without changing its result, filtering, or statistics semantics.
- `IndexHNSW::Search1` now performs graph search instead of reporting that the
  operation is unsupported. HNSW graph search rejects non-positive
  `ef_search` values consistently with the registry-backed search API.
- Added a typed C++ index configuration, strict JSON parser, thread-safe index
  registry, and `Engine` factory. The factory creates all current Flat, PQ,
  LVQ, IVF, and HNSW CPU variants, applies algorithm-specific defaults, rejects
  misplaced parameters, and optionally wraps indexes for external IDs.
- Added registry-backed runtime search configuration for per-query IVF/HNSW
  overrides and deterministic index/search configuration serialization.
- Corrected installed public-header paths from `include/include/...` to
  `include/...`, including the new factory API.
- Added a common dense-dataset `Index::Build` entry point, stable index type
  names, and runtime capability declarations across the current index
  families. `IndexIDMap::Build` now accepts external IDs while keeping mapping
  publication after a successful underlying build.
- The new virtual `Index` methods change the C++ ABI. Downstream subclasses
  must implement `TypeName()` and `Capabilities()` and must be rebuilt against
  this version.
- Made build and test dependency resolution offline by default. Missing test
  dependencies require an explicit `HYPERVEC_FETCH_DEPS=ON` opt-in.
- Limited `HYPERVEC_OPT_LEVEL` to the implemented portable `generic` mode.
  Legacy SIMD/DD values and the unimplemented `HYPERVEC_ENABLE_MKL` and
  `HYPERVEC_ENABLE_C_API` options now fail during configuration instead of
  being silently accepted.
