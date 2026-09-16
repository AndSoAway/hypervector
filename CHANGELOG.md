# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

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
