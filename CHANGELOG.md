# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

- Made build and test dependency resolution offline by default. Missing test
  dependencies require an explicit `HYPERVEC_FETCH_DEPS=ON` opt-in.
- Limited `HYPERVEC_OPT_LEVEL` to the implemented portable `generic` mode.
  Legacy SIMD/DD values and the unimplemented `HYPERVEC_ENABLE_MKL` and
  `HYPERVEC_ENABLE_C_API` options now fail during configuration instead of
  being silently accepted.
