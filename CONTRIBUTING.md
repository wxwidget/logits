# Contributing

Start with [development setup](docs/development.md). Use the existing checkout in a cloud task; create a Git worktree only when explicitly requested. Keep C++ style consistent with `.clang-format` and keep optimization math in the C++ core rather than duplicating it in Python.

Before proposing a change:

1. Describe the concrete behavior or numerical contract being changed.
2. Add a meaningful check that fails for the defect or unsupported behavior; use independent references for gradient/solution changes.
3. Run the affected Python/C++ tests. Run sanitizer checks for memory/storage changes and wheel/source-package checks for binding or build changes.
4. Update API documentation and runnable examples when behavior changes.

For performance work record dataset dimensions/nonzeros, objective normalization, compiler flags, seed, stopping condition and machine context. Compare quality and time to a matched optimization target across multiple runs. Keep failed, unconverged and successful results distinct. Refer to [the benchmark protocol](docs/benchmarking.md).

New dependencies should have a documented purpose and fixed C++ source revision. Preserve TLS/signature verification. Do not commit generated binaries, environments, credentials or machine-specific build caches. The project-wide license remains undeclared; do not infer a license grant from third-party notices.
