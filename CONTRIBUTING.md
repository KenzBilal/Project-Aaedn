# Contributing to Aaedn

Thanks for your interest in contributing.

This project is performance-sensitive and constraint-driven. Please prioritize correctness, determinism, and measurable impact over broad refactors.

## Development Setup

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

Run from the project root or `build/` as needed based on your local setup.

## Contribution Principles

- Keep hot paths allocation-free after initialization.
- Preserve memory alignment and arena discipline.
- Prefer explicit, readable systems code over abstraction-heavy patterns.
- Avoid introducing external runtime dependencies unless explicitly discussed.
- Include rationale for performance-sensitive changes.

## Pull Request Guidelines

Include the following in each PR:

1. **Problem statement:** what issue is being solved.
2. **Change summary:** concise technical description.
3. **Performance impact:** expected or measured impact (if relevant).
4. **Validation:** what you ran locally (build/tests/benchmarks).
5. **Risk notes:** edge cases or known limitations.

## Code Style Expectations

- C++17-compatible changes only.
- Keep interfaces minimal and explicit.
- Use descriptive names over clever shorthand.
- Add comments only where logic is non-obvious or constraint-driven.

## Benchmark-Aware Changes

If your change affects performance-critical paths (`matmul`, `attention`, `threading`, `quantization`):

- Provide before/after benchmark data when possible.
- Report hardware and build flags used.
- Avoid regressions in memory ceiling and decode stability.

## Issue Reporting

When filing an issue, include:

- OS and CPU details
- Build flags used
- Repro steps
- Expected vs actual behavior
- Relevant logs or error output
