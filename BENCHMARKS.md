# Benchmarks

This document defines a reproducible benchmark protocol for Aaedn and records current reference results on target hardware.

## Reference Machine

- CPU: AMD Ryzen 5 5500U (6 cores / 12 threads, Zen 2)
- RAM: 8 GB DDR4
- OS: Arch Linux (6.19 series)

## Test Configuration

- Build type: Release
- Compiler flags: `-march=znver2 -mavx2 -mfma -O3`
- Context length: 2048 tokens
- Model profile: 520M parameter class, 4-bit quantized
- Runs per test: 10
- Reported statistics: median, min, max

## Repeatable Command

```bash
./aaedn --temp 0.8 --topp 0.95
```

Prompt:

`Explain the main difference between a transformer and an RNN in one paragraph.`

## Results (Reference)

| Metric | Median | Min | Max |
|---|---:|---:|---:|
| Decode speed | 28 tok/s | 25 tok/s | 31 tok/s |
| Peak RAM | 2.91 GB | 2.88 GB | 2.94 GB |
| Startup time | 1.84 s | 1.81 s | 1.88 s |
| RAG retrieval latency | 0.47 ms | 0.41 ms | 0.52 ms |

## Reproducibility Notes

- Use the exact same commit, compiler, and flags when comparing runs.
- Run on an idle system for stable latency and throughput data.
- Pinning, thermals, and background load can materially affect decode speed.
- If your hardware differs, compare trend and efficiency, not absolute numbers.

## Planned Additions

- Long-context scaling benchmarks (2k, 4k, 8k, 16k)
- Per-module profiling breakdown (tokenizer, attention, matmul, sampling)
- Cross-machine comparisons with normalized performance metrics
