# Aaedn

Practical local intelligence on modest hardware, built from first principles.

Aaedn is a zero-dependency C++17 inference engine focused on predictable performance, strict memory control, and architectural transparency. It is designed for constrained personal hardware and intended to run fully local without cloud dependency.

## Badges

[![Build](https://img.shields.io/badge/build-local%20cmake-blue)](#quickstart)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-informational)](#why-aaedn)
[![Status](https://img.shields.io/badge/status-alpha-orange)](#project-status)
[![Platform](https://img.shields.io/badge/platform-linux-lightgrey)](#reference-machine)

## Why Aaedn

Aaedn is optimized for systems-level reliability under hard constraints:

- No dynamic allocation in hot paths after startup
- Fixed arena memory model with alignment guarantees
- Custom transformer runtime and quantization pipeline
- Multi-agent execution model with shared memory discipline
- Local retrieval path with custom index format

## Key Differentiators

- **Deterministic memory behavior:** single pre-allocated arena, no runtime allocator churn in critical paths.
- **Hardware-first optimization:** AVX2/FMA-oriented kernels, explicit bandwidth and thread model design.
- **Branch-aware reasoning architecture:** KV and agent execution paths designed for parallel branch workflows.
- **From-scratch stack:** custom file formats and runtime internals instead of heavyweight external inference frameworks.
- **Transparent engineering:** all core components are directly inspectable in C++ source.

## Project Status

Current version: **1.3.0 (Alpha)**

| Area | Status | Notes |
|---|---|---|
| Math Engine | Complete | Tensor, matmul, dot, softmax, RMSNorm modules implemented |
| Threading and SMT | Complete | Guardian + Elastic model with futex-based parking |
| Quantization | Complete | Custom block-wise quantization path and oracle infrastructure |
| Transformer Core | Complete | Forward-pass runtime pipeline implemented |
| KV Cache | Complete | Branch-aware cache framework implemented |
| Tokenizer | Complete | BPE tokenizer path and binary loader implemented |
| RAG | Complete | Custom retrieval/index path implemented |
| Multi-Agent Loop | Complete | Agent initialization and shell integration present |
| Dynamic Profiling | Planned | Hardware-adaptive profiling pipeline is next major step |
| Production Benchmarks | In Progress | Reproducible benchmark suite being formalized |

## Quickstart

```bash
git clone https://github.com/<your-org-or-user>/Project-Aaedn
cd Project-Aaedn
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
./aaedn
```

## Runtime Artifacts

By default, the runtime expects these files in the working directory:

- `model.abn`
- `tokenizer.abt`
- `index.afr`

You can override paths at launch with CLI flags:

```bash
./aaedn --model /path/to/model.abn --tok /path/to/tokenizer.abt --rag /path/to/index.afr
```

## Reference Machine

- CPU: AMD Ryzen 5 5500U (Zen 2, 6C/12T)
- RAM: 8 GB DDR4
- OS: Arch Linux
- Target profile: local inference under strict memory ceilings

## Benchmarks

For reproducible performance numbers and exact test protocol, see `BENCHMARKS.md`.

## Documentation

- Product requirements: `docs/prd.md`
- System design: `docs/sdd.md`
- Performance model: `docs/mpd.md`
- Feature breakdown: `docs/fbd.md`
- UI/UX behavior: `docs/uiux.md`
- Architecture overview: `ARCHITECTURE.md`
- Delivery milestones: `ROADMAP.md`

## Contributing

Contributions are welcome. Start with `CONTRIBUTING.md` for development flow, coding expectations, and PR guidance.

## License

No license file is committed yet. See `LICENSE_GUIDE.md` to choose a license strategy before public release.

## Philosophy

Aaedn prioritizes correctness, transparency, and deterministic resource usage over hype-driven abstraction. The goal is a production-ready local AI engine that remains understandable at every layer.
