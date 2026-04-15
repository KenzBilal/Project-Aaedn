---
name: aaedn-enforcer
description: Strict guardian and single source of truth for Project Aaedn — enforces all specifications, invariants, and documentation-only status until green light
license: MIT
compatibility: opencode
---

# aaedn-enforcer

This skill is the strict guardian and single source of truth for Project Aaedn. It enforces all project specifications, invariants, and ensures 100% consistency with the five master documents:

- PRD.md (Product Requirements)
- MPD.md (Master Project Document)  
- SDD.md (System Design)
- FBD.md (Feature Breakdown)
- UIUX.md (Interface)

## Project Status

- **Status:** Documentation-only until "Green light – begin Phase 1"
- **Version:** 1.8.0 (as of latest document update)
- **Zero C++ code allowed** until user explicitly triggers green light
- Python tools/ scripts (converter) are the only code allowed before implementation

## Hardware Covenant (MPD §1)

| Property | Value |
|---|---|
| CPU | AMD Ryzen 5 5500U, Zen 2 |
| Cores/Threads | 6/12 |
| L1d cache | 32 KB/core |
| L2 cache | 512 KB/core (3 MB total) |
| L3 cache | 8 MB shared |
| RAM | 8 GB DDR4 |
| Usable RAM | 5.5–6.0 GB |
| Instruction Set | AVX2 + FMA3 |

## Model Architecture (MPD §1.5)

| Property | Value |
|---|---|
| Architecture | LLaMA-style decoder-only |
| Layers | 24 |
| Hidden dimension | 1024 |
| Heads | 16 |
| Head dimension | 64 |
| KV heads | 16 (multi-query) |
| FFN intermediate | 4096 (SwiGLU) |
| Vocab size | 32768 |
| Context length | 16384 |
| Parameters | 520 M (FP16 baseline) |
| Quantization | 4-bit with FP16 block scales |

## Non-Negotiable Invariants (MPD §2)

1. **Zero `new` / `malloc` after startup** in any hot path
2. **Total peak RAM ≤ 6 GB** under any workload
3. **Every AVX2 load 32-byte aligned** by construction
4. **No external libraries** except: `<cstddef>`, `<atomic>`, `<thread>`, `<immintrin.h>`, `<cstdint>`, `<cstdio>`, `<vector>`, `<string>` (tokenizer only)
5. **All benchmarks run on exact Ryzen 5 5500U** hardware, not emulated
6. **Elastic thread bandwidth ≤ 9 GB/s** when Decode Guardian active
7. **Residual path always FP16** (3-bit gated by entropy oracle H < 3.5, prohibited on final 4 layers)
8. **Documentation-only project** — no C++ code until green light

## Memory Budget (MPD §3)

| Component | Size | Notes |
|---|---|---|
| Model weights | 281 MB | 520 M params, 4-bit + scales |
| KV cache | 1.61 GB | 16384 tokens × 24 layers × 16 heads × 64 |
| RAG index | 224 MB | 1 M tokens |
| Tokenizer | 2 MB | Binary blob |
| RoPE tables | 32 MB | FP16 |
| Activations + scratch | 512 MB | Reused |
| Agent state + branches | 64 MB | Fixed |
| Arena overhead | 128 MB | Alignment |
| **Total** | **~2.82 GB** | Leaves ~2.68 GB headroom under 6 GB |

## Thread Model

- **Decode Guardians:** 4 threads, pinned to CPUs 0, 2, 4, 6
- **Elastic Compute:** 8 threads, pinned to CPUs 1, 3, 5, 7, 8, 9, 10, 11
- **SMT Ternary State Machine:** IDLE, DECODE_ACTIVE, ELASTIC_PARKED
- **Bandwidth cap:** ≤ 9 GB/s on Elastic during decode

## Core Algorithms (MPD §4)

All algorithms must be implemented exactly as specified:

- **Arena Allocation:** O(1) bump pointer
- **Blocked MatMul:** 64×64 tiles, AVX2 fmadd
- **Softmax:** Numerically stable, single-pass
- **RMSNorm:** Fused compute + normalize
- **RoPE:** AVX2 complex multiply
- **4-bit Dequant:** AVX2 kernel
- **SwiGLU FFN:** gate × SiLU(up) @ down
- **Entropy Oracle:** 256-bin histogram, Living Quantization
- **Branch-Aware KV Cache:** 7-step atomic compaction
- **Temporal Fractal Index:** Coarse + fine AVX2 scan

## Enforcement Rules

1. **Never output C++ code** until user says "Green light – begin Phase 1"
2. **Always stay 100% consistent** with five master documents
3. **Update version number** on every document change
4. **Add Revision History entry** on every update
5. **Output only changed file(s)** when modifying documents
6. **Use direct, professional language** — bullet points and tables only
7. **No poetic language, no filler words, no unverified claims**
8. **Never invent or assume** unstated behavior

## Performance Targets (Aspirational)

- Decode throughput: 25–40 tokens/second (validated on Ryzen 5 5500U)
- RAG retrieval: < 1 ms per query (validated on Ryzen 5 5500U)
- Actual numbers documented after kernel optimization

## File Locations

- Master docs: `/home/kenz/Projects/Project-Aaedn/docs/`
- This skill: `/home/kenz/Projects/Project-Aaedn/.opencode/sk/aaedn-enforcer/SKILL.md`

## After Green Light

When user triggers "Green light – begin Phase 1":

- **Act as professional systems engineer**
- Write clean, cache-aware, zero-allocation C++
- AVX2-optimized kernels
- Unique design, not derivative of llama.cpp/ggml
- Arena-first memory architecture
- Exact thread pinning as specified
- Enforce all invariants throughout implementation