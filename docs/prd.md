# PRD — Product Requirements Document
## Project Aaedn
**Version:** 2.3.0 | **Status:** Complete | **Last Updated:** 2026-04-15

---

## 1. Product Definition

**Name:** Aaedn
**Type:** Local AI inference engine + agentic system
**Language:** C++17, zero external libraries
**Target User:** Single developer running on constrained personal hardware

### 1.1 Problem Statement
No existing open-source inference engine delivers practical LLM intelligence on 8 GB RAM without cloud dependency, external libraries (llama.cpp, ggml, HuggingFace), or dynamic allocation in hot paths. All current solutions make one of three tradeoffs: quality, speed, or memory safety. Aaedn makes none of them.

### 1.2 Product Goal
A fully local, zero-dependency C++ AI inference engine that runs a 520 M parameter quantized Transformer with integrated RAG and multi-agent control on a Ryzen 5 5500U laptop, targeting 25–40 tokens/second decode throughput (aspirational design goal to be validated during implementation on Ryzen 5 5500U. Actual measured performance will be documented after kernel optimization.)

---

## 2. Hardware Target (Locked — Never Changes)

| Property | Value |
|---|---|
| CPU | AMD Ryzen 5 5500U, Zen 2, 6 cores / 12 threads |
| Instruction Set | AVX2 + FMA3 (native), scalar fallback |
| L1d Cache | 32 KB per core |
| L2 Cache | 512 KB per core (3 MB total) |
| L3 Cache | 8 MB shared |
| RAM | 8 GB DDR4, ~35–45 GB/s sustained bandwidth |
| OS | Arch Linux 6.19 |
| Usable RAM for Aaedn | 5.5–6 GB after OS |

---

## 3. Functional Requirements

### 3.1 Inference Engine
- Run a LLaMA-style decoder-only Transformer, 520 M parameters (FP16 baseline), quantized to 4-bit
- Architecture: 24 layers, d_model=1024, 16 heads, head_dim=64, KV heads=16
- FFN: SwiGLU, 4096 intermediate size
- RoPE positional encoding (base 10000), RMSNorm (ε=1e-5)
- Context length: up to 16 384 tokens
- Decode throughput target: 25–40 tokens/second (aspirational design goal to be validated during implementation)
- Forward pass fully fused where possible
- All computation stays inside fixed pre-allocated arena
- Model loaded from .abn binary with validated magic, format version, payload size, and checksum
- No Python required at runtime

### 3.x Documentation Status
- This is a documentation-only project at this stage
- No C++ implementation code is included in any document
- Documentation contains complete specifications, algorithms, file formats, and test plans
- Actual C++ source code begins only after explicit "Green light – begin Phase 1" command
- Python conversion scripts (tools/) are the only code allowed before green light

### 3.2 Quantization
- Block-wise 4-bit and 8-bit weight storage with per-block FP16 scales
- Dynamic per-layer precision selection via entropy oracle (Living Quantization)
- Residual path always FP16 — no quantization noise cascade across layers
- No weight re-layout at runtime — precision change is kernel-only

### 3.3 Memory Management
- Single fixed arena allocated at startup via `std::aligned_alloc(32, total)`
- Zero `new` / `malloc` after startup in any hot path
- Peak total RAM usage ≤ 6 GB (model + KV + RAG + agent state + OS)
- All KV, RAG, activations, branch metadata live inside the arena

### 3.4 Multi-Agent System
- 4–6 parallel micro-agents with divergent reasoning paths
- ReAct-style planning loop (Reason → Act → Observe)
- Tool registry for agent action dispatch
- Agent merge/prune without allocation or KV duplication

### 3.5 RAG System
- 1 million token long-term memory
- 384-dimensional embeddings (from first 4 model layers, mean-pooled), 4-bit quantized
- Target: < 1 ms retrieval latency (aspirational design goal to be validated on Ryzen 5 5500U)
- Embeddings generated via tools/rag.py (standard early-layer pooling technique)
- No HNSW, no graph structures — flat fractal index only

### 3.6 Tokenizer
- BPE tokenizer, vocab size 32 768
- Zero external libraries — pure C++ with `<vector>`, `<string>`, `<cstdint>`, `<cstdio>`
- Load time < 5 ms from binary blob
- Special tokens: BOS = 32765, EOS = 32766, PAD = 32767
- Tokenizer artifact (.abt) must include validated magic, format version, payload size, and checksum

### 3.7 Thread Model
- 4 Decode Guardian threads: pinned to logical CPUs 0, 2, 4, 6 (SMT slot 0, cores 0–3)
- 8 Elastic Compute threads: remaining SMT slots + cores 4–5
- Elastic threads capped at ≤ 9 GB/s bandwidth when decode is active
- Futex-based parking with ternary state machine — zero spurious contention

### 3.8 Deterministic Artifact Pipeline
- Conversion tools must produce deterministic outputs for identical inputs
- Binary artifacts must include header fields: magic, version, payload_size, checksum
- Runtime loaders must reject artifacts on header mismatch, version mismatch, payload size mismatch, or checksum mismatch
- Runtime artifact set is fixed: `model.abn`, `tokenizer.abt`, `index.afr`

### 3.9 Production Validation and Release Hardening
- Deterministic artifact round-trip validation is mandatory before release
- Startup/load failure-mode tests are mandatory for header mismatch and checksum mismatch
- Benchmark automation must enforce pass/fail gates for throughput and artifact size constraints
- CI must validate artifact integrity and reproducibility checks on every pull request

### 3.10 Runtime Profiles and Operational Reliability
- Runtime must support `--profile safe`, `--profile balanced`, `--profile max`
- Profile selection must deterministically configure arena size, quantization aggressiveness, thread counts, and bandwidth cap
- Structured logging with levels (`DEBUG`, `INFO`, `WARN`, `ERROR`) is mandatory
- Crash diagnostics must persist to log file on fatal signal
- Artifact loader must support rollback-safe version handling: current version + approved compatibility window

---

## 4. Non-Functional Requirements

| Requirement | Specification |
|---|---|
| RAM ceiling | ≤ 6 GB peak, hard limit |
| Decode throughput | Target: 25–40 tokens/second (aspirational, to be validated) |
| RAG retrieval latency | < 1 ms |
| Tokenizer load time | < 5 ms |
| Dynamic allocation in hot path | Zero |
| External libraries | Zero |
| Alignment of all AVX2 loads | 32-byte guaranteed by construction |
| Tokenizer encoding for 512-token prompt | < 50 ms acceptable |

---

## 5. Memory Budget (Verified)

| Component | Size | Notes |
|---|---|---|
| Model weights (520M params, 4-bit + scales) | 281 MB | |
| KV cache (16384 tokens, 24 layers, FP16) | 1.61 GB | 16×64×2×2×16384×24 |
| RAG index (1 M tokens, 4-bit) | 224 MB | |
| Tokenizer binary | 2 MB | |
| Arena overhead + activations + agent state | < 1 GB | |
| **Total** | **~2.82 GB** | **Leaves ~2.68 GB under 6 GB wall** |

---

## 6. Success Criteria

- Full forward pass produces correct logits on reference input
- Decode throughput ≥ 25 tokens/second measured on target hardware
- 4 parallel micro-agents diverge and merge without memory error or allocation
- RAG retrieval returns correct top-K in < 1 ms on 1 M token index
- Peak RAM never exceeds 6 GB under any workload
- All unit tests pass on Ryzen 5 5500U (not emulated)

---

## 7. Out of Scope

- GPU inference
- Cloud or network access of any kind
- Windows or macOS support
- Batch inference (single-user interactive only)
- Training or fine-tuning
- Web UI or GUI

---

## 8. Revision History

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-04-14 | Initial specification locked |
| 1.1.0 | 2026-04-14 | Corrected model weight size to ~281 MB. Corrected total budget to ~2.4 GB. |
| 1.2.0 | 2026-04-14 | Added model architecture specification: LLaMA-style 24-layer 520M params, d_model=1024, 16 heads, 64 head_dim, 4096 FFN. Updated weight size to ~347 MB. |
| 1.3.0 | 2026-04-14 | Added model conversion pipeline: custom block-wise 4-bit quantization, tokenizer from HF, RAG embedding from first 4 layers. |
| 1.4.0 | 2026-04-14 | Confirmed documentation-only status: no C++ code included until green light. Python tools/ scripts are the only code allowed before implementation. |
| 1.5.0 | 2026-04-14 | Revised decode throughput from 50-90 to 25-40 tokens/second (aspirational, to be validated). Added clarification that actual performance documented after kernel optimization. |
| 1.6.0 | 2026-04-14 | Added RAG embedding generation pipeline: mean-pool first 4 layers (early-layer pooling). Marked retrieval latency <1ms as aspirational. |
| 1.7.0 | 2026-04-14 | Verified KV cache calculation: exact 1.61 GB for 24-layer model. Updated total budget to ~2.82 GB. |
| 2.1.0 | 2026-04-15 | Added deterministic artifact pipeline requirement. Locked mandatory header validation fields (magic, version, payload_size, checksum) for .abn/.abt/.afr and required runtime integrity checks at load time. |
| 2.2.0 | 2026-04-15 | Added production validation and release hardening requirements: deterministic round-trip tests, startup/load failure-mode tests, benchmark pass/fail gates, and CI integrity/reproducibility enforcement. |
| 2.3.0 | 2026-04-15 | Added runtime profile requirements (`safe`, `balanced`, `max`), structured logging and crash diagnostics, reproducible tagged release artifact requirements, and rollback-safe artifact format upgrade path policy. |
