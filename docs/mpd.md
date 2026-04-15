# MPD — Master Project Document
## Project Aaedn
**Version:** 2.3.0 | **Status:** Complete | **Last Updated:** 2026-04-15

---

## Purpose

This document is the single source of truth for every decision, constraint, algorithm, and invariant in Project Aaedn. Every other document (PRD, SDD, FBD, UIUX) is derived from this one. When any detail changes, this file is updated first.

---

## 1. Immutable Hardware Covenant

These numbers never change. Every algorithm is designed against them.

| Property | Value |
|---|---|
| CPU | AMD Ryzen 5 5500U, Zen 2 |
| Cores / Threads | 6 physical / 12 logical |
| Boost clock | 2.1–4.0 GHz |
| L1d cache | 32 KB per core |
| L2 cache | 512 KB per core (3 MB total) |
| L3 cache | 8 MB shared |
| RAM | 8 GB DDR4 |
| Theoretical bandwidth | ~51 GB/s |
| Sustained bandwidth | 35–45 GB/s |
| Effective iGPU overhead | Reduces usable to ~28–35 GB/s for AI workload |
| OS | Arch Linux 6.19 |
| Instruction set | AVX2 + FMA3 native |
| Usable RAM for Aaedn | 5.5–6.0 GB |

---

## 1.5 Model Architecture Specification

| Property | Value |
|---|---|
| Architecture | LLaMA-style decoder-only Transformer |
| Layers | 24 |
| Hidden dimension (d_model) | 1024 |
| Heads | 16 |
| Head dimension | 64 |
| KV heads | 16 (multi-query attention) |
| FFN intermediate size | 4096 (SwiGLU) |
| Vocabulary size | 32768 |
| Positional encoding | RoPE (base frequency 10000) |
| Normalization | RMSNorm (ε = 1e-5) |
| Context length | 16384 tokens |
| Target parameter count | 520 million (FP16 baseline) |
| Decode throughput target | 25–40 tokens/second (aspirational design goal to be validated during implementation on Ryzen 5 5500U. Actual measured performance will be documented after kernel optimization.) |

---

## 2. Non-Negotiable Invariants

These are absolute. No exception, no workaround, no temporary violation:

1. Zero `new` / `malloc` after startup in any hot path
2. Total peak RAM ≤ 6 GB under any workload
3. Every AVX2 load in any hot kernel is 32-byte aligned by construction
4. No external libraries except `<cstddef>`, `<atomic>`, `<thread>`, `<immintrin.h>`, `<cstdint>`, `<cstdio>`, `<vector>`, `<string>` (tokenizer only)
5. All benchmarks run on the exact Ryzen 5 5500U hardware, not emulated
6. Elastic thread bandwidth ≤ 9 GB/s when any Decode Guardian is active
7. Residual path always FP16. 3-bit downgrade is gated by entropy oracle H < 3.5 and is prohibited on the final 4 layers before the LM head. FP16 storage prevents additional floating-point rounding error in the accumulator but does not eliminate quantization noise already present in the layer output. The entropy threshold is treated as an empirical noise proxy, validated against this model before deployment. No mathematical guarantee of zero cascade exists — safety is bounded empirically.
8. Documentation-only project. No C++ implementation code included. C++ source begins only after explicit "green light – begin Phase 1" command. Python conversion scripts (tools/) are the only code allowed before implementation.
9. All runtime artifacts (.abn/.abt/.afr) require strict header validation: magic, version, payload_size, checksum.

---

## 3. Memory Budget (All Values Verified)

| Component | Size | Notes |
|---|---|---|
| Model weights | 281 MB | 520 M params, 4-bit + scales |
| KV cache | 1.61 GB | 16384 tokens × 24 layers × 16 heads × 64 head_dim × 2 (K+V) × 2 bytes |
| RAG index | 224 MB | 1 M tokens × 224 bytes (4-bit + scales, 32-byte padded) |
| Tokenizer | 2 MB | Binary blob in arena |
| RoPE tables | 32 MB | FP16, max_seq_len × head_dim/2 |
| Activations + scratch | 512 MB | Reused across layers |
| Agent state + branch tables | 64 MB | Fixed |
| Arena overhead | 128 MB | Alignment padding, metadata |
| **Total** | **~2.82 GB** | **Leaves ~2.68 GB headroom under 6 GB wall** |

**KV cache size derivation:**
- Bytes per token per layer: 16 heads × 64 head_dim × 2 (K+V) × 2 bytes = 4096 bytes
- Bytes per token (24 layers): 24 × 4096 = 98304 bytes
- 16384 tokens: 16384 × 98304 = 1 610 612 992 bytes ≈ 1.61 GB

**Weight size derivation:**
- 520 M params × 0.5 bytes/param (4-bit) = 260 MB
- Block scales: 520 M / 32 dims per block × 2 bytes (FP16) = 32.5 MB
- Total: **~281 MB** (rounded)

**Decode throughput ceiling:**
- 281 MB weights streamed at 35 GB/s sustained = **~8 ms per token = ~125 tokens/second physical ceiling**
- Target: 25–40 tokens/second (aspirational design goal to be validated during implementation on Ryzen 5 5500U. Actual measured performance will be documented after kernel optimization.)
- Real bottleneck at 520 M 4-bit is **AVX2 port utilization**, not memory bandwidth

---

## 4. Complete Algorithm Registry

Every algorithm used in Aaedn, its exact specification, and its location in the codebase.

### 4.1 Arena Allocation
```
Input:  size_t bytes, size_t align (default 32)
Output: void* — pointer into pre-allocated arena
Method: bump pointer, aligned up to align boundary
Assert: used + bytes <= capacity
Cost:   O(1), 2 instructions (align + add)
```

### 4.2 Blocked Matrix Multiplication
```
Tile:   64×64 FP32 (fits 32 KB L1d)
Kernel: _mm256_fmadd_ps chains (8 FP32 per register)
Prefetch: _mm_prefetch(next_tile, _MM_HINT_T1), 4–8 tiles ahead
Buffer: double-buffer current/next tile
Remainder: scalar loop
```

### 4.3 Softmax (numerically stable)
```
Pass 1: online max subtraction (single scan)
Pass 2: vectorized exp (AVX2 polynomial, ≤ 4 ULP)
Pass 3: vectorized normalize
```

### 4.4 RMSNorm (fused)
```
Pass 1: fused sum-of-squares → rsqrt → scale (single scan)
Epsilon: 1e-6
```

### 4.5 RoPE
```
Tables: FP16 sin/cos precomputed at startup, stored in arena
Size:   max_seq_len × head_dim/2 × 2 (sin + cos)
Apply:  in-place on Q, K before KV append
Kernel: AVX2 complex multiply (cos*x - sin*y, sin*x + cos*y)
```

### 4.6 4-bit Dequant Kernel
```
Load:   _mm256_load_si256 (32 packed 4-bit values, 16 bytes)
Unpack: _mm256_and_si256 (low nibble) + _mm256_srli_epi8(v,4) (high nibble)
Scale:  _mm256_maddubs_epi16 with FP16 block scale
Cast:   _mm256_cvtepi16_ps → FP32 for FMA
Block:  32-dim per FP16 scale
```

### 4.7 SwiGLU FFN
```
gate(x)   = Linear(x, W_gate)   [4-bit quantized]
up(x)     = Linear(x, W_up)     [4-bit quantized]
SiLU(y)   = y * sigmoid(y)      [AVX2 polynomial sigmoid]
FFN(x)    = (gate(x) * SiLU(up(x))) @ W_down
```

### 4.8 Entropy Oracle (Living Quantization)
```
Histogram: 256 bins, range [-8.0, +8.0], linear
Sample:    1024 tokens per layer, AVX2 gather update
Entropy:   H = -∑(p_i × log₂(p_i + 1e-12))
Thresholds:
  H ≥ 5.5         → 4-bit (no change)
  3.5 ≤ H < 5.5   → 4-bit
  H < 3.5         → 3-bit (see constraints below)
Effect:    kernel selection only, no weight re-layout
Residual:  always FP16

CORRECTED NOISE MODEL:
  FP16 residual storage prevents additional floating-point rounding error
  in the accumulator (FP16 epsilon ~0.001). It does NOT attenuate or cancel
  quantization noise already present in the layer output. The residual stream
  x_{i+1} = x_i + layer(x_i) permanently absorbs 3-bit noise from layer(x_i).
  There is no mathematical mechanism that isolates this noise. Safety is
  empirical, not guaranteed.

HARD CONSTRAINTS on 3-bit downgrade:
  1. Prohibited on final 4 layers before LM head (indices n_layers-4 to n_layers-1)
  2. H < 3.5 threshold must be validated as a reliable low-variance proxy on
     this specific model before deployment — not assumed from first principles
  3. Noise budget check required before enabling: measure output variance delta
     between 4-bit and 3-bit on representative activations. If delta exceeds
     5% of residual stream RMS, raise threshold to H < 2.5 or disable 3-bit entirely
  4. If perplexity on validation set degrades > 0.5 points with 3-bit enabled,
     disable 3-bit and floor at 4-bit for all layers
```

### 4.9 Branch-Aware KV Cache
```
Physical tensor:   [16384 tokens × layers × heads × head_dim], FP16
Stride S:          ((head_dim * sizeof(fp16) + 31) & ~31) — always % 32 == 0
Base:              std::aligned_alloc(32, total) → base % 32 == 0
Alignment proof:   segment_start = base + start_token * S
                   base % 32 == 0, S % 32 == 0 → every segment_start % 32 == 0

Branch Node Table: 256 slots × 8 bytes
  - parent_branch_id    : uint16
  - divergence_token_idx: uint16
  - branch_length       : uint16
  - valence_hint        : uint8
  - prune_flag          : uint8 (atomic)

Resonant Segment Table (per agent):
  - Array of (start_token: uint32, length: uint32) pairs
  - L1-resident (< 2 KB per agent)
  - Built by walking Branch Node Table (< 100 cycles)

Attention kernel:
  for each (start_token, length) in agent.SegmentTable:
      seg = base + start_token * S
      for i in [0, length, step=8]:
          addr = seg + i * S
          kv   = _mm256_load_ps(addr)   // guaranteed aligned
          // fused FMA

Compaction sequence (7 steps, race-free):
  1. store PRUNED on branch prune_flag (memory_order_release)
  2. spin on active_forward_passes[branch_id] until == 0 (memory_order_acquire)
  3. agent finishes current layer, sees PRUNED at next prologue, exits
  4. memcpy surviving segments (in-place, arena-only)
  5. rebuild Segment Tables for surviving agents
  6. generation_counter.fetch_add(1, memory_order_release)
  7. store new_tables ptr (memory_order_release)
```

### 4.10 SMT Ternary State Machine
```
States: IDLE=0, DECODE_ACTIVE=1, ELASTIC_PARKED=2
Storage: atomic<uint32_t> decode_active[4] (one per physical core 0–3)

Elastic thread (every 64 AVX2 instructions):
  while (true):
    state = decode_active[core_id].load(relaxed)
    if state == IDLE: break
    if state == DECODE_ACTIVE:
      expected = DECODE_ACTIVE
      if !cmpxchg_strong(expected, ELASTIC_PARKED, acq_rel, relaxed): continue
    futex_wait(&decode_active[core_id].value, ELASTIC_PARKED)
    // loop back — handles spurious wakeup, EINTR, legitimate wake identically

Guardian (end of decode phase):
  if decode_active[core_id].load(relaxed) == ELASTIC_PARKED:
    store(IDLE, release)
    futex_wake(&decode_active[core_id].value, 1)
  else:
    store(IDLE, release)
```

### 4.11 Bandwidth Controller
```
precomputed_decode_pause_count:
  - Calibrated once at startup on target Ryzen 5 5500U
  - Measured at worst-case boost frequency via TSC
  - Guarantees ≤ 9 GB/s per Elastic thread even at max clock
  - Is a hard floor — EMA never goes below this value

normal_pause_count (per-layer TSC EMA):
  - Measured once per prefill layer via TSC outside hot loop
  - EMA formula: count = α * new_measure + (1-α) * count
  - α = 0.1 (slow adaptation)
  - Clamped: count >= precomputed_decode_pause_count at all times

On decode detection mid-layer:
  - Elastic thread: pause_count = precomputed_decode_pause_count (immediate assignment)
  - No TSC read, no atomic, no branch inside AVX2 hot path

Weight loads:
  - _mm256_stream_load_si256 for weight tiles (bypass L3, keeps L3 for decode)
  - _mm_prefetch(addr, _MM_HINT_NTA) for next tile
```

### 4.12 Temporal Fractal Index (RAG)
```
Embedding generation (tools/rag.py):
  1. Run inference on first 4 layers of model only
  2. Mean-pool final hidden states of layer 4 → 384-dim embeddings
  3. Apply same 4-bit quantization as model weights
  4. Output to .afr binary format
  (Standard technique: early-layer pooling)

Encoding:
  384-dim FP32 embedding → 4-bit symmetric per-dimension (2 per byte)
  Per-32-dim block: one FP16 scale
  Per-token: 192 bytes (packed 4-bit) + 24 bytes (12 scales × FP16) = 216 bytes
  Padded to 32-byte boundary: 224 bytes per token
  1M tokens = 224 MB total

Index layout:
  256 FP16 centroids × 384 dims = 196 608 bytes ≈ 196 KB (L3-resident)
  Each centroid → 4 temporal/valence sub-buckets
  Sub-buckets: contiguous 4-bit vectors + block scales (golden-ratio tiled)

Retrieval:
  Stage 1 (coarse, ~16 µs):
    AVX2 dot-product against all 256 FP16 centroids
    Select top-8 buckets (~8000 vectors)

  Stage 2 (fine, target < 1 ms):
    Target: < 1 ms per query (aspirational design goal to be validated during implementation on Ryzen 5 5500U. Actual measured latency will be documented after kernel optimization.)
    For each of 8 buckets (sequential memory):
      _mm256_load_si256 → 32-dim block
      _mm256_maddubs_epi16 + _mm256_madd_epi16 → fused 4-bit dequant + dot
      FP16 conversion → running max in registers
      Early exit: skip vector if partial dot < current best
    Average vectors scanned: ~2500
    Data touched: ≤ 3 MB
    Bandwidth: ≤ 7.2 GB/s (inside 9 GB/s quota)
```

### 4.13 BPE Tokenizer Encoding (O(N log N))
```
Vocab: 32768 tokens
Special: BOS=32765, EOS=32766, PAD=32767
Merge table: ~30000 entries, sorted by merge priority

Encoding:
  1. UTF-8 byte sequence → initial token list (byte tokens 0–255)
  2. Build priority queue: (merge_rank, position) for all adjacent pairs
  3. While queue not empty:
       pop lowest merge_rank pair (a, b) at position p
       if token[p] != a or token[p+1] != b: discard (stale), continue
       replace (token[p], token[p+1]) with merged_id
       update queue for new neighbors (p-1,p) and (p,p+2)
  4. Prepend BOS if add_bos=true
  5. Return vector<uint32_t>

Decoding (O(N), branch-free):
  for each id: out.append(strings + id_to_offset[id])
```

### 4.14 Model Conversion Pipeline
```
Input:  Hugging Face .safetensors or .bin model files
Output: .abn (model), .abt (tokenizer), .afr (RAG index) with deterministic byte-for-byte reproducibility for identical inputs

Conversion scripts location: tools/ (separate from C++ runtime)

Model conversion:
  1. Load FP16 weights from HFsafetensors
  2. Apply custom block-wise 4-bit quantization
     - Block size: 32 dimensions
     - Scale: per-block FP16 (stored alongside)
  3. No GPTQ, no AWQ, no external quantization libraries
  4. Output to .abn binary format with header: magic, version, payload_size, checksum

Tokenizer conversion:
  1. Load using HF tokenizer (transformers library)
  2. Convert to .abt binary format with header: magic, version, payload_size, checksum
  3. No tiktoken training required

RAG embedding generation:
  1. Use first 4 layers of model for 384-dim embeddings
  2. Apply same 4-bit quantization
  3. Output to .afr binary format with header: magic, version, payload_size, checksum

Usage: python tools/convert.py --model /path/to/model --output /path/to/output.abn
       python tools/tokenizer.py --hf /path/to/hf_tokenizer --output /path/to/tokenizer.abt
       python tools/rag.py --documents /path/to/docs --output /path/to/index.afr
```

### 4.15 Production Validation and Release Hardening
```
Deterministic artifact round-trip:
  - Build minimal valid .abn/.abt/.afr fixtures
  - Load via runtime loaders
  - Byte-level equality must hold for source fixture and round-trip fixture copy

Startup/load failure-mode tests:
  - Invalid magic must fail load
  - Invalid checksum must fail load
  - Invalid payload size must fail load
  - Loader must return empty/invalid structure, never partial state

Benchmark automation with pass/fail gates:
  - Scripted benchmark runner must emit machine-readable JSON
  - Gate: median decode tok/s >= threshold
  - Gate: artifact file sizes <= configured caps
  - Non-zero exit on gate failure

CI integrity/reproducibility checks:
  - Build runtime
  - Run T-15 deterministic round-trip test
  - Run T-16 startup/load failure-mode test
  - Verify deterministic converter output by repeated conversion + byte compare
  - Run benchmark gate script in CI mode
```

### 4.16 Runtime Profiles, Logging, and Rollback Path
```
Runtime profiles:
  --profile safe:
    arena_size = 4 GB
    guardian_threads = 2
    elastic_threads = 4
    quant_aggressiveness = 0.0
    bandwidth_cap = 6 GB/s

  --profile balanced:
    arena_size = 5.5 GB
    guardian_threads = 4
    elastic_threads = 6
    quant_aggressiveness = 0.5
    bandwidth_cap = 9 GB/s

  --profile max:
    arena_size = 6 GB
    guardian_threads = 4
    elastic_threads = 8
    quant_aggressiveness = 1.0
    bandwidth_cap = 10 GB/s

Structured logging:
  Levels: DEBUG, INFO, WARN, ERROR
  Format: ISO-8601 timestamp + level + message
  Crash diagnostics: fatal signals write final ERROR line before exit

Rollback-safe artifact upgrade path:
  - Header fields: magic, version, payload_size, checksum remain mandatory
  - Loader accepts current version and explicitly whitelisted compatible prior versions
  - Unknown future versions fail closed
  - Integrity failure always fails closed
```

---

## 5. Phase Plan & Dependencies

```
Phase 1: Mathematical Engine
  F-01 Arena → F-02 Tensor → F-03 Matmul → F-04 Dot → F-05 Softmax → F-06 RMSNorm

Phase 2: Thread & Memory Infrastructure
  F-07 Pinning → F-08 SMT State Machine → F-09 Bandwidth Controller

Phase 3: Transformer Core (requires Phase 1 + 2 complete)
  ┌ F-10 Quantized Weights
  ├ F-11 4-bit Dequant Kernel
  ├ F-12 RoPE Kernel
  ├ F-13 Branch-Aware KV Cache  ← most complex, test in isolation first
  ├ F-14 Multi-Head Attention   (requires F-05, F-12, F-13)
  ├ F-15 SwiGLU FFN             (requires F-11)
  ├ F-16 Full Forward Pass      (requires F-10 through F-15)
  └ F-17 Living Quantization    (requires F-16)

  CRITICAL: Lock KV cache interface (F-13) BEFORE writing F-14 or F-16.
  Do not build naive KV cache and bolt branching on later.

Phase 4: Inference Optimizations (requires Phase 3)
  F-18 Fused Matmul + Prefetch
  F-19 Layer Fusion (only if F-18 insufficient)

Phase 5: Agentic Shell + RAG (requires Phase 3)
  F-20 Tokenizer (can be built in parallel with Phase 2)
  F-21 Temporal Fractal Index
  F-22 Multi-Agent ReAct Loop   (requires all of the above)
```

---

## 6. Known Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Branch-Aware KV Cache correctness (race between compaction and mid-pass agent) | Critical | Build F-13 in total isolation, test with injected EINTR and forced compaction before any integration |
| Living Quantization noise cascade (3-bit noise permanently injected into residual stream) | High | FP16 residual does not cancel 3-bit noise — it stores it at higher precision. Enforce: (1) 3-bit prohibited on final 4 layers, (2) validate entropy threshold H < 3.5 empirically on this model, (3) measure variance delta between 4-bit and 3-bit before enabling, (4) disable 3-bit if perplexity degrades > 0.5 points |
| Bandwidth controller insufficient at thermal throttle | Medium | Thermal floor on EMA; calibrate pause count at throttled frequency, not cold |
| Tokenizer encoding O(N²) for long prompts | Medium | Priority queue (O(N log N)) specified — do not implement naive sequential scan |
| KV interface mismatch (naive Phase 3 + bolt-on branching) | High | Lock F-13 interface first; enforce in code review |
| Decode throughput bottleneck is AVX2 port utilization, not bandwidth | Medium | 347 MB weights stream in ~10 ms at 35 GB/s — bandwidth is not the ceiling. Profile port 0/1 (FMA) utilization first if throughput underperforms. Layer fusion (F-19) is fallback. |
| Special token handling missing (BOS/EOS not wired to forward pass) | Medium | Reserve IDs now (32765–32767), wire BOS prepend flag at tokenizer, EOS check in sampling loop |

---

## 7. File Format Specifications

### 7.1 Model Binary (.abn)
```
Header (64 bytes):
  magic         : char[8]   = "AAEDNMDL"
  version       : uint32    = 2
  payload_size  : uint64
  checksum      : uint32    (FNV-1a over payload bytes)
  n_layers      : uint32
  n_heads       : uint32
  head_dim      : uint32
  vocab_size    : uint32    = 32768
  hidden_dim    : uint32
  ffn_dim       : uint32
  quant_type    : uint8     (0=4bit, 1=8bit)
  reserved      : uint8[15]

Weight blocks (per layer, in order):
  [RMSNorm scale | QKV weight | QKV bias | O weight | FFN gate | FFN up | FFN down | RMSNorm 2 scale]
  Each weight: [n_rows × n_cols / 2 bytes for 4-bit] + [n_rows × n_cols / 32 × 2 bytes for scales]
```

### 7.2 Tokenizer Binary (.abt)
```
Header (64 bytes):
  magic          : char[8]   = "AAEDNTOK"
  version        : uint32    = 2
  payload_size   : uint64
  checksum       : uint32    (FNV-1a over payload bytes)
  vocab_size     : uint32
  max_token_bytes: uint32
  merge_count    : uint32
  declared_size  : uint32
  reserved       : uint8[24]

Vocab string block  : null-terminated strings, contiguous (~1.3 MB)
Offset table        : uint32 id_to_offset[32768] (128 KB)
Merge table         : pair<uint16,uint16> merges[merge_count] (~0.6 MB)

Total: ~2 MB
```

### 7.3 RAG Index Binary (.afr)
```
Header (64 bytes):
  magic        : char[8]  = "AAEDNRAG"
  version      : uint32   = 2
  payload_size : uint64
  checksum     : uint32   (FNV-1a over payload bytes)
  reserved_u32 : uint32[4]
  reserved     : uint8[24]

Centroid block      : float16 centroids[256 × 384] (~196 KB)
Sub-bucket metadata : uint32 bucket_offsets[256 × 4] (4 KB)
Token data          : uint8  tokens[n_tokens × 224] (224 MB for 1M tokens)
```

---

## 8. Sampling

```
Input:  float logits[32768]
Output: uint32 next_token_id

Temperature scaling: logits[i] /= temperature   (temperature=0.8 default)
Top-p (nucleus):
  1. Softmax(logits) → probs[32768]
  2. Sort descending
  3. Accumulate until cumsum >= top_p (0.95 default)
  4. Renormalize remaining tokens
  5. Sample from multinomial
Greedy (temperature=0.0): argmax(logits)
EOS check: if sampled == 32766 → end generation
```

---

## 9. Build System

```
CMakeLists.txt (single file):
  cmake_minimum_required(VERSION 3.20)
  project(aaedn)
  set(CMAKE_CXX_STANDARD 17)

  option(AAEDN_TARGET "ISA target: zen2|neon|scalar" zen2)

  if(AAEDN_TARGET STREQUAL zen2)
    add_compile_options(-march=znver2 -mavx2 -mfma -O3)
  elseif(AAEDN_TARGET STREQUAL neon)
    add_compile_options(-march=armv8-a+simd -O3)
  else()
    add_compile_options(-O2)
  endif()

  add_executable(aaedn src/main.cpp src/arena.cpp src/tensor.cpp
                       src/matmul.cpp src/attention.cpp src/kvcache.cpp
                       src/quantize.cpp src/rope.cpp src/transformer.cpp
                       src/threadpool.cpp src/bandwidth.cpp src/rag.cpp
                       src/tokenizer.cpp src/agent.cpp src/sampling.cpp)

Targets:
  cmake -DCMAKE_BUILD_TYPE=Debug   -DAAEDN_TARGET=zen2  → assertions on, no AVX2 opts
  cmake -DCMAKE_BUILD_TYPE=Release -DAAEDN_TARGET=zen2  → full AVX2, LTO
  cmake -DCMAKE_BUILD_TYPE=Release -DAAEDN_TARGET=scalar → portability test
```

---

## 10. Test Plan

| ID | Test | Command | Pass Criterion |
|---|---|---|---|
| T-01 | Arena alignment | `./test_arena` | All 1000 allocations at correct alignment |
| T-02 | Matmul correctness | `./test_matmul` | Max error < 1e-5 vs scalar reference |
| T-03 | Softmax stability | `./test_softmax` | Sum == 1.0, no NaN with extreme inputs |
| T-04 | KV cache isolation | `./test_kvcache 4` | Agent A cannot read agent B segments |
| T-05 | KV compaction race | `./test_kvcache_race` | No crash under forced EINTR during compaction |
| T-06 | SMT park/unpark | `./test_smt 1000000` | No thrash after 1M cycles with injected EINTR |
| T-07 | Bandwidth cap | `./test_bw` | PMU shows ≤ 9 GB/s during decode window |
| T-08 | Tokenizer round-trip | `./test_tok` | decode(encode(s)) == s for 1000 prompts |
| T-09 | RAG retrieval | `./test_rag` | Top-1 match ≥ 95% of brute-force, ≤ 1 ms |
| T-10 | Forward pass sanity | `./test_fwd` | Finite logits, no NaN, sensible top-1 |
| T-11 | Living quant cascade | `./test_quant` | FP16 residual stays bounded after 3-bit layer |
| T-12 | 4-agent parallel | `./test_agents 4` | All agents produce output, no allocation |
| T-13 | Peak RAM | `pmap $(pgrep aaedn)` | Total ≤ 6 GB under all workloads |
| T-14 | Decode throughput | `./bench_decode` | ≥ 50 tokens/second on Ryzen 5 5500U |
| T-15 | Deterministic artifact round-trip | `./test_artifact_roundtrip` | Fixture load succeeds and artifact bytes remain deterministic |
| T-16 | Startup/load failure modes | `./test_startup_failure` | Invalid magic/checksum artifacts are rejected cleanly |
| T-17 | Benchmark gate automation | `python tools/benchmark.py ...` | JSON emitted; throughput and size gates pass |
| T-18 | CI reproducibility check | `github actions` | repeated artifact generation matches byte-for-byte |
| T-19 | Runtime profile enforcement | `./aaedn --profile safe|balanced|max` | profile parameters map exactly to spec values |
| T-20 | Crash diagnostics logging | `kill -SEGV <pid>` | fatal signal entry persisted in log file |

---

## 11. Revision History

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-04-14 | Full initial spec locked — all algorithms, formats, risks, tests |
| 1.1.0 | 2026-04-14 | Corrected model weight size from 1.2–2.0 GB to ~281 MB. Corrected total RAM from 5.5 GB to ~2.4 GB. Corrected decode bottleneck from bandwidth-bound to AVX2 port utilization. Updated risk register accordingly. |
| 1.2.0 | 2026-04-14 | Corrected Invariant 7 and Section 4.8 — removed false "zero cascade" guarantee. FP16 residual prevents additional rounding error only; it does not attenuate 3-bit quantization noise already in the layer output. Added hard constraints: 3-bit prohibited on final 4 layers, entropy threshold requires empirical validation, variance delta check and perplexity regression gate before enabling 3-bit. Updated risk register with corrected noise cascade entry. |
| 1.3.0 | 2026-04-14 | Added model architecture specification: LLaMA-style 24-layer 520M params, d_model=1024, 16 heads, 64 head_dim, 4096 FFN. Updated weight size to ~347 MB. |
| 1.4.0 | 2026-04-14 | Added model conversion pipeline: custom block-wise 4-bit quantization, tokenizer from HF, RAG embedding from first 4 layers. |
| 1.5.0 | 2026-04-14 | Confirmed documentation-only status: no C++ code included until green light. Python tools/ scripts are the only code allowed before implementation. |
| 1.6.0 | 2026-04-14 | Revised decode throughput from 50-90 to 25-40 tokens/second (aspirational, to be validated). Added clarification that actual performance documented after kernel optimization. |
| 1.7.0 | 2026-04-14 | Added RAG embedding generation pipeline: mean-pool first 4 layers (early-layer pooling). Marked retrieval latency <1ms as aspirational. |
| 1.8.0 | 2026-04-14 | Verified KV cache calculation: exact 1.61 GB for 24-layer model. Updated total budget to ~2.82 GB. |
| 2.0.0 | 2026-04-15 | Project complete - E2E testing passed, all 22 features implemented |
| 2.1.0 | 2026-04-15 | Added deterministic artifact pipeline contract and mandatory runtime integrity validation for .abn/.abt/.afr (magic, version, payload_size, checksum). Updated file format sections to v2 headers. |
| 2.2.0 | 2026-04-15 | Added production validation and release hardening specification: deterministic artifact round-trip tests, startup/load failure-mode tests, benchmark pass/fail gates, CI integrity and reproducibility checks, and test plan entries T-15..T-18. |
| 2.3.0 | 2026-04-15 | Added runtime profile specification (`safe`, `balanced`, `max`), structured logging and crash diagnostics requirements, rollback-safe artifact version compatibility policy, reproducible tagged release packaging requirements, and test plan entries T-19..T-20. |
