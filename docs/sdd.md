# SDD — System Design Document
## Project Aaedn
**Version:** 2.2.0 | **Status:** Complete | **Last Updated:** 2026-04-15

---

## 1. Architecture Overview

Aaedn is a five-layer tightly integrated system. Every layer shares a single pre-allocated memory arena. No layer allocates independently after startup.

```
┌─────────────────────────────────────────────────┐
│              Agentic Shell + RAG                │  Layer 5
├─────────────────────────────────────────────────┤
│           Inference Engine (AVX2)               │  Layer 4
├─────────────────────────────────────────────────┤
│             Transformer Core                    │  Layer 3
├─────────────────────────────────────────────────┤
│         Memory + Thread Management              │  Layer 2
├─────────────────────────────────────────────────┤
│          Mathematical Engine (Tensors)          │  Layer 1
└─────────────────────────────────────────────────┘
          Single Fixed Arena (~2.5 GB typical, ≤ 6 GB hard limit)

**Documentation Status:** This document is specification-only. No C++ code included. All algorithms, interfaces, and data structures are documented in pseudocode and ASCII diagrams. Actual C++ implementation begins after "green light – begin Phase 1" command.

---

## 2. Layer 1 — Mathematical Engine

### 2.1 Tensor Class
- Row-major storage, 32-byte aligned (`std::aligned_alloc(32, n)`)
- Shape stored as `uint32_t dims[4]` (max 4D)
- Data pointer into arena — no ownership, no destructor allocation
- Stride computed from shape at construction, never recomputed

### 2.2 Core Operations (all manually implemented, no BLAS)

**Matrix Multiplication (blocked)**
- Block size tuned to Ryzen 5 5500U: 64×64 tiles fit L1d (32 KB)
- Inner loop uses `_mm256_fmadd_ps` chains
- Prefetch next tile with `_mm_prefetch(addr, _MM_HINT_T1)` 4–8 tiles ahead
- Double-buffered: current tile computes while next tile loads

**Dot Product**
- AVX2 horizontal reduction: `_mm256_hadd_ps` + `_mm256_permute2f128_ps`
- Remainder handled with scalar loop

**Softmax**
- Online max subtraction for numerical stability (single-pass)
- AVX2 exp approximation (polynomial, 4 ULP accuracy)
- Final division vectorized

**RMSNorm**
- Fused: compute RMS + normalize + scale in one pass
- AVX2 horizontal sum for RMS computation

---

## 3. Layer 2 — Memory & Thread Management

### 3.1 Arena Allocator
```cpp
struct Arena {
    uint8_t* base;          // std::aligned_alloc(32, total_bytes)
    size_t   capacity;      // total bytes
    size_t   used;          // bump pointer offset
    // No free(), no destructor, no fragmentation
};

void* arena_alloc(Arena* a, size_t bytes, size_t align = 32) {
    size_t aligned_used = (a->used + align - 1) & ~(align - 1);
    assert(aligned_used + bytes <= a->capacity);
    a->used = aligned_used + bytes;
    return a->base + aligned_used;
}
```
All subsystem allocations (KV tensor, RAG index, activations, branch tables, agent state) call `arena_alloc` once at startup. Zero runtime allocation after that point.

### 3.2 Thread Model

**Decode Guardian Threads (4 threads)**
- Pinned to logical CPUs 0, 2, 4, 6 via `pthread_setaffinity_np`
- Handle all autoregressive decode steps
- Own exclusive use of their SMT slot 0 during decode

**Elastic Compute Threads (8 threads)**
- Pinned to logical CPUs 1, 3, 5, 7 (SMT siblings of Guardians on cores 0–3)
- Plus logical CPUs 8, 9, 10, 11 (cores 4–5, both SMT slots)
- Handle prefill, RAG scan, agent reasoning

### 3.3 SMT Isolation — Ternary State Machine

Three states per core (stored as `std::atomic<uint32_t>`):

```
STATE_IDLE          = 0   // decode not active
STATE_DECODE_ACTIVE = 1   // Guardian started decode
STATE_ELASTIC_PARKED = 2  // Elastic thread is in futex_wait
```

**Elastic thread loop (every 64 AVX2 instructions):**
```cpp
while (true) {
    uint32_t state = decode_active[core_id].value.load(memory_order_relaxed);
    if (state == STATE_IDLE) break;
    if (state == STATE_DECODE_ACTIVE) {
        uint32_t expected = STATE_DECODE_ACTIVE;
        if (!decode_active[core_id].value.compare_exchange_strong(
                expected, STATE_ELASTIC_PARKED,
                memory_order_acq_rel, memory_order_relaxed))
            continue;
    }
    futex_wait(&decode_active[core_id].value, STATE_ELASTIC_PARKED);
    // Any return (spurious, EINTR, or real) → loop back and re-evaluate
}
```

**Guardian thread (end of decode phase):**
```cpp
if (decode_active[core_id].value.load(memory_order_relaxed) == STATE_ELASTIC_PARKED) {
    decode_active[core_id].value.store(STATE_IDLE, memory_order_release);
    futex_wake(&decode_active[core_id].value, 1);
} else {
    decode_active[core_id].value.store(STATE_IDLE, memory_order_release);
}
```

This completely evacuates the ROB, front-end, FPU, and LSU from the sibling SMT thread during decode. No unconditional wakes, no spurious resume.

### 3.4 Bandwidth Controller

**Goal:** Elastic prefill threads stay ≤ 9 GB/s when decode is active.

**Mechanism:**
- `precomputed_decode_pause_count`: static constant, calibrated once at startup on target hardware at worst-case boost frequency. Guarantees ≤ 9 GB/s even at maximum clock.
- Per-layer TSC-based EMA (measured outside hot loop, once per prefill layer) updates `normal_pause_count` for non-decode periods.
- On decode detection mid-layer: Elastic threads immediately assign `pause_count = precomputed_decode_pause_count` (no TSC read, no atomic, no branch inside AVX2 loop).
- Non-temporal loads for weights: `_mm256_stream_load_si256` + `_mm_prefetch(addr, _MM_HINT_NTA)` to keep L3 clean for decode.
- **Thermal floor:** `precomputed_decode_pause_count` is a hard floor — EMA never goes below it to handle asymmetric boost under thermal throttle.

---

## 4. Layer 3 — Transformer Core

### 4.1 Architecture
- LLaMA-style decoder-only Transformer
- Layers: 24, d_model: 1024, heads: 16, head_dim: 64, KV heads: 16
- RMSNorm pre-normalization (ε = 1e-5)
- Rotary Positional Embedding (RoPE, base frequency 10000)
- SwiGLU feed-forward network (FFN intermediate: 4096)
- Multi-head causal attention with KV cache
- Target: 520 M parameters (FP16 baseline, quantized to ~347 MB)
- Context: up to 16 384 tokens
- Decode throughput target: 25–40 tokens/second (aspirational design goal to be validated on Ryzen 5 5500U)
- Model input: .abn binary (converted via tools/convert.py, see MPD §4.14)

### 4.2 Forward Pass Sequence (per token)
1. Token embedding lookup → FP16 activation vector
2. For each layer:
   a. RMSNorm(input)
   b. QKV projection (fused matmul, 4-bit weights → FP16 output)
   c. Apply RoPE to Q and K
   d. Append K, V to KV cache for this agent's branch
   e. Compute causal attention using Branch-Aware KV Cache
   f. Output projection
   g. Residual add (FP16)
   h. RMSNorm
   i. SwiGLU FFN (gate × up → activation → down projection)
   j. Residual add (FP16)
3. Final RMSNorm → LM head → logits (FP32)
4. Sampling (greedy or top-p)

### 4.3 Branch-Aware KV Cache

**Physical layout:**
- Single contiguous KV tensor, pre-allocated for 16 384 tokens
- Shape: `[max_tokens × layers × heads × head_dim × 2 (K,V)]`, FP16
- Per-token size: 16 heads × 64 head_dim × 2 × 2 bytes = 4096 bytes
- Total: 16384 × 4096 = 1.61 GB (24 layers)
- Per-token stride: `S = ((head_dim * sizeof(fp16) + 31) & ~31)` — always multiple of 32
- Base address: `std::aligned_alloc(32, total_bytes)` → `base % 32 == 0`
- Every segment_start = `base + start_token * S` → 32-byte aligned by construction
- **No `_mm256_loadu_ps`, no masking, no padding between segments**

**Branch Node Table:**
- Fixed 256 slots, pre-allocated in arena, ~4 KB total
- Per slot (8 bytes): `parent_branch_id (uint16)`, `divergence_token_index (uint16)`, `branch_length (uint16)`, `valence_hint (uint8)`, `prune_flag (uint8, atomic)`

**Resonant Segment Table (per agent):**
- L1-resident list of `(start_token, length)` pairs
- Describes exactly which physical segments this agent may attend to
- Generated by fast walk of Branch Node Table (< 100 cycles, 256-node tree)

**Attention kernel (per segment):**
```cpp
for each segment (start_token, length) in agent's SegmentTable:
    uint8_t* seg = base + (uint64_t)start_token * S;
    for (size_t i = 0; i < length; i += 8) {
        __m256* addr = (__m256*)(seg + i * S);
        __m256 kv = _mm256_load_ps((float*)addr);  // guaranteed 32-byte aligned
        // fused FMA dot-product
    }
```

**Compaction (branch merge/prune) — exact atomic sequence:**
1. Management thread: `branch.prune_flag.store(1, memory_order_release)`
2. Management thread: spin on `active_forward_passes[branch_id].load(memory_order_acquire)` until == 0
3. Third agent finishes current layer, sees PRUNED at next layer prologue, exits branch
4. Management thread: in-place `memcpy` compaction of surviving segments into arena
5. Management thread: rebuild Segment Tables for all surviving agents
6. Management thread: `generation_counter.fetch_add(1, memory_order_release)`
7. Management thread: `global_segment_table_ptr.store(new_tables, memory_order_release)`

No agent reads deallocated KV memory. All operations stay inside fixed arena.

### 4.4 Living Quantization (Entropy Oracle)

**Histogram:** 256 bins, linear scale over `[-8.0, +8.0]`, computed on 1 024-token sample window per layer using AVX2 gather + atomic update.

**Shannon entropy:**
```
H = -∑(p_i × log₂(p_i + 1e-12))   where p_i = bin_count[i] / total_tokens
```

**Precision thresholds:**
| Entropy | Precision |
|---|---|
| H ≥ 5.5 bits | 4-bit (keep current) |
| 3.5 ≤ H < 5.5 | 4-bit (downgrade if higher) |
| H < 3.5 | 3-bit |

**Residual protection (zero cascade):**
- All residuals stored and added in FP16 regardless of layer precision
- 3-bit / 4-bit weights dequantized on-the-fly inside fused matmul kernel
- Dequant path: `_mm256_maddubs_epi16` → `_mm256_cvtepi16_ps` → FMA in FP16
- Layer output immediately cast back to FP16 before residual add
- No re-layout, no extra RAM, no cascade error

---

## 5. Layer 4 — Inference Engine (AVX2 Kernels)

### 5.1 Fused QKV Projection Kernel
- Reads 4-bit quantized weights block by block
- Dequantizes with per-32-dim FP16 block scales
- Accumulates dot-product in FP32 registers
- Outputs Q, K, V in FP16 directly into arena buffers
- No intermediate FP32 weight materialization

### 5.2 Fused Matmul Kernel (weight streaming)
- Tile weights to 64 KB blocks (fits L2 per core)
- `_mm_prefetch(next_tile, _MM_HINT_NTA)` issued 4–8 tiles ahead
- Double-buffer: current tile computes while next tile loads via Elastic threads at low decode load
- Raises effective bandwidth from ~65% to >90% of DDR4 bus
- Scratch buffer: 128 KB in arena (no extra allocation)

### 5.3 4-bit Dequant AVX2 Kernel
```
_mm256_load_si256   → load 32 packed 4-bit values (16 bytes → 256-bit reg)
_mm256_and_si256    → mask low nibble
_mm256_srli_epi8    → shift high nibble
_mm256_maddubs_epi16 → accumulate
_mm256_cvtepi16_ps  → convert to FP32 for FMA
```

### 5.4 RoPE Kernel
- Precomputed sin/cos tables in FP16, stored in arena
- Applied to Q and K in-place before KV cache append
- AVX2 complex multiply: `cos*x - sin*y`, `sin*x + cos*y` vectorized

---

## 6. Layer 5 — Agentic Shell + RAG

### 6.1 Multi-Agent Loop
- 4–6 micro-agents running in parallel
- Each agent has its own branch ID, Segment Table, and tool queue
- ReAct loop: `[Reason] → [Act] → [Observe] → repeat`
- Tool registry: fixed table of function pointers, no dynamic dispatch
- Agent state (branch ID, step count, tool history) stored in arena

### 6.2 Temporal Fractal Index (RAG)

**Embedding generation (tools/rag.py):**
- Run inference on first 4 layers of model only
- Mean-pool final hidden states of layer 4 → 384-dim embeddings
- Apply same 4-bit quantization as model weights
- Output to .afr binary format
- Standard technique: early-layer pooling

**Encoding:**
- 384-dim embeddings → 4-bit symmetric per-dimension (2 values per byte)
- Per-32-dim FP16 block scales (12 × 2 = 24 bytes of scales per token)
- Per-token size: `384 × 0.5 + 24 = 216 bytes` → padded to 32-byte boundary = 224 bytes
- 1 million tokens = 224 MB (within budget)

**Index layout (single flat arena slab):**
- 256 coarse FP16 centroids (8 KB) — permanently L3-resident
- Each centroid owns 4 temporal/valence sub-buckets
- Sub-bucket contents: contiguous 4-bit vectors + block scales, golden-ratio tiled across cache lines for sequential prefetch

**Retrieval algorithm:**
1. **Coarse stage (< 20 µs, L3-resident):** AVX2 fused dot-product against all 256 centroids. Select top-8 buckets (~8 000 vectors total).
2. **Fine stage (target < 1 ms):**
   - Target: < 1 ms per query (aspirational design goal to be validated on Ryzen 5 5500U. Actual measured latency will be documented after kernel optimization.)
   - For each of 8 buckets (contiguous memory):
   - `_mm256_load_si256` → load 32-dim block
   - `_mm256_maddubs_epi16` + `_mm256_madd_epi16` → fused 4-bit dequant + dot-product
   - FP16 conversion → running max-score in registers
   - Early exit per vector when partial dot < current best → ~2 500 vectors scanned on average

**Performance target:**
- Data touched: ≤ 3 MB
- Bandwidth: ≤ 7.2 GB/s (inside Elastic quota)
- All loads sequential — no gather, no random access

### 6.3 Tokenizer

**Binary file format (~2 MB):**
```
Header     (32 bytes):  magic "AAEDNTOK", vocab_size=32768, max_token_bytes, merge_count
Vocab table            : null-terminated strings (~1.3 MB)
Offset table           : uint32_t id_to_offset[32768] (128 KB)
Merge table            : pair<uint16_t,uint16_t> merges[merge_count] (~0.6 MB)
```

**In-memory struct:**
```cpp
struct Tokenizer {
    char*                          strings;      // base of string block
    uint32_t*                      id_to_offset; // 32k offsets
    std::pair<uint16_t,uint16_t>*  merges;       // sorted by priority
    uint32_t                       vocab_size;   // 32768
    uint32_t                       merge_count;  // ~30000
};
```

**Special token IDs (reserved):**
| Token | ID |
|---|---|
| BOS | 32765 |
| EOS | 32766 |
| PAD | 32767 |

**Encoding — BPE with priority queue (O(N log N)):**
1. Convert input to UTF-8 bytes → initialize as byte token list
2. Build priority queue of adjacent pairs keyed on merge rank
3. Pop lowest-rank merge, apply, update neighbors in queue
4. Repeat until no merge applies
5. Prepend BOS if `add_bos = true`
6. Return `std::vector<uint32_t>` (temporary, discarded after use)

**Decoding (O(N), branch-free):**
```cpp
std::string decode(const std::vector<uint32_t>& ids) {
    std::string out;
    for (uint32_t id : ids)
        out.append(strings + id_to_offset[id]);
    return out;
}
```

**Load:** single `std::fread` into arena, < 5 ms. Only `<vector>`, `<string>`, `<cstdint>`, `<cstdio>` used.

### 6.4 Artifact Integrity Validation
- All loaders (`.abn`, `.abt`, `.afr`) must validate:
  - `magic` (8-byte identifier)
  - `version` (exact format match)
  - `payload_size` (must match file payload length)
  - `checksum` (FNV-1a over payload bytes)
- Loader behavior on failure:
  - Reject artifact
  - Return empty/invalid structure
  - Do not continue with partially loaded data
- Conversion behavior:
  - Same input produces byte-identical output artifacts
  - Any nondeterministic source (random centroid selection, unordered map traversal, unsorted tensor keys) is prohibited

---

## 7. Build System

- Single CMake project
- Targets: `debug` (assertions on, no AVX2 opts), `release` (full AVX2, LTO)
- CMake flag `AAEDN_TARGET=zen2|neon|scalar` controls ISA (zen2 default)
- All code compiled and profiled on target Ryzen 5 5500U
- **Build system documentation only** – actual CMakeLists.txt created during implementation phase

## 7.1 Release Packaging
- Tagged release workflow builds a Release binary from a clean checkout
- Release artifact naming is stable: `aaedn-linux-x86_64`
- SHA256 checksum manifest (`SHA256SUMS.txt`) is generated and verified before publishing
- Published release contains binary + checksum file only

## 7.2 Runtime Configuration and Logging
- `config.hpp/config.cpp` define `safe`, `balanced`, `max` profiles and deterministic parameter maps
- Runtime profile controls:
  - arena size
  - guardian thread count
  - elastic thread count
  - quantization aggressiveness
  - bandwidth cap
- `logger.hpp/logger.cpp` provide level-based structured logging with crash signal diagnostics
- Fatal signals must write a terminal error record to the configured log file before exit

---

## 8. Revision History

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-04-14 | Initial design locked |
| 1.1.0 | 2026-04-14 | Corrected arena size annotation. Weight size corrected to ~281 MB (from 1.2–2.0 GB). Decode bottleneck is AVX2 port utilization, not bandwidth. |
| 1.2.0 | 2026-04-14 | Added model architecture specification: LLaMA-style 24-layer 520M params, d_model=1024, 16 heads, 64 head_dim, 4096 FFN. Updated weight size to ~347 MB. |
| 1.3.0 | 2026-04-14 | Added model conversion pipeline: custom block-wise 4-bit quantization, tokenizer from HF, RAG embedding from first 4 layers. |
| 1.4.0 | 2026-04-14 | Confirmed documentation-only status: no C++ code included until green light. |
| 1.5.0 | 2026-04-14 | Revised decode throughput from 50-90 to 25-40 tokens/second (aspirational, to be validated). |
| 1.6.0 | 2026-04-14 | Added RAG embedding generation pipeline: mean-pool first 4 layers. Marked retrieval latency <1ms as aspirational. |
| 1.7.0 | 2026-04-14 | Verified KV cache calculation: exact 1.61 GB for 24-layer model. |
| 2.1.0 | 2026-04-15 | Added deterministic artifact pipeline and strict runtime loader validation requirements for .abn/.abt/.afr (magic, version, payload_size, checksum). |
| 2.2.0 | 2026-04-15 | Added release packaging architecture (tagged binary + SHA256), runtime profile subsystem design, structured logging design, crash diagnostics behavior, and rollback-safe artifact compatibility policy. |
