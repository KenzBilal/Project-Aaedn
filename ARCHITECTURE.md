# Architecture

Aaedn is built as a deterministic local runtime: fixed memory, explicit threading, and architecture-aware kernels.

## High-Level Flow

```text
User Prompt
  -> Tokenizer
  -> Agentic Shell (diverge/merge workflow)
  -> Branch-Aware KV Cache
  -> Transformer Forward Pass
  -> Quantized Compute Path
  -> Sampling
  -> Output
```

## Core Runtime Layers

1. **Input and Tokenization**
   - Converts prompt text into token IDs using local tokenizer artifacts.
   - Keeps runtime behavior local and deterministic.

2. **Agentic Shell**
   - Manages parallel reasoning branches within bounded resources.
   - Coordinates branch lifecycle and merge decisions.

3. **KV Cache Subsystem**
   - Maintains per-branch logical views over shared cache structures.
   - Reduces duplication pressure during multi-agent execution.

4. **Transformer Execution**
   - Runs attention and feed-forward stages with quantized kernels.
   - Uses architecture-aware math paths designed for AVX2/FMA-class CPUs.

5. **Sampling and Response**
   - Produces next-token output and streams decoded text.
   - Maintains predictable memory and control flow behavior.

## Memory Model

- Single fixed arena allocated at startup.
- Core runtime structures live inside arena-managed memory.
- Hot paths are designed to avoid dynamic allocation after initialization.
- Alignment constraints are enforced for vectorized compute safety.

## Thread Model

- **Decode Guardians:** latency-critical decode-oriented workers.
- **Elastic Compute:** opportunistic workers for heavier auxiliary compute.
- **SMT coordination:** contention-aware scheduling and parking strategy.
- **Bandwidth discipline:** control mechanisms to protect decode latency.

## Distinctive Design Choices

- **Branch-Aware KV Cache**
  - Shared physical layout with branch-specific logical context.
  - Enables parallel branch reasoning under tight memory budgets.

- **Custom RAG Index Path**
  - Retrieval structure optimized for local latency and memory constraints.
  - Designed for predictable behavior on commodity hardware.

- **Quantization-Centric Runtime**
  - Runtime and artifact formats are built around low-bit inference from the start.
  - Prioritizes practical throughput per watt and per GB RAM.

## Design Objective

Aaedn is engineered to deliver useful local intelligence on modest machines by combining strict systems constraints with transparent implementation.
