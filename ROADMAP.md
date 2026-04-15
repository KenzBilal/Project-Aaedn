# Roadmap

Goal: production-ready local multi-agent AI engine for constrained hardware.

Current version: **1.3.0 (Alpha)**

## Milestones

| Milestone | Status | Target Window | Outcome |
|---|---|---|---|
| Core inference engine | Complete | Done | End-to-end runtime skeleton and core modules implemented |
| Multi-agent and RAG integration | Complete | Done | Agent shell + retrieval path integrated into runtime |
| Dynamic hardware profiling | In Progress | Next | Auto-tuning hooks for CPU/RAM-aware scheduling |
| Model conversion tooling hardening | In Progress | Next | Stable conversion pipeline and artifact validation |
| Benchmark suite and examples | Planned | Soon | Reproducible performance and quality baselines |
| Production stability pass | Planned | Q3 2026 | Robustness, fault handling, and release confidence |
| Optional fine-tuning support | Planned | Future | Evaluate scope without compromising core constraints |

## Current Priorities

1. Improve matmul and attention kernel efficiency under real-world decode workloads.
2. Final-validate with production model artifacts and realistic prompt distributions.
3. Complete dynamic profiling without violating zero-allocation hot-path constraints.
4. Expand benchmark coverage for latency, memory ceiling, and throughput variance.

## Risks and Constraints

- Kernel optimization can regress numerical stability if not carefully guarded.
- Artifact conversion quality directly affects benchmark credibility.
- Dynamic adaptation must preserve deterministic behavior and memory discipline.
- Production readiness requires failure-mode testing, not just nominal performance.

## Exit Criteria for Production Readiness

- Stable reproducible benchmarks across repeated runs on target hardware.
- No allocator activity in critical decode path after startup.
- Verified memory ceiling under representative multi-agent workloads.
- Documented recovery behavior for malformed inputs and runtime faults.
