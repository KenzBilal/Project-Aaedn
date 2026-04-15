# FBD — Feature Breakdown Document
## Project Aaedn
**Version:** 1.6.0 | **Status:** Complete | **Last Updated:** 2026-04-15

---

## Revision History
| Version | Date | Change |
|---------|------|--------|
| 1.6.0 | 2026-04-15 | Added release packaging and operational reliability features: runtime profiles, structured logging, crash diagnostics, reproducible tagged release workflow |
| 1.5.0 | 2026-04-15 | Added production validation and release hardening test plan entries T-15 to T-18 |
| 1.4.0 | 2026-04-15 | Added deterministic artifact pipeline feature set with mandatory header integrity fields and runtime validation checks |
| 1.3.0 | 2026-04-15 | Project complete - all 22 features implemented |
| 1.2.0 | 2026-04-14 | Phase 3 transformer core features |
| 1.1.0 | 2026-04-14 | Phase 2 threading features |
| 1.0.0 | 2026-04-14 | Phase 1 mathematical engine |

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-04-14 | Initial feature breakdown locked |
| 1.1.0 | 2026-04-14 | Added model architecture specification: LLaMA-style 24-layer 520M params, d_model=1024, 16 heads, 64 head_dim, 4096 FFN. Updated weight size to ~347 MB. |
| 1.2.0 | 2026-04-14 | Added model conversion pipeline: custom block-wise 4-bit quantization, tokenizer from HF, RAG embedding from first 4 layers. |
| 1.3.0 | 2026-04-14 | Confirmed documentation-only status: no C++ code included until green light. Added Phase 0 documentation status section. |
| 1.4.0 | 2026-04-14 | Revised decode throughput from 50-90 to 25-40 tokens/second (aspirational, to be validated). Updated F-18/F-19 test thresholds. |
| 1.5.0 | 2026-04-14 | Added RAG embedding generation pipeline: mean-pool first 4 layers (early-layer pooling). Marked retrieval latency <1ms as aspirational. |
| 1.6.0 | 2026-04-14 | Verified KV cache calculation: exact 1.61 GB for 24-layer model. Updated total budget to ~2.82 GB. |
| 1.7.0 | 2026-04-15 | Added deterministic artifact pipeline feature requirement: reproducible `.abn/.abt/.afr` generation and strict loader verification for magic, version, payload size, and checksum. |

## Test Plan
| ID | Test | Command | Pass Criterion |
|---|---|---|---|
| T-15 | Deterministic artifact round-trip | `./test_artifact_roundtrip` | .abn/.abt/.afr fixtures load correctly and remain byte-identical |
| T-16 | Startup/load failure modes | `./test_startup_failure` | Invalid magic/checksum artifacts fail cleanly without partial state |
| T-17 | Benchmark automation gates | `python tools/benchmark.py --command "./aaedn --temp 0.8 --topp 0.95"` | JSON report generated and configured gates pass |
| T-18 | CI integrity and reproducibility | `.github/workflows/ci.yml` | CI runs tests, integrity checks, and reproducibility checks on PR/push |
| T-19 | Runtime profile behavior | `./aaedn --profile safe|balanced|max` | arena/thread/bandwidth/quant settings match profile map |
| T-20 | Crash diagnostics log persistence | crash injection test | fatal signal writes structured error line to log file |
| T-21 | Tagged release reproducibility | `.github/workflows/release.yml` | release binary + SHA256 checksum are published and checksum verifies |
