# UI/UX — Interface & Interaction Design Document
## Project Aaedn
**Version:** 2.2.0 | **Status:** Complete | **Last Updated:** 2026-04-15

---

## 1. Interface Philosophy

Aaedn has no graphical interface. The interface is the terminal. Every interaction, diagnostic, and control surface is text-based, designed for a single developer who understands what is happening inside the engine at all times. No abstraction layers, no progress spinners hiding detail, no friendly wrappers over raw state.

The interface must satisfy two users simultaneously: the developer debugging internals, and the user sending prompts and reading responses.

---

## 2. Startup Sequence (Terminal Output)

On launch, Aaedn prints a structured startup report to stdout. Format is fixed-width, machine-readable if piped.

```
AAEDN v1.0.0 — AMD Ryzen 5 5500U / Arch Linux
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[ARENA]  Allocated   5632 MB at 0x7f3a00000000 (32-byte aligned)
[MODEL]  Loaded      model.abn  520M params  347 MB  4-bit
[TOK]    Loaded      tokenizer.abt  32768 vocab  1.98 MB  4.2 ms
[RAG]    Loaded      index.afr  1000000 tokens  224 MB  38.1 ms
[THR]    Guardians   CPUs 0,2,4,6 (pinned)
[THR]    Elastic     CPUs 1,3,5,7,8,9,10,11 (pinned)
[BW]     Pause count 142 (calibrated, decode cap 9 GB/s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[OK]     Ready.  Peak RAM budget remaining: 1.51 GB
```

**Rules:**
- Tags are always 6 characters, left-bracket, right-bracket, two spaces
- Numbers are right-aligned within their field
- Any failure line begins with `[FAIL]` and causes non-zero exit
- No color codes unless `AAEDN_COLOR=1` env var is set

---

## 3. Interactive Prompt Interface

### 3.1 Prompt Mode (default)

After startup, Aaedn enters interactive mode:

```
> _
```

Single `> ` prefix. User types prompt and presses Enter. Aaedn streams tokens to stdout as they are decoded, one character at a time (no buffering until EOS).

```
> What is the capital of France?
Paris.
> _
```

### 3.2 Streaming Behavior
- Tokens printed immediately as decoded — no line buffering
- EOS token triggers newline + new `> ` prompt
- If agent is using RAG, no indicator is shown unless `--verbose` flag is active
- If agent spawns sub-agents, no indicator shown unless `--verbose`

### 3.3 Multi-line Input
- Input ending with `\` continues on next line
- Blank line submits
```
> Tell me about the three laws \
  of thermodynamics.
```

### 3.4 Special Commands (prefix with `/`)

| Command | Action |
|---|---|
| `/quit` or `/q` | Graceful shutdown (flushes all threads, frees arena) |
| `/reset` | Clears KV cache, resets all agents to fresh state |
| `/stat` | Print current runtime statistics (see §4) |
| `/agents` | Print current agent branch states |
| `/ram` | Print current RAM usage breakdown |
| `/bw` | Print current bandwidth controller state |
| `/verbose` | Toggle verbose mode on/off |
| `/ctx` | Print current context token count |

---

## 4. Runtime Statistics (`/stat` output)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[STAT]  Uptime          00:04:23
[STAT]  Tokens decoded  1842
[STAT]  Decode speed    67.3 tok/s  (last 64 tokens)
[STAT]  Prefill speed   1240 tok/s  (last prefill)
[STAT]  KV tokens       4096 / 16384  (25%)
[STAT]  Agents active   3
[STAT]  RAG queries     14  avg 0.47 ms
[STAT]  Quant profile   L0:4bit L1:4bit L2:3bit L3:4bit ...
[STAT]  BW Elastic      6.2 GB/s  (cap: 9.0)
[STAT]  Arena used      4.21 GB / 5.50 GB  (76%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 5. Verbose Mode (`--verbose` flag or `/verbose`)

Verbose mode adds inline annotations during generation:

```
> Summarize the attached document.
[RAG] query 0.41ms → 8 segments retrieved (tokens 14200–14891, 22100–22340)
[AGT] agent_0 → Reason: summarize document
[AGT] agent_1 → Act: retrieve(section=introduction)
[AGT] agent_0 → Observe: retrieved 512 tokens
[AGT] merge: agent_1 branch pruned, compaction 2.1ms
The document describes...
```

Verbose annotations are prefixed with `[RAG]`, `[AGT]`, `[BW]`, `[KV]`. Annotations go to stderr so stdout remains clean for piping.

---

## 6. Error Output

All errors go to stderr. Format:

```
[ERROR] KV cache full: 16384/16384 tokens used. Use /reset to clear.
[ERROR] Model file model.abn: magic mismatch (expected AAEDNMDL, got AAEDNOLD)
[WARN]  Decode throughput 31.2 tok/s below target (50). Check thermal throttle.
```

`[ERROR]` = fatal for that request, engine continues.
`[WARN]` = non-fatal advisory, engine continues.
`[FAIL]` = startup failure only, causes exit(1).

---

## 7. File Arguments & Flags

```
Usage: aaedn [OPTIONS]

Options:
  --model   PATH     Path to .abn model file (required)
  --tok     PATH     Path to .abt tokenizer file (required)
  --rag     PATH     Path to .afr RAG index file (optional)
  --arena   MB       Arena size in MB (default: 5632)
  --ctx     TOKENS   Max context length (default: 16384, max: 16384)
  --agents  N        Number of micro-agents (default: 4, max: 6)
  --verbose          Enable verbose agent/RAG annotations on stderr
  --color            Enable ANSI color output
  --no-rag           Disable RAG even if index file provided
  --seed    N        Sampling seed (default: 0 = random)
  --temp    F        Sampling temperature (default: 0.8)
  --topp    F        Top-p nucleus sampling threshold (default: 0.95)
  --profile MODE     Runtime profile: safe|balanced|max
```

Profile semantics:
- `safe`: lower arena + lower thread count + conservative quantization + lower bandwidth cap
- `balanced`: default production profile
- `max`: highest throughput profile within hardware covenant

---

## 8. Binary File Format Identifiers

Each binary file has a magic header so the engine fails fast on wrong file type:

| File Type | Extension | Magic |
|---|---|---|
| Model weights | `.abn` | `AAEDNMDL` |
| Tokenizer | `.abt` | `AAEDNTOK` |
| RAG index | `.afr` | `AAEDNRAG` |

Runtime validates all three files for:
- `magic` match
- `version` match
- `payload_size` match
- `checksum` match

---

## 9. Exit Codes

| Code | Meaning |
|---|---|
| 0 | Clean exit via `/quit` or EOF |
| 1 | Startup failure (model/tokenizer load failed) |
| 2 | Arena allocation failed |
| 3 | Thread pinning failed |
| 130 | SIGINT (Ctrl+C) |

---

## 10. Developer Debug Mode (`AAEDN_DEBUG=1`)

When env var `AAEDN_DEBUG=1` is set:
- Every forward pass prints layer-by-layer entropy oracle decisions
- Every KV append prints physical token index and branch ID
- Every RAG query prints coarse centroid scores and fine scan count
- Thread park/unpark events printed to stderr with TSC timestamp
- Arena watermark printed after every forward pass

This mode is never enabled in normal use. It is the complete internal visibility surface for debugging.

## 10.1 Structured Logging and Crash Diagnostics

Runtime log format:
`YYYY-MM-DDTHH:MM:SSZ [LEVEL] message`

Levels:
- `DEBUG`
- `INFO`
- `WARN`
- `ERROR`

Crash diagnostics:
- On fatal signals (`SIGSEGV`, `SIGABRT`, `SIGILL`, `SIGFPE`), runtime writes an `ERROR` record to log file before process exit.
- Log file path is configured at startup and retained across session for post-mortem inspection.

---

## 11. Revision History

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-04-14 | Initial UI/UX spec locked |
| 2.1.0 | 2026-04-15 | Added deterministic artifact validation behavior to interface contract and startup/load failure rules for version/integrity mismatch. |
| 2.2.0 | 2026-04-15 | Added profile CLI (`safe|balanced|max`), structured logging level contract, and crash diagnostics behavior with persisted fatal signal records. |
