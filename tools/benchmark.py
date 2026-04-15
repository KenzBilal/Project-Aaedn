#!/usr/bin/env python3

import argparse
import json
import os
import re
import statistics
import subprocess
import time


def run_once(command: str):
    start = time.perf_counter()
    proc = subprocess.run(command, shell=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    return proc.returncode, proc.stdout + proc.stderr, elapsed


def parse_tokens_per_sec(output: str):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*tok/s", output)
    if not m:
        return None
    return float(m.group(1))


def bytes_of(path: str):
    if not os.path.exists(path):
        return None
    return os.path.getsize(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", default="./aaedn --temp 0.8 --topp 0.95")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--min-tok-s", type=float, default=25.0)
    parser.add_argument("--max-model-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--max-tokenizer-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-rag-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--out", default="benchmark-results.json")
    args = parser.parse_args()

    model_bytes = bytes_of("model.abn")
    tok_bytes = bytes_of("tokenizer.abt")
    rag_bytes = bytes_of("index.afr")

    failures = []
    if model_bytes is not None and model_bytes > args.max_model_bytes:
        failures.append(f"model.abn exceeds cap ({model_bytes} > {args.max_model_bytes})")
    if tok_bytes is not None and tok_bytes > args.max_tokenizer_bytes:
        failures.append(f"tokenizer.abt exceeds cap ({tok_bytes} > {args.max_tokenizer_bytes})")
    if rag_bytes is not None and rag_bytes > args.max_rag_bytes:
        failures.append(f"index.afr exceeds cap ({rag_bytes} > {args.max_rag_bytes})")

    runs = []
    tok_s = []
    for _ in range(args.runs):
        rc, output, elapsed = run_once(args.command)
        rate = parse_tokens_per_sec(output)
        runs.append({"return_code": rc, "elapsed_s": elapsed, "tok_s": rate})
        if rc != 0:
            failures.append("benchmark command returned non-zero exit code")
        if rate is not None:
            tok_s.append(rate)

    median_tok = statistics.median(tok_s) if tok_s else None
    if median_tok is None:
        failures.append("no tok/s metric found in command output")
    elif median_tok < args.min_tok_s:
        failures.append(f"decode throughput gate failed ({median_tok:.2f} < {args.min_tok_s:.2f})")

    result = {
        "command": args.command,
        "runs": runs,
        "median_tok_s": median_tok,
        "artifact_sizes": {
            "model.abn": model_bytes,
            "tokenizer.abt": tok_bytes,
            "index.afr": rag_bytes,
        },
        "failures": failures,
        "passed": len(failures) == 0,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        raise SystemExit(1)

    print("[PASS] benchmark gates satisfied")


if __name__ == "__main__":
    main()
