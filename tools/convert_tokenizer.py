#!/usr/bin/env python3

import argparse
import struct
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

MAGIC = b"AAEDNTOK"
VERSION = 2
HEADER_SIZE = 64


def fnv1a32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def parse_merges(tokenizer, vocab: Dict[str, int]) -> List[Tuple[int, int]]:
    merges = getattr(tokenizer, "merges", None)
    if not merges:
        return []
    pairs: List[Tuple[int, int]] = []
    for merge in merges:
        if not isinstance(merge, str):
            continue
        parts = merge.split(" ")
        if len(parts) != 2:
            parts = merge.split("##")
        if len(parts) != 2:
            continue
        a = vocab.get(parts[0], 0)
        b = vocab.get(parts[1], 0)
        pairs.append((int(a) & 0xFFFF, int(b) & 0xFFFF))
    return pairs


def build_payload(tokenizer) -> Tuple[bytes, int, int, int]:
    vocab = tokenizer.get_vocab()
    sorted_tokens = sorted(vocab.items(), key=lambda kv: kv[1])
    vocab_size = len(sorted_tokens)

    token_bytes = [tok.encode("utf-8") for tok, _ in sorted_tokens]
    max_token_bytes = max((len(t) for t in token_bytes), default=0)

    offsets: List[int] = []
    strings = bytearray()
    for tb in token_bytes:
        offsets.append(len(strings))
        strings.extend(tb)
        strings.append(0)

    merge_pairs = parse_merges(tokenizer, vocab)
    merge_count = len(merge_pairs)

    payload = bytearray()
    payload.extend(struct.pack("<IIII", vocab_size, max_token_bytes, merge_count, len(strings)))
    payload.extend(strings)
    for off in offsets:
        payload.extend(struct.pack("<I", off))
    for a, b in merge_pairs:
        payload.extend(struct.pack("<HH", a, b))
    return bytes(payload), vocab_size, max_token_bytes, merge_count


def convert_tokenizer(hf_path: str, output_path: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    payload, vocab_size, max_token_bytes, merge_count = build_payload(tokenizer)
    payload_size = len(payload)
    checksum = fnv1a32(payload)

    header = struct.pack(
        "<8sIQI4I24s",
        MAGIC,
        VERSION,
        payload_size,
        checksum,
        vocab_size,
        max_token_bytes,
        merge_count,
        len(payload),
        b"\x00" * 24,
    )
    if len(header) != HEADER_SIZE:
        raise ValueError("tokenizer header size mismatch")

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(payload)

    print(f"Wrote {output_path}")
    print(f"header.version={VERSION} payload_size={payload_size} checksum={checksum:08x}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", required=True, help="HF tokenizer path or name")
    parser.add_argument("--output", required=True, help="Output .abt path")
    args = parser.parse_args()
    convert_tokenizer(args.hf, args.output)
