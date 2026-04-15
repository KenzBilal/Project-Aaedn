#!/usr/bin/env python3

import argparse
import hashlib
import struct
from typing import List

import numpy as np

MAGIC = b"AAEDNRAG"
VERSION = 2
N_DIMS = 384
N_CENTROIDS = 256
TOKEN_STRIDE = 224
HEADER_SIZE = 64


def fnv1a32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def deterministic_embedding(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(N_DIMS, dtype=np.float32).astype(np.float16)


def deterministic_centroids(embeddings: np.ndarray) -> np.ndarray:
    n_tokens = embeddings.shape[0]
    centroids = np.zeros((N_CENTROIDS, N_DIMS), dtype=np.float16)
    if n_tokens == 0:
        return centroids
    for i in range(N_CENTROIDS):
        idx = (i * n_tokens) // N_CENTROIDS
        if idx >= n_tokens:
            idx = n_tokens - 1
        centroids[i] = embeddings[idx]
    return centroids


def build_payload(documents: List[str]) -> bytes:
    cleaned = [d.strip() for d in documents if d.strip()]
    embeddings = [deterministic_embedding(doc) for doc in cleaned]
    token_data_rows = [doc.encode("utf-8")[:TOKEN_STRIDE].ljust(TOKEN_STRIDE, b"\x00") for doc in cleaned]
    n_tokens = len(embeddings)

    if n_tokens == 0:
        all_emb = np.zeros((1, N_DIMS), dtype=np.float16)
        token_data_rows = [b"\x00" * TOKEN_STRIDE]
        n_tokens = 1
    else:
        all_emb = np.stack(embeddings).astype(np.float16)

    centroids = deterministic_centroids(all_emb)
    bucket_offsets = [(i * n_tokens) // N_CENTROIDS for i in range(N_CENTROIDS)]

    payload = bytearray()
    payload.extend(struct.pack("<4I", n_tokens, N_DIMS, N_CENTROIDS, TOKEN_STRIDE))
    payload.extend(centroids.tobytes(order="C"))
    for off in bucket_offsets:
        payload.extend(struct.pack("<I", off))
    for row in token_data_rows:
        payload.extend(row)
    return bytes(payload)


def generate_rag_index(_model_path: str, docs_path: str, output_path: str) -> None:
    with open(docs_path, "r", encoding="utf-8") as f:
        documents = f.read().split("\n\n")

    payload = build_payload(documents)
    payload_size = len(payload)
    checksum = fnv1a32(payload)

    header = struct.pack(
        "<8sIQI4I24s",
        MAGIC,
        VERSION,
        payload_size,
        checksum,
        0,
        0,
        0,
        0,
        b"\x00" * 24,
    )
    if len(header) != HEADER_SIZE:
        raise ValueError("rag header size mismatch")

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(payload)

    print(f"Wrote {output_path}")
    print(f"header.version={VERSION} payload_size={payload_size} checksum={checksum:08x}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model path (kept for interface stability)")
    parser.add_argument("--documents", required=True, help="input documents path")
    parser.add_argument("--output", required=True, help="output .afr path")
    args = parser.parse_args()
    generate_rag_index(args.model, args.documents, args.output)
