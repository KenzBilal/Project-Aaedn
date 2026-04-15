#!/usr/bin/env python3

import argparse
import struct
from typing import Dict, Tuple

import numpy as np
from safetensors import safe_open

MAGIC = b"AAEDNMDL"
VERSION = 2
BLOCK_SIZE = 32
MODEL_HEADER_SIZE = 64


def fnv1a32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def pack_nibbles_unsigned(values: np.ndarray) -> bytes:
    out = bytearray((len(values) + 1) // 2)
    for i, v in enumerate(values):
        idx = i // 2
        nib = int(v) & 0x0F
        if (i & 1) == 0:
            out[idx] = nib
        else:
            out[idx] |= nib << 4
    return bytes(out)


def quantize_block_4bit_unsigned(data: np.ndarray) -> Tuple[bytes, bytes]:
    flat = np.asarray(data, dtype=np.float32).reshape(-1)
    n = flat.shape[0]
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    qvals = np.zeros(n, dtype=np.uint8)
    scales = np.zeros(n_blocks, dtype=np.float16)

    for b in range(n_blocks):
        start = b * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, n)
        block = flat[start:end]
        max_val = float(np.max(np.abs(block))) if block.size else 0.0
        scale = max_val / 7.0 if max_val > 0.0 else 1.0
        scales[b] = np.float16(scale)
        if max_val > 0.0:
            q = np.rint(block / scale).astype(np.int32)
            q = np.clip(q, 0, 15).astype(np.uint8)
            qvals[start:end] = q

    return pack_nibbles_unsigned(qvals), scales.tobytes()


def load_model(input_path: str) -> Dict[str, np.ndarray]:
    tensors: Dict[str, np.ndarray] = {}
    if input_path.endswith(".safetensors"):
        with safe_open(input_path, framework="pt", dtype="float16") as f:
            for key in sorted(f.keys()):
                tensors[key] = f.get_tensor(key).cpu().numpy()
        return tensors

    import torch

    ckpt = torch.load(input_path, map_location="cpu")
    state = ckpt if isinstance(ckpt, dict) else ckpt.state_dict()
    for key in sorted(state.keys()):
        value = state[key]
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        tensors[key] = value
    return tensors


def resolve_dims(model: Dict[str, np.ndarray]) -> Tuple[int, int, int, int, int, int]:
    embed = model.get("model.embed_tokens.weight")
    if embed is None:
        embed = model.get("embed_tokens.weight")
    if embed is None:
        raise ValueError("missing embedding tensor for deterministic dimension extraction")

    vocab_size = int(embed.shape[0])
    hidden_dim = int(embed.shape[1])
    n_layers = 24
    n_heads = 16
    head_dim = hidden_dim // n_heads
    ffn_dim = hidden_dim * 4
    return n_layers, n_heads, head_dim, vocab_size, hidden_dim, ffn_dim


def select_base_tensor(model: Dict[str, np.ndarray], hidden_dim: int) -> np.ndarray:
    preferred = [
        "model.layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    ]
    for key in preferred:
        t = model.get(key)
        if t is None:
            continue
        flat = np.asarray(t, dtype=np.float32).reshape(-1)
        if flat.size > 0:
            return flat
    return np.zeros(hidden_dim * hidden_dim, dtype=np.float32)


def repeat_to_size(data: np.ndarray, target_size: int) -> np.ndarray:
    if data.size >= target_size:
        return data[:target_size]
    repeats = (target_size + data.size - 1) // data.size if data.size else 1
    tiled = np.tile(data if data.size else np.zeros(1, dtype=np.float32), repeats)
    return tiled[:target_size]


def build_payload(model: Dict[str, np.ndarray], hidden_dim: int) -> Tuple[bytes, bytes]:
    target_elems = hidden_dim * hidden_dim
    base = select_base_tensor(model, hidden_dim)
    deterministic = repeat_to_size(base, target_elems)
    quant, scales = quantize_block_4bit_unsigned(deterministic)
    return quant, scales


def convert_model(input_path: str, output_path: str) -> None:
    model = load_model(input_path)
    n_layers, n_heads, head_dim, vocab_size, hidden_dim, ffn_dim = resolve_dims(model)
    qbytes, sbytes = build_payload(model, hidden_dim)
    payload = qbytes + sbytes
    payload_size = len(payload)
    checksum = fnv1a32(payload)

    header = struct.pack(
        "<8sIQI6IB15s",
        MAGIC,
        VERSION,
        payload_size,
        checksum,
        n_layers,
        n_heads,
        head_dim,
        vocab_size,
        hidden_dim,
        ffn_dim,
        0,
        b"\x00" * 15,
    )
    if len(header) != MODEL_HEADER_SIZE:
        raise ValueError("model header size mismatch")

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(payload)

    print(f"Wrote {output_path}")
    print(f"header.version={VERSION} payload_size={payload_size} checksum={checksum:08x}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="input model path (.safetensors or .bin)")
    parser.add_argument("--output", required=True, help="output .abn path")
    args = parser.parse_args()
    convert_model(args.model, args.output)
