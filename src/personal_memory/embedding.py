"""Deterministic lexical embeddings for an offline, reproducible baseline."""

import hashlib
import math
from operator import mul
import re
import struct

DIMENSIONS = 256
EMBEDDING_ID = "blake2b-word-bigram-256-v1"


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)


def embed(text: str) -> list[float]:
    words = tokens(text)
    features = words + [a + " " + b for a, b in zip(words, words[1:])]
    result = [0.0] * DIMENSIONS
    for term in features:
        digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
        position = int.from_bytes(digest[:4], "little") % DIMENSIONS
        result[position] += 1.0 if digest[4] & 1 else -1.0
    length = math.sqrt(sum(x * x for x in result))
    return [x / length for x in result] if length else result


def pack(vector: list[float]) -> bytes:
    return struct.pack("<" + "f" * DIMENSIONS, *vector)


def unpack(blob: bytes) -> tuple[float, ...]:
    return struct.unpack("<" + "f" * DIMENSIONS, blob)


def cosine_distance(a: list[float], b: tuple[float, ...]) -> float:
    # map(mul) adds the same products in the same order as a generator, only faster.
    aa = sum(map(mul, a, a))
    bb = sum(map(mul, b, b))
    if not aa or not bb:
        return 1.0
    return max(0.0, min(2.0, 1 - sum(map(mul, a, b)) / math.sqrt(aa * bb)))
