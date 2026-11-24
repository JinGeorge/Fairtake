"""Seeds and shuffled orders (spec §3, "Seeds").

The production seed comes from a SHA-256 hash of the poll id, so it is the same
in every process. Python's built-in ``hash()`` is randomised per process and
must not be used for this.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence


def base_seed_for(poll_id: str) -> int:
    digest = hashlib.sha256(poll_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def call_seeds(base_seed: int) -> dict[str, int]:
    """Pass A uses the base seed, pass B base + 1, Call 1 base + 2."""
    return {"pass_a": base_seed, "pass_b": base_seed + 1, "call1": base_seed + 2}


def shuffled(items: Sequence[str], seed: int) -> list[str]:
    out = list(items)
    random.Random(seed).shuffle(out)
    return out
