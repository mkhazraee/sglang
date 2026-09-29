# SPDX-License-Identifier: Apache-2.0
"""Object keys, compatibility identity and inventory hashes for the KVCR linker."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    from kvcr.types import BlockKey
else:
    # kvcr's BlockKey is a NewType over bytes; aliasing keeps this module
    # importable without the wheel.
    BlockKey = bytes

# Storage keys are ``<page-hash>#kvcr-linker-v1#<digest>#<pool>``. The router
# indexes the leading 16 hex chars of the page hash as an int64, so that prefix
# is the only identity the two sides share; the full key stays authoritative.
KEY_NAMESPACE = "kvcr-linker-v1"
_KEY_SEPARATOR = "#"
_EVENT_HASH_HEX_WIDTH = 16


def encode_object_key(page_hash: str, digest: str, pool: str) -> BlockKey:
    """One KVCR object per physical pool page."""
    return BlockKey(
        _KEY_SEPARATOR.join((page_hash, KEY_NAMESPACE, digest, pool)).encode("utf-8")
    )


def decode_object_key(key: bytes) -> tuple[str, str, str, str]:
    """``(page_hash, namespace, digest, pool)`` of an object key."""
    parts = key.decode("utf-8").split(_KEY_SEPARATOR, 3)
    if len(parts) != 4:
        raise ValueError(f"malformed KVCR linker key: {key!r}")
    return parts[0], parts[1], parts[2], parts[3]


def page_hash_event_int(page_hash: str) -> int:
    """Unsigned 64-bit event identity of a full page hash."""
    return int(page_hash[:_EVENT_HASH_HEX_WIDTH], 16)


def page_hash_to_int64(page_hash: str) -> int:
    """Signed int64 event hash, matching ``hash_str_to_int64``."""
    value = page_hash_event_int(page_hash)
    return value - (1 << 64) if value >= 1 << 63 else value


def compatibility_digest(identity: Mapping[str, Any]) -> str:
    """Short digest over the byte-compatibility identity of a cache layout.

    Two ranks whose digests agree hold byte-identical KV for the same page
    hash. The caller supplies a JSON-serializable mapping; ordering is
    canonicalized here so field insertion order cannot change the digest.
    """
    canonical = json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_EVENT_HASH_HEX_WIDTH]


def unique_page_hashes_from_keys(keys: Iterable[bytes]) -> list[str]:
    """Page hashes named by a sequence of object keys, deduplicated in order."""
    seen: dict[str, None] = {}
    for key in keys:
        try:
            page_hash = decode_object_key(key)[0]
        except (UnicodeDecodeError, ValueError):
            continue
        seen.setdefault(page_hash, None)
    return list(seen)
