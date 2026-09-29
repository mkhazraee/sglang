# SPDX-License-Identifier: Apache-2.0
"""Configuration for the KVCR direct linker.

Parsed from ``--hicache-storage-backend-extra-config`` (JSON, or ``@file``),
the same channel the Mooncake and UMBP linkers use despite its name.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import msgspec


class KVCRLinkerConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Operator-facing settings for ``--unified-cache-external-linker-backend kvcr``."""

    # Total KVCR-owned DRAM for this worker; each local scheduler rank gets an
    # equal share. Required: there is no sensible default for a cache tier.
    local_dram_bytes_per_worker: int
    pin_local_dram: bool = True
    nixl_backend: str = "UCX"
    # KVCR core knobs.
    operation_timeout_ms: int = 20000
    abandon_timeout_ms: int = 60000

    # Preparation bounds. A request stops waiting at the deadline and admits
    # whatever prefix was confirmed; late completions are drained afterwards.
    preparation_deadline_ms: int = 2000
    max_inflight_prepare_requests: int = 64
    max_inflight_prepare_bytes: int = 8 << 30
    max_prepare_bytes_per_request: int = 2 << 30
    fetch_chunk_pages: int = 32
    # Bound the amount of work submitted by a single offload operation.
    offload_chunk_pages: int = 8
    # Offloads beyond this many in-flight bytes are declined; the tree retries.
    max_inflight_offload_bytes: int = 8 << 30
    # Late (abandoned) work above this stops new preparation until it drains.
    max_abandoned_bytes: int = 4 << 30
    # Owner-thread poll interval while KVCR operations are in flight; shorter
    # finishes small transfers sooner.
    poll_interval_ms: float = 0.5
    stats_log_interval_s: float = 30.0

    def __post_init__(self) -> None:
        if self.local_dram_bytes_per_worker <= 0:
            raise ValueError("KVCR linker requires local_dram_bytes_per_worker > 0.")
        if self.operation_timeout_ms <= 0:
            raise ValueError("KVCR operation_timeout_ms must be positive.")
        if self.abandon_timeout_ms < 2 * self.operation_timeout_ms:
            raise ValueError(
                "KVCR abandon_timeout_ms must be at least twice operation_timeout_ms."
            )
        if self.preparation_deadline_ms <= 0:
            raise ValueError("KVCR preparation_deadline_ms must be positive.")
        if self.fetch_chunk_pages <= 0:
            raise ValueError("KVCR fetch_chunk_pages must be positive.")
        if self.offload_chunk_pages <= 0:
            raise ValueError("KVCR offload_chunk_pages must be positive.")
        for name in (
            "max_inflight_prepare_requests",
            "max_inflight_prepare_bytes",
            "max_prepare_bytes_per_request",
            "max_inflight_offload_bytes",
            "max_abandoned_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"KVCR {name} must be positive.")
        if self.poll_interval_ms <= 0:
            raise ValueError("KVCR poll_interval_ms must be positive.")

    @classmethod
    def from_extra_config(
        cls, extra_config: Optional[Mapping[str, Any]]
    ) -> KVCRLinkerConfig:
        extra_config = dict(extra_config or {})
        known = set(cls.__struct_fields__)
        unknown = sorted(set(extra_config) - known)
        if unknown:
            raise ValueError(
                f"KVCR linker config has unknown options {unknown}; known "
                f"options: {sorted(known)}."
            )
        if "local_dram_bytes_per_worker" not in extra_config:
            raise ValueError(
                "KVCR linker config requires local_dram_bytes_per_worker in "
                "--hicache-storage-backend-extra-config."
            )
        try:
            return msgspec.convert(extra_config, cls)
        except msgspec.ValidationError as error:
            raise ValueError(f"KVCR linker config is invalid: {error}") from error
