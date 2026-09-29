# SPDX-License-Identifier: Apache-2.0
"""Map device pages to KVCR objects with labeled component/layer pieces.

Equal-size pieces share a pool; labels identify each piece within an object.
Geometry comes from ``DevicePoolEntry.buffer_meta``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import msgspec
import torch

from sglang.srt.mem_cache.storage.kvcr.router_hint import compatibility_digest

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
        DevicePoolEntry,
        DevicePoolGroup,
    )

# NIXL segment type per torch device type.
_MEM_TYPE_BY_DEVICE = {"cuda": "VRAM", "cpu": "DRAM"}


def nixl_mem_type(device: torch.device) -> str:
    mem_type = _MEM_TYPE_BY_DEVICE.get(device.type)
    if mem_type is None:
        raise ValueError(
            f"KVCR linker cannot address {device.type!r} memory; supported "
            f"device types: {sorted(_MEM_TYPE_BY_DEVICE)}."
        )
    return mem_type


def nixl_device_id(device: torch.device) -> int:
    return device.index if device.index is not None else 0


class PoolObjectLayout(msgspec.Struct, frozen=True, kw_only=True):
    """How one physical pool's page maps to a KVCR object."""

    pool: str
    mem_type: str
    device_id: int
    # Pool:part label per span, in descriptor order.
    labels: tuple[str, ...]
    # (base_ptr, row_stride_bytes, span_bytes) per span.
    spans: tuple[tuple[int, int, int], ...]

    @property
    def span_sizes(self) -> tuple[int, ...]:
        return tuple(size for _, _, size in self.spans)

    @property
    def object_bytes(self) -> int:
        return sum(self.span_sizes)

    @property
    def expected_layout(self) -> list[str]:
        return list(self.labels)


class CapacityPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved per-rank KVCR DRAM allocation."""

    budget_bytes: int
    # Capacity of the all-pages anchor, in cache pages.
    page_capacity: int
    # Physical-object capacity per pool. Sparse checkpoint pools can have a
    # smaller capacity than the all-pages KV anchor.
    pool_capacities: dict[str, int]
    # (pool name, slot bytes) in pool_layouts order.
    pool_layouts: tuple[tuple[str, int], ...]
    # Pool name -> byte length of its region.
    pool_bytes: dict[str, int]
    total_bytes: int

    @property
    def unused_bytes(self) -> int:
        return self.budget_bytes - self.total_bytes


def build_pool_object_layouts(
    pool_group: DevicePoolGroup,
) -> dict[str, PoolObjectLayout]:
    """One object layout per physical pool entry, in group order."""
    layouts: dict[str, PoolObjectLayout] = {}
    for entry in pool_group.entries:
        layouts[str(entry.name)] = _layout_for_entry(entry)
    return layouts


def _layout_for_entry(entry: DevicePoolEntry) -> PoolObjectLayout:
    if not entry.packed:
        raise ValueError(
            f"KVCR linker requires packed device pools; pool {entry.name} "
            "stores components as separate objects."
        )
    devices = {buffer.device for buffer in entry.kv_buffer}
    if len(devices) != 1:
        raise ValueError(
            f"KVCR linker pool {entry.name} spans several devices: {devices}."
        )
    device = devices.pop()
    spans: list[tuple[int, int, int]] = []
    labels: list[str] = []
    mixed_sizes = (
        len({size for component in entry.buffer_meta for _, _, size in component}) > 1
    )
    for component_index, component in enumerate(entry.buffer_meta):
        for buffer_index, (base_ptr, row_stride, size) in enumerate(component):
            if size <= 0:
                raise ValueError(
                    f"KVCR linker pool {entry.name} has an empty span at "
                    f"component {component_index}, buffer {buffer_index}."
                )
            spans.append((int(base_ptr), int(row_stride), int(size)))
            pool = f"{entry.name}/{size}" if mixed_sizes else str(entry.name)
            labels.append(f"{pool}:{component_index}.{buffer_index}")
    return PoolObjectLayout(
        pool=str(entry.name),
        mem_type=nixl_mem_type(device),
        device_id=nixl_device_id(device),
        labels=tuple(labels),
        spans=tuple(spans),
    )


def plan_capacity(
    layouts: Mapping[str, PoolObjectLayout],
    budget_bytes: int,
    *,
    capacity_divisors: Mapping[str, int] | None = None,
) -> CapacityPlan:
    """Size physical pools within one per-rank DRAM budget.

    By default every pool stores one object per cache page. A sparse checkpoint
    pool can provide a divisor ``N`` to reserve one object per ``N`` anchor
    pages. This matters for hybrid Mamba models: the dense KV anchor is paged at
    64 tokens while resume state exists only on the 256-token checkpoint grid.
    Giving both pools the same object count wastes nearly the entire budget on
    duplicate checkpoint capacity and prematurely caps the KV prefix.
    """
    if budget_bytes <= 0:
        raise ValueError("KVCR linker DRAM budget must be positive.")
    if not layouts or sum(layout.object_bytes for layout in layouts.values()) <= 0:
        raise ValueError("KVCR linker pool layouts describe no bytes.")
    divisors = {name: 1 for name in layouts}
    for name, divisor in (capacity_divisors or {}).items():
        if name not in layouts:
            raise ValueError(f"Unknown KVCR capacity pool: {name}")
        if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor <= 0:
            raise ValueError(
                f"KVCR capacity divisor for pool {name} must be a positive integer."
            )
        divisors[name] = divisor

    def pool_capacity(pool: str, anchor_pages: int) -> int:
        divisor = divisors[pool]
        return (anchor_pages + divisor - 1) // divisor

    def required_bytes(anchor_pages: int) -> int:
        return sum(
            layout.object_bytes * pool_capacity(name, anchor_pages)
            for name, layout in layouts.items()
        )

    # Integer search avoids rounding a sparse pool's final partial interval.
    low, high = 0, 1
    while required_bytes(high) <= budget_bytes:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if required_bytes(middle) <= budget_bytes:
            low = middle
        else:
            high = middle
    page_capacity = low
    if page_capacity <= 0:
        raise ValueError(
            f"KVCR linker DRAM budget of {budget_bytes} bytes per rank holds no "
            "complete page across all pools. Raise "
            "local_dram_bytes_per_worker."
        )
    pool_capacities = {name: pool_capacity(name, page_capacity) for name in layouts}
    pool_layouts: dict[str, int] = {}
    pool_bytes: dict[str, int] = {}
    for name, layout in layouts.items():
        for label, size in zip(layout.labels, layout.span_sizes):
            pool = label.partition(":")[0]
            pool_layouts[pool] = size
            pool_bytes[pool] = pool_bytes.get(pool, 0) + size * pool_capacities[name]
    return CapacityPlan(
        budget_bytes=budget_bytes,
        page_capacity=page_capacity,
        pool_capacities=pool_capacities,
        pool_layouts=tuple(pool_layouts.items()),
        pool_bytes=pool_bytes,
        total_bytes=sum(pool_bytes.values()),
    )


def carve_local_dram(
    plan: CapacityPlan, buffer: torch.Tensor
) -> list[tuple[str, int, int]]:
    """``(name, address, length)`` per pool carved from one host buffer."""
    if buffer.numel() * buffer.element_size() < plan.total_bytes:
        raise ValueError("KVCR linker local DRAM buffer is smaller than the plan.")
    base = buffer.data_ptr()
    offset = 0
    regions = []
    for name, _ in plan.pool_layouts:
        length = plan.pool_bytes[name]
        regions.append((name, base + offset, length))
        offset += length
    return regions


def page_descriptors(
    layout: PoolObjectLayout, row: int, agent_name: str, descriptor_type: Any
) -> list:
    """Descriptor list for one page row of a pool, in layout order."""
    return [
        descriptor_type(
            end_point_name=agent_name,
            mem_type=layout.mem_type,
            addr=base_ptr + row * row_stride,
            size=size,
            device_Id=layout.device_id,
            info=label,
        )
        for (base_ptr, row_stride, size), label in zip(layout.spans, layout.labels)
    ]


def compatibility_identity(
    *,
    model_path: str,
    revision: str | None,
    dtype: str,
    kv_cache_dtype: str,
    quantization: str | None,
    page_size: int,
    is_eagle: bool,
    layouts: Mapping[str, PoolObjectLayout],
    shard: Mapping[str, int],
    speculative: Mapping[str, Any],
) -> dict[str, Any]:
    """Identity whose digest namespaces every stored key.

    Covers what changes the bytes of a page: checkpoint, KV representation,
    page geometry and physical span layout, target/draft identity, and the
    physical shard. Runtime addresses, worker identity, and pure DP replica
    identity are excluded so byte-compatible replicas share keys.
    """
    return {
        "namespace": "kvcr-linker-v1",
        "model_path": model_path,
        "revision": revision,
        "dtype": dtype,
        "kv_cache_dtype": kv_cache_dtype,
        "quantization": quantization,
        "page_size": page_size,
        "is_eagle": is_eagle,
        "pools": {
            name: {
                "mem_type": layout.mem_type,
                "labels": list(layout.labels),
                "span_sizes": list(layout.span_sizes),
            }
            for name, layout in layouts.items()
        },
        "shard": dict(shard),
        "speculative": dict(speculative),
    }


def compatibility_digest_for(identity: Mapping[str, Any]) -> str:
    return compatibility_digest(identity)


def restorable_boundaries(
    present: Mapping[str, Sequence[bool]],
    policies: Mapping[str, tuple[str, int]],
    num_pages: int,
) -> list[int]:
    """Prefix lengths (in pages) every pool can restore at that boundary.

    ``present[pool][i]`` says whether page ``i`` of ``pool`` is confirmed;
    ``policies[pool]`` is ``("all_pages", _)`` or ``("trailing_pages", window)``.
    Mirrors the UMBP hit policy so the tree's cross-rank intersection sees the
    same sparse set semantics.
    """
    valid = list(range(1, num_pages + 1))
    for pool, (policy, window) in policies.items():
        flags = list(present.get(pool, ()))
        flags.extend([False] * (num_pages - len(flags)))
        prefix = [0]
        for flag in flags[:num_pages]:
            prefix.append(prefix[-1] + int(flag))
        if policy == "all_pages":
            valid = [end for end in valid if prefix[end] == end]
        elif policy == "trailing_pages":
            span = max(1, window)
            valid = [
                end
                for end in valid
                if prefix[end] - prefix[max(0, end - span)] == end - max(0, end - span)
            ]
        else:
            raise ValueError(f"Unsupported pool hit policy: {policy}")
        if not valid:
            break
    return valid
