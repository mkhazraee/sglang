"""Hybrid geometry and progressive delivery regressions (no GPU required)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.mem_cache.storage.kvcr.kvcr_layout import plan_capacity
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_sparse_checkpoint_capacity_uses_physical_object_counts():
    layouts = {
        "kv": SimpleNamespace(object_bytes=8, subpools=("kv/0",), span_sizes=(8,)),
        "mamba": SimpleNamespace(object_bytes=32, subpools=("m/0",), span_sizes=(32,)),
    }
    dense = plan_capacity(layouts, 128)
    sparse = plan_capacity(layouts, 128, capacity_divisors={"mamba": 4})
    assert dense.pool_capacities == {"kv": 3, "mamba": 3}
    assert sparse.pool_capacities == {"kv": 8, "mamba": 2}
    assert sparse.total_bytes == 128
    # Crossing a checkpoint interval requires an entire additional state object.
    assert (
        plan_capacity(layouts, 135, capacity_divisors={"mamba": 4}).page_capacity == 8
    )
    for divisor in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            plan_capacity(layouts, 128, capacity_divisors={"mamba": divisor})
    with pytest.raises(ValueError, match="Unknown"):
        plan_capacity(layouts, 128, capacity_divisors={"missing": 4})


def test_mamba_external_transfer_owns_only_its_checkpoint_on_abort():
    component = MambaComponent.__new__(MambaComponent)
    component._alloc_mamba_slot = lambda: torch.tensor([7])
    component._free_mamba_value = Mock()
    transfer = component.build_external_linker_transfer(
        LinkerTransferPhase.LOAD, None, ["a", "b"]
    )
    assert transfer.keys == ["b"]
    req = SimpleNamespace(
        kv=SimpleNamespace(
            holds_mamba=True, mamba_pool_idx=torch.tensor(3), mamba_cow_src_index=None
        )
    )
    component.update_external_linker_load(
        ExternalLinkerLoadPhase.ABORT, req, None, transfer, 4
    )
    component._free_mamba_value.assert_called_once()
    assert req.kv.mamba_pool_idx.item() == 3


def test_mamba_commit_does_not_deliver_into_freed_duplicate_checkpoint():
    component = MambaComponent.__new__(MambaComponent)
    canonical = torch.tensor([9])
    node = SimpleNamespace(
        component_data={ComponentType.MAMBA: SimpleNamespace(value=canonical)}
    )
    component.tree_core = SimpleNamespace(node_by_id=lambda _: node)
    req = SimpleNamespace(kv=SimpleNamespace(mamba_cow_src_index=None))
    result = component.update_external_linker_load(
        ExternalLinkerLoadPhase.COMMIT,
        req,
        None,
        SimpleNamespace(device_indices=torch.tensor([7])),
        4,
        insert_result=SimpleNamespace(last_device_node=1, mamba_exist=True),
    )
    assert result is None
    assert req.kv.mamba_cow_src_index is canonical
