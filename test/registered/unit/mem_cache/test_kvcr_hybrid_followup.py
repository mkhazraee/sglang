"""Hybrid geometry and progressive delivery regressions (no GPU required)."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
