"""Contract and fault tests for the KVCR direct linker.

The real ``KVCRDirectLinker`` runs over CPU tensors with a fake NIXL agent that
moves bytes with ``memmove`` and can hold or fail individual transfers, so the
owner-thread preparation, claim, load, offload, and teardown paths execute
exactly as in production without a GPU.
"""

from __future__ import annotations

import ctypes
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from queue import Empty, SimpleQueue
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("kvcr")

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.kvcr import kvcr_direct_linker as linker_module
from sglang.srt.mem_cache.storage.kvcr.kvcr_config import KVCRLinkerConfig
from sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker import KVCRDirectLinker
from sglang.srt.mem_cache.storage.kvcr.kvcr_layout import restorable_boundaries
from sglang.srt.mem_cache.storage.kvcr.router_hint import (
    KVCRFetchHint,
    KVCRLinkerKeyAdapter,
    encode_object_key,
    normalize_block_hash,
    parse_fetch_hint,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LinkerRequestContext,
)
from sglang.srt.mem_cache.utils import hash_str_to_int64
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

PAGE = 2
LAYERS = 3
ROW_BYTES = 8
TIMEOUT_S = 10.0
# Deliberately unreachable peers cannot prove quiescence. Retain their fake
# transport harnesses through process exit just as production retains buffers.
_UNQUIESCED_HARNESSES = []


class FakePeerNetwork:
    """In-process control and notification transport for two real KVCR cores."""

    def __init__(self):
        self.agents = {}
        self.controls = {}

    def control(self, endpoint):
        network = self
        incoming = self.controls[endpoint] = SimpleQueue()

        class Control:
            def send(self, endpoint, message):
                inbox = network.controls.get(endpoint)
                if inbox is None:
                    return False
                inbox.put(message)
                return True

            def recv(self):
                messages = []
                while True:
                    try:
                        messages.append(incoming.get_nowait())
                    except Empty:
                        return messages

        control = Control()
        control.endpoint = endpoint
        return control


class FakeNixlAgent:
    """Fake NIXL agent: WRITE copies CPU bytes; transfers can be held."""

    def __init__(self, network=None):
        self.network = network
        self.peer_notifs = SimpleQueue()
        self.xfer_notifs = {}
        self.name = ""
        self.registrations = []
        self.prep_calls = []
        self.deregistered = []
        self.xfers = []
        self.released = []
        self.notifs = {}
        # handle -> forced state; default DONE. Tests set "PROC" to hold a
        # transfer and "ERR" to fail it.
        self.states: dict[int, str] = {}
        self.default_state = "DONE"
        self.transferred = []
        self.landed = set()

    def register_memory(self, descs, mem_type="DRAM"):
        self.registrations.append((list(descs), mem_type))
        return len(self.registrations)

    def deregister_memory(self, handle):
        self.deregistered.append(handle)

    def get_agent_metadata(self):
        return self.name.encode() if self.network is not None else b"metadata"

    def add_remote_agent(self, metadata):
        if self.network is not None:
            assert metadata.decode() in self.network.agents
            return metadata
        return b"remote"

    def get_xfer_descs(self, descs, mem_type="DRAM"):
        return list(descs)

    def prep_xfer_dlist(self, agent_name, descriptors, *, mem_type, backends):
        assert descriptors.dtype.name == "uint64"
        assert descriptors.flags.c_contiguous and descriptors.shape[1] == 5
        rows = descriptors.tolist()
        self.prep_calls.append((agent_name, mem_type, rows))
        return agent_name, rows

    def make_prepped_xfer(
        self,
        op,
        local_handle,
        local_indices,
        remote_handle,
        remote_indices,
        *,
        notif_msg=b"",
        backends=None,
    ):
        def selected(handle, indices):
            result = []
            for index in indices:
                for addr, size, device, stride, count in handle[1]:
                    if 0 <= index < count:
                        result.append((addr + index * stride, size, device))
                        break
                    index -= count
                else:
                    raise AssertionError("invalid prepared descriptor index")
            return result

        return self.initialize_xfer(
            op,
            selected(local_handle, local_indices),
            selected(remote_handle, remote_indices),
            remote_handle[0],
            notif_msg=notif_msg,
            backends=backends,
        )

    def release_dlist_handle(self, handle):
        pass

    def initialize_xfer(
        self, op, local_descs, remote_descs, remote_agent, notif_msg=b"", backends=None
    ):
        local_descs, remote_descs = list(local_descs), list(remote_descs)
        assert len(local_descs) == len(remote_descs)
        self.xfers.append((op, local_descs, remote_descs, remote_agent))
        self.xfer_notifs[len(self.xfers)] = notif_msg
        return len(self.xfers)

    def transfer(self, handle, notif_msg=b""):
        if notif_msg:
            self.xfer_notifs[handle] = notif_msg
        op, local_descs, remote_descs, remote_agent = self.xfers[handle - 1]
        if self.states.get(handle, self.default_state) == "ERR":
            return "ERR"
        self.transferred.append(handle)
        return "PROC"

    def check_xfer_state(self, handle):
        state = self.states.get(handle, self.default_state)
        if state == "DONE" and handle not in self.landed:
            op, local_descs, remote_descs, remote_agent = self.xfers[handle - 1]
            if op == "WRITE" and (
                remote_agent == self.name
                or self.network is not None
                and remote_agent in self.network.agents
            ):
                for (src, size, _), (dst, dst_size, _) in zip(
                    local_descs, remote_descs
                ):
                    assert size == dst_size
                    ctypes.memmove(dst, src, size)
            self.landed.add(handle)
            if self.xfer_notifs[handle]:
                self.send_notif(remote_agent, self.xfer_notifs[handle])
        return state

    def release_xfer_handle(self, handle):
        self.released.append(handle)

    def send_notif(self, agent_name, notif_msg):
        if self.network is not None:
            self.network.agents[
                agent_name.decode() if isinstance(agent_name, bytes) else agent_name
            ].peer_notifs.put((self.name, notif_msg))

    def get_new_notifs(self, backends=None):
        notifs, self.notifs = self.notifs, {}
        while True:
            try:
                sender, message = self.peer_notifs.get_nowait()
            except Empty:
                return notifs
            notifs.setdefault(sender, []).append(message)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _pool_group(*, with_swa: bool, rows: int = 64) -> tuple[DevicePoolGroup, dict]:
    """CPU stand-ins for device pools: rows are token slots, pages are PAGE rows."""
    buffers = {
        "k": [torch.zeros(rows, ROW_BYTES, dtype=torch.uint8) for _ in range(LAYERS)],
        "v": [torch.zeros(rows, ROW_BYTES, dtype=torch.uint8) for _ in range(LAYERS)],
    }
    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[buffers["k"], buffers["v"]],
            layer_mapping={i: i for i in range(LAYERS)},
            page_size=PAGE,
            rows_are_pages=False,
        )
    ]
    if with_swa:
        buffers["swa"] = [torch.zeros(rows, ROW_BYTES, dtype=torch.uint8)]
        entries.append(
            DevicePoolEntry(
                name=PoolName.SWA,
                indices_from_pool=PoolName.SWA,
                device_pool=None,
                components=[buffers["swa"]],
                layer_mapping={0: 0},
                page_size=PAGE,
                rows_are_pages=False,
            )
        )
    return DevicePoolGroup(entries, LAYERS, PAGE), buffers


def _publish_args(extra: dict) -> None:
    config = {
        "local_dram_bytes_per_worker": 1 << 20,
        "pin_local_dram": False,
        "preparation_deadline_ms": 2000,
        "operation_timeout_ms": 500,
        "abandon_timeout_ms": 1000,
        "poll_interval_ms": 0.2,
        "fetch_chunk_pages": 2,
        "offload_chunk_pages": 2,
    }
    config.update(extra)
    args = ServerArgs(
        model_path="dummy",
        page_size=PAGE,
        enable_unified_cache_external_linker=True,
        unified_cache_external_linker_backend="kvcr",
        hicache_storage_backend_extra_config=json.dumps(config),
    )
    # Project explicit test configuration without loading a model or GPU.
    get_context().set_server_args(args)


class Harness:
    """A linker over fake pools plus the fake agent it talks to."""

    def __init__(
        self,
        *,
        with_swa: bool = False,
        extra: dict | None = None,
        network: FakePeerNetwork | None = None,
        device_pools: tuple[DevicePoolGroup, dict] | None = None,
        req_to_token_pool=None,
    ):
        extra = dict(extra or {})
        control_patch = nullcontext()
        if network is not None:
            port = 26000 + len(network.controls)
            extra.update(
                enable_remote_hint=True,
                control_port=port,
                control_advertise_host="127.0.0.1",
            )

            def build_control(linker):
                control = network.control(f"tcp://127.0.0.1:{port}")
                linker.control_endpoint = control.endpoint
                return control

            control_patch = patch.object(
                KVCRDirectLinker, "_build_control_channel", new=build_control
            )
        _publish_args(extra)
        self.agent = FakeNixlAgent(network)
        self.group, self.buffers = device_pools or _pool_group(with_swa=with_swa)
        params = SimpleNamespace(
            page_size=PAGE,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            attn_tp_cache_group=None,
            tp_cache_group=None,
            attn_cp_rank=0,
            attn_cp_size=1,
            pp_rank=0,
            pp_size=1,
            is_eagle=False,
            mtp_draft_device_pools=(),
        )
        import kvcr.progress as kvcr_progress
        from kvcr import KVCR

        agent = self.agent

        def factory(config, bindings, backend_configs):
            def make_agent(name, *_):
                agent.name = name
                if network is not None:
                    network.agents[name] = agent
                return agent

            with patch.multiple(
                kvcr_progress,
                nixl_agent=make_agent,
                nixl_agent_config=lambda **kwargs: kwargs,
            ):
                return KVCR(config, bindings, backend_configs)

        with (
            patch.object(
                linker_module,
                "resolve_hybrid_device_pool_group",
                return_value=self.group,
            ),
            control_patch,
        ):
            self.linker = KVCRDirectLinker(
                None,
                params,
                components=set(),
                _kvcr_factory=factory,
                _nixl_probe=lambda backend: {"DRAM_SEG", "VRAM_SEG"},
            )

    # -- helpers --------------------------------------------------------

    def page_indices(self, first_page: int, num_pages: int) -> torch.Tensor:
        return torch.arange(first_page * PAGE, (first_page + num_pages) * PAGE)

    def fill(self, first_page: int, num_pages: int, seed: int) -> None:
        for name, layers in self.buffers.items():
            for layer, buffer in enumerate(layers):
                rows = buffer[first_page * PAGE : (first_page + num_pages) * PAGE]
                rows.copy_(
                    torch.arange(rows.numel(), dtype=torch.int64)
                    .add(seed * 7 + layer * 13 + hash(name) % 5)
                    .remainder(251)
                    .to(torch.uint8)
                    .reshape(rows.shape)
                )

    def snapshot(self, first_page: int, num_pages: int) -> dict:
        return {
            name: [
                buffer[first_page * PAGE : (first_page + num_pages) * PAGE].clone()
                for buffer in layers
            ]
            for name, layers in self.buffers.items()
        }

    def offload(self, hashes: list[str], first_page: int, *, swa_tail: int = 0) -> None:
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                device_indices=self.page_indices(first_page, len(hashes)),
                keys=list(hashes),
            )
        ]
        if swa_tail:
            transfers.append(
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=self.page_indices(
                        first_page + len(hashes) - swa_tail, swa_tail
                    ),
                    keys=list(hashes[-swa_tail:]),
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        assert self.linker.offload(transfers)

    def wait_offloads(self, count: int) -> list[bool]:
        self.wait(lambda: self.linker.num_completed_offloads() >= count)
        return [self.linker.pop_completed_offload() for _ in range(count)]

    def lookup_transfers(self, hashes: list[str], *, swa_window: int = 0):
        transfers = [PoolTransfer(name=PoolName.KV, keys=list(hashes))]
        if swa_window:
            transfers.append(
                PoolTransfer(
                    name=PoolName.SWA,
                    keys=list(hashes[-swa_window:]),
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        return transfers

    def prepare(
        self,
        rid: str,
        hashes: list[str],
        *,
        attempt: int = 0,
        hint=None,
        swa_window: int = 0,
    ):
        handle = CacheRequestHandle(rid=rid, attempt_id=attempt)
        self.linker.prepare_request(
            LinkerRequestContext(request=handle, router_hint=hint),
            self.lookup_transfers(hashes, swa_window=swa_window),
        )
        return handle

    def hint(self, hashes: list[str]):
        return KVCRFetchHint(
            source_control_endpoint=self.linker.control_endpoint,
            block_hashes=tuple(hash_str_to_int64(page) for page in hashes),
        ).to_kvcr_hint()

    def wait_ready(self, handle: CacheRequestHandle) -> None:
        self.wait(lambda: self.linker.preparation_ready(handle))

    def load(
        self, rid: str, hashes: list[str], first_page: int, *, swa_tail: int = 0
    ) -> int:
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                device_indices=self.page_indices(first_page, len(hashes)),
                keys=list(hashes),
            )
        ]
        if swa_tail:
            transfers.append(
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=self.page_indices(
                        first_page + len(hashes) - swa_tail, swa_tail
                    ),
                    keys=list(hashes[-swa_tail:]),
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        assert self.linker.load(rid, transfers)
        return self.linker.start_layer_wise_loading()

    def wait_loads(self, count: int) -> list[list[str]]:
        self.wait(lambda: self.linker.num_completed_loads() >= count)
        return [self.linker.pop_completed_load() for _ in range(count)]

    def public_claims(self) -> int:
        return len(self.linker._kvcr._core._local_dram._public_claims)

    @staticmethod
    def wait(predicate, timeout: float = TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.002)
        raise AssertionError("condition not reached")

    def close(self) -> None:
        self.linker.close()


def _hashes(prefix: str, count: int) -> list[str]:
    """Page hashes shaped like production ones: 64 hex chars each."""
    import hashlib

    return [hashlib.sha256(f"{prefix}{i}".encode()).hexdigest() for i in range(count)]


@pytest.fixture
def harness():
    created = []

    def make(**kwargs):
        h = Harness(**kwargs)
        created.append(h)
        return h

    yield make
    for h in created:
        expected = getattr(h, "expected_close_error", None)
        if expected is None:
            h.close()
            continue
        core, buffer = h.linker._kvcr, h.linker._local_dram
        with pytest.raises(RuntimeError, match=expected):
            h.close()
        assert h.linker._kvcr is core and h.linker._local_dram is buffer
        assert not h.linker._closed and h.linker._unhealthy is not None
        _UNQUIESCED_HARNESSES.append(h)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_offload_prepare_lookup_load_round_trip_moves_bytes(harness):
    from kvcr.types import RegDescriptor

    h = harness()
    regions = h.linker._framework_regions
    assert len(regions) == 2 * LAYERS
    assert all(isinstance(region, RegDescriptor) for region in regions)
    assert {(r.size, r.stride, r.count) for r in regions} == {
        (PAGE * ROW_BYTES, PAGE * ROW_BYTES, 64 // PAGE)
    }
    assert {r.info for r in regions} == {"kv"}
    assert len(set(h.linker.layouts["kv"].labels)) == 2 * LAYERS
    assert h.linker.plan.pool_layouts == (("kv", PAGE * ROW_BYTES),)
    assert len(h.agent.prep_calls) == 2  # Initiator and loopback, once at startup.
    hashes = _hashes("a", 4)
    h.fill(0, 4, seed=1)
    expected = h.snapshot(0, 4)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]

    handle = h.prepare("r1", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r1", h.lookup_transfers(hashes)) == [1, 2, 3, 4]
    # Claims are held between lookup and load so the pages cannot be evicted.
    assert h.public_claims() == 4

    index = h.load("r1", hashes, first_page=8)
    assert index >= 0
    h.linker.layer_done_counter.set_consumer(index)
    assert h.wait_loads(1) == [["r1"]]
    h.linker.layer_done_counter.wait_until(LAYERS - 1)
    restored = h.snapshot(8, 4)
    for name in expected:
        for got, want in zip(restored[name], expected[name]):
            assert torch.equal(got, want), name
    h.wait(lambda: h.public_claims() == 0)
    stats = h.linker.snapshot_stats()
    assert stats["restored_pages"] == 4
    assert stats["offload_bytes"] > 0
    assert stats["gpu_restore_bytes"] == stats["offload_bytes"]
    assert len(h.agent.prep_calls) == 2


@pytest.mark.parametrize("rows_are_pages", [False, True])
def test_framework_registration_matches_page_geometry(rows_are_pages):
    from kvcr.types import RegDescriptor

    from sglang.srt.mem_cache.storage.kvcr.kvcr_layout import build_pool_object_layouts

    # Whole pages remain contiguous; page-row views may have gaps between pages.
    parent = torch.zeros(LAYERS, 8, ROW_BYTES * 2, dtype=torch.uint8)
    buffers = [view[:, :ROW_BYTES] if rows_are_pages else view for view in parent]
    entry = DevicePoolEntry(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        device_pool=None,
        components=[buffers],
        layer_mapping={i: i for i in range(LAYERS)},
        page_size=PAGE,
        rows_are_pages=rows_are_pages,
    )
    linker = KVCRDirectLinker.__new__(KVCRDirectLinker)
    linker.pool_group = DevicePoolGroup([entry], LAYERS, PAGE)
    linker.layouts = build_pool_object_layouts(linker.pool_group)
    regions = linker._build_framework_regions()
    row_span = 1 if rows_are_pages else PAGE
    assert len(regions) == LAYERS
    for layer, region in enumerate(regions):
        assert isinstance(region, RegDescriptor)
        assert region.addr == buffers[layer].data_ptr()
        assert region.size == buffers[layer].shape[1] * row_span
        assert region.stride == buffers[layer].stride(0) * row_span
        assert region.count == len(buffers[layer]) // row_span
        assert region.info == "kv"
        assert (region.mem_type, region.device_Id) == ("DRAM", 0)
        last = (region.count - 1) * row_span
        assert (
            region.addr + (region.count - 1) * region.stride
            == buffers[layer][last].data_ptr()
        )


def test_mamba_restore_copies_checkpoint_and_gates_cow_until_all_spans_land(
    harness, monkeypatch
):
    import threading
    from unittest.mock import Mock

    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    monkeypatch.setattr(
        linker_module, "mamba_track_grid", lambda page_size: 2 * page_size
    )
    group, buffers = _pool_group(with_swa=False)
    states = [torch.zeros(16, width, dtype=torch.uint8) for width in (3, 3, 5, 5)]
    mamba = DevicePoolEntry(
        name=PoolName.MAMBA,
        indices_from_pool=PoolName.MAMBA,
        device_pool=None,
        components=[states],
        layer_mapping={1: [0, 2], 2: [1, 3]},
        page_size=1,
        rows_are_pages=True,
    )
    group = DevicePoolGroup([*group.entries, mamba], LAYERS, PAGE)
    req_pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    req_pool.start_layer = 0
    req_pool.mamba_map = {1: 0, 2: 1}
    req_pool.mamba_pool = SimpleNamespace(copy_from=Mock())
    h = harness(
        device_pools=(group, buffers),
        req_to_token_pool=req_pool,
        extra={"operation_timeout_ms": 5000, "abandon_timeout_ms": 10000},
    )
    hashes = _hashes("mamba-restore", 2)
    h.fill(0, 2, seed=41)
    expected_kv = h.snapshot(0, 2)
    for index, state in enumerate(states):
        state[1].copy_(torch.arange(state.shape[1], dtype=torch.uint8) + index + 1)
    expected_state = [state[1].clone() for state in states]

    def transfers(kv_indices=None, checkpoint_indices=None):
        return [
            PoolTransfer(name=PoolName.KV, keys=hashes, device_indices=kv_indices),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=hashes[-1:],
                device_indices=checkpoint_indices,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]

    assert h.linker.offload(transfers(h.page_indices(0, 2), torch.tensor([1])))
    assert h.wait_offloads(1) == [True]
    handle = CacheRequestHandle(rid="mamba-restore", attempt_id=0)
    h.linker.prepare_request(LinkerRequestContext(request=handle), transfers())
    h.wait_ready(handle)
    assert h.linker.lookup(handle.rid, transfers()) == [2]
    assert h.public_claims() == 3  # Two KV pages and one boundary checkpoint.

    first_restore = len(h.agent.xfers)
    h.agent.default_state = "PROC"
    assert h.linker.load(handle.rid, transfers(h.page_indices(8, 2), torch.tensor([6])))
    counter = h.linker.start_layer_wise_loading()
    h.linker.layer_done_counter.set_consumer(counter)
    destinations = {state[6].data_ptr() for state in states}
    h.wait(
        lambda: destinations.issubset(
            dst[0] for _, _, descs, _ in h.agent.xfers[first_restore:] for dst in descs
        )
    )
    restore_handles = list(range(first_restore + 1, len(h.agent.xfers) + 1))
    held = next(
        transfer
        for transfer in restore_handles
        if states[0][6].data_ptr() in {dst[0] for dst in h.agent.xfers[transfer - 1][2]}
    )
    cow = None
    try:
        for transfer in restore_handles:
            if transfer != held:
                h.agent.states[transfer] = "DONE"
        h.wait(lambda: all(t in h.agent.landed for t in restore_handles if t != held))
        assert h.linker.num_completed_loads() == 0
        assert h.public_claims() == 3
        # Use the real COW hook: later-layer completion cannot release earlier state.
        cow = threading.Thread(
            target=req_pool.copy_mamba_state,
            args=(torch.tensor([6]), torch.tensor([7])),
            daemon=True,
        )
        cow.start()
        cow.join(0.05)
        assert cow.is_alive()
        req_pool.mamba_pool.copy_from.assert_not_called()
        h.agent.states[held] = "DONE"
        assert h.wait_loads(1) == [[handle.rid]]
        cow.join(TIMEOUT_S)
        assert not cow.is_alive()
        req_pool.mamba_pool.copy_from.assert_called_once()
        h.wait(lambda: h.public_claims() == 0)
        restored = h.snapshot(8, 2)
        for name, layers in expected_kv.items():
            for got, want in zip(restored[name], layers):
                assert torch.equal(got, want)
        for state, want in zip(states, expected_state):
            assert torch.equal(state[6], want)
    finally:
        h.agent.default_state = "DONE"
        for transfer in restore_handles:
            h.agent.states[transfer] = "DONE"
        if cow is not None:
            cow.join(TIMEOUT_S)


def test_unprepared_and_unknown_pages_never_hit(harness):
    h = harness()
    hashes = _hashes("b", 3)
    # KVCR query is the authority for both local and remote availability.
    handle = h.prepare("r2", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r2", h.lookup_transfers(hashes)) == []
    assert h.public_claims() == 0
    stats = h.linker.snapshot_stats()
    assert stats["miss_no_candidates"] == 1
    assert stats.get("prepared_requests", 0) == 0
    # A lookup for a request that was never prepared is a miss, not a hang.
    assert h.linker.lookup("never-prepared", h.lookup_transfers(hashes)) == []


def test_hinted_but_absent_pages_never_become_hits(harness):
    port = _free_port()
    h = harness(
        extra={
            "enable_remote_hint": True,
            "control_port": port,
            "control_advertise_host": "127.0.0.1",
            "preparation_deadline_ms": 300,
        }
    )
    # No peer can acknowledge abandonment, so close must fail closed.
    h.expected_close_error = "unresolved operations"
    hashes = _hashes("c", 2)
    hint = {
        "protocol_version": "0.1",
        "message_id": "m",
        "actions": [
            {
                "action_id": "a",
                "action_type": "kv.fetch",
                "action_version": "1.0",
                # A peer that does not exist: nothing ever arrives.
                "payload": {
                    "source_control_endpoint": f"tcp://127.0.0.1:{_free_port()}",
                    "block_hashes": [hash_str_to_int64(x) for x in hashes],
                },
            }
        ],
    }
    handle = h.prepare("r3", hashes, hint=hint)
    started = time.monotonic()
    h.wait_ready(handle)
    assert time.monotonic() - started < TIMEOUT_S
    assert h.linker.lookup("r3", h.lookup_transfers(hashes)) == []
    stats = h.linker.snapshot_stats()
    assert stats["hinted_requests"] == 1
    assert stats.get("prepared_pages", 0) == 0
    assert stats["prepare_deadlines"] == 1
    # The dead-peer fetch is abandoned, not assumed finished: it stays tracked
    # as late work, counted against the abandoned bound, until KVCR's own
    # operation timeout resolves it as failed. Only then is the accounting
    # released; the core keeps the destination slots quarantined internally.
    assert stats["abandoned_bytes"] > 0
    h.wait(lambda: h.linker.snapshot_stats().get("late_completions", 0) >= 1, timeout=5)
    late = h.linker.snapshot_stats()
    assert late["abandoned_bytes"] == 0
    assert late["kvcr_pending_ops"] == 0
    assert h.public_claims() == 0
    # New requests are not blocked by the quarantined work below the bound.
    other = h.prepare("r3b", _hashes("c2", 1))
    h.wait_ready(other)
    assert h.linker.lookup("r3b", h.lookup_transfers(_hashes("c2", 1))) == []


def test_partial_pool_success_selects_only_valid_boundaries(harness):
    h = harness(with_swa=True)
    hashes = _hashes("d", 4)
    h.fill(0, 4, seed=3)
    # Store KV for every page but SWA only for the trailing page of the node.
    h.offload(hashes, first_page=0, swa_tail=1)
    assert h.wait_offloads(1) == [True]

    handle = h.prepare("r4", hashes, swa_window=1)
    h.wait_ready(handle)
    # Only the boundary whose trailing SWA window is present is restorable.
    assert h.linker.lookup("r4", h.lookup_transfers(hashes, swa_window=1)) == [4]


def test_restorable_boundaries_are_sparse_for_trailing_pools():
    present = {"kv": [True] * 5, "swa": [False, True, False, True, False]}
    policies = {"kv": ("all_pages", 0), "swa": ("trailing_pages", 1)}
    assert restorable_boundaries(present, policies, 5) == [2, 4]
    policies["swa"] = ("trailing_pages", 2)
    assert restorable_boundaries(present, policies, 5) == []
    assert restorable_boundaries(
        {"kv": [True, False, True]}, {"kv": ("all_pages", 0)}, 3
    ) == [1]


def test_out_of_order_offload_completions_keep_fifo_results(harness):
    h = harness()
    first, second = _hashes("e", 2), _hashes("f", 2)
    # Hold the first offload's transfer; the second completes immediately.
    h.agent.states[1] = "PROC"
    h.offload(first, first_page=0)
    h.wait(lambda: len(h.agent.xfers) >= 1)
    h.offload(second, first_page=4)
    h.wait(lambda: len(h.agent.transferred) >= 2)
    time.sleep(0.05)
    assert h.linker.num_completed_offloads() == 0
    h.agent.states[1] = "DONE"
    assert h.wait_offloads(2) == [True, True]


def test_offload_submission_yields_between_chunks_and_still_completes(harness):
    h = harness(extra={"fetch_chunk_pages": 1, "offload_chunk_pages": 1})
    hashes = _hashes("y", 4)
    h.fill(0, 4, seed=31)
    expected = h.snapshot(0, 4)
    # Pretend a command is always waiting: every deposit after the first is
    # deferred to a later owner-loop iteration, and the task must still finish
    # exactly once with every page resident.
    with patch.object(h.linker._adapter, "has_pending_commands", return_value=True):
        h.offload(hashes, first_page=0)
        assert h.wait_offloads(1) == [True]
    assert h.linker.num_completed_offloads() == 0
    handle = h.prepare("ry", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("ry", h.lookup_transfers(hashes)) == [1, 2, 3, 4]
    index = h.load("ry", hashes, first_page=8)
    h.linker.layer_done_counter.set_consumer(index)
    assert h.wait_loads(1) == [["ry"]]
    h.linker.layer_done_counter.wait_until(LAYERS - 1)
    restored = h.snapshot(8, 4)
    for name in expected:
        for got, want in zip(restored[name], expected[name]):
            assert torch.equal(got, want), name
    stats = h.linker.snapshot_stats()
    assert stats["offload_tasks"] == 1
    assert stats["offload_inflight_bytes"] == 0


def test_release_and_finish_drain_claims(harness):
    h = harness()
    hashes = _hashes("g", 3)
    h.fill(0, 3, seed=5)
    h.offload(hashes, first_page=0)
    h.wait_offloads(1)
    handle = h.prepare("r5", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r5", h.lookup_transfers(hashes)) == [1, 2, 3]
    assert h.public_claims() == 3
    h.linker.release_request("r5")
    h.wait(lambda: h.public_claims() == 0)
    assert h.linker.lookup("r5", h.lookup_transfers(hashes)) == []

    handle = h.prepare("r6", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r6", h.lookup_transfers(hashes)) == [1, 2, 3]
    h.linker.finish_request("r6")
    h.wait(lambda: h.public_claims() == 0)


def test_stale_attempt_cannot_satisfy_a_new_attempt(harness):
    h = harness()
    hashes = _hashes("h", 2)
    h.fill(0, 2, seed=6)
    # Hold the deposit so the pages are still filling: both attempts' fetches
    # then wait on the same fill, and only the live attempt may keep claims.
    h.agent.default_state = "PROC"
    h.offload(hashes, first_page=0)
    h.wait(lambda: len(h.agent.transferred) >= 1)
    first = h.prepare("r7", hashes, attempt=0)
    h.wait(lambda: h.linker.snapshot_stats()["kvcr_pending_ops"] >= 2)
    second = h.prepare("r7", hashes, attempt=1)
    assert h.linker.preparation_ready(first)  # retired, never blocks admission
    h.agent.default_state = "DONE"
    h.wait_ready(second)
    assert h.wait_offloads(1) == [True]
    assert h.linker.lookup("r7", h.lookup_transfers(hashes)) == [1, 2]
    stats = h.linker.snapshot_stats()
    assert stats["prepare_retired_superseded"] == 1
    assert stats["late_claims_released"] == 2
    # Exactly the new attempt's claims remain; the stale ones were released.
    h.wait(lambda: h.linker.snapshot_stats()["kvcr_pending_ops"] == 0)
    assert h.public_claims() == 2


def test_lookup_realigns_to_a_grown_device_prefix(harness):
    h = harness()
    hashes = _hashes("i", 4)
    h.fill(0, 4, seed=7)
    h.offload(hashes, first_page=0)
    h.wait_offloads(1)
    handle = h.prepare("r8", hashes)
    h.wait_ready(handle)
    # Another request inserted the first two pages meanwhile: the tail is
    # shorter, boundaries shift, and the now-resident pages' claims drop.
    assert h.linker.lookup("r8", h.lookup_transfers(hashes[2:])) == [1, 2]
    h.wait(lambda: h.public_claims() == 2)


def test_lookup_after_device_eviction_recomputes_instead_of_exposing_gaps(harness):
    h = harness()
    hashes = _hashes("j", 4)
    h.fill(0, 4, seed=8)
    h.offload(hashes, first_page=0)
    h.wait_offloads(1)
    handle = h.prepare("r9", hashes[1:])
    h.wait_ready(handle)
    # The device prefix shrank: page 0 was never prepared, so nothing is hit.
    assert h.linker.lookup("r9", h.lookup_transfers(hashes)) == []
    h.wait(lambda: h.public_claims() == 0)
    assert h.linker.snapshot_stats()["prepare_retired_tail_shrunk"] == 1


def test_deadline_yields_miss_and_late_claims_are_released(harness):
    h = harness(extra={"preparation_deadline_ms": 200, "enable_telemetry": True})
    hashes = _hashes("k", 2)
    h.fill(0, 2, seed=9)
    # A held deposit keeps the pages filling, so the fetch cannot confirm
    # before the preparation deadline.
    h.agent.default_state = "PROC"
    h.offload(hashes, first_page=0)
    h.wait(lambda: len(h.agent.transferred) >= 1)
    handle = h.prepare("r10", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r10", h.lookup_transfers(hashes)) == []
    stats = h.linker.snapshot_stats()
    assert stats["prepare_deadlines"] == 1
    assert stats["prepare_latency_seconds_count"] == 1
    assert stats["fetch_seconds_count"] == 1
    assert stats["admission_wait_seconds_count"] == 1
    assert stats["abandoned_bytes"] > 0
    h.agent.default_state = "DONE"
    assert h.wait_offloads(1) == [True]
    h.wait(lambda: h.linker.snapshot_stats().get("late_completions", 0) >= 1)
    h.wait(lambda: h.public_claims() == 0)
    assert h.linker.snapshot_stats()["late_claims_released"] == 2
    assert h.linker.snapshot_stats()["abandoned_bytes"] == 0
    late_stats = h.linker.snapshot_stats()
    for stage in ("prepare_latency", "fetch", "admission_wait"):
        assert late_stats[f"{stage}_seconds_count"] == 1
        assert late_stats[f"{stage}_seconds_sum"] == stats[f"{stage}_seconds_sum"]


def test_failed_gpu_load_fails_the_layer_counter_and_stops_new_work(harness):
    h = harness()
    hashes = _hashes("l", 2)
    h.fill(0, 2, seed=10)
    h.offload(hashes, first_page=0)
    h.wait_offloads(1)
    handle = h.prepare("r11", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r11", h.lookup_transfers(hashes)) == [1, 2]
    # The deliver transfer errors: after admission this is not a miss.
    h.agent.default_state = "ERR"
    index = h.load("r11", hashes, first_page=8)
    h.linker.layer_done_counter.set_consumer(index)
    h.wait_loads(1)
    with pytest.raises(RuntimeError, match="KVCR layer-wise KV load failed"):
        h.linker.layer_done_counter.wait_until(0)
    assert h.linker.snapshot_stats()["uncertain_loads"] == 1
    # The backend refuses further work rather than recomputing over it.
    h.agent.default_state = "DONE"
    assert not h.linker.offload(
        [
            PoolTransfer(
                name=PoolName.KV, device_indices=h.page_indices(0, 2), keys=hashes
            )
        ]
    )
    new = h.prepare("r12", hashes)
    assert h.linker.preparation_ready(new)
    assert h.linker.lookup("r12", h.lookup_transfers(hashes)) == []


def test_reset_clears_local_residency(harness):
    h = harness()
    hashes = _hashes("m", 2)
    h.fill(0, 2, seed=11)
    h.offload(hashes, first_page=0)
    h.wait_offloads(1)
    h.linker.reset()
    handle = h.prepare("r13", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r13", h.lookup_transfers(hashes)) == []
    # The rebuilt core is live: a new offload round-trips again.
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    handle = h.prepare("r14", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r14", h.lookup_transfers(hashes)) == [1, 2]


def test_inventory_removal_reports_page_event_hashes(harness):
    # A tiny tier: the third page evicts the first, and the eviction surfaces
    # as an EXTERNAL removal keyed by the page's event hash.
    h = harness(
        extra={"local_dram_bytes_per_worker": ROW_BYTES * PAGE * (2 * LAYERS) * 2}
    )
    hashes = _hashes("n", 3)
    h.fill(0, 3, seed=12)
    h.offload(hashes[:2], first_page=0)
    assert h.wait_offloads(1) == [True]
    h.offload(hashes[2:], first_page=2)
    assert h.wait_offloads(1) == [True]
    h.wait(lambda: h.linker.snapshot_stats()["inventory_removed_pages"] >= 1)
    removed = h.linker.take_removed_page_hashes()
    assert hash_str_to_int64(hashes[0]) in removed
    assert h.linker.take_removed_page_hashes() == []


def test_offload_backpressure_declines_beyond_inflight_bytes(harness):
    h = harness(extra={"max_inflight_offload_bytes": ROW_BYTES * PAGE * (2 * LAYERS)})
    h.agent.default_state = "PROC"
    h.offload(_hashes("o", 1), first_page=0)
    assert not h.linker.offload(
        [
            PoolTransfer(
                name=PoolName.KV,
                device_indices=h.page_indices(2, 1),
                keys=_hashes("p", 1),
            )
        ]
    )
    assert h.linker.snapshot_stats()["offload_declined_backpressure"] == 1
    h.agent.default_state = "DONE"
    h.wait_offloads(1)


# ---------------------------------------------------------------------------
# Identity, hints, and configuration
# ---------------------------------------------------------------------------


def test_full_key_identity_survives_event_hash_conversion():
    page = "7f" * 32
    for pool in ("kv", "swa"):
        key = encode_object_key(page, "digest", pool)
        decoded = KVCRLinkerKeyAdapter().decode(key)
        assert decoded == normalize_block_hash(hash_str_to_int64(page))
        assert key.decode().startswith(page + "#kvcr-linker-v1#digest#")
    # Different digests keep incompatible layouts apart on the full key.
    assert encode_object_key(page, "a", "kv") != encode_object_key(page, "b", "kv")


def test_hint_parser_reads_kv_fetch_and_ignores_unknown_actions():
    envelope = {
        "protocol_version": "0.1",
        "message_id": "m",
        "actions": [
            {
                "action_id": "x",
                "action_type": "kv.other",
                "action_version": "1.0",
                "payload": {},
            },
            {
                "action_id": "y",
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": "tcp://h:1",
                    "block_hashes": [5, -1],
                },
            },
        ],
    }
    hint = parse_fetch_hint(envelope)
    assert hint is not None
    assert hint.source_control_endpoint == "tcp://h:1"
    assert hint.block_hashes == (5, (1 << 64) - 1)
    assert (
        parse_fetch_hint(
            {
                "actions": [
                    {"action_type": "kv.fetch", "action_version": "2.0", "payload": {}}
                ]
            }
        )
        is None
    )
    assert parse_fetch_hint(None) is None
    assert parse_fetch_hint({"actions": "nope"}) is None
    assert (
        parse_fetch_hint(
            {
                "actions": [
                    {
                        "action_type": "kv.fetch",
                        "action_version": "1.0",
                        "payload": {
                            "source_control_endpoint": "tcp://h:1",
                            "block_hashes": ["zz"],
                        },
                    }
                ]
            }
        )
        is None
    )


def test_config_rejects_unknown_and_unsafe_options():
    with pytest.raises(ValueError, match="unknown options"):
        KVCRLinkerConfig.from_extra_config({"local_dram_bytes": 1})
    with pytest.raises(ValueError, match="unknown options"):
        KVCRLinkerConfig.from_extra_config(
            {"local_dram_bytes_per_worker": 1, "bogus": 1}
        )
    with pytest.raises(ValueError, match="requires local_dram_bytes_per_worker"):
        KVCRLinkerConfig.from_extra_config({})
    with pytest.raises(ValueError, match="explicit control_port"):
        KVCRLinkerConfig.from_extra_config(
            {"local_dram_bytes_per_worker": 1, "enable_remote_hint": True}
        )
    with pytest.raises(ValueError, match="cannot advertise"):
        KVCRLinkerConfig.from_extra_config(
            {
                "local_dram_bytes_per_worker": 1,
                "enable_remote_hint": True,
                "control_port": 25000,
                "control_advertise_host": "0.0.0.0",
            }
        )
    with pytest.raises(ValueError, match="abandon_timeout_ms"):
        KVCRLinkerConfig.from_extra_config(
            {
                "local_dram_bytes_per_worker": 1,
                "operation_timeout_ms": 1000,
                "abandon_timeout_ms": 1000,
            }
        )


def test_startup_rejects_unsupported_arrangements(harness):
    with pytest.raises(RuntimeError, match="does not support memory types"):
        _publish_args({})
        agent = FakeNixlAgent()
        group, _ = _pool_group(with_swa=False)
        params = SimpleNamespace(
            page_size=PAGE,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            attn_tp_cache_group=None,
            tp_cache_group=None,
            attn_cp_rank=0,
            attn_cp_size=1,
            pp_rank=0,
            pp_size=1,
            is_eagle=False,
            mtp_draft_device_pools=(),
        )
        with patch.object(
            linker_module, "resolve_hybrid_device_pool_group", return_value=group
        ):
            KVCRDirectLinker(
                None, params, components=set(), _nixl_probe=lambda b: {"VRAM_SEG"}
            )
    with pytest.raises(ValueError, match="pipeline parallelism"):
        params = SimpleNamespace(
            page_size=PAGE,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            attn_tp_cache_group=None,
            tp_cache_group=None,
            attn_cp_rank=0,
            attn_cp_size=1,
            pp_rank=0,
            pp_size=2,
            is_eagle=False,
            mtp_draft_device_pools=(),
        )
        with patch.object(
            linker_module, "resolve_hybrid_device_pool_group", return_value=group
        ):
            KVCRDirectLinker(
                None,
                params,
                components=set(),
                _nixl_probe=lambda b: {"DRAM_SEG", "VRAM_SEG"},
            )
    with pytest.raises(RuntimeError, match="ROCm"):
        params = SimpleNamespace(
            page_size=PAGE,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            attn_tp_cache_group=None,
            tp_cache_group=None,
            attn_cp_rank=0,
            attn_cp_size=1,
            pp_rank=0,
            pp_size=1,
            is_eagle=False,
            mtp_draft_device_pools=(),
        )
        with (
            patch.object(
                linker_module, "resolve_hybrid_device_pool_group", return_value=group
            ),
            patch.object(linker_module, "is_hip", return_value=True),
        ):
            KVCRDirectLinker(
                None,
                params,
                components=set(),
                _nixl_probe=lambda b: {"DRAM_SEG", "VRAM_SEG"},
            )


def test_startup_rejects_speculative_without_draft_pools():
    args = ServerArgs(
        model_path="dummy",
        page_size=PAGE,
        enable_unified_cache_external_linker=True,
        unified_cache_external_linker_backend="kvcr",
        hicache_storage_backend_extra_config=json.dumps(
            {"local_dram_bytes_per_worker": 1 << 20, "pin_local_dram": False}
        ),
        speculative_algorithm="EAGLE",
        speculative_draft_model_path="dummy-draft",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=2,
    )
    # Project explicit test configuration without loading a model or GPU.
    get_context().set_server_args(args)
    group, _ = _pool_group(with_swa=False)
    params = SimpleNamespace(
        page_size=PAGE,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
        attn_tp_cache_group=None,
        tp_cache_group=None,
        attn_cp_rank=0,
        attn_cp_size=1,
        pp_rank=0,
        pp_size=1,
        is_eagle=True,
        mtp_draft_device_pools=(),
    )
    with patch.object(
        linker_module, "resolve_hybrid_device_pool_group", return_value=group
    ):
        with pytest.raises(ValueError, match="draft state"):
            KVCRDirectLinker(
                None,
                params,
                components=set(),
                _nixl_probe=lambda b: {"DRAM_SEG", "VRAM_SEG"},
            )


def test_owner_thread_stats_tick_during_startup_does_not_fault(harness, caplog):
    # A stats interval shorter than core construction makes the first owner
    # tick fire before __init__ returns; it must find the adapter in place.
    import logging

    with caplog.at_level(logging.WARNING):
        h = harness(extra={"stats_log_interval_s": 0.001})
        h.wait(lambda: h.linker.snapshot_stats() is not None)
        time.sleep(0.05)
    assert h.linker._adapter.healthy
    assert not [r for r in caplog.records if "owner loop fault" in r.getMessage()]


@pytest.mark.parametrize("failure", ["entry", "submission"])
@pytest.mark.parametrize("direct_remote", [False, True])
def test_delivery_failure_drains_submitted_work_before_releasing_claims(
    harness, failure, direct_remote
):
    import threading

    h = harness(extra={"direct_remote_restore": direct_remote})
    hashes = _hashes("failure", 2)
    h.fill(0, 2, seed=24)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    h.wait_ready(h.prepare("failure", hashes))
    first = len(h.agent.xfers) + 1
    h.agent.default_state = "PROC"
    if failure == "submission":
        attempted = threading.Event()
        deliver = h.linker._kvcr.deliver
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                attempted.set()
                raise RuntimeError("injected second-layer submission failure")
            return deliver(*args, **kwargs)

        with patch.object(h.linker._kvcr, "deliver", side_effect=fail_second):
            index = h.load("failure", hashes, first_page=4)
            assert attempted.wait(TIMEOUT_S)
        h.wait(lambda: len(h.agent.xfers) == first)
    else:
        index = h.load("failure", hashes, first_page=4)
        h.wait(lambda: len(h.agent.xfers) == first + LAYERS - 1)
        h.agent.states[first] = "ERR"
        h.wait(lambda: first in h.agent.released)
    assert h.public_claims() == (0 if direct_remote else 2)
    assert h.linker.num_completed_loads() == 0
    h.agent.default_state = "DONE"
    assert h.wait_loads(1) == [["failure"]]
    h.wait(lambda: h.public_claims() == 0)
    h.linker.layer_done_counter.set_consumer(index)
    with pytest.raises(RuntimeError):
        h.linker.layer_done_counter.wait_until(LAYERS - 1)


def test_multi_pool_fetches_keep_each_expected_layout_homogeneous(harness):
    h = harness(with_swa=True)
    hashes = _hashes("pool-layout", 4)
    h.fill(0, 4, seed=25)
    h.offload(hashes, first_page=0, swa_tail=2)
    assert h.wait_offloads(1) == [True]
    # The real KVCR public fetch accepts one layout per operation.
    h.wait_ready(h.prepare("pools", hashes, swa_window=2))
    assert h.linker.lookup("pools", h.lookup_transfers(hashes, swa_window=2)) == [4]
    assert h.public_claims() == 6
    h.linker.release_request("pools")
    h.wait(lambda: h.public_claims() == 0)


def test_drain_waits_for_queued_load_submission(harness):
    import threading

    h = harness()
    hashes = _hashes("queued-drain", 2)
    h.fill(0, 2, seed=26)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    h.wait_ready(h.prepare("queued-drain", hashes))
    blocked, unblock = threading.Event(), threading.Event()

    def block_owner(adapter):
        blocked.set()
        assert unblock.wait(TIMEOUT_S)

    h.linker._adapter.post(block_owner)
    assert blocked.wait(TIMEOUT_S)
    h.agent.default_state = "PROC"
    first = len(h.agent.xfers) + 1
    h.load("queued-drain", hashes, first_page=4)
    drained = []
    drain = threading.Thread(target=lambda: drained.append(h.linker._drain(TIMEOUT_S)))
    drain.start()
    try:
        drain.join(0.05)
        assert drain.is_alive()
        unblock.set()
        h.wait(lambda: len(h.agent.xfers) == first + LAYERS - 1)
        assert h.public_claims() == 2 and drain.is_alive()
        h.agent.default_state = "DONE"
        drain.join(TIMEOUT_S)
        assert drained == [True]
        assert h.public_claims() == 0
    finally:
        unblock.set()
        h.agent.default_state = "DONE"
        drain.join(TIMEOUT_S)


def test_reset_retains_core_buffers_and_claims_when_delivery_has_not_drained(harness):
    h = harness(extra={"operation_timeout_ms": 5000, "abandon_timeout_ms": 10000})
    hashes = _hashes("reset-drain", 2)
    h.fill(0, 2, seed=28)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    h.wait_ready(h.prepare("reset-drain", hashes))
    h.agent.default_state = "PROC"
    first = len(h.agent.xfers) + 1
    h.load("reset-drain", hashes, first_page=4)
    h.wait(lambda: len(h.agent.xfers) == first + LAYERS - 1)
    core, buffer, adapter = h.linker._kvcr, h.linker._local_dram, h.linker._adapter
    drain = h.linker._drain
    with patch.object(h.linker, "_drain", side_effect=lambda timeout: drain(0.02)):
        with pytest.raises(RuntimeError, match="did not drain"):
            h.linker.reset()
    assert h.linker._kvcr is core and h.linker._local_dram is buffer
    assert h.linker._adapter is adapter and adapter.healthy
    assert h.public_claims() == 2
    assert h.linker._unhealthy is not None
    h.agent.default_state = "DONE"
    h.wait(lambda: h.public_claims() == 0)
    h.linker.reset()
    assert h.linker._kvcr is not core
    assert h.linker._unhealthy is None
    h.close()
    assert h.linker._closed


def test_owner_callback_failure_marks_adapter_unhealthy():
    import threading

    from sglang.srt.mem_cache.storage.kvcr.kvcr_adapter import KVCRAdapter

    completions = [[(1, {})]]
    failed = threading.Event()
    errors = []

    def on_unhealthy(error):
        errors.append(error)
        failed.set()

    def completion(entries):
        raise RuntimeError("injected completion callback failure")

    adapter = KVCRAdapter(
        SimpleNamespace(
            poll_completed=lambda: completions.pop() if completions else []
        ),
        poll_interval_s=0.001,
        name="callback-failure-test",
        on_unhealthy=on_unhealthy,
    )
    adapter.track(1, completion)
    adapter.start()
    try:
        assert failed.wait(TIMEOUT_S)
        assert not adapter.healthy
        assert len(errors) == 1
        assert str(errors[0]) == "injected completion callback failure"
    finally:
        assert adapter.stop(TIMEOUT_S)


def test_peer_preparation_stages_bytes_before_admission(harness):
    network = FakePeerNetwork()
    source = harness(network=network)
    target = harness(network=network)
    hashes = _hashes("staged-peer", 1)
    source.fill(0, 1, seed=29)
    expected = source.snapshot(0, 1)
    source.offload(hashes, first_page=0)
    assert source.wait_offloads(1) == [True]
    before = len(source.agent.xfers)
    source.agent.default_state = "PROC"
    try:
        handle = target.prepare("staged-peer", hashes, hint=source.hint(hashes))
        source.wait(lambda: len(source.agent.xfers) == before + 1)
        assert not target.linker.preparation_ready(handle)
        assert (
            target.linker.lookup("staged-peer", target.lookup_transfers(hashes)) == []
        )
        assert target.public_claims() == 0 and target.agent.xfers == []
        source.agent.default_state = "DONE"
        target.wait_ready(handle)
        assert target.linker.lookup("staged-peer", target.lookup_transfers(hashes)) == [
            1
        ]
        assert target.public_claims() == 1
        # The peer wrote staged DRAM; device destinations still need delivery.
        assert target.agent.xfers == []
        assert all(
            not torch.count_nonzero(layer)
            for layers in target.snapshot(4, 1).values()
            for layer in layers
        )
        target.load("staged-peer", hashes, first_page=4)
        assert target.wait_loads(1) == [["staged-peer"]]
        target.wait(lambda: target.public_claims() == 0)
        for name, layers in target.snapshot(4, 1).items():
            assert all(
                torch.equal(got, want) for got, want in zip(layers, expected[name])
            )
    finally:
        source.agent.default_state = "DONE"


@pytest.mark.parametrize("progressive_restore", [False, True])
def test_layer_delivery_releases_early_layer_and_holds_claims_until_all_drain(
    harness, progressive_restore
):
    import threading

    h = harness(extra={"progressive_restore": progressive_restore})
    hashes = _hashes("layers", 2)
    h.fill(0, 2, seed=22)
    expected = h.snapshot(0, 2)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    h.wait_ready(h.prepare("layers", hashes))
    first = len(h.agent.xfers) + 1
    h.agent.default_state = "PROC"
    index = h.load("layers", hashes, first_page=4)
    h.wait(lambda: len(h.agent.xfers) == first + LAYERS - 1)
    # Both modes submit the identical per-layer NIXL descriptor lists.
    for layer, transfer in enumerate(h.agent.xfers[first - 1 :]):
        destinations = {address for address, _, _ in transfer[2]}
        assert destinations == {
            h.buffers[name][layer][page * PAGE].data_ptr()
            for name in ("k", "v")
            for page in (4, 5)
        }
    futures = h.linker.layer_done_counter.futures[index]
    assert h.public_claims() == 2
    assert not any(f.done() for f in futures)
    h.agent.states[first] = "DONE"
    h.wait(lambda: h.linker._adapter.pending_ops == LAYERS - 1)
    observed = threading.Event()
    h.linker._adapter.post(lambda adapter: observed.set())
    assert observed.wait(TIMEOUT_S)
    assert futures[0].done() is progressive_restore
    assert not futures[1].done() and not futures[2].done()
    restored = h.snapshot(4, 2)
    for name in ("k", "v"):
        assert torch.equal(restored[name][0], expected[name][0])
        assert torch.count_nonzero(restored[name][1]) == 0
        assert torch.count_nonzero(restored[name][2]) == 0
    assert h.public_claims() == 2 and h.linker.num_completed_loads() == 0
    h.agent.default_state = "DONE"
    assert h.wait_loads(1) == [["layers"]]
    h.wait(lambda: h.public_claims() == 0)
    assert all(future.done() and future.exception() is None for future in futures)
    for name, layers in h.snapshot(4, 2).items():
        assert all(torch.equal(got, want) for got, want in zip(layers, expected[name]))


def test_layer_waits_for_every_request_and_chunk(harness):
    h = harness()
    hashes = _hashes("joined-layers", 4)
    h.fill(0, 4, seed=23)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    for rid in ("a", "b"):
        h.wait_ready(h.prepare(rid, hashes))
        assert h.linker.load(
            rid,
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=hashes,
                    device_indices=h.page_indices(8 if rid == "a" else 16, 4),
                )
            ],
        )
    first = len(h.agent.xfers) + 1
    h.agent.default_state = "PROC"
    index = h.linker.start_layer_wise_loading()
    h.wait(lambda: len(h.agent.xfers) == first + 12 - 1)
    futures = h.linker.layer_done_counter.futures[index]
    # Two requests, two page chunks each: all four transfers gate layer zero.
    for handle in range(first, first + 3):
        h.agent.states[handle] = "DONE"
    h.wait(
        lambda: all(handle in h.agent.released for handle in range(first, first + 3))
    )
    assert not futures[0].done()
    h.agent.states[first + 3] = "DONE"
    h.wait(lambda: futures[0].done())
    assert not futures[1].done() and h.public_claims() == 8
    h.agent.default_state = "DONE"
    assert h.wait_loads(1) == [["a", "b"]]
    h.wait(lambda: h.public_claims() == 0)


def test_wait_for_later_layer_also_waits_for_incomplete_earlier_layers(harness):
    import threading

    h = harness()
    hashes = _hashes("out-of-order", 2)
    h.fill(0, 2, seed=27)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    h.wait_ready(h.prepare("out-of-order", hashes))
    first = len(h.agent.xfers) + 1
    h.agent.default_state = "PROC"
    index = h.load("out-of-order", hashes, first_page=4)
    h.wait(lambda: len(h.agent.xfers) == first + LAYERS - 1)
    for handle in range(first + 1, first + LAYERS):
        h.agent.states[handle] = "DONE"
    futures = h.linker.layer_done_counter.futures[index]
    h.wait(lambda: futures[-1].done())
    h.linker.layer_done_counter.set_consumer(index)
    ready = threading.Event()

    def wait_all_layers():
        h.linker.layer_done_counter.wait_until(LAYERS - 1)
        ready.set()

    waiter = threading.Thread(target=wait_all_layers)
    waiter.start()
    try:
        assert not ready.wait(0.05)
        assert h.public_claims() == 2
        h.agent.states[first] = "DONE"
        assert ready.wait(TIMEOUT_S)
        assert h.wait_loads(1) == [["out-of-order"]]
        h.wait(lambda: h.public_claims() == 0)
    finally:
        h.agent.default_state = "DONE"
        waiter.join(TIMEOUT_S)


def test_fetch_completion_rechecks_retirement_after_acquiring_lock():
    import threading
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager

    lock, waiting = threading.Lock(), threading.Event()

    @contextmanager
    def observed_lock():
        waiting.set()
        with lock:
            yield

    prep = SimpleNamespace(
        state=linker_module._State.FETCHING,
        claims={},
        pools={"kv": SimpleNamespace(present=[False])},
        bytes_confirmed=0,
        outstanding_ops=2,
    )
    released = []
    owner = SimpleNamespace(
        _lock=observed_lock(),
        layouts={"kv": SimpleNamespace(object_bytes=8)},
        _release_handles=released.extend,
        _abandoned_bytes=8,
        stats=defaultdict(int),
    )
    completion = KVCRDirectLinker._fetch_completion(owner, prep, [("kv", 0, "page")])
    with ThreadPoolExecutor(max_workers=1) as executor:
        with lock:
            done = executor.submit(
                completion, {"page": SimpleNamespace(success=True, release_handle=17)}
            )
            assert waiting.wait(TIMEOUT_S)
            # The scheduler retires this preparation while completion waits.
            prep.state = linker_module._State.MISS
        done.result(timeout=TIMEOUT_S)
    assert prep.claims == {}
    assert released == [17]
    assert owner._abandoned_bytes == 0


@pytest.mark.parametrize("retirement_point", ["queued", "query"])
def test_retired_preparation_does_not_start_fetch_or_charge_bytes(
    harness, retirement_point
):
    import threading

    h = harness()
    hashes = _hashes("retired-before-fetch", 1)
    h.fill(0, 1, seed=31)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    blocked, resume, observed = (threading.Event() for _ in range(3))
    query = h.linker._kvcr.query

    def block_owner(adapter=None):
        blocked.set()
        assert resume.wait(TIMEOUT_S)

    def query_while_blocked(*args, **kwargs):
        if retirement_point == "query":
            block_owner()
        return query(*args, **kwargs)

    with (
        patch.object(h.linker._kvcr, "query", side_effect=query_while_blocked),
        patch.object(h.linker._kvcr, "fetch", wraps=h.linker._kvcr.fetch) as fetch,
    ):
        try:
            if retirement_point == "queued":
                h.linker._adapter.post(block_owner)
            handle = h.prepare("retired-before-fetch", hashes)
            assert blocked.wait(TIMEOUT_S)
            h.linker.release_request(handle.rid)
            h.linker._adapter.post(lambda adapter: observed.set())
            resume.set()
            assert observed.wait(TIMEOUT_S)
            assert fetch.call_count == 0
            with h.linker._lock:
                assert h.linker._inflight_prepare_bytes == 0
                assert h.linker._abandoned_bytes == 0
            assert h.public_claims() == 0
        finally:
            resume.set()


@pytest.mark.parametrize("pages", [1, 2, 4])
def test_direct_budget_counts_only_selected_window(harness, pages):
    h = harness(
        with_swa=True,
        extra={
            "direct_remote_restore": True,
            "max_prepare_bytes_per_request": pages
            * PAGE
            * ROW_BYTES
            * (2 * LAYERS + 1),
        },
    )
    hashes = _hashes("direct-budget", 4)
    h.fill(0, 4, seed=31)
    h.offload(hashes, first_page=0, swa_tail=4)
    assert h.wait_offloads(1) == [True]
    h.wait_ready(h.prepare("budget", hashes, swa_window=4))
    assert h.linker.lookup("budget", h.lookup_transfers(hashes, swa_window=4)) == list(
        range(1, pages + 1)
    )


@pytest.mark.parametrize("evict_source", [False, True])
def test_direct_remote_layers_use_peer_bytes_and_retain_hints(harness, evict_source):
    from kvcr.types import QueryStatus

    network = FakePeerNetwork()
    source = harness(
        network=network,
        extra={
            "local_dram_bytes_per_worker": PAGE * ROW_BYTES * 2 * LAYERS,
            "operation_timeout_ms": 5000,
            "abandon_timeout_ms": 10000,
        },
    )
    target = harness(
        network=network,
        extra={
            "direct_remote_restore": True,
            "operation_timeout_ms": 5000,
            "abandon_timeout_ms": 10000,
        },
    )
    hashes = _hashes("peer", 1)
    source.fill(0, 1, seed=29)
    expected = source.snapshot(0, 1)
    source.offload(hashes, first_page=0)
    assert source.wait_offloads(1) == [True]
    core = target.linker._kvcr
    hints = core._core._remote_fw_dram._request_hints
    keys = [target.linker._key(page, str(PoolName.KV)) for page in hashes]

    with patch.object(core, "fetch", wraps=core.fetch) as fetch:
        unused = target.prepare("unused", hashes, hint=source.hint(hashes))
        target.wait_ready(unused)
        unused_id = target.linker._preparations[unused].request_id
        assert unused_id in hints
        target.linker.release_request("unused")
        target.wait(lambda: unused_id not in hints)

        handle = target.prepare("peer", hashes, hint=source.hint(hashes))
        target.wait_ready(handle)
        request_id = target.linker._preparations[handle].request_id
        assert target.linker.lookup("peer", target.lookup_transfers(hashes)) == [1]
        fetch.assert_not_called()
        assert target.public_claims() == 0 and target.agent.xfers == []
        assert core.query(keys) == [(QueryStatus.MISS, None)]
        assert request_id in hints

        if evict_source:
            source.fill(1, 1, seed=30)
            source.offload(_hashes("replacement", 1), first_page=1)
            assert source.wait_offloads(1) == [True]
            assert source.linker._kvcr.query(keys) == [(QueryStatus.MISS, None)]
        else:
            source.agent.default_state = "PROC"
        first = len(source.agent.xfers) + 1
        try:
            index = target.load("peer", hashes, first_page=4)
            futures = target.linker.layer_done_counter.futures[index]
            if not evict_source:
                source.wait(lambda: len(source.agent.xfers) == first + LAYERS - 1)
                layer_zero_address = target.buffers["k"][0][4 * PAGE].data_ptr()
                first_layer = next(
                    number
                    for number in range(first, first + LAYERS)
                    if any(
                        address == layer_zero_address
                        for address, _, _ in source.agent.xfers[number - 1][2]
                    )
                )
                source.agent.states[first_layer] = "DONE"
                target.wait(lambda: futures[0].done())
                assert futures[0].exception() is None
                assert not futures[1].done() and not futures[2].done()
                for name, layers in target.snapshot(4, 1).items():
                    assert torch.equal(layers[0], expected[name][0])
                    assert not torch.count_nonzero(layers[1])
                    assert not torch.count_nonzero(layers[2])
                assert target.linker.num_completed_loads() == 0
                assert request_id in hints
                source.agent.default_state = "DONE"
            assert target.wait_loads(1) == [["peer"]]
            target.wait(lambda: request_id not in hints)
            fetch.assert_not_called()
            assert target.public_claims() == 0 and target.agent.xfers == []
            assert core.query(keys) == [(QueryStatus.MISS, None)]
            target.linker.layer_done_counter.set_consumer(index)
            if evict_source:
                with pytest.raises(RuntimeError, match="layer-wise KV load failed"):
                    target.linker.layer_done_counter.wait_until(LAYERS - 1)
                assert target.linker._unhealthy is not None
                assert all(
                    not torch.count_nonzero(layer)
                    for layers in target.snapshot(4, 1).values()
                    for layer in layers
                )
            else:
                target.linker.layer_done_counter.wait_until(LAYERS - 1)
                for name, layers in target.snapshot(4, 1).items():
                    assert all(
                        torch.equal(got, want)
                        for got, want in zip(layers, expected[name])
                    )
                source.wait(
                    lambda: (
                        source.linker._kvcr._core._block_record_map[
                            keys[0]
                        ].local_dram.claim_count
                        == 0
                    )
                )
        finally:
            source.agent.default_state = "DONE"


def test_telemetry_summaries_are_cumulative_bounded_and_thread_safe():
    from kvcr import DURATION_METRIC, STATE_METRIC, TRANSFER_BYTES_METRIC

    stats = linker_module._LinkerTelemetry()
    assert stats.is_empty()

    def record(_):
        stats.increase_counter(TRANSFER_BYTES_METRIC, 8, ("local_fill",))
        stats.observe_histogram(DURATION_METRIC, 0.25, ("local_fill", "success"))

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(record, range(1000)))
    stats.set_gauge(STATE_METRIC, 2, ("in_flight_ops",))
    stats.set_gauge(STATE_METRIC, 0, ("in_flight_ops",))
    snapshot = stats.reduce()
    assert snapshot[f"{TRANSFER_BYTES_METRIC}[local_fill]"] == 8000
    assert snapshot[f"{DURATION_METRIC}[local_fill,success]_count"] == 1000
    assert snapshot[f"{DURATION_METRIC}[local_fill,success]_sum"] == 250
    assert snapshot[f"{DURATION_METRIC}[local_fill,success]_max"] == 0.25
    assert snapshot[f"{STATE_METRIC}[in_flight_ops]"] == 0
    assert len(snapshot) == 5
    assert not stats.is_empty()
    assert stats.reduce() == snapshot


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("direct_remote", [False, True])
@pytest.mark.parametrize("progressive", [False, True])
def test_telemetry_records_real_nixl_path_and_once_only_timings(
    harness, enabled, direct_remote, progressive
):
    from kvcr import DURATION_METRIC, TRANSFER_BYTES_METRIC

    h = harness(
        extra={
            "enable_telemetry": enabled,
            "direct_remote_restore": direct_remote,
            "progressive_restore": progressive,
            "stats_log_interval_s": 3600,
        }
    )
    hashes = _hashes("telemetry", 2)
    h.fill(0, 2, seed=31)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    handle = h.prepare("telemetry", hashes)
    h.wait_ready(handle)
    for _ in range(3):
        assert h.linker.preparation_ready(handle)
    h.load("telemetry", hashes, first_page=4)
    assert h.wait_loads(1) == [["telemetry"]]
    h.wait(lambda: h.public_claims() == 0)
    snapshot = h.linker.snapshot_stats()
    if not enabled:
        assert h.linker._telemetry is None
        assert not any(key.startswith("kvcr_duration_seconds") for key in snapshot)
        assert not any(key.endswith("_seconds_count") for key in snapshot)
        return

    for stage in (
        "prepare_queue",
        "query",
        "prepare_latency",
        "admission_wait",
        "restore_wait",
        "restore_build",
        "restore_submit",
        "restore_copy",
        "offload_wait",
        "offload_submit",
        "offload",
    ):
        assert snapshot[f"{stage}_seconds_count"] == 1, stage
        assert 0 <= snapshot[f"{stage}_seconds_max"] <= snapshot[f"{stage}_seconds_sum"]
    if direct_remote:
        assert not any(key.startswith("fetch_seconds_") for key in snapshot)
    else:
        assert snapshot["fetch_seconds_count"] == 1
        assert 0 <= snapshot["fetch_seconds_max"] <= snapshot["fetch_seconds_sum"]
    assert snapshot[f"{TRANSFER_BYTES_METRIC}[local_fill]"] == snapshot["offload_bytes"]
    assert (
        snapshot[f"{TRANSFER_BYTES_METRIC}[local_deliver]"]
        == snapshot["gpu_restore_bytes"]
    )
    assert snapshot[f"{DURATION_METRIC}[local_deliver,success]_count"] == LAYERS
    for layer in range(LAYERS):
        metric = f"restore_layer_completion_seconds[{layer}]"
        assert snapshot[f"{metric}_count"] == 1
        assert 0 <= snapshot[f"{metric}_max"] <= snapshot[f"{metric}_sum"]
    # The owner refreshes core gauges, while scheduler snapshots remain read-only.
    refreshed = SimpleQueue()

    def refresh(adapter):
        h.linker._collect_telemetry()
        refreshed.put(True)

    h.linker._adapter.post(refresh)
    assert refreshed.get(timeout=TIMEOUT_S)
    first = h.linker.snapshot_stats()
    h.linker._adapter.post(refresh)
    assert refreshed.get(timeout=TIMEOUT_S)
    second = h.linker.snapshot_stats()
    assert second == first
    assert "kvcr_state[in_flight_ops]" in second
    assert (
        second[f"{TRANSFER_BYTES_METRIC}[local_deliver]"]
        == snapshot["gpu_restore_bytes"]
    )


@pytest.mark.parametrize("direct_remote", [False, True])
@pytest.mark.parametrize("progressive", [False, True])
def test_layer_telemetry_waits_for_all_chunks_independently_of_release_mode(
    harness, direct_remote, progressive
):
    h = harness(
        extra={
            "enable_telemetry": True,
            "direct_remote_restore": direct_remote,
            "progressive_restore": progressive,
        }
    )
    hashes = _hashes("timed-layer-chunks", 4)
    h.fill(0, 4, seed=41)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    for rid, destination in (("a", 8), ("b", 16)):
        h.wait_ready(h.prepare(rid, hashes))
        assert h.linker.load(
            rid,
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=hashes,
                    device_indices=h.page_indices(destination, 4),
                )
            ],
        )
    first = len(h.agent.xfers) + 1
    h.agent.default_state = "PROC"
    try:
        index = h.linker.start_layer_wise_loading()
        h.wait(lambda: len(h.agent.xfers) == first + 12 - 1)
        futures = h.linker.layer_done_counter.futures[index]
        metric = "restore_layer_completion_seconds[0]_count"
        # Two requests and two chunks per request contribute to layer zero.
        for handle in range(first, first + 3):
            h.agent.states[handle] = "DONE"
        h.wait(
            lambda: all(
                handle in h.agent.released for handle in range(first, first + 3)
            )
        )
        assert metric not in h.linker.snapshot_stats()
        h.agent.states[first + 3] = "DONE"
        h.wait(lambda: h.linker.snapshot_stats().get(metric) == 1)
        if progressive:
            h.wait(lambda: futures[0].done())
        else:
            assert not any(future.done() for future in futures)
        assert not futures[1].done()
        assert h.linker.num_completed_loads() == 0
        assert h.linker.snapshot_stats()[metric] == 1
    finally:
        h.agent.default_state = "DONE"
    assert h.wait_loads(1) == [["a", "b"]]
    h.wait(lambda: h.public_claims() == 0)
    for layer in range(LAYERS):
        assert (
            h.linker.snapshot_stats()[
                f"restore_layer_completion_seconds[{layer}]_count"
            ]
            == 1
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
