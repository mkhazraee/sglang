"""Direct GPU delivery ordering and ownership, with a controlled KVCR boundary."""

import threading
import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class Event:
    def __init__(self):
        self.ready = True

    def record(self):
        pass

    def query(self):
        return self.ready


class Client:
    def __init__(self):
        self.calls = []
        self.results = deque()
        self.hints = {}
        self.status = {}
        self.next_handle = 0
        self.polls = 0
        self.fail_poll = False

    def submit_hint(self, hint, request_id=None):
        self.hints[request_id] = hint

    def discard_hint(self, request_id):
        self.hints.pop(request_id, None)

    def query(self, keys, request_id=None):
        return [self.status.get(k, (SimpleNamespace(value="MISS"), None)) for k in keys]

    def deliver(self, blocks, request_id=None):
        return self._submit("deliver", blocks, request_id)

    def deposit(self, blocks):
        return self._submit("deposit", blocks, None)

    def _submit(self, kind, blocks, request_id):
        self.next_handle += 1
        self.calls.append(
            (self.next_handle, kind, blocks, request_id, threading.get_ident())
        )
        return self.next_handle

    def complete(self, handle, success=True):
        blocks = next(call[2] for call in self.calls if call[0] == handle)
        self.results.append(
            (handle, {key: SimpleNamespace(success=success) for key in blocks})
        )

    def poll_completed(self):
        self.polls += 1
        if self.fail_poll:
            raise RuntimeError("owner poll failed")
        while self.results:
            yield self.results.popleft()


class Runtime:
    def __init__(self, *args, **kwargs):
        self.client = Client()
        self.events = deque()
        self.removals = {}
        self.closed = False
        self.close_error = False

    def close(self):
        if self.close_error:
            raise RuntimeError("DMA still active")
        self.closed = True


def eventually(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("owner did not make progress")
        time.sleep(0.001)


class TestKVCRDirectLinker(CustomTestCase):
    def setUp(self):
        from sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker import (
            KVCRDirectLinker,
        )

        self.scope = get_context().override_server_args(model_path="dummy")
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)
        self.group = DevicePoolGroup(
            [
                DevicePoolEntry(
                    name=PoolName.KV,
                    indices_from_pool=PoolName.KV,
                    device_pool=None,
                    components=[[torch.zeros((4096, 4)) for _ in range(3)]],
                    layer_mapping={0: 0, 1: 1, 2: 2},
                    page_size=1,
                    rows_are_pages=True,
                )
            ],
            num_layers=3,
            page_size=1,
        )
        self.runtime = Runtime()
        module = "sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker"
        self.patches = [
            patch.dict(
                "sys.modules",
                {
                    "kvcr.types": SimpleNamespace(
                        RegionDescriptor=SimpleNamespace, MemoryRef=SimpleNamespace
                    )
                },
            ),
            patch(
                module + ".resolve_hybrid_device_pool_group", return_value=self.group
            ),
            patch(module + ".torch.cuda.Event", side_effect=Event),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.params = SimpleNamespace(
            page_size=1,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            tp_cache_group=None,
            attn_tp_cache_group=None,
        )
        self.linker = KVCRDirectLinker(
            SimpleNamespace(enable_linker_mla_dedup=False),
            self.params,
            components=set(),
            runtime_factory=lambda *args, **kwargs: self.runtime,
        )
        self.addCleanup(self.linker.close)
        self.client = self.runtime.client

    def transfer(self, keys, start=0):
        return [
            PoolTransfer(
                name=PoolName.KV,
                keys=keys,
                device_indices=torch.arange(start, start + len(keys)),
            )
        ]

    def test_layer_first_window_and_no_implicit_deposit(self):
        self.linker.load("a", self.transfer(["same"], 0))
        self.linker.load("b", self.transfer(["same"], 1))
        index = self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 4)
        self.assertTrue(all(call[1] == "deliver" for call in self.client.calls))
        first = next(iter(self.client.calls[0][2].values()))[0]
        second = next(iter(self.client.calls[1][2].values()))[0]
        self.assertEqual((first.element_index, second.element_index), (0, 1))
        # The second request's first layer may finish before the first request's.
        self.client.complete(2)
        eventually(lambda: len(self.client.calls) == 5)
        self.client.complete(1)
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(0)
        self.assertEqual(self.linker.num_completed_loads(), 0)
        eventually(lambda: len(self.client.calls) == 6)
        for handle in [6, 4, 5, 3]:
            self.client.complete(handle)
        eventually(lambda: self.linker.num_completed_loads() == 1)
        self.assertEqual(self.linker.pop_completed_load(), ["a", "b"])
        self.assertEqual(
            {call[4] for call in self.client.calls}, {self.linker._thread.ident}
        )
        self.assertTrue(all(call[1] == "deliver" for call in self.client.calls))

    def test_large_delivery_is_not_page_chunked_and_snapshots_indices(self):
        transfers = self.transfer([f"{i:064x}" for i in range(2051)])
        self.linker.load("large", transfers)
        transfers[0].device_indices.fill_(0)
        self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 3)
        self.assertEqual([len(call[2]) for call in self.client.calls], [2051] * 3)
        refs = [refs[0].element_index for refs in self.client.calls[0][2].values()]
        self.assertEqual(refs, list(range(2051)))
        for handle in [1, 2, 3]:
            self.client.complete(handle)

    def test_active_progress_skips_idle_wait_then_resumes_peer_polling(self):
        """An empty command queue must not pace completion and window refills."""
        active_waits, idle_waits = [], []
        get_command, submit = self.linker._commands.get, self.client._submit

        def get(block=True, timeout=None):
            if self.client.calls:
                waits = (
                    idle_waits if self.linker.num_completed_offloads() else active_waits
                )
                waits.append(timeout if block else 0)
            return get_command(block=block, timeout=timeout)

        def complete_immediately(kind, blocks, request_id):
            handle = submit(kind, blocks, request_id)
            self.client.complete(handle)
            return handle

        with (
            patch.object(self.linker._commands, "get", side_effect=get),
            patch.object(self.client, "_submit", side_effect=complete_immediately),
        ):
            for rid in ("a", "b"):
                self.linker.load(rid, self.transfer([rid]))
            self.linker.offload(self.transfer(["a"]))
            self.linker.start_layer_wise_loading()
            eventually(lambda: len(idle_waits) >= 2)
        self.assertEqual(self.linker.pop_completed_load(), ["a", "b"])
        self.assertTrue(self.linker.pop_completed_offload())
        self.assertEqual(len(self.client.calls), 7)
        self.assertTrue(active_waits)
        self.assertEqual(set(active_waits), {0})
        self.assertTrue(all(wait is not None and wait > 0 for wait in idle_waits))

    def test_offload_waits_for_producer_and_earlier_queued_load(self):
        self.linker.load("a", self.transfer(["a"]))
        first_ready, second_ready, load_ready = Event(), Event(), Event()
        second_ready.ready = load_ready.ready = False
        with patch(
            "torch.cuda.Event", side_effect=[first_ready, second_ready, load_ready]
        ):
            self.linker.offload(self.transfer(["a"]))
            self.linker.offload(self.transfer(["b"], 1))
            self.linker.start_layer_wise_loading()
        before = self.client.polls
        eventually(lambda: self.client.polls > before + 2)
        self.assertEqual(self.client.calls, [])
        load_ready.ready = True
        eventually(lambda: len(self.client.calls) == 3)
        self.assertTrue(all(call[1] == "deliver" for call in self.client.calls))
        for handle in [1, 3, 2]:
            self.client.complete(handle)
        eventually(lambda: len(self.client.calls) == 4)
        self.assertEqual(self.client.calls[-1][1], "deposit")
        self.client.complete(4)
        eventually(lambda: self.linker.num_completed_offloads() == 1)
        self.assertTrue(self.linker.pop_completed_offload())
        before = self.client.polls
        eventually(lambda: self.client.polls > before + 2)
        self.assertEqual(len(self.client.calls), 4)
        second_ready.ready = True
        eventually(lambda: len(self.client.calls) == 5)
        self.assertEqual(self.client.calls[-1][1], "deposit")
        self.client.complete(5)
        eventually(lambda: self.linker.num_completed_offloads() == 1)
        self.assertTrue(self.linker.pop_completed_offload())

    def test_hint_scope_cleanup_waits_for_accepted_load(self):
        a, b = CacheRequestHandle("a", 1), CacheRequestHandle("b", 2)
        hints = {"actions": [{"source": ["a"]}]}
        self.linker.set_request_context(a, hints)
        scope = next(iter(self.client.hints))
        hints["actions"][0]["source"].append("changed")
        self.assertEqual(self.client.hints[scope], {"actions": [{"source": ["a"]}]})
        self.linker.set_request_context(b, {"actions": [{"source": "b"}]})
        self.assertEqual(len(self.client.hints), 2)
        self.linker.load("a", self.transfer(["a"]))
        self.linker.release_request_context(a)
        next_attempt = CacheRequestHandle("a", 2)
        self.linker.set_request_context(next_attempt, {"actions": [{"source": "new"}]})
        self.linker.release_request_context(b)
        self.assertEqual(len(self.client.hints), 2)
        self.linker.release_request_context(next_attempt)
        self.assertEqual(self.client.hints, {scope: {"actions": [{"source": ["a"]}]}})
        self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 3)
        self.assertEqual({call[3] for call in self.client.calls}, {scope})
        for handle in [1, 2, 3]:
            self.client.complete(handle)
        eventually(lambda: not self.client.hints)

    def test_failure_holds_dma_ownership_until_quiesced(self):
        self.linker.load("a", self.transfer(["a"]))
        index = self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 3)
        self.runtime.events.append(SimpleNamespace(state="uncertain", op_handle=-9))
        for handle in [1, 2, 3]:
            self.client.complete(handle, success=False)
        self.linker.layer_done_counter.set_consumer(index)
        with self.assertRaisesRegex(RuntimeError, "KVCR"):
            self.linker.layer_done_counter.wait_until(0)
        self.assertEqual(self.linker.num_completed_loads(), 0)
        self.runtime.events.append(SimpleNamespace(state="quiesced", op_handle=-9))
        eventually(lambda: self.linker.num_completed_loads() == 1)
        self.assertEqual(self.linker.pop_completed_load(), ["a"])

    def test_owner_failure_rejects_commands_and_close_retains_resources(self):
        self.client.fail_poll = True
        eventually(lambda: self.linker._fatal is not None)
        with self.assertRaisesRegex(RuntimeError, "owner poll failed"):
            self.linker.lookup("a", self.transfer(["a"]))
        self.runtime.close_error = True
        with self.assertRaisesRegex(RuntimeError, "DMA still active"):
            self.linker.close()
        self.assertFalse(self.runtime.closed)
        self.assertTrue(self.linker._thread.is_alive())
        self.runtime.close_error = False
        self.linker.close()
        self.assertTrue(self.runtime.closed)

    def test_lookup_checks_all_and_trailing_boundaries_for_physical_pools(self):
        from sglang.srt.mem_cache.storage.kvcr.layout import KVCRLayout

        base = self.group.entries[0]
        # DeepSeek-V4's logical KV component has no physical pool named KV.
        base.name = PoolName.DEEPSEEK_V4_C4
        swa = DevicePoolEntry(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            device_pool=None,
            components=base.components,
            layer_mapping=base.layer_mapping,
            page_size=1,
            rows_are_pages=True,
        )
        self.linker.layout = KVCRLayout(
            DevicePoolGroup([base, swa], 3, 1), model_name="dummy", endpoint_name="test"
        )
        layout = self.linker.layout
        keys = ["a", "b", "c", "d", "e"]
        for key in keys:
            self.client.status[layout.encode_key(key, base.name)] = (
                SimpleNamespace(value="HIT"),
                SimpleNamespace(value="DRAM"),
            )
        for key in ["b", "c", "e"]:
            self.client.status[layout.encode_key(key, PoolName.SWA)] = (
                SimpleNamespace(value="FETCHABLE"),
                SimpleNamespace(value="REMOTE_G2"),
            )
        transfers = [
            PoolTransfer(PoolName.KV, keys=keys),
            PoolTransfer(
                PoolName.SWA, keys=keys[-2:], hit_policy=PoolHitPolicy.TRAILING_PAGES
            ),
        ]
        self.assertEqual(self.linker.lookup("a", transfers), [3])
        self.client.status[layout.encode_key("b", base.name)] = (
            SimpleNamespace(value="FETCHING"),
            SimpleNamespace(value="DRAM"),
        )
        self.assertEqual(self.linker.lookup("a", transfers), [])

    def test_failure_stops_unsubmitted_work_and_fails_dependent_offload(self):
        for i in range(3):
            self.linker.load(str(i), self.transfer([str(i)], i))
        index = self.linker.start_layer_wise_loading()
        self.linker.offload(self.transfer(["0"]))
        eventually(lambda: len(self.client.calls) == 4)
        self.client.complete(1, success=False)
        self.linker.layer_done_counter.set_consumer(index)
        with self.assertRaisesRegex(RuntimeError, "KVCR"):
            self.linker.layer_done_counter.wait_until(0)
        self.assertEqual(self.linker.num_completed_loads(), 0)
        for handle in [2, 3, 4]:
            self.client.complete(handle)
        eventually(lambda: self.linker.num_completed_offloads() == 1)
        self.assertFalse(self.linker.pop_completed_offload())
        self.assertEqual(len(self.client.calls), 4)

    def test_reset_quiesces_and_preserves_registered_counter(self):
        counter = self.linker.layer_done_counter
        self.linker.load("a", self.transfer(["a"]))
        self.linker.start_layer_wise_loading()
        replacement = Runtime()
        with patch.object(self.linker, "_make_runtime", return_value=replacement):
            self.linker.reset()
        self.assertTrue(self.runtime.closed)
        self.assertIs(self.linker.layer_done_counter, counter)
        self.assertEqual(self.linker.num_completed_loads(), 0)
        self.assertEqual(self.linker.start_layer_wise_loading(), -1)
        self.linker.load("b", self.transfer(["b"]))
        self.linker.start_layer_wise_loading()
        eventually(lambda: len(replacement.client.calls) == 3)
        for handle in [1, 2, 3]:
            replacement.client.complete(handle)
        eventually(lambda: self.linker.num_completed_loads() == 1)
        self.assertEqual(self.linker.pop_completed_load(), ["b"])

    def test_request_cleanup_after_close_and_shutdown_rejects_waiting_commands(self):
        handle = CacheRequestHandle("a", 0)
        self.linker.set_request_context(handle, {"actions": []})
        entered, release = threading.Event(), threading.Event()
        original_close = self.runtime.close

        def gated_close():
            entered.set()
            release.wait(5)
            original_close()

        self.runtime.close = gated_close
        errors = []
        close = threading.Thread(target=self.linker.close)
        close.start()
        self.assertTrue(entered.wait(5))

        def query():
            try:
                self.linker.lookup("a", self.transfer(["a"]))
            except RuntimeError as error:
                errors.append(str(error))

        waiter = threading.Thread(target=query)
        waiter.start()
        try:
            eventually(lambda: not self.linker._commands.empty())
        finally:
            release.set()
        close.join(5)
        waiter.join(5)
        self.assertFalse(close.is_alive())
        self.assertFalse(waiter.is_alive())
        self.assertEqual(errors, ["KVCR linker is closed"])
        self.linker.release_request_context(handle)

    def test_batches_complete_in_admission_order_and_idle_source_keeps_polling(self):
        self.linker.load("a", self.transfer(["a"]))
        first = self.linker.start_layer_wise_loading()
        self.linker.load("b", self.transfer(["b"], 1))
        second = self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 4)
        for handle in [4, 5, 6]:
            eventually(lambda: len(self.client.calls) >= handle)
            self.client.complete(handle)
        self.linker.layer_done_counter.set_consumer(second)
        self.linker.layer_done_counter.wait_until(2)
        self.assertEqual(self.linker.num_completed_loads(), 0)
        for handle in [3, 1, 2]:
            self.client.complete(handle)
        eventually(lambda: self.linker.num_completed_loads() == 2)
        self.assertEqual(self.linker.pop_completed_load(), ["a"])
        self.assertEqual(self.linker.pop_completed_load(), ["b"])
        self.linker.layer_done_counter.set_consumer(first)
        self.linker.layer_done_counter.wait_until(2)
        polls = self.client.polls
        eventually(lambda: self.client.polls > polls + 2)

    def test_removal_waits_for_its_offload_ack_but_not_unrelated_offloads(self):
        self.linker.offload(self.transfer(["a"]))
        eventually(lambda: len(self.client.calls) == 1)
        self.client.complete(1)
        eventually(lambda: self.linker.num_completed_offloads() == 1)
        key = self.linker.layout.encode_key("a", PoolName.KV)
        other_key = self.linker.layout.encode_key("b", PoolName.KV)
        self.runtime.removals.update({key: "a", other_key: "b"})
        # Another rank has not acknowledged this offload yet.
        self.assertEqual(self.linker.pop_storage_removals(), ["b"])
        self.assertTrue(self.linker.pop_completed_offload())
        self.assertEqual(self.linker.pop_storage_removals(), ["a"])

    def test_unaddressed_remote_shard_hints_do_not_admit_failed_deliveries(self):
        from sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker import (
            KVCRDirectLinker,
        )

        self.linker.close()
        self.params.attn_cp_size = 2
        runtime = Runtime()
        linker = KVCRDirectLinker(
            None,
            self.params,
            components=set(),
            runtime_factory=lambda *args, **kwargs: runtime,
        )
        self.addCleanup(linker.close)
        linker.set_request_context(
            CacheRequestHandle("a", 0), {"actions": [{"action_type": "kv.fetch"}]}
        )
        self.assertEqual(runtime.client.hints, {})
        key = linker.layout.encode_key("a", PoolName.KV)
        runtime.client.status[key] = (
            SimpleNamespace(value="HIT"),
            SimpleNamespace(value="DRAM"),
        )
        self.assertEqual(linker.lookup("a", self.transfer(["a"])), [1])

    def test_unrelated_hint_actions_preserve_local_lookup(self):
        from sglang.srt.managers.kv_hints import decode_kv_hints_envelope

        hint = decode_kv_hints_envelope(
            {"protocol_version": "0.1", "message_id": "other", "actions": []}
        )
        with patch.object(
            self.client, "submit_hint", side_effect=ValueError("no kv.fetch action")
        ):
            self.linker.set_request_context(CacheRequestHandle("a", 0), hint)
        key = self.linker.layout.encode_key("a", PoolName.KV)
        self.client.status[key] = (
            SimpleNamespace(value="HIT"),
            SimpleNamespace(value="DRAM"),
        )
        self.assertEqual(self.linker.lookup("a", self.transfer(["a"])), [1])


class TestKVCRRuntime(CustomTestCase):
    def setUp(self):
        import sys
        from types import ModuleType

        from sglang.srt.mem_cache.storage.kvcr.runtime import KVCRRuntime

        self.runtime_type = KVCRRuntime
        self.startup_error = type("KVCRStartupError", (RuntimeError,), {})
        self.local_tier = object()
        self.constructor_error = None
        self.close_error = False

        def close():
            if self.close_error:
                raise RuntimeError("DMA is still active")

        def client(config, bindings, backends):
            if self.constructor_error is not None:
                raise self.constructor_error
            return SimpleNamespace(
                config=config, bindings=bindings, backends=backends, close=close
            )

        modules = {
            name: ModuleType(name)
            for name in ("kvcr", "kvcr.config", "kvcr.control_channels", "kvcr.types")
        }
        modules["kvcr"].KVCR = client
        modules["kvcr"].KVCRBindings = SimpleNamespace
        config_module = modules["kvcr.config"]
        config_module.KVCRConfig = SimpleNamespace
        config_module.KVCRBackendConfigs = SimpleNamespace
        config_module.LocalDramOptions = lambda pools: SimpleNamespace(pools=pools)
        config_module.RemoteFWDramOptions = SimpleNamespace
        modules["kvcr.control_channels"].ZmqPeerControlChannel = (
            lambda host, port, advertise: SimpleNamespace(
                host=host, port=port, advertise=advertise
            )
        )
        modules["kvcr.types"].KVCRStartupError = self.startup_error
        modules["kvcr.types"].CacheTier = SimpleNamespace(LOCAL_G2=self.local_tier)
        scoped_modules = patch.dict(sys.modules, modules)
        scoped_modules.start()
        self.addCleanup(scoped_modules.stop)
        self.layout = SimpleNamespace(
            endpoint_name="test-rank",
            pool_layouts=[("small", 8), ("large", 24)],
            pool_counts={"small": 2, "large": 2},
            registrations=[],
            decode_key=lambda key: tuple(key.decode().split(":", 1)),
        )

    def make_runtime(self):
        runtime = self.runtime_type(
            self.layout,
            {"dram_size_gb": 128 / (1 << 30), "advertise_host": "127.0.0.1"},
            rank=2,
        )
        self.addCleanup(runtime.close)
        return runtime

    def test_capacity_preserves_equal_page_counts_and_shifts_ports(self):
        runtime = self.make_runtime()
        self.assertEqual(
            [size for _, _, size in runtime.client.backends.local_dram.pools],
            [32, 96],
        )
        self.assertEqual(runtime.client.config.nixl_listen_port, 20502)
        self.assertEqual(runtime.client.bindings.framework_control.port, 19502)
        self.assertFalse(runtime.client.backends.remote_fw_dram.opportunistic_query)

    def test_declines_pins_and_cancels_removals_per_physical_key(self):
        runtime = self.make_runtime()
        bindings = runtime.client.bindings
        request = bindings.request_pin([b"small:1234"])
        cancelled = bindings.request_pin([b"large:1234"])
        bindings.cancel_pin_request(cancelled)
        self.assertEqual(list(bindings.poll_pin_results()), [(request, None)])
        self.assertEqual(list(bindings.poll_pin_results()), [])
        for keys, removed in (
            ((b"small:1234", b"large:1234"), True),
            ((b"small:1234",), False),
        ):
            bindings.inventory_sink(
                SimpleNamespace(keys=keys, removed=removed, tier=self.local_tier)
            )
        self.assertEqual(list(runtime.removals.values()), ["1234"])
        runtime.removals.clear()
        bindings.inventory_sink(
            SimpleNamespace(keys=(b"large:1234",), removed=False, tier=self.local_tier)
        )
        self.assertFalse(runtime.removals)

    def test_failed_close_retains_mappings_until_success(self):
        runtime = self.make_runtime()
        buffers = tuple(runtime._buffers)
        self.close_error = True
        try:
            with self.assertRaisesRegex(RuntimeError, "DMA is still active"):
                runtime.close()
            self.assertTrue(all(not buffer.closed for buffer in buffers))
        finally:
            self.close_error = False
        runtime.close()
        self.assertTrue(all(buffer.closed for buffer in buffers))

    def test_nonquiescent_startup_attaches_retained_runtime(self):
        self.constructor_error = self.startup_error("native resources retained")
        with self.assertRaises(self.startup_error) as caught:
            self.make_runtime()
        runtime = caught.exception.kvcr_runtime
        self.addCleanup(runtime._close_buffers)
        self.assertTrue(runtime._buffers)
        self.assertTrue(all(not buffer.closed for buffer in runtime._buffers))
        with self.assertRaisesRegex(RuntimeError, "startup"):
            runtime.close()

    def test_invalid_options_fail_before_allocation(self):
        for options in (
            {"unknown_option": 1},
            {"dram_size_gb": 0},
            {"dram_size_gb": float("nan")},
            {"dram_size_gb": 1e-20},
            {"control_port": 65535},
            {"operation_timeout_ms": 0},
            {"abandon_timeout_ms": 1000},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.runtime_type(self.layout, options, rank=2)


if __name__ == "__main__":
    unittest.main()
