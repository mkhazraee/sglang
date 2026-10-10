from __future__ import annotations

import ctypes
import math
import mmap
import socket
from collections import deque


class KVCRRuntime:
    """KVCR resources constructed, polled, and closed by the linker owner thread."""

    def __init__(self, layout, options: dict, *, rank: int):
        defaults = {
            "dram_size_gb": 1,
            "control_host": "0.0.0.0",
            "advertise_host": None,
            "control_port": 19500,
            "nixl_port": 20500,
            "operation_timeout_ms": 1000,
            "abandon_timeout_ms": 5000,
        }
        unknown = options.keys() - defaults.keys()
        if unknown:
            raise ValueError(f"Unknown KVCR options: {sorted(unknown)}")
        config = defaults | options
        if type(rank) is not int or rank < 0:
            raise ValueError("KVCR rank must be a nonnegative integer")
        size_gb = config["dram_size_gb"]
        if (
            isinstance(size_gb, bool)
            or not isinstance(size_gb, (int, float))
            or not math.isfinite(size_gb)
            or size_gb <= 0
        ):
            raise ValueError("KVCR dram_size_gb must be finite and positive")
        page_bytes = sum(
            size * layout.pool_counts[name] for name, size in layout.pool_layouts
        )
        page_count = int(size_gb * (1 << 30)) // page_bytes
        if page_count < 1:
            raise ValueError("KVCR DRAM capacity must fit at least one complete page")
        for name in ("control_port", "nixl_port"):
            port = config[name]
            if type(port) is not int or not 1 <= port + rank <= 65535:
                raise ValueError(f"KVCR {name} plus rank must be in [1, 65535]")
            config[name] += rank
        if config["control_port"] == config["nixl_port"]:
            raise ValueError("KVCR control_port and nixl_port must differ")
        for name in ("operation_timeout_ms", "abandon_timeout_ms"):
            if type(config[name]) is not int or config[name] <= 0:
                raise ValueError(f"KVCR {name} must be a positive integer")
        if config["abandon_timeout_ms"] < 2 * config["operation_timeout_ms"]:
            raise ValueError(
                "KVCR abandon_timeout_ms must be at least twice operation_timeout_ms"
            )
        if config["advertise_host"] is None:
            config["advertise_host"] = socket.gethostbyname(socket.gethostname())
        for name in ("control_host", "advertise_host"):
            if not isinstance(config[name], str) or not config[name]:
                raise ValueError(f"KVCR {name} must be a nonempty host")

        from kvcr import KVCR, KVCRBindings
        from kvcr.config import (
            KVCRBackendConfigs,
            KVCRConfig,
            LocalDramOptions,
            RemoteFWDramOptions,
        )
        from kvcr.control_channels import ZmqPeerControlChannel
        from kvcr.types import CacheTier, KVCRStartupError

        self._layout = layout
        self.events = deque()
        self.removals = {}
        self._buffers = []
        self._pin_results = {}
        self._next_pin = 0
        self.client = None

        def inventory(event):
            if event.tier is CacheTier.LOCAL_G2:
                for key in event.keys:
                    decoded = layout.decode_key(key)
                    if decoded is not None:
                        if event.removed:
                            self.removals[key] = decoded[1]
                        else:
                            self.removals.pop(key, None)

        try:
            pools = []
            for name, size in layout.pool_layouts:
                length = page_count * layout.pool_counts[name] * size
                buffer = mmap.mmap(-1, length)
                self._buffers.append(buffer)
                address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
                pools.append((name, address, length))
            self.client = KVCR(
                KVCRConfig(
                    nixl_agent_name=layout.endpoint_name,
                    pool_layouts=layout.pool_layouts,
                    nixl_listen_port=config["nixl_port"],
                    operation_timeout_ms=config["operation_timeout_ms"],
                    abandon_timeout_ms=config["abandon_timeout_ms"],
                ),
                KVCRBindings(
                    request_pin=self._decline_pin,
                    poll_pin_results=self._poll_pin_results,
                    release_pin=lambda _: True,
                    cancel_pin_request=lambda request: self._pin_results.pop(
                        request, None
                    ),
                    framework_control=ZmqPeerControlChannel(
                        config["control_host"],
                        config["control_port"],
                        config["advertise_host"],
                    ),
                    key_adapter=self,
                    inventory_sink=inventory,
                    on_resilience_event=self.events.append,
                ),
                KVCRBackendConfigs(
                    framework_regions=layout.registrations,
                    local_dram=LocalDramOptions(pools),
                    remote_fw_dram=RemoteFWDramOptions(opportunistic_query=False),
                ),
            )
        except BaseException as error:
            if isinstance(error, KVCRStartupError) or not isinstance(error, Exception):
                # The owner retains this error until process exit; native work may survive.
                error.kvcr_runtime = self
            else:
                self._close_buffers()
            raise

    def _decline_pin(self, keys):
        self._next_pin += 1
        self._pin_results[self._next_pin] = None
        return self._next_pin

    def _poll_pin_results(self):
        results = list(self._pin_results.items())
        self._pin_results.clear()
        return results

    def encode(self, key: object) -> bytes:
        if not isinstance(key, bytes):
            raise TypeError("KVCR keys must be encoded layout keys")
        return key

    def decode(self, key: bytes) -> int:
        decoded = self._layout.decode_key(key)
        if decoded is None:
            raise ValueError("Unknown KVCR layout key")
        return int(decoded[1][:16], 16)

    def close(self) -> None:
        if self.client is None:
            raise RuntimeError("KVCR startup did not establish safe resource ownership")
        self.client.close()
        self._close_buffers()

    def _close_buffers(self) -> None:
        while self._buffers:
            self._buffers[-1].close()
            self._buffers.pop()
