from __future__ import annotations

import dataclasses
import logging
import struct
import threading
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    filter_kv_indices_for_cp_rank,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

GUARD = b"NixlMsgGuard"


def _compute_pool_group_boundaries(kv_item_lens: List[int]) -> List[tuple]:
    """Group consecutive layers by item_len into pool groups.

    Returns list of (start, end) index ranges into kv_data_ptrs / kv_item_lens.

    Uniform models  → [(0, num_layers)]          — single group, no change.
    V4 multi-pool   → [(0, N_c4), (N_c4, N_tot)] — two groups (c4 + c128).
    """
    if not kv_item_lens:
        return []
    groups: List[tuple] = []
    start = 0
    for i in range(1, len(kv_item_lens)):
        if kv_item_lens[i] != kv_item_lens[start]:
            groups.append((start, i))
            start = i
    groups.append((start, len(kv_item_lens)))
    return groups


def _pack_extra_pool_indices(arrays: List[npt.NDArray[np.int32]]) -> bytes:
    """Pack a list of int32 arrays into a single byte string for ZMQ transport.

    Format: uint32 n_arrays, then n_arrays × (uint32 nbytes, raw bytes).
    """
    parts = [struct.pack("I", len(arrays))]
    for arr in arrays:
        b = arr.astype(np.int32).tobytes()
        parts.append(struct.pack("I", len(b)))
        parts.append(b)
    return b"".join(parts)


def _unpack_extra_pool_indices(data: bytes) -> List[npt.NDArray[np.int32]]:
    """Inverse of _pack_extra_pool_indices."""
    view = memoryview(data)
    offset = 0
    n = struct.unpack_from("I", view, offset)[0]
    offset += 4
    arrays = []
    for _ in range(n):
        nbytes = struct.unpack_from("I", view, offset)[0]
        offset += 4
        arrays.append(np.frombuffer(bytes(view[offset : offset + nbytes]), dtype=np.int32))
        offset += nbytes
    return arrays


@dataclasses.dataclass
class TransferInfo:
    """Contains indices for a transfer, sent by KVReceiver. Received by prefill bootstrap thread."""

    room: int
    endpoint: str
    dst_port: int
    agent_name: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    dst_state_indices: List[int]
    # Per-pool dst kv indices for V4 multi-pool models (pool groups beyond the first).
    # Empty for uniform models (backward compat). Index 0 is always dst_kv_indices.
    dst_kv_indices_extra: List[npt.NDArray[np.int32]] = dataclasses.field(
        default_factory=list
    )

    def is_dummy(self):
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        # Parse state_indices from msg[7] if present
        if len(msg) > 7 and msg[7] != b"":
            dst_state_indices = list(np.frombuffer(msg[7], dtype=np.int32))
        else:
            dst_state_indices = []

        # msg[8] (new, V4): extra pool indices packed as:
        #   uint32 n_extra, then n_extra × (uint32 nbytes, bytes)
        dst_kv_indices_extra = []
        if len(msg) > 8 and msg[8] != b"":
            dst_kv_indices_extra = _unpack_extra_pool_indices(msg[8])

        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
            dst_state_indices=dst_state_indices,
            dst_kv_indices_extra=dst_kv_indices_extra,
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Contains base pointers and other info which only needs to be sent once by KVReceiver. Received by prefill bootstrap thread."""

    room: str
    endpoint: str
    dst_port: int
    agent_name: str
    agent_metadata: bytes
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: list[int]
    gpu_id: int
    decode_tp_size: int
    decode_tp_rank: int
    dst_kv_item_len: int
    dst_num_slots: Optional[int] = None
    dst_num_slots_state: Optional[int] = None
    dst_num_slots_per_layer: list[int] = dataclasses.field(default_factory=list)
    dst_state_item_lens: list[int] = dataclasses.field(default_factory=list)
    dst_state_dim_per_tensor: list[int] = dataclasses.field(default_factory=list)

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        # Parse state_data_ptrs from msg[7] if present
        if len(msg) > 7 and msg[7] != b"":
            dst_state_data_ptrs = list(struct.unpack(f"{len(msg[7]) // 8}Q", msg[7]))
        else:
            dst_state_data_ptrs = []

        dst_state_item_lens = []
        dst_state_dim_per_tensor = []
        dst_num_slots_per_layer = []
        if len(msg) > 12 and len(msg[12]) > 0:
            dst_state_item_lens = list(struct.unpack(f"{len(msg[12]) // 4}I", msg[12]))
        if len(msg) > 13 and len(msg[13]) > 0:
            dst_state_dim_per_tensor = list(
                struct.unpack(f"{len(msg[13]) // 4}I", msg[13])
            )
        if len(msg) > 16 and len(msg[16]) > 0:
            dst_num_slots_per_layer = list(struct.unpack(f"{len(msg[16]) // 4}I", msg[16]))

        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            agent_metadata=msg[4],
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])),
            dst_state_data_ptrs=dst_state_data_ptrs,
            gpu_id=int(msg[8].decode("ascii")),
            decode_tp_size=int(msg[9].decode("ascii")),
            decode_tp_rank=int(msg[10].decode("ascii")),
            dst_kv_item_len=int(msg[11].decode("ascii")),
            dst_num_slots=int(msg[14].decode("ascii")) if len(msg) > 14 else None,
            dst_num_slots_state=int(msg[15].decode("ascii")) if len(msg) > 15 else None,
            dst_num_slots_per_layer=dst_num_slots_per_layer,
            dst_state_item_lens=dst_state_item_lens,
            dst_state_dim_per_tensor=dst_state_dim_per_tensor,
        )


@dataclasses.dataclass
class TransferStatus:
    """Used by KV Receiver to know when a transfer is done."""

    # KV chunks received per pp_rank: {pp_rank: set of chunk_ids}
    received_kvs_per_pp: Dict[int, Set[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )
    # Expected chunk count per pp_rank (set when is_last=True): {pp_rank: expected_count}
    expected_kvs_per_pp: Dict[int, int] = dataclasses.field(default_factory=dict)
    # Number of PP ranks expected to send data.
    num_pp_ranks_expected: Optional[int] = None
    # Whether aux data has been received.
    received_aux: bool = False
    # PP ranks that have sent state data (state is layer-specific, each PP rank sends its portion).
    received_state_per_pp: Set[int] = dataclasses.field(default_factory=set)
    # Whether state data is expected (set based on state_type).
    expects_state: bool = False
    # Mark as failed
    is_failure: bool = False

    def is_done(self):
        if self.is_failure:
            return True
        if self.num_pp_ranks_expected is None or not self.received_aux:
            return False
        # If state data is expected, check all PP ranks have sent it
        if (
            self.expects_state
            and len(self.received_state_per_pp) < self.num_pp_ranks_expected
        ):
            return False
        # All PP ranks must have reported their expected count
        if len(self.expected_kvs_per_pp) < self.num_pp_ranks_expected:
            return False
        # Each PP rank must have received all expected chunks
        for pp_rank, expected in self.expected_kvs_per_pp.items():
            if len(self.received_kvs_per_pp[pp_rank]) != expected:
                return False
        return True


def expand_page_indices_for_slice(
    page_indices: npt.NDArray[np.int32],
    num_ptr_pairs: int,
    num_slots: int,
    page_size: int,
    num_groups: int = 1,
    head_group_idx: int = 0,
) -> npt.NDArray[np.int32]:
    """Map page slot indices to flat dlist indices for the slice prepped path.

    Dlist layout: num_ptr_pairs blocks of (num_slots * page_size * num_groups),
    with [slot, token, group] interleaving. head_group_idx selects one group (0 for dst).

    E.g. page_indices=[1], page_size=2, num_slots=4, num_groups=2, head_group_idx=1:
      → [1*(2*2)+0*2+1, 1*(2*2)+1*2+1] = [5, 7]
    """
    token_offsets = np.arange(page_size, dtype=np.int32)
    pair_stride = num_slots * page_size * num_groups
    within_pair = (
        page_indices[:, None] * (page_size * num_groups)
        + token_offsets[None, :] * num_groups
        + head_group_idx
    ).ravel()
    pair_offsets = np.arange(num_ptr_pairs, dtype=np.int64) * pair_stride
    return (pair_offsets[:, None] + within_pair[None, :]).ravel().astype(np.int32)


def repeat_indices_over_layers(
    indices: npt.NDArray[np.int32], layer_lengths: List[int]
) -> npt.NDArray[np.int32]:
    """Map per-slot token indices to flat indices in a pre-built descriptor list.

    Given indices [1, 3] and layer_lengths [M, M] (two layers with M slots each),
    returns [1, 3, M+1, M+3] — the flat dlist positions across all layers concatenated.
    Works uniformly for both MLA (one ptr/layer) and MHA (K+V ptrs, 2×N entries).
    """
    offsets = np.cumsum([0] + layer_lengths[:-1])
    return (offsets[:, None] + indices[None, :]).ravel().astype(np.int32)


def _make_req_array(
    addr_chunks: list, len_chunks: list, gpu: int
) -> npt.NDArray[np.int64]:
    """Build a NIXL request array from per-layer address and length arrays."""
    if not addr_chunks:
        return np.empty((0, 3), dtype=np.int64)
    flat_addrs = np.concatenate(addr_chunks)
    flat_lens = np.concatenate(len_chunks)
    return np.column_stack((flat_addrs, flat_lens, np.full_like(flat_addrs, gpu)))


class NixlKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError as e:
            raise ImportError(
                "Please install NIXL by following the instructions at "
                "https://github.com/ai-dynamo/nixl/blob/main/README.md "
                "to run SGLang with NixlTransferEngine."
            ) from e

        backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
        agent_config = nixl_agent_config(
            backends=[backend],
            num_threads=(8 if disaggregation_mode == DisaggregationMode.PREFILL else 0),
        )
        self.agent = nixl_agent(str(uuid.uuid4()), agent_config)

        available_plugins = self.agent.get_plugin_list()
        if backend not in available_plugins:
            raise ValueError(
                f"NIXL backend '{backend}' not found. Available: {available_plugins}. "
                f"Please install the required NIXL plugin or choose from: {available_plugins}"
            )
        logger.info(f"NIXL KVManager initialized with backend: {backend}")

        self.register_buffer_to_engine()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Pool group boundaries: [(start, end), ...] grouping layers by item_len.
            # Single entry for uniform models; multiple for V4 multi-pool.
            self._pool_group_boundaries: List[tuple] = _compute_pool_group_boundaries(
                self.kv_args.kv_item_lens
            )
            self.use_prepped_transfer = True  # Set to False to fall back to non-prepped path (for debugging).
            if self.use_prepped_transfer:
                self.prep_handles = {}  # peer_name -> combined KV+state dlist handle
                self.prep_handle_slice_src: Optional[tuple] = None  # (handle, num_groups, num_ptr_pairs, num_slots)
                self.prep_handles_slice_dst: Dict[str, tuple] = {}  # peer_name -> (handle, num_slots)
                self.peer_head_group: Dict[str, int] = {}  # peer_name -> head_group_idx
                # Per-layer slot counts (V4: non-uniform item_lens across pool groups).
                # Scalar _num_slots_src kept as alias of [0] for backward compatibility.
                self._slots_per_kv_layer: list[int] = [
                    dl // il
                    for dl, il in zip(self.kv_args.kv_data_lens, self.kv_args.kv_item_lens)
                ]
                self._num_slots_src: int = self._slots_per_kv_layer[0]
                self._num_slots_src_state: Optional[int] = None
                self._slots_per_state_layer: list[int] = []
                # Offset of state entries in the combined dlist (= sum of per-layer KV slot counts).
                self._kv_dlist_src_size: int = sum(self._slots_per_kv_layer)
                self._kv_dlist_dst_size: Dict[str, int] = {}  # peer_name -> KV entry count in dst dlist
                self._dst_slots_per_kv_layer: Dict[str, list[int]] = {}  # peer_name -> per-layer dst slot counts
                state_type = getattr(self.kv_args, "state_type", "none")
                if state_type in ("nsa", "swa") and self.kv_args.state_data_ptrs:
                    self._slots_per_state_layer = [
                        dl // il
                        for dl, il in zip(self.kv_args.state_data_lens, self.kv_args.state_item_lens)
                    ]
                    self._num_slots_src_state = self._slots_per_state_layer[0]
                self._init_prep_handle(
                    "", self.kv_args.kv_data_ptrs, self.kv_args.gpu_id,
                    num_slots=self._slots_per_kv_layer,
                    state_ptrs=self.kv_args.state_data_ptrs if self._num_slots_src_state is not None else None,
                    num_slots_state=self._slots_per_state_layer if self._slots_per_state_layer else None,
                )
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                TransferStatus
            )
            self._start_heartbeat_checker_thread()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def _start_heartbeat_checker_thread(self):
        """
        Start the heartbeat checker thread for Decode worker.
        TODO (smor): unite nixl heartbeat checker with mooncake's.
        """

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_info_table.keys())

                for bootstrap_addr in addresses:
                    failed = False
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            failed = True
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        failed = True
                    if failed:
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=heartbeat_checker, daemon=True).start()

    def _handle_node_failure(self, failed_bootstrap_addr):
        """Handle failure of a prefill node."""
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            self.prefill_info_table.pop(failed_bootstrap_addr, None)

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            self.addr_to_rooms_tracker.pop(failed_bootstrap_addr, None)

        # Mark all pending transfers associated with the failed node as failed
        affected_rooms = []
        for room in possible_affected_rooms:
            status = self.transfer_statuses.get(room)
            if status is not None and not status.is_done():
                status.is_failure = True
                affected_rooms.append(room)

        logger.error(
            f"Lost connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), "
            f"{len(affected_rooms)} transfers affected"
        )
        for room in affected_rooms:
            logger.error(f"Let room {room} be failed due to prefill down")
            self.update_status(room, KVPoll.Failed)

    def register_buffer_to_engine(self):
        kv_addrs = [
            (ptr, length, self.kv_args.gpu_id, "")
            for ptr, length in zip(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens)
        ]
        self.kv_descs = self.agent.register_memory(kv_addrs, "VRAM")
        logger.debug(f"Register kv tensors, len(kv_addr)= {len(kv_addrs)}")
        if not self.kv_descs:
            raise Exception("NIXL memory registration failed for kv tensors")
        aux_addrs = [
            (ptr, length, 0, "")
            for ptr, length in zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens)
        ]
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM")
        logger.debug(f"Register aux tensors, len(aux_addrs)= {len(aux_addrs)}")
        if not self.aux_descs:
            raise Exception("NIXL memory registration failed for aux tensors")

        # Register state/extra pool data buffers if present
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            state_addrs = [
                (ptr, length, self.kv_args.gpu_id, "")
                for ptr, length in zip(
                    self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
                )
            ]
            self.state_descs = self.agent.register_memory(state_addrs, "VRAM")
            logger.debug(
                f"Register state tensors, len(state_addrs)= {len(state_addrs)}"
            )
            if not self.state_descs:
                raise Exception("NIXL memory registration failed for state tensors")

    def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo):
        agent_name = decode_kv_args.agent_name
        if agent_name in self.decode_kv_args_table:
            logger.info(f"Peer {agent_name} was already registered, ignoring.")
            return
        self.decode_kv_args_table[agent_name] = decode_kv_args
        self.agent.add_remote_agent(decode_kv_args.agent_metadata)
        if self.use_prepped_transfer:
            if self.is_mla_backend or decode_kv_args.decode_tp_size == self.attn_tp_size:
                # Safe to use prefill's kv_item_lens for the dst dlist stride:
                # equal_tp guarantees identical heads-per-rank (same item_len);
                # MLA latent shape is TP-invariant.
                #
                # Per-layer dst slot counts: use wire-provided list if present (V4),
                # else fall back to scalar broadcast.
                if decode_kv_args.dst_num_slots_per_layer:
                    dst_slots_per_kv_layer = decode_kv_args.dst_num_slots_per_layer
                else:
                    dst_num_slots = (
                        decode_kv_args.dst_num_slots
                        if decode_kv_args.dst_num_slots is not None
                        else self._num_slots_src
                    )
                    dst_slots_per_kv_layer = [dst_num_slots] * len(self.kv_args.kv_item_lens)
                self._dst_slots_per_kv_layer[agent_name] = dst_slots_per_kv_layer
                self._kv_dlist_dst_size[agent_name] = sum(dst_slots_per_kv_layer)
                self._init_prep_handle(
                    agent_name, decode_kv_args.dst_kv_ptrs, decode_kv_args.gpu_id,
                    num_slots=dst_slots_per_kv_layer,
                    state_ptrs=decode_kv_args.dst_state_data_ptrs if self._num_slots_src_state is not None else None,
                    num_slots_state=decode_kv_args.dst_num_slots_state,
                )
            else:
                self._init_prep_handle_slice(agent_name, decode_kv_args)

    def _build_prep_entries(
        self,
        ptrs: list[int],
        item_lens: list[int],
        data_lens: list[int],
        gpu_id: int,
        num_slots: Optional[Union[int, list[int]]] = None,
    ) -> list:
        """Build descriptor arrays for a set of buffers (one array per layer/ptr).

        num_slots can be a scalar (uniform), a per-layer list (V4 multi-pool),
        or None (inferred from data_len // item_len).
        """
        arrays = []
        for i, (base_ptr, item_len, data_len) in enumerate(zip(ptrs, item_lens, data_lens)):
            if num_slots is None:
                n = data_len // item_len
            elif isinstance(num_slots, list):
                n = num_slots[i]
            else:
                n = num_slots
            addrs = np.arange(n, dtype=np.int64) * item_len + base_ptr
            arrays.append(np.column_stack([
                addrs,
                np.full(n, item_len, dtype=np.int64),
                np.full(n, gpu_id, dtype=np.int64),
            ]))
        return arrays

    def _init_prep_handle(
        self,
        peer_name: str,
        kv_ptrs: list[int],
        gpu_id: int,
        num_slots: Optional[Union[int, list[int]]] = None,
        state_ptrs: Optional[list[int]] = None,
        num_slots_state: Optional[Union[int, list[int]]] = None,
    ):
        """Pre-build a combined NIXL dlist: KV layers followed by state layers (if any).

        peer_name="" = src side; agent name = dst side. num_slots overrides the local
        KV slot count; num_slots_state overrides the state pool slot count.
        State entries are appended after all KV entries — the offset is _kv_dlist_src_size
        (src) or _kv_dlist_dst_size[peer_name] (dst), used during index computation.
        """
        arrays = self._build_prep_entries(
            kv_ptrs, self.kv_args.kv_item_lens, self.kv_args.kv_data_lens, gpu_id, num_slots,
        )
        if state_ptrs:
            arrays += self._build_prep_entries(
                state_ptrs, self.kv_args.state_item_lens, self.kv_args.state_data_lens,
                gpu_id, num_slots_state,
            )
        handle = self.agent.prep_xfer_dlist(peer_name, np.vstack(arrays), "VRAM")
        assert handle is not None, f"prep_xfer_dlist returned None for peer '{peer_name}'"
        self.prep_handles[peer_name] = handle

    def _init_prep_handle_slice(
        self, peer_name: str, decode_kv_args: KVArgsRegisterInfo
    ):
        """Pre-build NIXL dlists for TP-heterogeneous slice transfers.

        Src dlist shared across decode peers (same TP size). prefill_tp < decode_tp:
        interleave num_groups per token, peers select via head_group_idx.
        prefill_tp > decode_tp: num_groups=1. Dst dlist is per-peer.
        """
        decode_tp_size = decode_kv_args.decode_tp_size
        decode_tp_rank = decode_kv_args.decode_tp_rank
        dst_kv_item_len = decode_kv_args.dst_kv_item_len
        dst_gpu_id = decode_kv_args.gpu_id
        prefill_tp_size = self.attn_tp_size

        src_kv_item_len = self.kv_args.kv_item_lens[0]
        page_size = self.kv_args.page_size
        num_slots = self.kv_args.kv_data_lens[0] // src_kv_item_len

        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * prefill_tp_size

        src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // decode_tp_size)
        bytes_per_head_slice = dst_kv_item_len // page_size // dst_heads_per_rank

        if prefill_tp_size > decode_tp_size:
            # Multiple prefill ranks feed one decode rank: each prefill rank sends
            # all its src heads to a specific head-range in the decode rank.
            src_replication = max(1, prefill_tp_size // total_kv_heads)
            local_tp_rank_in_group = self.kv_args.engine_rank % prefill_tp_size
            num_groups = 1
            num_heads_to_send = src_heads_per_rank
            head_group_idx = 0
            unique_head_idx = local_tp_rank_in_group // src_replication
            dst_head_start = (unique_head_idx * src_heads_per_rank) % dst_heads_per_rank
            dst_head_offset = dst_head_start * bytes_per_head_slice
        else:
            # One prefill rank feeds multiple decode ranks: interleave num_groups
            # head-groups in the src dlist so each decode rank picks its slice.
            dst_tp_rank_in_group = decode_tp_rank % decode_tp_size
            num_groups = decode_tp_size // prefill_tp_size
            num_heads_to_send = dst_heads_per_rank
            src_head_start = (dst_tp_rank_in_group * dst_heads_per_rank) % src_heads_per_rank
            head_group_idx = src_head_start // dst_heads_per_rank
            dst_head_offset = 0

        bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice
        bytes_per_token_src = src_kv_item_len // page_size
        bytes_per_token_dst = dst_kv_item_len // page_size

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_pp = (
            self.get_mha_kv_ptrs_with_pp(
                self.kv_args.kv_data_ptrs, decode_kv_args.dst_kv_ptrs
            )
        )
        src_ptrs = list(src_k_ptrs[:layers_pp]) + list(src_v_ptrs[:layers_pp])
        dst_ptrs = list(dst_k_ptrs[:layers_pp]) + list(dst_v_ptrs[:layers_pp])
        num_ptr_pairs = len(src_ptrs)

        slots = np.arange(num_slots, dtype=np.int64)
        tokens = np.arange(page_size, dtype=np.int64)  # reused in dst dlist below
        groups = np.arange(num_groups, dtype=np.int64)

        # Src dlist built once and shared. For heterogeneous decode TP, key by (decode_tp_size, layers_pp).
        if self.prep_handle_slice_src is None:
            # (ptr, slot, token, group) → ravel; groups interleaved per token.
            src_ptrs_arr = np.array(src_ptrs, dtype=np.int64)
            addrs = (
                src_ptrs_arr[:, None, None, None]
                + slots[None, :, None, None] * src_kv_item_len
                + tokens[None, None, :, None] * bytes_per_token_src
                + groups[None, None, None, :] * bytes_per_token_to_send
            ).ravel()
            src_array = np.column_stack([
                addrs,
                np.full(len(addrs), bytes_per_token_to_send, dtype=np.int64),
                np.full(len(addrs), self.kv_args.gpu_id, dtype=np.int64),
            ])
            src_handle = self.agent.prep_xfer_dlist("", src_array, "VRAM")
            assert src_handle is not None, (
                f"prep_xfer_dlist returned None for slice src (decode_tp_size={decode_tp_size})"
            )
            self.prep_handle_slice_src = (src_handle, num_groups, num_ptr_pairs, num_slots)

        # Dst dlist per-peer; use decode's slot count (may exceed prefill's).
        num_slots_dst = decode_kv_args.dst_num_slots if decode_kv_args.dst_num_slots is not None else num_slots
        dst_slots = np.arange(num_slots_dst, dtype=np.int64)
        # (ptr, slot, token) → ravel.
        dst_ptrs_arr = np.array(dst_ptrs, dtype=np.int64)
        addrs = (
            dst_ptrs_arr[:, None, None]
            + dst_slots[None, :, None] * dst_kv_item_len
            + tokens[None, None, :] * bytes_per_token_dst
            + dst_head_offset
        ).ravel()
        dst_array = np.column_stack([
            addrs,
            np.full(len(addrs), bytes_per_token_to_send, dtype=np.int64),
            np.full(len(addrs), dst_gpu_id, dtype=np.int64),
        ])
        dst_handle = self.agent.prep_xfer_dlist(peer_name, dst_array, "VRAM")
        assert dst_handle is not None, (
            f"prep_xfer_dlist returned None for slice dst for peer '{peer_name}'"
        )
        self.prep_handles_slice_dst[peer_name] = (dst_handle, num_slots_dst)
        self.peer_head_group[peer_name] = head_group_idx

    def _send_kvcache_generic(
        self,
        peer_name: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
        prefill_state_indices: Optional[npt.NDArray[np.int32]] = None,
        dst_state_indices: Optional[npt.NDArray[np.int32]] = None,
        src_kv_indices_per_pool: Optional[List[npt.NDArray[np.int32]]] = None,
        dst_kv_indices_per_pool: Optional[List[npt.NDArray[np.int32]]] = None,
    ):
        """Generic KV cache transfer supporting both MHA and MLA architectures.
        Used by both send_kvcache and maybe_send_extra.

        When prefill_state_indices/dst_state_indices are provided, state is bundled
        into the same make_prepped_xfer call using the combined KV+state dlist handle.

        src_kv_indices_per_pool / dst_kv_indices_per_pool: per-pool-group index lists for
        V4 multi-pool models.  For uniform models these are both [prefill_data_indices] /
        [dst_data_indices] (a single-element list) and the code path is identical to before.
        """
        # Prepped path: use the combined KV+state dlist handle when available.
        if (
            self.use_prepped_transfer
            and src_data_ptrs is self.kv_args.kv_data_ptrs
            and peer_name in self.prep_handles
        ):
            src_prep = self.prep_handles[""]
            dst_prep = self.prep_handles[peer_name]
            info = self.decode_kv_args_table[peer_name]
            slots_per_kv_layer_src = self._slots_per_kv_layer
            slots_per_kv_layer_dst = self._dst_slots_per_kv_layer[peer_name]

            # Build flat dlist indices per pool group, then concatenate.
            # For uniform models: single group → same as the old single call.
            # For V4 multi-pool: each group uses its own src/dst index array.
            _src_per_pool = src_kv_indices_per_pool or [prefill_data_indices]
            _dst_per_pool = dst_kv_indices_per_pool or [dst_data_indices]
            src_parts: List[npt.NDArray] = []
            dst_parts: List[npt.NDArray] = []
            src_offset = 0
            dst_offset = 0
            for g_idx, (g_start, g_end) in enumerate(self._pool_group_boundaries):
                src_g = _src_per_pool[min(g_idx, len(_src_per_pool) - 1)]
                dst_g = _dst_per_pool[min(g_idx, len(_dst_per_pool) - 1)]
                layer_slots_src = slots_per_kv_layer_src[g_start:g_end]
                layer_slots_dst = slots_per_kv_layer_dst[g_start:g_end]
                flat_src = repeat_indices_over_layers(src_g, layer_slots_src)
                flat_dst = repeat_indices_over_layers(dst_g, layer_slots_dst)
                if src_offset:
                    flat_src = (flat_src + src_offset).astype(np.int32)
                if dst_offset:
                    flat_dst = (flat_dst + dst_offset).astype(np.int32)
                src_parts.append(flat_src)
                dst_parts.append(flat_dst)
                src_offset += sum(layer_slots_src)
                dst_offset += sum(layer_slots_dst)

            src_kv_flat = np.concatenate(src_parts) if len(src_parts) > 1 else src_parts[0]
            dst_kv_flat = np.concatenate(dst_parts) if len(dst_parts) > 1 else dst_parts[0]

            if (
                prefill_state_indices is not None
                and dst_state_indices is not None
                and self._num_slots_src_state is not None
            ):
                # Append state indices, shifted past the KV entries in the combined dlist.
                num_slots_dst_state = (
                    info.dst_num_slots_state
                    if info.dst_num_slots_state is not None
                    else self._num_slots_src_state
                )
                src_state_flat = repeat_indices_over_layers(
                    prefill_state_indices, self._slots_per_state_layer
                )
                dst_state_flat = repeat_indices_over_layers(
                    dst_state_indices, [num_slots_dst_state] * len(self.kv_args.state_item_lens)
                )
                src_indices = np.concatenate(
                    [src_kv_flat, (src_state_flat + self._kv_dlist_src_size).astype(np.int32)]
                )
                dst_indices = np.concatenate(
                    [dst_kv_flat, (dst_state_flat + self._kv_dlist_dst_size[peer_name]).astype(np.int32)]
                )
            else:
                src_indices = src_kv_flat
                dst_indices = dst_kv_flat

            xfer_handle = self.agent.make_prepped_xfer(
                "WRITE", src_prep, src_indices, dst_prep, dst_indices, notif.encode("ascii"),
            )
            if not xfer_handle:
                raise Exception("KVSender failed to create prepped transfer")
            state = self.agent.transfer(xfer_handle)
            if state == "ERR":
                raise Exception("KVSender failed to post prepped transfer")
            return xfer_handle

        # Non-prepped path: fallback for slice transfers.
        # group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_data_indices, dst_data_indices
        )

        logger.debug(f"sending kvcache to {peer_name} with notif {notif}")
        # Make descs
        if self.is_mla_backend:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            L = layers_current_pp_stage
            layers_params = list(zip(src_kv_ptrs[:L], dst_kv_ptrs[:L], item_lens[:L]))
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            L = layers_current_pp_stage
            layers_params = (
                list(zip(src_k_ptrs[:L], dst_k_ptrs[:L], item_lens[:L]))
                + list(zip(src_v_ptrs[:L], dst_v_ptrs[:L], item_lens[:L]))
            )

        src_addrs = []
        dst_addrs = []
        # Transfer lengths are identical for src and dst (same item_len * block_lens per layer).
        lens = []

        # Precompute block starts/lengths to reduce Python-level loops.
        prefill_starts = np.array([block[0] for block in prefill_kv_blocks], dtype=np.int64)
        dst_starts = np.array([block[0] for block in dst_kv_blocks], dtype=np.int64)
        block_lens = np.array([len(block) for block in prefill_kv_blocks], dtype=np.int64)

        for src_ptr, dst_ptr, item_len in layers_params:
            lens.append(item_len * block_lens)
            src_addrs.append(src_ptr + prefill_starts * item_len)
            dst_addrs.append(dst_ptr + dst_starts * item_len)

        src_reqs = _make_req_array(src_addrs, lens, self.kv_args.gpu_id)
        dst_reqs = _make_req_array(dst_addrs, lens, dst_gpu_id)

        logger.debug(
            f"len(src_addrs): before group: {len(prefill_data_indices)}, after group: {len(prefill_kv_blocks)}"
        )
        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def send_kvcache(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
        prefill_state_indices: Optional[npt.NDArray[np.int32]] = None,
        dst_state_indices: Optional[npt.NDArray[np.int32]] = None,
        src_kv_indices_per_pool: Optional[List[npt.NDArray[np.int32]]] = None,
        dst_kv_indices_per_pool: Optional[List[npt.NDArray[np.int32]]] = None,
    ):
        return self._send_kvcache_generic(
            peer_name=peer_name,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=self.kv_args.kv_item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            dst_gpu_id=dst_gpu_id,
            notif=notif,
            prefill_state_indices=prefill_state_indices,
            dst_state_indices=dst_state_indices,
            src_kv_indices_per_pool=src_kv_indices_per_pool,
            dst_kv_indices_per_pool=dst_kv_indices_per_pool,
        )

    def send_kvcache_slice(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
        prefill_tp_size: int,
        decode_tp_size: int,
        decode_tp_rank: int,
        dst_kv_item_len: int,
    ):
        # Prepped path: src dlist is shared per decode_tp_size; dst is per peer.
        if (
            self.use_prepped_transfer
            and self.prep_handle_slice_src is not None
            and peer_name in self.prep_handles_slice_dst
        ):
            src_handle, num_groups, num_ptr_pairs, num_slots_src = self.prep_handle_slice_src
            dst_handle, num_slots_dst = self.prep_handles_slice_dst[peer_name]
            head_group_idx = self.peer_head_group[peer_name]
            page_size = self.kv_args.page_size
            src_indices = expand_page_indices_for_slice(
                np.asarray(prefill_kv_indices, dtype=np.int32),
                num_ptr_pairs, num_slots_src, page_size,
                num_groups=num_groups, head_group_idx=head_group_idx,
            )
            dst_indices = expand_page_indices_for_slice(
                np.asarray(dst_kv_indices, dtype=np.int32),
                num_ptr_pairs, num_slots_dst, page_size,
            )
            xfer_handle = self.agent.make_prepped_xfer(
                "WRITE", src_handle, src_indices, dst_handle, dst_indices, notif.encode("ascii")
            )
            if not xfer_handle:
                raise Exception("KVSender failed to create prepped slice transfer")
            state = self.agent.transfer(xfer_handle)
            if state == "ERR":
                raise Exception("KVSender failed to post prepped slice transfer")
            return xfer_handle

        # Non-prepped path.
        local_tp_rank_in_group = self.kv_args.engine_rank % prefill_tp_size
        dst_tp_rank_in_group = decode_tp_rank % decode_tp_size

        src_kv_item_len = self.kv_args.kv_item_lens[0]
        page_size = self.kv_args.page_size

        # Use total KV head count (not per-rank) for correct head distribution.
        # Per-rank kv_head_num is max(1, total//tp) which loses info when total < tp.
        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * prefill_tp_size

        src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // decode_tp_size)

        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # Determine which heads to send
        if prefill_tp_size > decode_tp_size:
            # Multiple prefill ranks to one decode rank
            src_replication = max(1, prefill_tp_size // total_kv_heads)
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            unique_head_idx = local_tp_rank_in_group // src_replication
            dst_head_start_offset = (
                unique_head_idx * src_heads_per_rank
            ) % dst_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            src_head_start_offset = (
                dst_tp_rank_in_group * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
        )
        # Calculate precise byte offset and length for the sub-slice within the token
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        prefill_indices = np.asarray(prefill_kv_indices, dtype=np.int64)
        dst_indices = np.asarray(dst_kv_indices, dtype=np.int64)
        bytes_per_token_prefill = src_kv_item_len // page_size
        bytes_per_token_decode = dst_kv_item_len // page_size
        token_offsets = np.arange(page_size, dtype=np.int64)

        # Vectorize across all K+V ptr pairs at once: shape (ptr, slot, token) → ravel.
        src_ptrs_arr = np.array(
            list(src_k_ptrs[:layers_current_pp_stage])
            + list(src_v_ptrs[:layers_current_pp_stage]),
            dtype=np.int64,
        )
        dst_ptrs_arr = np.array(
            list(dst_k_ptrs[:layers_current_pp_stage])
            + list(dst_v_ptrs[:layers_current_pp_stage]),
            dtype=np.int64,
        )
        src_addrs_flat = (
            src_ptrs_arr[:, None, None]
            + prefill_indices[None, :, None] * src_kv_item_len
            + token_offsets[None, None, :] * bytes_per_token_prefill
            + src_head_slice_offset
        ).ravel()
        dst_addrs_flat = (
            dst_ptrs_arr[:, None, None]
            + dst_indices[None, :, None] * dst_kv_item_len
            + token_offsets[None, None, :] * bytes_per_token_decode
            + dst_head_slice_offset
        ).ravel()

        src_reqs = np.column_stack((
            src_addrs_flat,
            np.full_like(src_addrs_flat, heads_bytes_per_token_to_send),
            np.full_like(src_addrs_flat, self.kv_args.gpu_id),
        ))
        dst_reqs = np.column_stack((
            dst_addrs_flat,
            np.full_like(dst_addrs_flat, heads_bytes_per_token_to_send),
            np.full_like(dst_addrs_flat, dst_gpu_id),
        ))

        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not xfer_handle:
            raise Exception("Failed to create sliced KV transfer")

        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post sliced KV transfer")

        return xfer_handle

    def send_aux(
        self,
        peer_name: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
        notif: str,
    ):
        src_addrs = []
        dst_addrs = []

        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for dst_ptr, src_ptr, length in zip(dst_aux_ptrs, prefill_aux_ptrs, prefill_aux_item_lens):
            src_addr = src_ptr + length * prefill_aux_index
            dst_addr = dst_ptr + length * dst_aux_index
            src_addrs.append((src_addr, length, 0))
            dst_addrs.append((dst_addr, length, 0))

        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def _send_mamba_state(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_gpu_id: int,
        notif: str,
    ):
        """Transfer Mamba states via RDMA."""
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"
        assert len(dst_state_indices) == len(prefill_state_indices), "State indices count mismatch between Prefill and Decode"

        src_addrs = []
        dst_addrs = []

        prefill_state_data_ptrs = self.kv_args.state_data_ptrs
        prefill_state_item_lens = self.kv_args.state_item_lens
        src_idx = int(prefill_state_indices[0])
        dst_idx = int(dst_state_indices[0])

        for dst_ptr, src_ptr, length in zip(
            dst_state_data_ptrs, prefill_state_data_ptrs, prefill_state_item_lens
        ):
            src_addrs.append((src_ptr + length * src_idx, length, self.kv_args.gpu_id))
            dst_addrs.append((dst_ptr + length * dst_idx, length, dst_gpu_id))

        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("Failed to create Mamba state transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post Mamba state transfer")
        return xfer_handle

    def _send_mamba_state_slice(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_gpu_id: int,
        notif: str,
        dst_state_item_lens: list[int],
        dst_state_dim_per_tensor: list[int],
        decode_tp_size: int,
        decode_tp_rank: int,
    ):
        """Transfer Mamba states with TP slice support via RDMA.

        When prefill and decode have different attn_tp_size, we slice the
        TP-sharded dimension (3rd dim) of conv_state and temporal_state
        accordingly, mirroring Mooncake's _send_mamba_state_slice.
        """
        logger.warning_once(
            "Using Mamba state slice transfer for different TP sizes. "
            f"Prefill attn_tp_size={self.attn_tp_size}, "
            f"Decode attn_tp_size={decode_tp_size}."
        )
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"

        prefill_state_data_ptrs = self.kv_args.state_data_ptrs
        prefill_state_item_lens = self.kv_args.state_item_lens
        src_state_dim_per_tensor = getattr(self.kv_args, "state_dim_per_tensor", [])

        if not src_state_dim_per_tensor or not dst_state_dim_per_tensor:
            return self._send_mamba_state(
                peer_name,
                prefill_state_indices,
                dst_state_data_ptrs,
                dst_state_indices,
                dst_gpu_id,
                notif,
            )

        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank_in_group = decode_tp_rank % decode_tp_size
        src_idx = int(prefill_state_indices[0])
        dst_idx = int(dst_state_indices[0])

        src_addrs = []
        dst_addrs = []

        if self.attn_tp_size > decode_tp_size:
            writers_per_decode = self.attn_tp_size // decode_tp_size
            local_writer_idx = local_tp_rank_in_group % writers_per_decode

        for dst_ptr, src_ptr, src_item_len, dst_item_len, src_dim, dst_dim in zip(
            dst_state_data_ptrs,
            prefill_state_data_ptrs,
            prefill_state_item_lens,
            dst_state_item_lens,
            src_state_dim_per_tensor,
            dst_state_dim_per_tensor,
        ):
            src_bytes_per_dim = src_item_len // src_dim
            dst_bytes_per_dim = dst_item_len // dst_dim

            if self.attn_tp_size > decode_tp_size:
                src_dim_start = 0
                num_dims_to_send = src_dim
                dst_dim_start = local_writer_idx * src_dim
            else:
                src_dim_start = (dst_tp_rank_in_group * dst_dim) % src_dim
                num_dims_to_send = dst_dim
                dst_dim_start = 0

            bytes_to_send = num_dims_to_send * src_bytes_per_dim
            src_addrs.append((
                src_ptr + src_item_len * src_idx + src_dim_start * src_bytes_per_dim,
                bytes_to_send,
                self.kv_args.gpu_id,
            ))
            dst_addrs.append((
                dst_ptr + dst_item_len * dst_idx + dst_dim_start * dst_bytes_per_dim,
                bytes_to_send,
                dst_gpu_id,
            ))

        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("Failed to create Mamba state slice transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post Mamba state slice transfer")
        return xfer_handle

    def maybe_send_extra(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_gpu_id: int,
        notif: str,
        decode_tp_size: int,
        decode_tp_rank: int = 0,
        dst_state_item_lens: list[int] | None = None,
        dst_state_dim_per_tensor: list[int] | None = None,
    ):
        """Send state or extra pool data with type-specific handling."""
        state_type = getattr(self.kv_args, "state_type", "none")

        if state_type == "mamba":
            if self.attn_tp_size != decode_tp_size:
                return self._send_mamba_state_slice(
                    peer_name,
                    prefill_state_indices,
                    dst_state_data_ptrs,
                    dst_state_indices,
                    dst_gpu_id,
                    notif,
                    dst_state_item_lens or [],
                    dst_state_dim_per_tensor or [],
                    decode_tp_size,
                    decode_tp_rank,
                )
            return self._send_mamba_state(
                peer_name,
                prefill_state_indices,
                dst_state_data_ptrs,
                dst_state_indices,
                dst_gpu_id,
                notif,
            )
        elif state_type in ["swa", "nsa"]:
            if not self.is_mla_backend and self.attn_tp_size != decode_tp_size:
                raise RuntimeError(
                    f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {state_type.upper()} hybrid models yet."
                )
            if len(prefill_state_indices) != len(dst_state_indices):
                raise RuntimeError(
                    f"State index length mismatch: prefill={len(prefill_state_indices)}, "
                    f"dst={len(dst_state_indices)}"
                )
            return self._send_kvcache_generic(
                peer_name=peer_name,
                src_data_ptrs=self.kv_args.state_data_ptrs,
                dst_data_ptrs=dst_state_data_ptrs,
                item_lens=self.kv_args.state_item_lens,
                prefill_data_indices=np.array(prefill_state_indices, dtype=np.int32),
                dst_data_indices=np.array(dst_state_indices, dtype=np.int32),
                dst_gpu_id=dst_gpu_id,
                notif=notif,
            )
        elif state_type != "none":
            raise RuntimeError(
                f"PD Disaggregation via NIXL does NOT support {state_type} hybrid models yet."
            )

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
        kv_indices_extra: Optional[List[npt.NDArray[np.int32]]] = None,
    ):
        """Issue KV transfers for one chunk.

        kv_indices_extra: per-pool src indices for pool groups beyond the first.
            None (or []) for uniform models — backward compatible.
            For V4 multi-pool: kv_indices_extra[i] is the (i+1)-th pool's src indices.
        """
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or aux_index is not None

        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        handles = []
        for req in reqs_to_be_processed:
            assert bootstrap_room == req.room
            if req.is_dummy():
                continue

            chunked_dst_kv_indice = req.dst_kv_indices[index_slice]
            assert len(chunked_dst_kv_indice) == len(kv_indices)
            assert req.agent_name in self.decode_kv_args_table

            dst_info = self.decode_kv_args_table[req.agent_name]
            decode_tp_size = dst_info.decode_tp_size
            state_type = getattr(self.kv_args, "state_type", "none")
            # Bundle NSA/SWA state with the last KV chunk when using the prepped path
            # (non-slice only — slice transfers don't use the combined dlist handle).
            bundle_state = (
                is_last
                and state_indices is not None
                and state_type in ("nsa", "swa")
                and self.use_prepped_transfer
                and self._num_slots_src_state is not None
                and (self.is_mla_backend or decode_tp_size == self.attn_tp_size)
            )

            if bundle_state:
                kv_notif = f"{req.room}_kvstate_{chunk_id}_{self.kv_args.engine_rank}"
            else:
                kv_notif = f"{req.room}_kv_{chunk_id}_{int(is_last)}_{self.kv_args.engine_rank}"

            # Build per-pool src/dst index lists.
            # For uniform models: single entry each — no change to logic.
            # For V4 multi-pool: first pool is kv_indices; extras come from kv_indices_extra
            # (src) and req.dst_kv_indices_extra (dst).
            src_kv_indices_per_pool = [kv_indices] + list(kv_indices_extra or [])
            dst_kv_indices_per_pool = [chunked_dst_kv_indice] + list(req.dst_kv_indices_extra)

            if self.is_mla_backend or (decode_tp_size == self.attn_tp_size):
                kv_xfer_handle = self.send_kvcache(
                    req.agent_name,
                    kv_indices,
                    dst_info.dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    dst_info.gpu_id,
                    kv_notif,
                    prefill_state_indices=np.array(state_indices, dtype=np.int32) if bundle_state else None,
                    dst_state_indices=np.array(req.dst_state_indices, dtype=np.int32) if bundle_state else None,
                    src_kv_indices_per_pool=src_kv_indices_per_pool,
                    dst_kv_indices_per_pool=dst_kv_indices_per_pool,
                )
            else:
                kv_xfer_handle = self.send_kvcache_slice(
                    req.agent_name,
                    kv_indices,
                    dst_info.dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    dst_info.gpu_id,
                    kv_notif,
                    prefill_tp_size=self.attn_tp_size,
                    decode_tp_size=decode_tp_size,
                    decode_tp_rank=dst_info.decode_tp_rank,
                    dst_kv_item_len=dst_info.dst_kv_item_len,
                )

            handles.append(kv_xfer_handle)
            # Aux/state sent only with the last chunk.
            if is_last:
                if state_indices is not None and not bundle_state:
                    state_xfer_handle = self.maybe_send_extra(
                        req.agent_name,
                        state_indices,
                        dst_info.dst_state_data_ptrs,
                        req.dst_state_indices,
                        dst_info.gpu_id,
                        f"{req.room}_state_{self.kv_args.engine_rank}",
                        decode_tp_size,
                        decode_tp_rank=dst_info.decode_tp_rank,
                        dst_state_item_lens=dst_info.dst_state_item_lens,
                        dst_state_dim_per_tensor=dst_info.dst_state_dim_per_tensor,
                    )
                    if state_xfer_handle is not None:
                        handles.append(state_xfer_handle)

                assert aux_index is not None
                aux_xfer_handle = self.send_aux(
                    req.agent_name,
                    aux_index,
                    dst_info.dst_aux_ptrs,
                    req.dst_aux_index,
                    f"{req.room}_aux",
                )
                handles.append(aux_xfer_handle)
        if is_last:
            del self.transfer_infos[bootstrap_room]
        return handles

    def update_transfer_status(self):
        notif_map = self.agent.get_new_notifs()
        for messages in notif_map.values():
            for msg in messages:
                components = msg.decode("ascii").split("_", 4)
                room = int(components[0])
                status = self.transfer_statuses[room]
                if components[1] == "kv":
                    chunk_id = int(components[2])
                    is_last = bool(int(components[3]))
                    pp_rank = int(components[4]) if len(components) > 4 else 0
                    status.received_kvs_per_pp[pp_rank].add(chunk_id)
                    if is_last:
                        status.expected_kvs_per_pp[pp_rank] = chunk_id + 1
                        if status.num_pp_ranks_expected is None:
                            status.num_pp_ranks_expected = (
                                self.required_prefill_response_num_table.get(room, 1)
                            )
                elif components[1] == "kvstate":
                    # Combined KV+state notification (last chunk with bundled NSA/SWA state).
                    chunk_id = int(components[2])
                    pp_rank = int(components[3]) if len(components) > 3 else 0
                    status.received_kvs_per_pp[pp_rank].add(chunk_id)
                    status.expected_kvs_per_pp[pp_rank] = chunk_id + 1
                    status.received_state_per_pp.add(pp_rank)
                    if status.num_pp_ranks_expected is None:
                        status.num_pp_ranks_expected = (
                            self.required_prefill_response_num_table.get(room, 1)
                        )
                elif components[1] == "aux":
                    status.received_aux = True
                elif components[1] == "state":
                    pp_rank = int(components[2]) if len(components) > 2 else 0
                    status.received_state_per_pp.add(pp_rank)

    def check_transfer_done(self, room: int):
        status = self.transfer_statuses.get(room)
        return status is not None and status.is_done()

    def _start_bootstrap_thread(self):
        def bootstrap_thread():
            """This thread recvs transfer info from the decode engine"""
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )
                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}. Foreign traffic?"
                waiting_req_bytes = waiting_req_bytes[1:]
                room = waiting_req_bytes[0].decode("ascii")
                agent_name = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    # Register new peer and save KV base pointers.
                    self._add_remote_peer(
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    logger.debug(f"Register KVArgs from {agent_name} successfully")
                    continue
                room = int(room)
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                transfer_info = TransferInfo.from_zmq(waiting_req_bytes)
                self.transfer_infos[room][agent_name] = transfer_info
                required_dst_info_num = transfer_info.required_dst_info_num
                logger.debug(f"got info {room=} {agent_name=} {required_dst_info_num=}")
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    logger.debug(f"{room=} is bootstrapped")
                    self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()


class NixlKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.xfer_handles = []
        self.has_sent = False
        self.chunk_id = 0

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
        kv_indices_extra: Optional[List[npt.NDArray[np.int32]]] = None,
    ):
        """Send one KV chunk.

        kv_indices_extra: per-pool src indices for pool groups beyond the first.
            None for uniform models (backward compat). For V4 multi-pool:
            kv_indices_extra[i] is the (i+1)-th pool's src page indices.
        """
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        # Special handling for cp
        if self.kv_mgr.enable_all_cp_ranks_for_transfer:
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            if is_last:
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
            return

        new_xfer_handles = self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.chunk_id,
            self.aux_index,
            state_indices,
            kv_indices_extra,
        )
        self.xfer_handles.extend(new_xfer_handles)
        self.chunk_id += 1
        if is_last:
            self.has_sent = True
            del self.kv_mgr.request_status[self.bootstrap_room]

    def poll(self) -> KVPoll:
        if not self.has_sent:
            return self.kv_mgr.check_status(self.bootstrap_room)
        for x in self.xfer_handles:
            state = self.kv_mgr.agent.check_xfer_state(x)
            if state == "ERR":
                raise Exception("KVSender transfer encountered an error.")
            if state != "DONE":
                return KVPoll.WaitingForInput  # type: ignore
        return KVPoll.Success  # type: ignore

    def failure_exception(self):
        raise RuntimeError("NIXL KVSender Exception")


class NixlKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.started_transfer = False
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self.init_time = None

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
        kv_indices_extra: Optional[List[npt.NDArray[np.int32]]] = None,
    ):
        """Send transfer metadata to the prefill bootstrap server.

        kv_indices_extra: per-pool dst indices for pool groups beyond the first.
            None (or []) for uniform models — msg[8] will be b"" (backward compat).
            For V4 multi-pool: kv_indices_extra[i] is the (i+1)-th pool's dst indices.
        """
        if self.bootstrap_infos is None:
            logger.error(
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        room_bytes = str(self.bootstrap_room).encode("ascii")
        local_ip_bytes = self.kv_mgr.local_ip.encode("ascii")
        rank_port_bytes = str(self.kv_mgr.rank_port).encode("ascii")
        agent_name_bytes = self.kv_mgr.agent.name.encode("ascii")
        aux_index_bytes = str(aux_index).encode("ascii")
        required_num_bytes = str(self.required_dst_info_num).encode("ascii")
        kv_bytes = kv_indices.tobytes()
        state_bytes = (
            np.array(state_indices, dtype=np.int32).tobytes()
            if state_indices is not None
            else b""
        )
        # msg[8]: extra pool dst indices (V4 multi-pool). Empty bytes for uniform models.
        extra_bytes = (
            _pack_extra_pool_indices(kv_indices_extra)
            if kv_indices_extra
            else b""
        )

        for bootstrap_info in self.bootstrap_infos:
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
            )
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            logger.debug(
                f"Sending to prefill server with bootstrap room {self.bootstrap_room} {is_dummy=}"
            )
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        room_bytes,
                        local_ip_bytes,
                        rank_port_bytes,
                        agent_name_bytes,
                        b"" if is_dummy else kv_bytes,
                        aux_index_bytes,
                        required_num_bytes,
                        b"" if is_dummy else state_bytes,
                        b"" if is_dummy else extra_bytes,
                    ]
                )

        if state_indices is not None:
            self.kv_mgr.transfer_statuses[self.bootstrap_room].expects_state = True

        self.started_transfer = True
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status
        if not self.started_transfer:
            return status

        now = time.time()
        elapsed = now - self.init_time

        if elapsed >= self.kv_mgr.waiting_timeout:
            logger.error(f"Request {self.bootstrap_room} waiting_timeout")
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
            )
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed

        self.kv_mgr.update_transfer_status()
        if self.kv_mgr.check_transfer_done(self.bootstrap_room):  # type: ignore
            ts = self.kv_mgr.transfer_statuses[self.bootstrap_room]
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(
                self.bootstrap_room
            )
            # Check if the transfer failed
            if ts.is_failure:
                self.conclude_state = KVPoll.Failed
                logger.error(
                    f"Transfer for room {self.bootstrap_room} failed due to node failure"
                )
            else:
                self.conclude_state = KVPoll.Success
            del self.kv_mgr.transfer_statuses[self.bootstrap_room]
            return self.conclude_state  # type: ignore
        return KVPoll.WaitingForInput  # type: ignore

    def _register_kv_args(self):
        kv_ptrs = self.kv_mgr.kv_args.kv_data_ptrs
        aux_ptrs = self.kv_mgr.kv_args.aux_data_ptrs
        state_ptrs = self.kv_mgr.kv_args.state_data_ptrs
        state_item_lens = self.kv_mgr.kv_args.state_item_lens
        state_dim_per_tensor = getattr(self.kv_mgr.kv_args, "state_dim_per_tensor", [])

        packed_kv_data_ptrs = struct.pack(f"{len(kv_ptrs)}Q", *kv_ptrs)
        packed_aux_data_ptrs = struct.pack(f"{len(aux_ptrs)}Q", *aux_ptrs)
        packed_state_data_ptrs = struct.pack(f"{len(state_ptrs)}Q", *state_ptrs) if state_ptrs else b""
        packed_state_item_lens = struct.pack(f"{len(state_item_lens)}I", *state_item_lens) if state_item_lens else b""
        packed_state_dim_per_tensor = struct.pack(f"{len(state_dim_per_tensor)}I", *state_dim_per_tensor) if state_dim_per_tensor else b""

        # Per-layer slot counts for V4 multi-pool support (msg[16]).
        # Empty when all layers have the same slot count (uniform models).
        kv_slots_per_layer = [
            dl // il
            for dl, il in zip(
                self.kv_mgr.kv_args.kv_data_lens, self.kv_mgr.kv_args.kv_item_lens
            )
        ]
        if len(set(kv_slots_per_layer)) == 1:
            # Uniform: omit the field so older receivers see an empty msg[16].
            packed_kv_slots_per_layer = b""
        else:
            packed_kv_slots_per_layer = struct.pack(f"{len(kv_slots_per_layer)}I", *kv_slots_per_layer)

        # Build the message template once; only agent metadata varies per bootstrap peer.
        msg = [
            GUARD,
            b"None",
            self.kv_mgr.local_ip.encode("ascii"),
            str(self.kv_mgr.rank_port).encode("ascii"),
            self.kv_mgr.agent.name.encode("ascii"),
            None,  # slot 5: filled per iteration by get_agent_metadata()
            packed_kv_data_ptrs,
            packed_aux_data_ptrs,
            packed_state_data_ptrs,
            str(self.kv_mgr.kv_args.gpu_id).encode("ascii"),
            str(self.kv_mgr.attn_tp_size).encode("ascii"),
            str(self.kv_mgr.kv_args.engine_rank).encode("ascii"),
            str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii"),
            packed_state_item_lens,
            packed_state_dim_per_tensor,
            str(
                self.kv_mgr.kv_args.kv_data_lens[0] // self.kv_mgr.kv_args.kv_item_lens[0]
            ).encode("ascii"),
            str(
                self.kv_mgr.kv_args.state_data_lens[0] // self.kv_mgr.kv_args.state_item_lens[0]
                if self.kv_mgr.kv_args.state_data_lens
                else 0
            ).encode("ascii"),
            packed_kv_slots_per_layer,  # msg[16]: per-layer slot counts (empty = uniform)
        ]
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            msg[5] = self.kv_mgr.agent.get_agent_metadata()
            with lock:
                sock.send_multipart(msg)

    def failure_exception(self):
        raise RuntimeError("NIXL KVReceiver Exception")


class NixlKVBootstrapServer(CommonKVBootstrapServer):
    pass
