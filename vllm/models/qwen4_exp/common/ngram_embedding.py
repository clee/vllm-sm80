# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Qwen4Exp n-gram embedding storage with device and pinned-host backends.

Both the NVIDIA and AMD Qwen4Exp implementations use these classes so the (large)
n-gram embedding table can be kept in pinned host memory and looked up through
Unified Virtual Addressing on any CUDA-alike platform.
"""

import threading
from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import get_dp_group, get_etp_group, get_tp_group
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.inc.inc import INCConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .ple import PLEVocabParallelEmbedding

logger = init_logger(__name__)


# [local patch] PLE gather runtime instrumentation (VLLM_PLE_DEBUG=1).
import hashlib as _hashlib
import os as _os
import threading as _threading

_ple_dbg_lock = _threading.Lock()


def _ple_debug() -> bool:
    return _os.environ.get("VLLM_PLE_DEBUG", "0") == "1"


def _ple_log(tag: str, *tensors) -> None:
    h = _hashlib.sha256()
    parts = [tag]
    for t in tensors:
        t = t.detach().contiguous().cpu()
        if t.dtype == torch.bfloat16:
            t = t.view(torch.uint16)
        b = t.numpy().tobytes()
        h.update(b)
        parts.append(f"{tuple(t.shape)}")
    with _ple_dbg_lock, open(
        _os.environ.get("VLLM_PLE_DEBUG_LOG", "/tmp/ple_debug.log"), "a"
    ) as f:
        f.write(f"{parts[0]} {h.hexdigest()[:16]} {' '.join(parts[1:])}\n")


class Qwen4ExpPLEEmbedding(PLEVocabParallelEmbedding, ABC):
    """ETP-sharded PLE table shared by device and pinned-host backends."""

    supports_prefetch: ClassVar[bool] = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: "Qwen4ExpPLEEmbeddingMethod",
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        del num_ngram_heads, max_total_tokens
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            quant_method=embedding_method,
            parallel_group=get_etp_group(),
        )
        self.embedding_method = embedding_method
        self.data_parallel_rank = data_parallel_rank
        tp_size = get_tp_group().world_size
        if self.tp_size % tp_size:
            raise ValueError(
                "ETP size must be divisible by TP size, but got "
                f"ETP={self.tp_size} and TP={tp_size}"
            )
        self.etp_data_parallel_size = self.tp_size // tp_size

    @abstractmethod
    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate storage for the complete embedding weight."""
        raise NotImplementedError

    def dequantize(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Delegate storage-format conversion to the embedding method."""
        return self.embedding_method.dequantize(self, embeddings, output_dtype)

    def _get_dp_gather_slot(self, local_num_tokens: int) -> tuple[int, int]:
        """Return the per-DP slot size and this rank's slot offset."""
        if self.etp_data_parallel_size == 1:
            return local_num_tokens, 0
        dp_metadata: DPMetadata | None = get_forward_context().dp_metadata
        if dp_metadata is None:
            raise RuntimeError("ETP spanning DP requires DP token metadata")
        group_start = (self.data_parallel_rank // self.etp_data_parallel_size) * (
            self.etp_data_parallel_size
        )
        group_end = group_start + self.etp_data_parallel_size
        token_counts = dp_metadata.num_tokens_across_dp_cpu.tolist()
        group_counts = token_counts[group_start:group_end]
        slot_size = max(group_counts)
        dp_rank = get_dp_group().rank_in_group
        return slot_size, dp_rank * slot_size

    def _gather_dp_ids(
        self,
        ngram_ids: torch.Tensor,
        slot_size: int,
    ) -> torch.Tensor:
        """Gather DP-local IDs that share one ETP-sharded PLE table."""
        if self.etp_data_parallel_size == 1:
            return ngram_ids
        if ngram_ids.shape[0] < slot_size:
            padding = ngram_ids.new_zeros(
                slot_size - ngram_ids.shape[0], ngram_ids.shape[1]
            )
            ngram_ids = torch.cat((ngram_ids, padding), dim=0)
        return get_dp_group().all_gather(ngram_ids, dim=0)

    def _select_embeddings(
        self,
        embeddings: torch.Tensor,
        local_num_tokens: int,
        slot_offset: int,
    ) -> torch.Tensor:
        """Select this DP rank's rows from the ETP-reduced embeddings."""
        if self.etp_data_parallel_size == 1:
            return embeddings
        return embeddings.narrow(0, slot_offset, local_num_tokens)

    @abstractmethod
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Start an asynchronous lookup when supported."""
        raise NotImplementedError


class Qwen4ExpPLEEmbeddingMethod(QuantizeMethodBase):
    """Quantization interface shared by resident and pinned PLE tables."""

    # PLE post-load processing only validates scales in their current storage.
    requires_device_loading: bool = False

    @staticmethod
    def from_quant_config(
        quant_config: QuantizationConfig | None,
        prefix: str,
        embedding_dtype: str | None = None,
    ) -> "Qwen4ExpPLEEmbeddingMethod":
        """Select the concrete PLE embedding format for a layer."""
        if embedding_dtype == "float8_e4m3fn":
            return Qwen4ExpPLEFp8EmbeddingMethod()
        if quant_config is None:
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(quant_config, ModelOptMixedPrecisionConfig):
            if quant_config._resolve_quant_algo(prefix) == "FP8":
                return Qwen4ExpPLEFp8EmbeddingMethod()
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(
            quant_config, ModelOptQuantConfigBase
        ) and quant_config.is_layer_excluded(prefix):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if (
            isinstance(quant_config, CompressedTensorsConfig)
            and quant_config.get_scheme_dict(None, layer_name=prefix) is None
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # [local patch] AutoRound/intel-neural-compressor checkpoints quantize
        # only the routed experts; the PLE table ships BF16 with no per-layer
        # entry. The config parser resolves unquantized layers to
        # layer_config.quantized=False, but this chooser runs before that, so
        # fall back to unquantized for any non-FP8 config whose PLE is not
        # explicitly targeted (mirrors the CompressedTensors branch above).
        if isinstance(quant_config, INCConfig):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not isinstance(quant_config, Fp8Config):
            raise NotImplementedError(
                "Qwen4Exp PLE embedding does not support quantization config "
                f"{type(quant_config).__name__}"
            )

        ignored_layers = quant_config.ignored_layers
        if is_layer_skipped(
            prefix,
            ignored_layers,
            quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # PLE checkpoint shards form one runtime embedding parameter.
        shard_prefix = f"{prefix}.shard_"
        if any(name.startswith(shard_prefix) for name in ignored_layers):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not quant_config.is_checkpoint_fp8_serialized:
            raise NotImplementedError(
                "Qwen4Exp PLE embedding only supports serialized FP8 checkpoints"
            )
        return Qwen4ExpPLEFp8EmbeddingMethod()

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)

    @abstractmethod
    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert looked-up PLE rows to the activation dtype."""
        raise NotImplementedError


class Qwen4ExpPLEUnquantizedEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Unquantized PLE embedding storage and lookup semantics."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        weight = nn.Parameter(
            layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer, output_dtype
        return embeddings


class Qwen4ExpPLEFp8EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """FP8 PLE embedding with one global checkpoint scale."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            scale_dtype=torch.float32,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject FP8 PLE checkpoints without a global scale."""
        sentinel = torch.finfo(torch.float32).min
        if torch.any(layer.weight_scale == sentinel):
            raise ValueError("FP8 PLE checkpoint is missing its global scale")

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        weight_scale = getattr(layer, "weight_scale", None)
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)


class Qwen4ExpPLEDeviceEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table allocated on the active model device."""

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight on the active device."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype)

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Resident embedding prefetch is a no-op."""
        return None

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather ETP inputs, look up embeddings, and select local rows."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = super().forward(gathered_ids)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )


@triton.jit
def _lookup_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Look up TP-owned PLE rows through a CUDA view of pinned host memory."""
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = offsets < embedding_dim
    load_mask = store_mask & in_range
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=load_mask,
        other=0.0,
    )
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        values,
        mask=store_mask,
    )


class Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table loaded into pinned CPU memory and looked up through UVA."""

    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        if not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        # [local patch] mmap/staged mode: never build the UVA view —
        # registering the placeholder would pin the whole table in host RAM;
        # the CPU staging path in _lookup (mmap) or the preadv gather in
        # model_state.prepare_inputs (staged) read the rows elsewhere.
        import os

        if os.environ.get("VLLM_PLE_MMAP", "0") == "1" or os.environ.get(
            "VLLM_PLE_STAGED", "0"
        ) == "1":
            self._ple_mmap = True
            self._uva_weight = None
            # Prefetch buffer/stream live on the active device; gathers go
            # through the CPU-staged _lookup, which copies into `output`
            # wherever it lives.
            self._ple_device = torch.accelerator.current_accelerator()
        else:
            self._ple_mmap = False
            self._ple_device = None
            self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)
        self._block_d = triton.next_power_of_2(self.embedding_dim)
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetch_buffer: torch.Tensor | None = None
        self._prefetch_alloc_lock = threading.Lock()
        self._prefetch_rows = max_total_tokens * self.etp_data_parallel_size
        self._num_ngram_heads = num_ngram_heads
        self._output_dim = num_ngram_heads * self.embedding_dim
        # [local patch] mmap mode: persistent pinned D2H/H2D scratch for the
        # CPU-staged lookup (allocated lazily outside cudagraph capture).
        self._mmap_scratch: dict[str, torch.Tensor] | None = None

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight directly in pinned CPU memory.

        [local patch] With VLLM_PLE_MMAP=1 or VLLM_PLE_STAGED=1 the pinned
        96 GiB allocation is skipped: mmap mode replaces this tensor with an
        np.memmap-backed view before any gather runs, and staged mode never
        reads it at all (rows come from the preadv shard reader). A
        non-pinned placeholder holds the Parameter slot; its pages are never
        touched.
        """
        import os

        if os.environ.get("VLLM_PLE_MMAP", "0") == "1" or os.environ.get(
            "VLLM_PLE_STAGED", "0"
        ) == "1":
            return torch.empty(
                num_embeddings,
                embedding_dim,
                dtype=dtype,
                device="cpu",
            )
        return torch.empty(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Look up local ETP rows while preserving the weight storage dtype."""
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=self.weight.dtype,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != self.weight.dtype
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape, weight dtype, "
                "and input device"
            )

        flat_ids = input_ids.reshape(-1).long()
        if _ple_debug() and not torch.cuda.is_current_stream_capturing():
            _ple_log("lookup-in", input_ids.detach().cpu())
        if flat_ids.numel():
            if getattr(self, "_ple_mmap", False):
                # [local patch] mmap mode: the weight is a page-cache mmap
                # view, not pinned RAM, so the UVA gather kernel cannot read
                # it. Gather on the CPU (page-cache hits for hot rows) into
                # persistent pinned staging, then copy to the GPU output.
                # All CPU tensors are pre-allocated pinned *outside* capture
                # (the memory profiler captures cudagraphs before any eager
                # warmup), so no allocation or D2H staging happens mid-capture.
                scratch = self._mmap_scratch
                if scratch is None or scratch["rows"].shape[0] < flat_ids.numel():
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "PLE mmap staging must be allocated before "
                            "cudagraph capture (run one eager forward)"
                        )
                    scratch = {
                        "ids": torch.empty(
                            flat_ids.numel(),
                            dtype=torch.int64,
                            device="cpu",
                            pin_memory=True,
                        ),
                        "rows": torch.empty(
                            (flat_ids.numel(), self.embedding_dim),
                            dtype=self.weight.dtype,
                            device="cpu",
                            pin_memory=True,
                        ),
                    }
                    self._mmap_scratch = scratch
                n = flat_ids.numel()
                cpu_ids = scratch["ids"][:n]
                # Capture-safe race-free D2H: the CPU consumes cpu_ids below.
                # Outside capture, issue an async D2H on the current stream and
                # synchronize before touching it on the CPU — a bare
                # non_blocking copy races the stream (under chunked-prefill
                # pressure the CPU ran ahead and gathered rows for stale ids).
                # During capture, the ids are dummies; zero-fill instead of
                # touching CUDA (any sync or D2H op is illegal mid-capture).
                if torch.cuda.is_current_stream_capturing():
                    cpu_ids.zero_()
                else:
                    cpu_ids.copy_(flat_ids, non_blocking=True)
                    torch.cuda.current_stream().synchronize()
                start = int(self.shard_indices.org_vocab_start_index)
                end = int(self.shard_indices.org_vocab_end_index)
                valid = (cpu_ids >= start) & (cpu_ids < end)
                safe_ids = cpu_ids.clamp(start, end - 1)
                rows = scratch["rows"][:n]
                if n:
                    # torch.index_select on the mmap-backed weight faults
                    # only the touched rows through the page cache.
                    rows.copy_(
                        torch.index_select(self.weight, 0, safe_ids),
                        non_blocking=False,
                    )
                    rows *= valid.to(rows.dtype).unsqueeze(1)
                if _os.environ.get("VLLM_PLE_ZERO", "0") == "1":
                    # [local patch] ablation knob: zero the PLE contribution
                    # (the CPU-staged delivery poisons the model; see notes)
                    rows.zero_()
                if _ple_debug():
                    _ple_log("lookup-rows", cpu_ids, rows)
                if output.device.type == "cuda":
                    # The output may be a 3D prefetch buffer view
                    # [ids, num_ngram_heads, embedding_dim]; index_select
                    # produced [ids, embedding_dim]. Copy through a flat view
                    # of the destination (same element count, same dtype).
                    output.view(-1, self.embedding_dim).copy_(
                        rows.to(output.dtype), non_blocking=True
                    )
                    # [local patch] the pinned scratch is reused on the next
                    # step; async scheduling can re-enter this Python before
                    # the GPU executes the H2D, and the next gather would
                    # overwrite `rows` mid-copy. Wait until the copy has been
                    # consumed (illegal mid-capture; the capture path never
                    # reaches a live consumer of `rows`).
                    if not torch.cuda.is_current_stream_capturing():
                        torch.cuda.current_stream().synchronize()
                else:
                    output.view(-1, self.embedding_dim).copy_(rows.to(output.dtype))
                return output
            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self._uva_weight,
                flat_ids,
                output,
                self.embedding_dim,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                BLOCK_D=self._block_d,
            )
        return output

    def sync_lookup(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Synchronous UVA lookup for platforms without prefetch wiring."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = self._lookup(gathered_ids)
        embeddings = self._reduce_etp_embeddings(embeddings)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )

    def _reduce_etp_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Combine pinned lookup results owned by different ETP ranks."""
        if self.tp_size == 1:
            return embeddings
        assert self.parallel_group is not None
        if embeddings.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Each vocabulary row has one owner, so reduce the raw FP8 bytes.
            reduced = self.parallel_group.all_reduce(embeddings.view(torch.int8))
            return reduced.view(embeddings.dtype)
        return self.parallel_group.all_reduce(embeddings)

    @eager_break_during_capture
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Gather ETP IDs and launch their UVA lookup on the side stream."""
        buffer = self._prefetch_buffer
        if buffer is None:
            # First use allocates. The eager profile run always precedes
            # cudagraph capture, so allocation never happens mid-capture;
            # the lock keeps concurrent first callers from tearing the
            # stream/buffer pair.
            with self._prefetch_alloc_lock:
                buffer = self._prefetch_buffer
                if buffer is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "pinned PLE prefetch buffer must be allocated "
                            "eagerly, before cudagraph capture"
                        )
                    # [local patch] mmap mode has no UVA weight; use the
                    # active device captured in __init__ (CPU-staged gathers
                    # copy into `output` wherever it lives).
                    dev = (
                        self._uva_weight.device
                        if self._uva_weight is not None
                        else self._ple_device
                    )
                    self._prefetch_stream = torch.cuda.Stream(device=dev)
                    buffer = torch.empty(
                        self._prefetch_rows,
                        self._num_ngram_heads,
                        self.embedding_dim,
                        dtype=self.weight.dtype,
                        device=dev,
                    )
                    self._prefetch_buffer = buffer
        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            raise RuntimeError("pinned PLE prefetch stream was not allocated")
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        if _ple_debug() and not torch.cuda.is_current_stream_capturing():
            _ple_log("prefetch-in", ngram_ids.detach().cpu())
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        if _ple_debug() and not torch.cuda.is_current_stream_capturing():
            import os as _o
            with open(_o.environ.get("VLLM_PLE_DEBUG_LOG", "/tmp/ple_debug.log"), "a") as f:
                f.write(f"prefetch-shapes ngram={tuple(ngram_ids.shape)} gathered={tuple(gathered_ids.shape)} slot_size={slot_size} etp={self.etp_data_parallel_size} buf={tuple(buffer.shape)}\n")
        if gathered_ids.shape[0] > buffer.shape[0]:
            raise ValueError(
                f"pinned PLE prefetch buffer holds {buffer.shape[0]} rows, "
                f"but the batch needs {gathered_ids.shape[0]}"
            )
        active_output = buffer[: gathered_ids.shape[0]]
        prefetch_stream.wait_stream(torch.cuda.current_stream())
        gathered_ids.record_stream(prefetch_stream)
        import os as _os

        if _os.environ.get("VLLM_PLE_MMAP_ASYNC", "0") == "1":
            # [local patch] async prefetch: CPU-staged gather on the side
            # stream. UNSAFE as written: the CPU portion of _lookup has no
            # ordering guarantee vs. the main stream's production of the ids
            # (record_stream pins lifetime, not readiness). Kept only for
            # future event-fenced overlap work.
            with torch.cuda.stream(prefetch_stream):
                self._lookup(gathered_ids, output=active_output)
        else:
            # Synchronous CPU-staged gather on the MAIN stream. Correct:
            # the D2H ids copy + sync inside _lookup orders against the
            # producing ops, and the H2D output copy lands on the stream
            # the consumer (_finalize_prefetch) waits on.
            self._lookup(gathered_ids, output=active_output)

    @eager_break_during_capture
    def _finalize_prefetch(
        self,
        prefetch_output: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Join the side stream, reduce ETP shards, and select local rows."""
        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            raise RuntimeError("pinned PLE finalize requires a prior start_prefetch")
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        slot_size, slot_offset = self._get_dp_gather_slot(output.shape[0])
        active_output = prefetch_output[: slot_size * self.etp_data_parallel_size]
        if _ple_debug() and not torch.cuda.is_current_stream_capturing():
            import os as _o
            with open(_o.environ.get("VLLM_PLE_DEBUG_LOG", "/tmp/ple_debug.log"), "a") as f:
                f.write(f"finalize-shapes out={tuple(output.shape)} slot_size={slot_size} etp={self.etp_data_parallel_size} read_n={slot_size * self.etp_data_parallel_size}\n")
        embeddings = self._reduce_etp_embeddings(active_output)
        embeddings = self._select_embeddings(
            embeddings,
            output.shape[0],
            slot_offset,
        )
        output.copy_(embeddings.flatten(-2))
        if _ple_debug() and not torch.cuda.is_current_stream_capturing():
            _ple_log("finalize-out", output.detach().cpu())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Finish the pinned lookup into graph-owned output storage."""
        buffer = self._prefetch_buffer
        if buffer is None:
            raise RuntimeError("pinned PLE lookup requires a prior start_prefetch")
        output = buffer.new_empty((hidden_states.shape[0], self._output_dim))
        self._finalize_prefetch(buffer, output)
        return output
