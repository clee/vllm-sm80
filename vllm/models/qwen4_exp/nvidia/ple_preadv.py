"""Shard-direct preadv PLE n-gram table (no assembly step).

Port of tonyd2wild/Qwen38-Flash-Next-4x3090 vllm-w4a16/patch/ple_mmap.py
(Apache-2.0, (c) Kai / 2Wild). The checkpoint's n-gram safetensors shards ARE
the table: rows are read positionally per shard, so no .bin is ever assembled
(the src302 mmap_swap._assemble layout turned out to be wrong; see notes).
"""

from __future__ import annotations

import json
import os
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

_TORCH_DTYPE = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


class PLEPreadvTable:
    """The checkpoint's n-gram shards laid end to end, read with preadv.

    Row r lives in shard ``r // shard_rows`` at local index ``r % shard_rows``.
    """

    def __init__(
        self,
        model_dir: str,
        layer_idx: int,
        *,
        threads: int | None = None,
    ) -> None:
        self.threads = threads or max(8, min(48, 2 * (os.cpu_count() or 8)))
        self.chunk_rows = int(os.environ.get("QWEN4EXP_PLE_MMAP_CHUNK", "8"))
        self._pool = ThreadPoolExecutor(
            max_workers=self.threads, thread_name_prefix="ple-preadv"
        )
        pat = re.compile(
            rf"(?:^|\.)layers\.{layer_idx}\.ple\..*ngram_embedding\.shard_(\d+)\.weight$"
        )
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            files = sorted({v for k, v in weight_map.items() if pat.search(k)})
        else:
            files = sorted(
                fn for fn in os.listdir(model_dir) if fn.endswith(".safetensors")
            )
        # shard_index -> (abs path, abs byte offset of row 0, rows, row_bytes, dtype)
        shards: dict[int, tuple[str, int, int, int, str]] = {}
        for fn in files:
            path = os.path.join(model_dir, fn)
            header, base = _read_header(path)
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                m = pat.search(name)
                if not m:
                    continue
                shard = int(m.group(1))
                rows, width = meta["shape"]
                dtype = meta["dtype"]
                row_bytes = width * (2 if dtype in ("BF16", "F16") else 1)
                begin, end = meta["data_offsets"]
                if end - begin != rows * row_bytes:
                    raise ValueError(f"PLE shard {name}: size mismatch")
                if shard in shards:
                    raise ValueError(f"PLE shard {shard} appears twice")
                shards[shard] = (path, base + begin, rows, row_bytes, dtype)
        if not shards:
            raise ValueError(
                f"no PLE n-gram shards for layer {layer_idx} under {model_dir}"
            )
        expected = list(range(len(shards)))
        if sorted(shards) != expected:
            raise ValueError(f"PLE shards are not contiguous: {sorted(shards)[:5]}...")
        self.num_shards = len(shards)
        self.row_bytes = shards[0][3]
        self.dtype = _TORCH_DTYPE[shards[0][4]]
        self.shard_rows = shards[0][2]
        if any(s[3] != self.row_bytes or s[4] != shards[0][4] for s in shards.values()):
            raise ValueError("PLE shards disagree on dtype/width")
        self.rows_total = sum(s[2] for s in shards.values())
        for i in range(self.num_shards - 1):
            if shards[i][2] != self.shard_rows:
                raise ValueError("PLE shards are not equal-sized")
        self._fds: dict[str, int] = {}
        self.shard_base: list[tuple[int, int]] = []  # (fd, abs byte offset of row 0)
        for i in range(self.num_shards):
            path, begin, _, _, _ = shards[i]
            self.shard_base.append((self._open(path), begin))
        self._staging: torch.Tensor | None = None

    def _open(self, path: str) -> int:
        if path in self._fds:
            return self._fds[path]
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except (AttributeError, OSError):
            pass
        self._fds[path] = fd
        return fd

    def describe(self) -> str:
        return (
            f"PLEPreadvTable(shards={self.num_shards}, rows={self.rows_total:,}, "
            f"row_bytes={self.row_bytes}, threads={self.threads})"
        )

    def _staging_for(self, n: int) -> torch.Tensor:
        if self._staging is None or self._staging.shape[0] < n:
            cap = max(n, 4096)
            self._staging = torch.empty(
                (cap, self.row_bytes), dtype=torch.uint8, pin_memory=True
            )
        return self._staging[:n]

    def _gather_chunk(self, ids: list[int], mv: memoryview, start: int) -> None:
        """Read rows for ids into staging rows start..start+len(ids)."""
        rb = self.row_bytes
        sr = self.shard_rows
        base = self.shard_base
        preadv = os.preadv
        for k, i in enumerate(ids):
            s, l = divmod(i, sr)
            fd, off = base[s]
            dst = (start + k) * rb
            n = preadv(fd, [mv[dst : dst + rb]], off + l * rb)
            if n != rb:
                raise IOError(f"short PLE row read: {n} of {rb} bytes (row {i})")

    def gather_cpu(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather rows for flat int64 CPU ids -> pinned uint8 [n, row_bytes]."""
        assert ids.device.type == "cpu"
        n = ids.numel()
        out = self._staging_for(n)
        if n == 0:
            return out
        flat = ids.view(-1)
        if int(flat.min()) < 0 or int(flat.max()) >= self.rows_total:
            raise IndexError("PLE row id out of range")
        id_list = flat.tolist()
        mv = memoryview(out.numpy()).cast("B")
        chunk = self.chunk_rows
        if n <= chunk:
            self._gather_chunk(id_list, mv, 0)
            return out
        futs = [
            self._pool.submit(self._gather_chunk, id_list[a : a + chunk], mv, a)
            for a in range(0, n, chunk)
        ]
        for f in futs:
            f.result()
        return out