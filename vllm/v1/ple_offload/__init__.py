"""vLLM PLE mmap swap for vllm-backport v0.11.0 (bp/Dockerfile installs).

Why: the PleOffloadWorker materializes the n-gram table as anonymous CPU RAM
(102 GiB BF16 for Intel AutoRound, 51 GiB FP8 for RadixArk/albucino). On a
swapless host that is unreclaimable. But a token touches only ~16 small rows,
so the table can live on NVMe and be served through the page cache — the
llama.cpp GGUF-mmap trick.

How: apply() wraps Qwen4ExpNGramEmbedding.load_weights in the offload process
only, gated by VLLM_PLE_MMAP=1. On the first call, one-time-assemble a flat
cache file laid out exactly like the runtime [padded_vocab, dim] weight (shard
n covers consecutive rows; vocab padding stays zero), then point weight.data
at an np.memmap view of it. Incoming shard tensors are dropped unread
(assembly streams bytes from the safetensors files via index offsets), so the
table's anonymous pages are never touched. F.embedding on the mmap tensor
faults rows on demand; the kernel caches the hot subset, all evictable.

The 102 GiB parameter is allocated by __init__ as untouched virtual pages
(overcommit — zero physical cost) and released the moment .data is swapped,
so peak RSS is a page cache we can lose.

Cache: $VLLM_PLE_MMAP_DIR/<key>.bin; key hashes model path, geometry and
per-shard file offsets. Stale entries are removed by size mismatch.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_GB = 1024**3
_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I32": torch.int32,
    "I64": torch.int64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _enabled() -> bool:
    if os.environ.get("VLLM_PLE_MMAP", "0") != "1":
        return False
    try:
        from vllm.config import get_current_vllm_config

        engram = get_current_vllm_config().engram_config
        return engram is not None and engram.cpu_offload
    except Exception:
        return False


def _model_dir() -> Path:
    from vllm.config import get_current_vllm_config

    src = get_current_vllm_config().model_config.model
    path = Path(src)
    if not path.is_dir():
        raise RuntimeError(f"VLLM_PLE_MMAP needs a local model dir, got {src}")
    return path


def _shard_regions(model_dir: Path) -> dict[int, tuple[Path, int, int, tuple[int, ...]]]:
    """shard_index -> (file, byte_offset, byte_len, shape) for ngram weights."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    headers: dict[Path, dict] = {}
    regions: dict[int, tuple[Path, int, int, tuple[int, ...]]] = {}
    for name, fname in index["weight_map"].items():
        if "ngram_embedding." not in name or not name.endswith(".weight"):
            continue
        leaf = name.rsplit(".", 2)[-2]
        if not (leaf.startswith("shard_") and leaf[6:].isdigit()):
            continue
        fpath = model_dir / fname
        if fpath not in headers:
            with open(fpath, "rb") as fh:
                hlen = int.from_bytes(fh.read(8), "little")
                headers[fpath] = json.loads(fh.read(hlen))
        entry = headers[fpath][name]
        start, end = entry["data_offsets"]
        regions[int(leaf[6:])] = (fpath, start, end - start, tuple(entry["shape"]))
    return regions


def _cache_key(model_dir: Path, regions, rows: int, dim: int, dtype) -> str:
    h = hashlib.sha256()
    h.update(str(model_dir.resolve()).encode())
    h.update(f"|{rows}x{dim}|{dtype}|".encode())
    for i in sorted(regions):
        f, off, ln, _ = regions[i]
        h.update(f"{i}:{f.name}:{off}:{ln}:{f.stat().st_size},".encode())
    return h.hexdigest()[:24]


def _assemble(cache: Path, regions, rows: int, dim: int, itemsize: int,
              split: int, org_vocab: int) -> None:
    expected = rows * dim * itemsize
    cache.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "PLE mmap: assembling %s (%.1f GiB, %d shards) — one-time sequential pass",
        cache.name, expected / _GB, len(regions),
    )
    tmp = cache.with_name(cache.name + ".tmp")
    shard_rows = (org_vocab + split - 1) // split
    with open(tmp, "wb") as out:
        out.truncate(expected)
        for i, (fpath, off, blen, shape) in sorted(regions.items()):
            if shape[-1] != dim:
                raise RuntimeError(f"PLE mmap: shard {i} dim {shape[-1]} != {dim}")
            start_row = i * shard_rows
            n_rows = min(shard_rows, org_vocab - start_row, shape[0])
            if n_rows <= 0:
                continue
            nbytes = min(n_rows * dim * itemsize, blen)
            with open(fpath, "rb") as src:
                src.seek(off)
                out.seek(start_row * dim * itemsize)
                remaining = nbytes
                while remaining > 0:
                    chunk = src.read(min(256 * 1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError(f"PLE mmap: short read in shard {i}")
                    out.write(chunk)
                    remaining -= len(chunk)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, cache)


def _swap(layer, model_dir: Path) -> None:
    embedding = layer.ngram_embedding
    weight = embedding.weight
    rows, dim = int(weight.shape[0]), int(weight.shape[1])
    dtype = weight.dtype
    itemsize = torch.empty(0, dtype=dtype).element_size()

    regions = _shard_regions(model_dir)
    if not regions:
        raise RuntimeError(f"PLE mmap: no ngram shard tensors indexed under {model_dir}")

    cache_dir = Path(os.environ.get("VLLM_PLE_MMAP_DIR", "/models/.ple-mmap"))
    key = _cache_key(model_dir, regions, rows, dim, dtype)
    cache = cache_dir / f"{key}.bin"
    expected = rows * dim * itemsize
    if cache.exists() and cache.stat().st_size != expected:
        logger.warning("PLE mmap: removing stale cache %s", cache)
        cache.unlink()
    if not cache.exists():
        _assemble(
            cache, regions, rows, dim, itemsize,
            int(layer.split_ngram_parts), int(embedding.org_vocab_size),
        )

    mm = np.memmap(cache, dtype=np.uint8, mode="r+", shape=(expected,))
    view = torch.from_numpy(mm).view(dtype).view(rows, dim)
    weight.data = view
    # Keep the memmap alive next to the tensor that aliases it.
    layer._ple_mmap_keepalive = mm
    # CPU-staged gathers (see _lookup) read the mmap through self.weight; no
    # UVA view is built over the file mapping — cudaHostRegister would pin the
    # whole table back into RAM. The __init__ env gate sets _ple_mmap on the
    # embedding already.
    embedding._ple_mmap_keepalive = mm
    logger.info(
        "PLE mmap: ngram weight -> file-backed mmap %s (%dx%d %s, %.1f GiB, "
        "page-cache served, evictable)",
        cache, rows, dim, dtype, expected / _GB,
    )


def apply(cls) -> None:
    """Wrap cls.load_weights (call at end of ple_layer.py)."""
    if getattr(cls, "_ple_mmap_wrapped", False):
        return
    orig = cls.load_weights

    def load_weights(self, weights):
        if not _enabled():
            return orig(self, weights)
        model_dir = _model_dir()
        if not hasattr(self, "_ple_mmap_done"):
            _swap(self, model_dir)
            self._ple_mmap_done = True
        # Drop shard tensors (their bytes are served by the mmap), defer
        # everything else to the stock loader.
        rest = []
        for name, tensor in weights:
            leaf = name.rsplit(".", 2)[-2] if name.count(".") >= 2 else ""
            if (
                "ngram_embedding." in name
                and name.endswith(".weight")
                and leaf.startswith("shard_")
            ):
                del tensor
                continue
            rest.append((name, tensor))
        loaded = orig(self, iter(rest))
        # The stock loader marks ngram_embedding.weight only through the shard
        # copies we skipped; the runner's materialization check needs it.
        loaded.add("ngram_embedding.weight")
        return loaded

    cls.load_weights = load_weights
    cls._ple_mmap_wrapped = True
