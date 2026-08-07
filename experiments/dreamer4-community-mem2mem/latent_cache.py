"""Exact window-keyed latent cache for the community Dreamer 4 tokenizer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CACHE_FORMAT = "dreamer4-community-window-latents-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path) -> tuple[dict, str]:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != CACHE_FORMAT:
        raise RuntimeError(f"unsupported latent cache format: {manifest.get('format')!r}")
    return manifest, sha256(path)


class WindowLatentCache:
    """Read-only mapping from flattened frame starts to exact W-frame latents."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest, self.manifest_sha256 = load_manifest(self.root)
        self.latents = np.load(self.root / self.manifest["latents_file"], mmap_mode="r")
        self.row_by_start = np.load(
            self.root / self.manifest["row_by_start_file"], mmap_mode="r"
        )
        if list(self.latents.shape) != self.manifest["shape"]:
            raise RuntimeError("latent cache shape differs from manifest")
        if str(self.latents.dtype) != self.manifest["dtype"]:
            raise RuntimeError("latent cache dtype differs from manifest")
        if self.row_by_start.ndim != 1:
            raise RuntimeError("row_by_start must be one-dimensional")

    def rows_for_starts(self, starts: np.ndarray) -> np.ndarray:
        starts = np.asarray(starts, dtype=np.int64)
        if starts.size == 0 or starts.min() < 0 or starts.max() >= self.row_by_start.shape[0]:
            raise IndexError(f"cache start outside [0,{self.row_by_start.shape[0]})")
        rows = np.asarray(self.row_by_start[starts], dtype=np.int64)
        if (rows < 0).any():
            bad = starts[rows < 0][:8].tolist()
            raise KeyError(f"latent cache has no valid W-frame window for starts {bad}")
        return rows


class CachedLatentClipDataset(Dataset):
    """Long clips backed by exact cached latents; pixel shards are never read per sample."""

    def __init__(self, base, cache_root: Path, *, window: int, clip_length: int):
        self.base = base
        self.cache_root = Path(cache_root)
        self.window = int(window)
        self.clip_length = int(clip_length)
        if self.window <= 0 or self.clip_length < self.window:
            raise ValueError("invalid cache rollout geometry")
        if self.window % 2 or (self.clip_length - self.window) % (self.window // 2):
            raise ValueError("cache geometry requires W/2-aligned slides")
        manifest, self.cache_manifest_sha256 = load_manifest(self.cache_root)
        if int(manifest["window"]) != self.window:
            raise RuntimeError("cache window differs from trainer window")
        if manifest["dtype"] != "float32":
            raise RuntimeError("trainer requires the exact FP32 latent cache")
        if len(base.tasks) != 1 or int(manifest["frame_count"]) != int(base.ep[0].numel()):
            raise RuntimeError("cache frame axis differs from the training metadata")
        self._manifest = manifest
        self._cache = None
        self._offsets = np.arange(
            0, self.clip_length - self.window + 1, self.window // 2, dtype=np.int64
        )

    def __len__(self):
        return len(self.base)

    def _get_cache(self) -> WindowLatentCache:
        if self._cache is None:
            self._cache = WindowLatentCache(self.cache_root)
        return self._cache

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state

    def __getitem__(self, index):
        task_idx, start = self.base._lookup(int(index))
        cache = self._get_cache()
        starts = start + self._offsets
        rows = cache.rows_for_starts(starts)
        # Copy into ordinary worker-owned memory. The DataLoader can then pin and
        # transfer it without retaining references to the read-only mmap pages.
        latents = torch.from_numpy(np.array(cache.latents[rows], copy=True))

        T, A = self.clip_length, self.base.A
        act = self.base.act[task_idx][start + 1:start + 1 + T]
        act_dim = int(self.base._act_dims[task_idx])
        act_padded = torch.zeros(T, A, dtype=torch.float32)
        if act_dim > 0:
            act_padded[:, :act_dim] = torch.nan_to_num(act[:, :act_dim], nan=0.0)
        act_mask = self.base._act_mask_1d[task_idx][None, :].expand(T, A).contiguous()
        return {
            "latents": latents,
            "act": act_padded,
            "act_mask": act_mask,
            "_source_window_index": torch.tensor(index, dtype=torch.long),
            "_global_start": torch.tensor(start, dtype=torch.long),
        }
