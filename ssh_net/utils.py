from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset


def pca_reduce_cube(cube: np.ndarray, n_components: int) -> np.ndarray:
    height, width, channels = cube.shape
    if n_components >= channels:
        return cube

    data = cube.reshape(-1, channels).astype(np.float32)
    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    reduced = centered @ components.T
    return reduced.reshape(height, width, n_components).astype(np.float32)


def predict_map_from_splits(
    model,
    extractor,
    splits: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 256,
    ignore_index: int = -1,
) -> np.ndarray:
    model.eval()

    height, width = extractor.height, extractor.width
    pred_map = np.full((height, width), fill_value=ignore_index, dtype=np.int32)

    coords_list = []
    for key in ["train", "test"]:
        if key not in splits or splits[key] is None:
            continue
        coords = np.asarray(splits[key])
        if coords.size == 0:
            continue
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"splits['{key}'] expected shape (N, 2), got {coords.shape}")
        coords_list.append(coords.astype(int))

    if not coords_list:
        return pred_map

    coords_all = np.unique(np.vstack(coords_list), axis=0)

    with torch.no_grad():
        for start in range(0, len(coords_all), batch_size):
            batch_coords = coords_all[start:start + batch_size]

            patches = []
            rc_list = []
            for row, col in batch_coords:
                patches.append(extractor.get_patch(int(row), int(col)))
                rc_list.append((int(row), int(col)))

            patches_tensor = torch.stack(patches, dim=0).to(device)
            logits = model(patches_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int32)

            for (row, col), pred in zip(rc_list, preds):
                if 0 <= row < height and 0 <= col < width:
                    pred_map[row, col] = pred

    return pred_map


def Draw_Classification_Map(
    label,
    name: str,
    scale: float = 4.0,
    dpi: int = 400,
    ignore_index: int = -1,
):
    save_path = Path(name)
    if save_path.suffix.lower() != ".png":
        save_path = save_path.with_suffix(".png")

    if save_path.parent and not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    color_map = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [200, 100, 0],
        [0, 200, 100],
        [100, 0, 200],
        [200, 0, 100],
        [100, 200, 0],
        [0, 100, 200],
        [150, 75, 75],
        [75, 150, 75],
        [75, 75, 150],
        [255, 100, 100],
        [100, 255, 100],
        [100, 100, 255],
        [255, 150, 75],
        [75, 255, 150],
    ], dtype=np.uint8)

    numlabel = np.array(label, dtype=np.int32)
    height, width = numlabel.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    valid_mask = (
        (numlabel != ignore_index) &
        (numlabel >= 0) &
        (numlabel < len(color_map))
    )
    rgb[valid_mask] = color_map[numlabel[valid_mask]]

    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.set_size_inches(width * scale / dpi, height * scale / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    fig.savefig(
        str(save_path),
        format="png",
        dpi=dpi,
        pad_inches=0,
    )
    plt.close(fig)
    print(f"Saved classification map to: {save_path}")
    return save_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_cube(cube: np.ndarray) -> np.ndarray:
    cube = cube.astype(np.float32)
    height, width, channels = cube.shape
    reshaped = cube.reshape(-1, channels)
    min_val = reshaped.min(axis=0, keepdims=True)
    max_val = reshaped.max(axis=0, keepdims=True)
    denom = np.where(max_val - min_val == 0, 1.0, max_val - min_val)
    normalized = (reshaped - min_val) / denom
    return normalized.reshape(height, width, channels)


def _wait_for_lock(lock_file: Path, timeout: float = 300.0, poll: float = 0.1) -> None:
    deadline = time.time() + timeout
    while lock_file.exists():
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for cache lock: {lock_file}")
        time.sleep(poll)


def _acquire_lock(lock_file: Path, timeout: float = 300.0, poll: float = 0.1) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out acquiring cache lock: {lock_file}")
            time.sleep(poll)


def load_hsi_data(dataset_cfg: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    mat_data = sio.loadmat(dataset_cfg["data_path"])
    cube = mat_data[dataset_cfg["data_key"]]
    gt = sio.loadmat(dataset_cfg["gt_path"])[dataset_cfg["gt_key"]]
    return normalize_cube(cube), gt.astype(np.int32)


class CubeCache:
    def __init__(self, dataset_configs: Dict[str, Dict[str, object]], cache_dir: Path):
        self.dataset_configs = dataset_configs
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        cache_file = self.cache_dir / f"{dataset}.npz"
        lock_file = cache_file.with_suffix(cache_file.suffix + ".lock")

        _wait_for_lock(lock_file)
        if cache_file.exists():
            data = np.load(cache_file)
            return data["cube"], data["gt"]

        _acquire_lock(lock_file)
        try:
            if cache_file.exists():
                data = np.load(cache_file)
                return data["cube"], data["gt"]
            cfg = self.dataset_configs[dataset]
            cube, gt = load_hsi_data(cfg)
            np.savez_compressed(cache_file, cube=cube, gt=gt)
            return cube, gt
        finally:
            lock_file.unlink(missing_ok=True)


def extract_indices(
    gt: np.ndarray,
    num_classes: int,
    train_ratio: float,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    splits = {"train": [], "test": []}
    for cls in range(1, num_classes + 1):
        positions = np.argwhere(gt == cls)
        if positions.size == 0:
            continue
        rng.shuffle(positions)
        total = len(positions)
        train_count = min(math.ceil(total * train_ratio), total)
        splits["train"].append(positions[:train_count])
        splits["test"].append(positions[train_count:])

    return {
        key: np.concatenate(value, axis=0) if value else np.zeros((0, 2), dtype=int)
        for key, value in splits.items()
    }


def compute_split_counts(
    gt: np.ndarray,
    splits: Dict[str, np.ndarray],
    num_classes: int,
) -> Dict[str, np.ndarray]:
    counts = {
        "train": np.zeros(num_classes, dtype=int),
        "test": np.zeros(num_classes, dtype=int),
    }
    for split_name, coords in splits.items():
        if coords.size == 0:
            continue
        cls_vals = gt[coords[:, 0], coords[:, 1]] - 1
        for cls in range(num_classes):
            counts[split_name][cls] = int(np.sum(cls_vals == cls))
    return counts


def print_sample_statistics(counts: Dict[str, np.ndarray]) -> None:
    train_total = counts["train"].sum()
    test_total = counts["test"].sum()
    print("Samples per class (Train / Test = Total):")
    for idx in range(len(counts["train"])):
        total = counts["train"][idx] + counts["test"][idx]
        print(
            f"Class {idx + 1:2d}: "
            f"{counts['train'][idx]:4d} / {counts['test'][idx]:5d} = {total:6d}"
        )
    total_all = train_total + test_total
    print(f"Total: {train_total} (train) / {test_total} (test) = {total_all}")


class SplitCache:
    def __init__(self, cache_dir: Path, dataset: str) -> None:
        self.cache_dir = cache_dir / "splits" / dataset
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        seed: int,
        gt: np.ndarray,
        num_classes: int,
        train_ratio: float,
    ) -> Dict[str, np.ndarray]:
        ratio_tag = f"tr{train_ratio:.4f}".replace(".", "p")
        file = self.cache_dir / f"seed_{seed}_{ratio_tag}.npz"
        lock = file.with_suffix(".lock")

        _wait_for_lock(lock)
        if file.exists():
            data = np.load(file)
            return {k: data[k] for k in ("train", "test")}

        _acquire_lock(lock)
        try:
            if file.exists():
                data = np.load(file)
                return {k: data[k] for k in ("train", "test")}
            rng = np.random.RandomState(seed)
            splits = extract_indices(gt, num_classes, train_ratio, rng)
            np.savez_compressed(file, **splits)
            return splits
        finally:
            lock.unlink(missing_ok=True)


class PatchExtractor:
    def __init__(self, cube: np.ndarray, patch_size: Tuple[int, int]) -> None:
        pad_h = patch_size[0] // 2
        pad_w = patch_size[1] // 2
        self.height = cube.shape[0]
        self.width = cube.shape[1]
        padded = np.pad(
            cube,
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="reflect",
        )
        self.tensor = torch.from_numpy(padded).permute(2, 0, 1).contiguous()
        self.patch_size = patch_size
        self.pad_h = pad_h
        self.pad_w = pad_w

    def get_patch(self, row: int, col: int) -> torch.Tensor:
        row_p = row + self.pad_h
        col_p = col + self.pad_w
        height, width = self.patch_size
        return self.tensor[
            :,
            row_p - self.pad_h: row_p - self.pad_h + height,
            col_p - self.pad_w: col_p - self.pad_w + width,
        ]


class FastHSIDataset(Dataset):
    def __init__(
        self,
        extractor: PatchExtractor,
        gt: np.ndarray,
        coords: np.ndarray,
        num_classes: int,
    ) -> None:
        self.extractor = extractor
        self.gt = gt
        self.coords = coords
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int):
        row, col = self.coords[idx]
        patch = self.extractor.get_patch(int(row), int(col))
        label = int(self.gt[row, col]) - 1
        if label < 0 or label >= self.num_classes:
            raise ValueError("Invalid label fetched from GT.")
        return patch, label


def build_dataloaders(
    cube: np.ndarray,
    gt: np.ndarray,
    splits: Dict[str, np.ndarray],
    num_classes: int,
    batch_size: int,
    test_batch_size: int,
    patch_size: Tuple[int, int],
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, PatchExtractor]:
    extractor = PatchExtractor(cube, patch_size)
    datasets = {
        split: FastHSIDataset(extractor, gt, coords, num_classes)
        for split, coords in splits.items()
    }
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id + 1
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=generator,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=generator,
    )
    return train_loader, test_loader, extractor


__all__ = [
    "CubeCache",
    "Draw_Classification_Map",
    "PatchExtractor",
    "SplitCache",
    "build_dataloaders",
    "compute_split_counts",
    "pca_reduce_cube",
    "predict_map_from_splits",
    "print_sample_statistics",
    "set_seed",
]
