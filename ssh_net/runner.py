from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .configs import (
    BETAS,
    CLIP_GRAD_NORM,
    DATASET_CONFIGS,
    ENABLE_VIS_AND_SAVE,
    EXPERIMENT_NAME,
    NUM_WORKERS,
    PCA_NUM,
    ROOT_LOG_DIR,
    SEEDS,
    TRAINING_PRESETS,
    USE_PCA,
    TrainingPreset,
)
from .model import SpectralSpatialHamiltonianNet
from .utils import (
    CubeCache,
    Draw_Classification_Map,
    SplitCache,
    build_dataloaders,
    compute_split_counts,
    pca_reduce_cube,
    predict_map_from_splits,
    print_sample_statistics,
    set_seed,
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad_norm: float = 0.0,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    loss_sum = torch.zeros((), device=device)
    num_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        if clip_grad_norm is not None and clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        batch_size = images.size(0)
        loss_sum = loss_sum + loss.detach() * batch_size
        num_samples += batch_size

    return (loss_sum / max(1, num_samples)).item()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[np.ndarray, float, float, float]:
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            preds_np = preds.cpu().numpy().astype(int)
            labels_np = labels.cpu().numpy().astype(int)
            for target, pred in zip(labels_np, preds_np):
                if 0 <= target < num_classes and 0 <= pred < num_classes:
                    conf_mat[target, pred] += 1

    per_class_acc = np.zeros(num_classes, dtype=np.float64)
    for cls_idx in range(num_classes):
        denom = conf_mat[cls_idx].sum()
        per_class_acc[cls_idx] = conf_mat[cls_idx, cls_idx] / denom if denom > 0 else 0.0

    total_correct = np.trace(conf_mat)
    total_samples = conf_mat.sum()
    oa = total_correct / total_samples if total_samples > 0 else 0.0
    aa = per_class_acc.mean()

    rows_sum = conf_mat.sum(axis=1)
    cols_sum = conf_mat.sum(axis=0)
    pe = (rows_sum * cols_sum).sum() / (total_samples**2) if total_samples > 0 else 0.0
    kappa = (oa - pe) / (1.0 - pe) if (1.0 - pe) != 0 else 0.0

    return per_class_acc, oa, aa, kappa


def main_one_seed(
    preset: TrainingPreset,
    dataset_name: str,
    seed: int,
    device: torch.device,
    cfg_num_heads: int,
    cfg_num_hamiltonian_layers: int,
    cfg_d_model: int,
    cfg_pos_drop: float,
    cfg_attn_drop: float,
    save_checkpoint_path: Optional[Path] = None,
) -> Tuple[np.ndarray, float, float, float]:
    cfg = DATASET_CONFIGS[dataset_name]
    short_name = cfg["short_name"]
    num_classes = int(cfg["num_classes"])
    train_ratio = float(cfg["train_ratio"])

    print("=" * 60)
    print(f"Dataset: {dataset_name}  (short_name={short_name}), Seed: {seed}")
    print("=" * 60)

    set_seed(seed)

    cache_root = ROOT_LOG_DIR / "cache"
    cube_cache = CubeCache(DATASET_CONFIGS, cache_root)
    cube, gt = cube_cache.load(dataset_name)
    _, _, channels = cube.shape
    print(f"Cube shape: {cube.shape}, GT shape: {gt.shape}")

    if USE_PCA:
        print(f"Applying PCA: C={channels} -> {PCA_NUM}")
        cube = pca_reduce_cube(cube, n_components=PCA_NUM)
        print(f"After PCA, cube shape: {cube.shape}")

    split_cache = SplitCache(cache_root, dataset_name)
    splits = split_cache.get(
        seed=seed,
        gt=gt,
        num_classes=num_classes,
        train_ratio=train_ratio,
    )

    counts = compute_split_counts(gt, splits, num_classes)
    print_sample_statistics(counts)

    train_loader, test_loader, extractor = build_dataloaders(
        cube=cube,
        gt=gt,
        splits=splits,
        num_classes=num_classes,
        batch_size=preset.batch_size,
        test_batch_size=preset.test_batch_size,
        patch_size=(preset.patch_size, preset.patch_size),
        num_workers=NUM_WORKERS,
        seed=seed,
    )

    in_channels = extractor.tensor.shape[0]
    model = SpectralSpatialHamiltonianNet(
        input_channels=in_channels,
        num_classes=num_classes,
        d_model=cfg_d_model,
        num_hamiltonian_layers=cfg_num_hamiltonian_layers,
        num_heads=cfg_num_heads,
        patchsize=preset.patch_size,
        pos_drop=float(cfg_pos_drop),
        attn_drop=float(cfg_attn_drop),
        evolution_mode="dynamic",
        evolution_steps=1,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=preset.learning_rate,
        weight_decay=preset.weight_decay,
        betas=BETAS,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=preset.num_epochs,
    )

    for _ in range(1, preset.num_epochs + 1):
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            clip_grad_norm=CLIP_GRAD_NORM,
        )
        scheduler.step()

    per_class, oa, aa, kappa = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=num_classes,
    )
    print("\n===== Test results =====")
    for idx, acc in enumerate(per_class):
        print(f"Class {idx + 1:02d}: {acc * 100:.2f}%")
    print(f"OA: {oa * 100:.2f}%")
    print(f"AA: {aa * 100:.2f}%")
    print(f"Kappa: {kappa * 100:.2f}%")

    if save_checkpoint_path is not None:
        save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_state = {
            "epoch": preset.num_epochs,
            "model": model.state_dict(),
            "seed": seed,
            "dataset": dataset_name,
        }
        torch.save(save_state, save_checkpoint_path)
        print(f"Saved checkpoint to: {save_checkpoint_path}")

    exp_dir = ROOT_LOG_DIR / EXPERIMENT_NAME / short_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if ENABLE_VIS_AND_SAVE:
        full_pred_map = predict_map_from_splits(
            model=model,
            extractor=extractor,
            splits=splits,
            device=device,
            batch_size=preset.test_batch_size,
            ignore_index=-1,
        )

        test_coords = np.asarray(splits["test"], dtype=np.int64)
        if test_coords.size > 0:
            test_rows = test_coords[:, 0]
            test_cols = test_coords[:, 1]
            preds_test = full_pred_map[test_rows, test_cols]
            gt_test = gt[test_rows, test_cols].astype(np.int32) - 1

            valid_mask = preds_test != -1
            preds_valid = preds_test[valid_mask]
            gt_valid = gt_test[valid_mask]

            correct = (preds_valid == gt_valid).sum()
            total = gt_valid.size
            oa_from_map = correct / total if total > 0 else 0.0
            print(f"[Seed {seed}] OA on test split from full_pred_map: {oa_from_map * 100:.2f}%")

        map_path = exp_dir / f"classification_map_seed_{seed}.png"
        Draw_Classification_Map(full_pred_map, str(map_path), ignore_index=-1)
        print(f"Full prediction map for seed={seed} saved to: {map_path}")

    return per_class, oa, aa, kappa


def _as_list(value: Union[int, float, List[Union[int, float]]]) -> List[Union[int, float]]:
    return value if isinstance(value, list) else [value]


def _src_tag(value: Union[int, float, List[Union[int, float]]]) -> str:
    return f"list{len(value)}" if isinstance(value, list) else "scalar"


def _mean_std_pct(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()) * 100.0, float(arr.std(ddof=0)) * 100.0


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return torch.device("cuda:1")
        return torch.device("cuda:0")
    return torch.device("cpu")


def main(preset_key: str) -> None:
    preset = TRAINING_PRESETS[preset_key]

    parser = argparse.ArgumentParser(description=preset.description)
    parser.add_argument("--dataset_name", type=str, choices=list(DATASET_CONFIGS), default=preset.dataset_name)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--pos_drop", type=float, default=None)
    parser.add_argument("--attn_drop", type=float, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 0,1,2,3,4")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--summary_file", type=str, default=None)
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default=None,
        help="Optional path template to save final checkpoint. Supports {seed}/{dataset} placeholders.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    dataset_name = args.dataset_name
    if dataset_name != preset.dataset_name:
        print(
            f"Warning: preset {preset.entry_name} defaults to {preset.dataset_name}, "
            f"but dataset_name was overridden to {dataset_name}. "
            "The wrapper keeps its original hyperparameter preset."
        )

    cfg = DATASET_CONFIGS[dataset_name]
    short_name = cfg["short_name"]

    seed_list = [int(x) for x in args.seeds.split(",")] if args.seeds else list(SEEDS)
    heads_src = args.num_heads if args.num_heads is not None else preset.num_heads
    layers_src = preset.num_hamiltonian_layers
    d_model_src = args.d_model if args.d_model is not None else preset.d_model
    pos_drop_src = args.pos_drop if args.pos_drop is not None else preset.pos_drop
    attn_drop_src = args.attn_drop if args.attn_drop is not None else preset.attn_drop

    heads_list = [int(x) for x in _as_list(heads_src)]
    layers_list = [int(x) for x in _as_list(layers_src)]
    d_model_list = [int(x) for x in _as_list(d_model_src)]
    pos_drop_list = [float(x) for x in _as_list(pos_drop_src)]
    attn_drop_list = [float(x) for x in _as_list(attn_drop_src)]

    sweep_mode = any(isinstance(x, list) for x in (heads_src, layers_src, d_model_src, pos_drop_src, attn_drop_src))

    log_file = Path(args.log_file) if args.log_file else (ROOT_LOG_DIR / f"{short_name}_sweep.log")
    summary_file = Path(args.summary_file) if args.summary_file else log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    if not log_file.exists():
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"Experiment: {EXPERIMENT_NAME}\n")
            handle.write(f"Dataset: {dataset_name} (short_name={short_name})\n")
            handle.write(f"Seeds: {seed_list}\n")
            handle.write("Format: cfg | per-class(mean+-std,%) | OA(mean+-std,%) | AA(mean+-std,%) | Kappa(mean+-std,%)\n")

    for num_heads in heads_list:
        for num_layers in layers_list:
            for d_model in d_model_list:
                for pos_drop in pos_drop_list:
                    for attn_drop in attn_drop_list:
                        all_oa: List[float] = []
                        all_aa: List[float] = []
                        all_kappa: List[float] = []
                        all_per_class: List[np.ndarray] = []

                        print("\n" + "#" * 72)
                        print(
                            f"Running config: num_heads={num_heads}, "
                            f"num_hamiltonian_layers={num_layers}, d_model={d_model}, "
                            f"pos_drop={pos_drop}, attn_drop={attn_drop}"
                        )
                        print("#" * 72)

                        for seed in seed_list:
                            save_ckpt_path = None
                            if args.save_checkpoint_path:
                                rendered = (
                                    str(args.save_checkpoint_path)
                                    .replace("{seed}", str(seed))
                                    .replace("{dataset}", str(short_name))
                                )
                                save_ckpt_path = Path(rendered)

                            per_class, oa, aa, kappa = main_one_seed(
                                preset=preset,
                                dataset_name=dataset_name,
                                seed=seed,
                                device=device,
                                cfg_num_heads=int(num_heads),
                                cfg_num_hamiltonian_layers=int(num_layers),
                                cfg_d_model=int(d_model),
                                cfg_pos_drop=float(pos_drop),
                                cfg_attn_drop=float(attn_drop),
                                save_checkpoint_path=save_ckpt_path,
                            )
                            all_per_class.append(per_class)
                            all_oa.append(oa)
                            all_aa.append(aa)
                            all_kappa.append(kappa)

                        oa_mean, oa_std = _mean_std_pct(all_oa)
                        aa_mean, aa_std = _mean_std_pct(all_aa)
                        kappa_mean, kappa_std = _mean_std_pct(all_kappa)

                        per_class_arr = np.stack(all_per_class, axis=0)
                        per_class_mean = per_class_arr.mean(axis=0) * 100.0
                        per_class_std = per_class_arr.std(axis=0, ddof=0) * 100.0
                        per_class_str = "; ".join(
                            [
                                f"C{i + 1:02d}={mean:.2f}+/-{std:.2f}"
                                for i, (mean, std) in enumerate(zip(per_class_mean, per_class_std))
                            ]
                        )

                        cfg_str = (
                            f"nh({_src_tag(heads_src)})={num_heads}, "
                            f"hl({_src_tag(layers_src)})={num_layers}, "
                            f"dm({_src_tag(d_model_src)})={d_model}, "
                            f"posdrop({_src_tag(pos_drop_src)})={pos_drop}, "
                            f"attndrop({_src_tag(attn_drop_src)})={attn_drop}"
                        )
                        cfg_str = ("sweep " if sweep_mode else "single ") + cfg_str

                        line = (
                            f"{cfg_str} | "
                            f"PerClass=[{per_class_str}] | "
                            f"OA={oa_mean:.2f}+/-{oa_std:.2f} | "
                            f"AA={aa_mean:.2f}+/-{aa_std:.2f} | "
                            f"Kappa={kappa_mean:.2f}+/-{kappa_std:.2f}"
                        )
                        compact_line = (
                            f"dataset={dataset_name} | "
                            f"OA={oa_mean:.2f}+/-{oa_std:.2f} | "
                            f"AA={aa_mean:.2f}+/-{aa_std:.2f} | "
                            f"Kappa={kappa_mean:.2f}+/-{kappa_std:.2f}"
                        )

                        output_file = summary_file if args.summary_only else log_file
                        output_line = compact_line if args.summary_only else line

                        with output_file.open("a", encoding="utf-8") as handle:
                            handle.write(output_line + "\n")

                        print("\n===== Aggregate over seeds =====")
                        print(output_line)
                        print(f"Logged to: {output_file}")
