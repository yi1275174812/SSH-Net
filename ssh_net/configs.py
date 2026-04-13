from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class TrainingPreset:
    entry_name: str
    dataset_name: str
    num_epochs: int
    batch_size: int
    test_batch_size: int
    learning_rate: float
    weight_decay: float
    patch_size: int
    num_heads: int
    num_hamiltonian_layers: int
    d_model: int
    pos_drop: float
    attn_drop: float
    description: str


DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "IP": {
        "train_ratio": 0.01,
        "num_classes": 16,
        "short_name": "IP",
        "data_path": "/mnt/data/yxf/lz/data/Indian_Pines/Indian_pines_corrected.mat",
        "gt_path": "/mnt/data/yxf/lz/data/Indian_Pines/Indian_pines_gt.mat",
        "data_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
    },
    "Hou13": {
        "train_ratio": 0.01,
        "num_classes": 15,
        "short_name": "Hou13",
        "data_path": "/mnt/data/yxf/lz/data/Houston2013/HustonU_IM.mat",
        "gt_path": "/mnt/data/yxf/lz/data/Houston2013/HustonU_gt.mat",
        "data_key": "hustonu",
        "gt_key": "hustonu_gt",
    },
    "WHU": {
        "train_ratio": 0.001,
        "num_classes": 9,
        "short_name": "WHU",
        "data_path": "/mnt/data/yxf/lz/data/WHU-Hi-LongKou/WHU_Hi_LongKou.mat",
        "gt_path": "/mnt/data/yxf/lz/data/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat",
        "data_key": "WHU_Hi_LongKou",
        "gt_key": "WHU_Hi_LongKou_gt",
    },
}

SEEDS = [0, 1, 2, 3, 4]
NUM_WORKERS = 0
ENABLE_VIS_AND_SAVE = False
BETAS: Tuple[float, float] = (0.9, 0.999)
CLIP_GRAD_NORM = 0.0
ROOT_LOG_DIR = Path("./result")
EXPERIMENT_NAME = "ssh_net"
USE_PCA = False
PCA_NUM = 150

TRAINING_PRESETS: Dict[str, TrainingPreset] = {
    "IP": TrainingPreset(
        entry_name="IP",
        dataset_name="IP",
        num_epochs=500,
        batch_size=64,
        test_batch_size=64,
        learning_rate=1e-3,
        weight_decay=5e-2,
        patch_size=13,
        num_heads=2,
        num_hamiltonian_layers=6,
        d_model=96,
        pos_drop=0.4,
        attn_drop=0.4,
        description="Run training on Indian Pines with the paper preset.",
    ),
    "Hou13": TrainingPreset(
        entry_name="Hou13",
        dataset_name="Hou13",
        num_epochs=500,
        batch_size=128,
        test_batch_size=64,
        learning_rate=5e-4,
        weight_decay=5e-2,
        patch_size=11,
        num_heads=8,
        num_hamiltonian_layers=3,
        d_model=128,
        pos_drop=0.5,
        attn_drop=0.5,
        description="Run training on Houston 2013 with the paper preset.",
    ),
    "WHU": TrainingPreset(
        entry_name="WHU",
        dataset_name="WHU",
        num_epochs=700,
        batch_size=64,
        test_batch_size=64,
        learning_rate=1e-3,
        weight_decay=5e-3,
        patch_size=13,
        num_heads=8,
        num_hamiltonian_layers=3,
        d_model=128,
        pos_drop=0.5,
        attn_drop=0.3,
        description="Run training on WHU with the paper preset.",
    ),
}
