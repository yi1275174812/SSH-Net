# SSH-Net

This repository contains the code release for the paper in SSH-Net: Symplectic-Inspired Discrete Dynamic Evolution for
Hyperspectral Image Classification.

## Repository layout

- `ssh_net/model.py` contains the main network definition.
- `ssh_net/runner.py` contains the shared training and evaluation pipeline.
- `ssh_net/configs.py` stores dataset paths and the three paper presets.
- `ssh_net/utils.py` contains data loading and patch extraction helpers.
- `IP.py`, `Hou.py`, `WHU.py` are preserved as entry scripts for the three datasets.

## Dataset presets

The three public entry scripts keep different default hyperparameters.

- `IP.py` uses the Indian Pines preset.
- `Hou.py` uses the Hou13 preset.
- `WHU.py` uses the WHU preset.

Please keep the matching entry script for each dataset. Mixing presets across datasets can noticeably degrade performance.

## Data paths

Edit the dataset paths in [ssh_net/configs.py] before training if your `.mat` files are stored elsewhere.

## Training

```bash
python IP.py
python Hou.py
python WHU.py
```

Optional arguments are still supported, for example:

```bash
python IP.py --seeds 0,1,2,3,4
python Hou.py --device cuda:0
```

## Dependencies

Install the packages listed in `requirements.txt`.
