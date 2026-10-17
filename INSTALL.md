# Installation

The code runs on Linux with Python 3.8 or newer. A CUDA GPU is strongly recommended for training;
inference also works on the CPU (`--set GPU_IDS=[]`).

## 1. Create an environment

```bash
python3 -m venv .venv            # or: conda create -n oss python=3.10 && conda activate oss
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install PyTorch for your CUDA driver

Check the driver version with `nvidia-smi` and pick the matching requirements file. The PyTorch
wheels bring their own CUDA runtime, so only the driver on the machine matters.

| Driver (as shown by `nvidia-smi`) | Command |
|-----------------------------------|---------|
| CUDA 11.x, e.g. 11.4 (R470 series) | `pip install -r requirements-cu118.txt` |
| CUDA 12.x, e.g. 12.2 (R535 series) | `pip install -r requirements-cu121.txt` |
| no GPU | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` then `pip install -r requirements.txt` |

Both requirements files install the remaining dependencies as well (`requirements.txt`).

## 3. Install the package

```bash
pip install -e .                 # editable install from the repository root
# or, for a fixed copy:
pip install .
```

This adds the commands

| Command | Same as |
|---------|---------|
| `oss-task1` | `python oss_suturing/task1_grs_classification.py` |
| `oss-task2` | `python oss_suturing/task2_osats_prediction.py` |
| `oss-task3` | `python oss_suturing/task3_keypoint_tracking.py` |
| `oss-infer-task1` / `-task2` / `-task3` | `python oss_suturing/inference_taskN.py` |

`pip install -e ".[torch]"` installs torch from PyPI as well, which is fine on machines where the
default PyPI build matches the driver. `pip install -e ".[dev]"` adds the linters used for the code.
`pip install -e ".[timm]"` adds timm for the `timm:<model name>` backbones and encoders.

## 4. Check the installation

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
oss-task1 --help
oss-infer-task1 --help
```

At start-up every training script compares `CONFIG.CUDA_BUILD` (`cu118`, `cu121` or `cpu`) with the
build of the installed torch and prints a warning when they differ.

## Older PyTorch installations

The scripts also run on older environments that cannot be upgraded:

- On torch versions without `torch.autocast` (older than 1.10), mixed precision falls back to
  `torch.cuda.amp.autocast`.
- On torchvision versions without weight enums (older than 0.13), pre-trained weights are loaded
  with `pretrained=True`. The ResNet / R(2+1)D / R3D / MC3 backbones work there. MViT, Swin3D, S3D
  and ConvNeXt need a recent torchvision.

Installing `requirements-cu118.txt` into a fresh environment is still the recommended path on a
CUDA 11.4 machine.

## Data

Place the challenge data anywhere and point the CONFIG entries to it, for example:

```
data/
├── OSATS_MICCAI_trainset.xlsx
├── Train/videos/<VIDEO>.mp4                       # Tasks 1 and 2
├── train/frames/<video>_frame_<id>.png            # Task 3
├── train/masks/<video>_frame_<id>_<object>_mask.png
├── train/mot/<video>_frame_<id>.txt
├── val/frames/<video>/<video>_frame_<id>.png
└── val/mot/<video>_frame_<id>.txt
```

Details of every file are in [PROJECT.md](PROJECT.md).

## Docker

Docker with the NVIDIA container toolkit is needed to build and test the challenge container:

```bash
docker build -f docker/Dockerfile -t oss-skill:cu118 .          # CUDA 11.8 base image
docker build -f docker/Dockerfile.cu121 -t oss-skill:cu121 .    # CUDA 12.1 base image
```

Build from the repository root: the image copies `oss_suturing/` to `/app/code` and `weights/` to
`/app/weights`. torch and torchvision come from the PyTorch base image; only the packages in
`docker/requirements-docker.txt` are added (with the headless OpenCV build, since the runtime image
has no GUI libraries).
