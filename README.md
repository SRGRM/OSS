# OSS Suturing Skills – video-based surgical skill assessment

Training and inference code for the three tasks of the **Open Suturing Skills (OSS)** challenge of the
MICCAI Endoscopic Vision (EndoVis) challenge series:

| Task | Input | Output | Ranking metric |
|------|-------|--------|----------------|
| 1 – GRS classification | suturing video | skill class (beginner / intermediate / proficient / expert) | macro F1 ↑, expected cost ↓ |
| 2 – OSATS prediction | suturing video | eight OSATS scores, 1–5 each | mean macro F1 ↑, mean expected cost ↓ |
| 3 – keypoint tracking | video frames | keypoints, visibility and track ids of hands and instruments | HOTA ↑ |

- Challenge: <https://opencas.dkfz.de/endovis/challenges/2025/>
- Paper: <https://arxiv.org/abs/2605.22200>

This repository is part of the **GRM (Generative Reward Machine)** idea described in the paper above.
Choosing a model and its hyper-parameters is treated as a sequential decision problem. The reward of
that problem (how well a configuration trains) is noisy and non-stationary. GRM therefore learns a
reward estimator from the training logs of probe runs and lets a Q-learning agent search the
configurations with that estimated reward. All three training scripts include the GRM search (see
[GRM hyper-parameter search](#grm-hyper-parameter-search)). The framework is described in detail in
the PhD thesis *Generative Reward Machines for Reinforcement Learning in Problems with Poor
Observability* (<https://cora.ucc.ie/items/6cf46796-173c-4426-99bd-676e11656485>). The thesis also
applies it to footstep recognition (StepUP) and to the genomic foundation-model benchmark (OGB).

[PROJECT.md](PROJECT.md) describes the tasks, data and evaluation in more detail.
[INSTALL.md](INSTALL.md) covers installation.

---

## Highlights

- **One CONFIG block per script.** Every option of a pipeline sits at the top of its file, with a
  comment giving its meaning and the accepted alternatives. Any entry can be overridden from the
  command line with `--set KEY=VALUE`.
- **Task 1 / Task 2 clip encoders** (`video_backbones.py`):
  - Kinetics-400 video networks: R(2+1)D-18, R3D-18, MC3-18, S3D, MViT-v1-B, MViT-v2-S and
    Swin3D-T/S/B.
  - ImageNet image networks applied frame by frame: ResNet-18/34/50/101, ResNeXt-50,
    Wide-ResNet-50, DenseNet-121, EfficientNet-B0–B3 and V2-S, MobileNet-V3, RegNet-Y, ConvNeXt-T/S/B,
    ViT-B/16, Swin-T/S and Swin-V2-T, or any timm model. These are combined with a temporal head:
    mean, max, attention pooling, bidirectional LSTM / GRU, a transformer encoder or a dilated TCN.
  - Frames are normalised with the statistics of the chosen pre-trained weights.
- **Task 1 heads and losses.** Heads: softmax, CORAL ordinal, GRS regression (binned into the
  classes) or hybrid. Losses: cross-entropy, class-balanced CE, CE + expected cost, focal, soft
  ordinal labels (SORD), earth mover's distance, CORAL, and Huber / MSE / L1 for the GRS value.
- **Task 2 heads and losses.** Regression, classification, CORAL or hybrid heads. Per-category CE,
  focal, SORD or EMD losses, with class balancing and an expected-cost penalty. A GRS-consistency
  loss and suture-count regression serve as auxiliary objectives.
- **Task 3 network.** Encoders: ResNet / ResNeXt / Wide-ResNet, ConvNeXt, EfficientNet / V2,
  MobileNet-V3, RegNet-Y, or timm models with multi-scale features. Decoders: FPN, U-Net or the
  deconvolution "simple baseline". Heads: 24 keypoint heatmaps, attention-pooled visibility and an
  auxiliary segmentation head.
- **Task 3 losses and decoding.** Focal / MSE / BCE / adaptive-wing heatmap losses plus an optional
  soft-argmax coordinate loss. Decoding uses peak NMS and sub-pixel refinement; needle instances are
  assembled and then tracked across frames with Hungarian matching.
- **Training machinery.** Eight optimisers (AdamW, Adam, NAdam, RAdam, Adamax, SGD, RMSprop, Adagrad)
  and ten learning-rate schedules with warm-up. Also: mixed precision, gradient accumulation and
  clipping, weight EMA, encoder freezing and early stopping.
- **Data handling.** Mixup / CutMix, rotation, grey-scale and blur augmentation, and class-balanced
  sampling. The validation split is grouped and stratified, so pre/post videos of one student stay
  together. Multi-clip plus flip test-time augmentation and checkpoint ensembles are supported.
- **Everything in one run folder.** Every epoch writes its checkpoint, its submission file and a row
  of the training report (CSV and JSON) to the run folder.
- **Strict data checks.** An unreadable video, frame, mask or annotation stops the program with a
  clear message.
- **Multi-GPU and CUDA support.** Set `GPU_IDS = [0, 1, ...]` to select GPUs (exported as
  `CUDA_VISIBLE_DEVICES`, with DataParallel across them). Supports CUDA 11.x and 12.x installs.
- **Challenge container.** A ready-made container with the `Process_SkillEval.sh` entry point that
  the evaluation system expects.

---

## Repository layout

```
oss-suturing-skills/
├── oss_suturing/
│   ├── task1_grs_classification.py   # Task 1 training / validation / prediction / GRM phases
│   ├── task2_osats_prediction.py     # Task 2 training / validation / prediction / GRM phases
│   ├── task3_keypoint_tracking.py    # Task 3 training / validation / prediction / GRM phases
│   ├── grm.py                        # Generative Reward Machine (task independent)
│   ├── video_backbones.py            # clip encoders of Tasks 1 and 2
│   ├── training_blocks.py            # optimisers, schedules, ordinal losses, mixing, sampling
│   ├── inference_task1.py            # container entry point, task "GRS"
│   ├── inference_task2.py            # container entry point, task "OSATS"
│   ├── inference_task3.py            # container entry point, keypoint tracking
│   └── inference_common.py           # helpers shared by the entry points
├── docker/
│   ├── Dockerfile                    # CUDA 11.8 runtime (base image recommended by the organisers)
│   ├── Dockerfile.cu121              # CUDA 12.1 runtime
│   ├── Process_SkillEval.sh          # entry point called by the evaluation system
│   ├── requirements-docker.txt       # packages installed on top of the PyTorch image
│   └── test_container.sh             # build the image and run all tasks locally
├── weights/README.md                 # which checkpoint goes where
├── requirements.txt                  # dependencies without torch
├── requirements-cu118.txt            # + torch / torchvision for CUDA 11.x drivers
├── requirements-cu121.txt            # + torch / torchvision for CUDA 12.x drivers
├── pyproject.toml                    # pip installation and command-line entry points
├── INSTALL.md
└── PROJECT.md
```

---

## Quick start

```bash
git clone <repository url> oss-suturing-skills
cd oss-suturing-skills
pip install -r requirements-cu121.txt      # or requirements-cu118.txt on a CUDA 11 machine
pip install -e .
```

### Training

The default paths in the CONFIG blocks follow the layout of the released data (see
[PROJECT.md](PROJECT.md)). Either edit the CONFIG block or override entries on the command line:

```bash
# Task 1 - video network
python oss_suturing/task1_grs_classification.py --mode train \
    --set VIDEO_DIR=data/Train/videos --set ANNOTATION_FILE=data/OSATS_MICCAI_trainset.xlsx \
    --set BACKBONE=r2plus1d_18 --set LOSS=sord --set NUM_EPOCHS=30 --set GPU_IDS=[0,1]

# Task 1 - image network per frame + transformer over time
python oss_suturing/task1_grs_classification.py --mode train \
    --set BACKBONE=convnext_tiny --set TEMPORAL_HEAD=transformer --set IMG_SIZE=224

# Task 2 - OSATS scores with a CORAL head
python oss_suturing/task2_osats_prediction.py --mode train --set TARGET_MODE=coral

# Task 3 - keypoint tracking with a U-Net decoder
python oss_suturing/task3_keypoint_tracking.py --mode train --set ENCODER=convnext_tiny --set DECODER_TYPE=unet \
    --set TRAIN_FRAMES_DIR=data/train/frames --set TRAIN_MOT_DIR=data/train/mot \
    --set TRAIN_MASKS_DIR=data/train/masks --set VAL_FRAMES_DIR=data/val/frames --set VAL_MOT_DIR=data/val/mot
```

After `pip install -e .` the same scripts are also available as `oss-task1`, `oss-task2` and `oss-task3`.

Each run creates `runs/<run name>/` with:

```
config_used.json               every option of the run
train.log                      full console log
training_report.csv / .json    one row per epoch (learning rate, losses, batch-loss spread, metrics, timing)
checkpoints/epoch_XX.pt        weights of every epoch (+ best.pt, last.pt)
submissions/epoch_XX/          predictions of that epoch
best_submission/               copy of the best epoch's predictions
val_predictions/ or val_metrics/   per-epoch validation details
```

---

## GRM hyper-parameter search

### Concept

| GRM element | In this repository |
|-------------|--------------------|
| true state | a configuration together with every source of training noise (not observable) |
| observation | the training log of a probe run (per-epoch losses, batch-loss spread, validation metrics) |
| mapping function | configuration → GRM state, the value of `GRM_STATE_KEY`: by default the batch size (as described for the OSS entry in the paper), or e.g. the architecture (as for StepUP); `GRM_STATE_BINS` groups values into classes |
| actions | combinations of the remaining keys of `GRM_SEARCH_SPACE` (models, losses, optimisers, schedules, learning rates, ...) |
| estimator | per GRM state, maps signals of the first `GRM_PROBE_EPOCHS` epochs to the final performance; linear (`Perf = Σ wᵢ·Pᵢ`), ridge, robust Huber, k-NN, global or shrinkage variants |
| reward machine | explore / exploit states; a labelling function marks each probe as *improved*, *plateau*, *diverged* or *invalid*; rewards are the estimate plus a novelty bonus while exploring, and a penalty for failed runs |
| agent | tabular Q-learning (or bandit updates) with epsilon-greedy, UCB or softmax exploration |

Three signal presets reproduce the earlier GRM entries; any column of the training report can also be
combined freely (`GRM_SIGNAL_PRESET = "custom"`):

| Preset | Signals | Used for |
|--------|---------|----------|
| `oss` | per-epoch train loss, validation loss and validation score, their slopes, best score, loss decrease | OSS challenge entry (early training dynamics) |
| `stepup` | P1 = mean epoch loss, P2 = standard deviation of all batch losses | StepUP footstep recognition |
| `ogb` | P1 = mean of a score metric, P2 = mean of an error metric | OGB genomic benchmark |

Setting `GRM_REWARD_SOURCE = "raw"` turns the search into the plain RL baseline. That baseline is
rewarded directly with a log signal, needs no calibration, and allows a direct GRM-versus-RL
comparison.

### Phases

Every phase reads and writes the GRM folder (`GRM_DIR`, default `runs/grm_task<N>`). A phase can
therefore run at a different time or on another machine, and an interrupted phase resumes where it
stopped.

```bash
python oss_suturing/task1_grs_classification.py --mode grm_calibrate   # phase 1: calibration probes
python oss_suturing/task1_grs_classification.py --mode grm_fit         # reward estimator per GRM state
python oss_suturing/task1_grs_classification.py --mode grm_search      # phase 2: GRM_SEARCH_STEPS more search steps
python oss_suturing/task1_grs_classification.py --mode grm_final       # choose and train the best configuration
python oss_suturing/task1_grs_classification.py --mode grm_report      # refresh grm_summary.md
python oss_suturing/task1_grs_classification.py --mode grm_full        # all phases in one go
```

- **Calibration.** `grm_calibrate` trains `GRM_CALIBRATION_PROBES` random actions per GRM state for
  `GRM_CALIBRATION_EPOCHS` epochs. It stores the early signals and the true performance (best, last,
  or mean of the last k epochs of the selection score).
- **Sharing calibration across machines.** `GRM_CALIBRATION_STATES` restricts a call to some states.
  Each machine then works in its own folder, and `grm_merge` (with `GRM_MERGE_DIRS`) combines the
  probes.
- **Estimators.** `grm_fit` reports the leave-one-out error and rank correlation of every estimator.
- **Search.** `grm_search` trains each chosen pair for `GRM_PROBE_EPOCHS` epochs only. The reward
  comes from the estimator and the reward machine. Steps continue across calls until
  `GRM_MAX_SEARCH_STEPS`; `GRM_MAX_HOURS` limits the time of one call.
- **Final choice.** `grm_final` picks the configuration by visited Q-value, by mean estimate, or by a
  short verification run of the top candidates. It then trains that configuration in full in
  `<GRM_DIR>/final_<model>`.
- **Failure handling.** Configurations that cannot be trained (incompatible options, out of GPU
  memory) are labelled *invalid*, penalised and excluded from the rest of the search.

GRM folder contents: `space.json`, `probes.jsonl` (every probe with its history and signals),
`estimators.json`, `search_state.json` (Q-table, visits, reward-machine state, random state),
`calibration_log.csv`, `search_log.csv`, `best_configuration.json`, `grm_summary.md` and `train.log`.

---

## Inference

```bash
python oss_suturing/inference_task1.py <video folder> <output folder> --checkpoint runs/<run>/checkpoints/best.pt
python oss_suturing/inference_task2.py <video folder> <output folder> --checkpoint runs/<run>/checkpoints/best.pt
python oss_suturing/inference_task3.py <video or frame folder> <output folder> --checkpoint runs/<run>/checkpoints/best.pt
```

Without `--checkpoint`, the files named in `CONFIG.CHECKPOINTS` are loaded from `weights/` (see
[weights/README.md](weights/README.md)). Repeating `--checkpoint` builds an ensemble for Tasks 1 and 2.

The inference scripts accept two checkpoint formats:

- checkpoints written by the training scripts, from which the network (including per-frame encoders,
  temporal heads and decoders), the clip geometry and the decoding are rebuilt automatically;
- plain PyTorch state dictionaries of the first submission models.

Output files:

| Task | File | Columns |
|------|------|---------|
| 1 | `task1_results.csv` | `VIDEO, GRS` (class 0–3); optionally class name and probabilities |
| 2 | `task2_results.csv` | `VIDEO, OSATS_RESPECT, …, OSATS_FINAL_QUALITY` (integers 1–5); optionally expected values and GRS |
| 3 | `<video>.txt`, `mot/<video>_frame_<id>.txt` | `frame, track, class, -1, -1, -1, -1, x1, y1, v1, …` |

### Challenge container

```bash
cp runs/<task1 run>/checkpoints/best.pt weights/task1_grs_model.pth
cp runs/<task2 run>/checkpoints/best.pt weights/task2_osats_model.pth
cp runs/<task3 run>/checkpoints/best.pt weights/task3_keypoint_model.pt

docker build -f docker/Dockerfile -t oss-skill:cu118 .
docker run --gpus '"device=0"' --ipc=host -v <videos>:/input -v <results>:/output \
    oss-skill:cu118 /usr/local/bin/Process_SkillEval.sh GRS      # or OSATS / TRACK
```

`docker/test_container.sh <videos> <task3 input>` builds the image and runs all three tasks the way
the evaluation system does.

---

## Configuration in short

| Area | Main entries |
|------|--------------|
| run control | `RUN_MODE`, `RUN_NAME`, `OUTPUT_ROOT`, `SEED`, `DETERMINISTIC` |
| hardware | `GPU_IDS`, `CUDA_BUILD`, `USE_AMP`, `NUM_WORKERS` |
| data | `VIDEO_DIR`, `ANNOTATION_FILE`, `FRAME_CACHE_DIR`, `DECODER`, `FRAME_POOL_SIZE`, `FRAME_SAMPLING`, `SAMPLER` |
| augmentation | `AUG_*`, `MIX_MODE`, `MIX_ALPHA`, `MIX_PROB` |
| model | `BACKBONE` / `ENCODER`, `TEMPORAL_HEAD`, `DECODER_TYPE`, `HEAD_TYPE` / `TARGET_MODE`, `PRETRAINED`, `NORMALIZATION` |
| loss | `LOSS`, `CLASS_LOSS`, `LABEL_MODE`, `EC_LOSS_WEIGHT`, `FOCAL_GAMMA`, `SORD_ALPHA`, `HEATMAP_LOSS`, `COORD_LOSS_WEIGHT`, … |
| optimisation | `OPTIMIZER`, `LEARNING_RATE`, `SCHEDULER`, `WARMUP_EPOCHS`, `NUM_EPOCHS`, `BATCH_SIZE`, `EMA_DECAY`, … |
| evaluation | `VAL_STRATEGY`, `SELECTION_METRIC`, `TTA_NUM_CLIPS`, `TTA_HFLIP`, … |
| GRM search | `GRM_SEARCH_SPACE`, `GRM_STATE_KEY`, `GRM_SIGNAL_PRESET`, `GRM_ESTIMATOR`, `GRM_REWARD_MACHINE`, `GRM_ACTION_POLICY`, `GRM_Q_UPDATE`, `GRM_FINAL_SELECTION`, … |

The comment next to every entry in the scripts lists the accepted values.

---

## Citation

If this code is useful for your work, please cite the paper:

> OSS challenge paper, arXiv:2605.22200 — <https://arxiv.org/abs/2605.22200>

For the Generative Reward Machine framework:

> S. Rezaei, *Generative Reward Machines for Reinforcement Learning in Problems with Poor Observability*,
> PhD thesis, University College Cork — <https://cora.ucc.ie/items/6cf46796-173c-4426-99bd-676e11656485>

## Acknowledgements

We thank the organisers of the Open Suturing Skills challenge for the data, the annotations and the
evaluation, and the EndoVis community for hosting the challenge.
