# Project information

This document summarises the Open Suturing Skills (OSS) challenge as far as it matters for the
code: what each task asks for, what the data look like, how the results are scored and how a
solution is submitted. The official description is on the challenge page
(<https://opencas.dkfz.de/endovis/challenges/2025/>) and in the paper
(<https://arxiv.org/abs/2605.22200>).

## Background

Open suturing is a basic surgical skill that is usually rated by experts with the OSATS scheme
(Objective Structured Assessment of Technical Skills). The challenge asks whether such ratings can
be predicted from video. The videos show participants suturing on a training pad, recorded from
above. Each video was rated by three independent raters.

## Annotations of Tasks 1 and 2

`OSATS_MICCAI_trainset.xlsx`, sheet `DATASET`, holds one row per video and rater:

| Column | Meaning |
|--------|---------|
| `STUDENT` | participant id; the same student appears in several videos (e.g. before and after training) |
| `GROUP` | training group of the student |
| `TIME` | recording time point (e.g. pre / post) |
| `SUTURES` | number of sutures placed |
| `INVESTIGATOR` | rater id |
| `VIDEO` | video id = file name without extension |
| `OSATS_RESPECT` … `OSATS_FINAL_QUALITY` | the eight OSATS items, 1–5 each |
| `GLOBA_RATING_SCORE` | global rating score (sum of the eight items; the column name is misspelled in the released sheet) |

The training scripts find the columns by name, so their order does not matter. The misspelled GRS
column is recognised as well. Because the same student can appear in several videos, the
validation split keeps all videos of one student on the same side (`GROUP_COLUMN = "STUDENT"`).

## Task 1 – GRS classification

The mean GRS of the three raters is mapped to four ordinal classes:

| Class | Name | GRS |
|-------|------|-----|
| 0 | beginner | < 16 |
| 1 | intermediate | 16 – 23 |
| 2 | proficient | 24 – 31 |
| 3 | expert | ≥ 32 |

**Metrics.** Macro-averaged F1 over the classes (higher is better) and expected cost with linear
weights |i − j| / (C − 1) (lower is better; also used as the tie-breaker). The training script
reports both after every epoch, and selects the best epoch by default with `rank_score = macro F1 − expected cost`.

## Task 2 – OSATS score prediction

The eight OSATS items are predicted on their 1–5 scale:

| Column | Item |
|--------|------|
| `OSATS_RESPECT` | respect for tissue |
| `OSATS_MOTION` | time and motion |
| `OSATS_INSTRUMENT` | instrument handling |
| `OSATS_SUTURE` | suture technique |
| `OSATS_FLOW` | flow of operation |
| `OSATS_KNOWLEDGE` | knowledge of the specific procedure |
| `OSATS_PERFORMANCE` | overall performance |
| `OSATS_FINAL_QUALITY` | quality of the final product |

The ground truth of an item is the rounded mean of the three raters.

**Metrics.** Macro F1 and expected cost (five ordinal classes) are computed per item and then
averaged over the eight items. The training script also reports the mean absolute error against
the mean rater score and the per-item breakdown.

## Task 3 – keypoint tracking

Six object classes are tracked with a fixed number of keypoints each:

| Class | Object | Keypoints |
|-------|--------|-----------|
| 0 | left hand | thumb, middle, index, ring, pinky, back of hand |
| 1 | right hand | same layout |
| 2 | scissors | left (sharp point), right (broad point), joint |
| 3 | tweezers | left (with nub), right (with hole), nub |
| 4 | needle holder | left, right, joint |
| 5 | needle | left (end), right (tip), middle — up to two needles per frame |

Visibility flags: `0` out of frame, `1` hidden (occluded), `2` visible.

**Data layout.**

```
train/frames/<video>_frame_<id>.png
train/masks/<video>_frame_<id>_<object>[<n>]_mask.png     binary mask per object instance
train/mot/<video>_frame_<id>.txt                            keypoint annotations
val/frames/<video>/<video>_frame_<id>.png
val/mot/<video>_frame_<id>.txt
```

Every line of an annotation (and of a prediction) file describes one object:

```
frame, track_id, class_id, -1, -1, -1, -1, x1, y1, v1, x2, y2, v2, ...
```

The four `-1` fields are unused bounding-box fields. Coordinates are in pixels of the original frame.

**Metric.** HOTA (Higher Order Tracking Accuracy), with the Euclidean distance between predicted
and annotated keypoints as similarity. During validation the training script computes a HOTA-style
score with that similarity, its detection and association parts (DetA, AssA), the PCK and the
visibility accuracy.

## Submission

Solutions are submitted as Docker images through Synapse. During testing the image is run as

```bash
docker run --gpus '"device=0"' --ipc=host \
    -v "<input folder>:/input" -v "<output folder>:/output" \
    <image> /usr/local/bin/Process_SkillEval.sh <task>
```

`<task>` is `GRS` (Task 1) or `OSATS` (Task 2). An image that solves a single task is called without
the argument. The script has to read the videos in `/input` and write the results to `/output`. The
organisers recommend `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` as the base image.

In this repository `docker/Process_SkillEval.sh` dispatches:

| Argument | Script | Result |
|----------|--------|--------|
| `GRS` | `inference_task1.py` | `/output/task1_results.csv` |
| `OSATS` | `inference_task2.py` | `/output/task2_results.csv` |
| `TRACK` (also `KEYPOINT`, `TRACKING`, `SEG`, `TASK3`) | `inference_task3.py` | `/output/<video>.txt`, `/output/mot/…` |
| none | `DEFAULT_TASK` (default `GRS`) | |

For the upload, create a Synapse project named `OSS Subchallenge <team name>`, push the image to it,
and give the challenge admin team (`Open Suturing Skills Challenge Admin`) download rights.

## Methods in this repository

| | Task 1 / Task 2 | Task 3 |
|-|-----------------|--------|
| input | clip of `NUM_FRAMES` frames sampled from a cached pool covering the whole video | single frame at `IMG_HEIGHT × IMG_WIDTH` |
| encoders | Kinetics-400 video networks (R(2+1)D, R3D, MC3, S3D, MViT, Swin3D) or ImageNet image networks per frame (ResNet, EfficientNet, RegNet, ConvNeXt, ViT, Swin, timm) with a temporal head (mean, max, attention, LSTM, GRU, transformer, TCN) | ResNet / ResNeXt / Wide-ResNet, ConvNeXt, EfficientNet, MobileNet-V3, RegNet-Y, timm |
| heads | Task 1: softmax, CORAL, GRS regression, hybrid; Task 2: regression, classification, CORAL, hybrid | FPN / U-Net / simple-baseline decoder; heatmap, visibility and segmentation heads |
| targets | hard, smoothed or rater-distribution labels | Gaussian heatmaps, visibility classes, masks, keypoint coordinates |
| losses | CE, class-balanced CE, CE + expected cost, focal, SORD, EMD, CORAL, Huber / MSE / L1, GRS consistency, suture count | focal / MSE / BCE / adaptive-wing heatmap loss, soft-argmax coordinate loss, weighted visibility CE, CE + Dice |
| optimisation | AdamW, Adam, NAdam, RAdam, Adamax, SGD, RMSprop, Adagrad; cosine, cosine restarts, linear, polynomial, step, multi-step, exponential, one-cycle, plateau | same |
| inference | multi-clip + flip TTA, checkpoint ensembles | peak NMS, sub-pixel refinement, needle assembly, Hungarian tracking, optional flip TTA |
| hyper-parameter search | GRM (grm.py): per-state reward estimator trained on early training dynamics + Q-learning search with a reward machine | same, probes scored with the validation HOTA |

## The GRM search and the challenge paper

The challenge paper describes the GRM entry only at the level of its two stages. It sets no further
constraints or rules for the search. That description is:

- an offline estimation stage, in which an estimator per batch-size class learns to predict final
  model performance from early training dynamics;
- an online search stage, in which a Q-learning agent explores batch-size / hyper-parameter pairs
  with an epsilon-greedy strategy;
- each selected configuration is trained for a limited number of epochs, its dynamics feature vector
  is extracted, and the estimator turns it into the reward that guides the agent.

The defaults of the training scripts follow this description: `GRM_STATE_KEY = "BATCH_SIZE"`,
`GRM_SIGNAL_PRESET = "oss"`, per-state estimators, epsilon-greedy Q-learning. The remaining parts of
the implementation (mapping function, labelling function, reward machine, estimator types, the
StepUP and OGB signal sets) follow the general GRM framework of the PhD thesis
(<https://cora.ucc.ie/items/6cf46796-173c-4426-99bd-676e11656485>). All of them are switchable in
the CONFIG blocks.
