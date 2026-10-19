<div align="center">

# Scene-R1
### Video-Grounded Large Language Models for 3D Scene Reasoning with Lightweight Annotations

[![arXiv](https://img.shields.io/badge/arXiv-2506.17545-b31b1b.svg)](https://arxiv.org/abs/2506.17545)
[![License](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

Junjie Wang*, Zhihao Yuan*, Shuyi Jiang, Chun-Mei Feng, Yaolun Zhang, Shuguang Cui, Zhen Li, Na Zhao

*Equal contribution. Author details follow the revised manuscript.

**From video observations to query-relevant 3D geometry.**

[Quick start](#quick-start) · [GRPO training](docs/TRAINING.md) · [SAI3D integration](docs/SAI3D.md) · [Data formats](docs/DATA_FORMATS.md) · [Release scope](docs/RELEASE_SCOPE.md)

</div>

## Overview

Scene-R1 studies video-grounded spatial reasoning through temporal grounding, image grounding, and geometric refinement. The paper describes a reinforcement-learning framework that exposes intermediate reasoning and localizes queried objects from RGB-D observations.

This repository provides **GRPO training, inference and 3D refinement code**: Qwen2.5-VL temporal/image grounding and question answering, RGB-D target-mask lifting, and SAI3D candidate generation, import, and matching. Please see the [paper](https://arxiv.org/abs/2506.17545) for experimental results.

```text
Video + query ──► temporal interval ──► selected RGB frame ──► 2D target box
                                                                  │
                                          external target masks (e.g. SAM2)
                                                                  │
Aligned depth + camera poses ───────────────────────────────► coarse 3D points
                                                                  │
RGB-D scene ──► external SAI3D candidate generation ─────────► candidate matching
                                                                  │
                                                 selected instance + tight AABB
```

The commands expose individual components. Supply the selected frames, propagated target masks (e.g. from SAM2), and compatible checkpoints externally. Training data and model weights are not bundled. The [training guide](docs/TRAINING.md) explains the three restored training flows, original annotation adapters, launch commands, and changes from the baseline.

## What is included

- **GRPO training:** baseline-derived temporal, image and QA training flows, explicit paper-aligned rewards, data validation, launchers and checkpoint saving/resume. See [training](docs/TRAINING.md) for setup and validation scope.
- **Single-example inference:** temporal grounding, 2D box prediction, and video/image question answering with a user-supplied Qwen2.5-VL-compatible checkpoint.
- **Metric geometry:** fuse target masks with aligned depth and camera-to-world poses; optionally select the original scene mesh vertices for consistent candidate matching.
- **SAI3D integration:** generate instance candidates through an external SAI3D checkout, import its real index/mask format, and match without reading any GT instance IDs or labels.
- **Reusable geometry utilities:** deterministic candidate matching, input validation, and coarse-geometry fallback when no candidate overlaps.

The default refinement uses **voxel overlap** between coarse target points and candidate points, following the matching branch of the supplied extended implementation. An explicitly selected `--metric aabb` mode matches their bounding boxes instead. These are different geometric matching rules, not interchangeable benchmark metrics. See [the integration notes](docs/SAI3D.md#matching-protocol).

## Quick start

For training, install `python -m pip install -e ".[train]"` in a separate environment and follow the [complete training guide](docs/TRAINING.md). The lightweight steps below remain sufficient for geometry and parsing.

### 1. Install the lightweight components

```bash
git clone https://github.com/819346675/Scene-R1.git
cd Scene-R1
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m scene_r1 --help
python -m unittest discover -s tests -v
```

Geometry utilities use NumPy; PLY reading is optional:

```bash
python -m pip install -e ".[ply]"
```

Tests use synthetic inputs and require no downloaded dataset or checkpoint. GPU model inference and upstream SAI3D execution require their own dependencies and have not been validated by the CPU unit tests.

### 2. Run a single inference example

Install the optional VLM dependencies in a suitable PyTorch/CUDA environment:

```bash
python -m pip install -e ".[vlm]"
python -m scene_r1 infer \
  --task temporal --video data/example/video.mp4 \
  --query "the cabinet next to the refrigerator" \
  --model checkpoints/temporal-grounding --duration 35.5 \
  --output runs/example/temporal.json

python -m scene_r1 infer \
  --task box --image data/example/selected_frame.jpg \
  --query "the cabinet next to the refrigerator" \
  --model checkpoints/image-grounding \
  --output runs/example/box.json
```

Paths above are examples: supply your own media and compatible checkpoint directories. For a base-model demonstration, use `--model Qwen/Qwen2.5-VL-7B-Instruct --allow-download`. A base model is **not** the paper's trained checkpoint. Downloads are disabled unless explicitly enabled; weights must be in safetensors format. No API key, telemetry, or W&B configuration is required by our commands.

Use `--task qa --video ...` or `--task qa --image ...` for a short final answer. Temporal outputs use seconds; 2D boxes are mapped back to the original image's pixel coordinates. Invalid answers produce `status: invalid_prediction` and a nonzero exit code, not a made-up box or a reused prior prediction.

### 3. Lift target masks into 3D

Prepare binary **target** masks using your mask-propagation implementation (for example, upstream SAM2). Supply aligned depth, intrinsics, and poses in a [frames manifest](docs/DATA_FORMATS.md#frames-manifest):

```bash
python -m scene_r1 lift \
  --frames data/example/frames.json \
  --scene-points data/example/scene_vh_clean_2.ply \
  --output runs/example/coarse.npy
```

With `--scene-points`, the command selects visible scene vertices inside the masks and checks depth consistency. Without it, it back-projects masked depth pixels. All points and SAI3D candidates must be in the **same world frame, in metres**. Do not mix ScanNet aligned and unaligned coordinates. The software does not infer that transform for you.

### 4. Generate, import, and match SAI3D candidates

Install SAI3D and its dependencies separately; prepare its Semantic-SAM masks and geometric primitives as described in [docs/SAI3D.md](docs/SAI3D.md). Plan a candidate-generation command first:

```bash
python -m scene_r1 sai3d \
  --sai3d-root third_party/SAI3D --data-root data/ScanNet \
  --scene-id scene0000_00 --output-dir runs/sai3d-candidates \
  --python /path/to/sai3d-environment/bin/python --use-torch
```

Add `--execute` to run that plan. Candidate generation is a segmentation step, **not an evaluation**. Then:

```bash
python -m scene_r1 import-candidates \
  --index runs/sai3d-candidates/scene0000_00.txt \
  --scene-points data/example/scene_vh_clean_2.ply \
  --coordinate-frame scene0000_00_raw \
  --output-dir runs/example/candidates

python -m scene_r1 refine \
  --coarse runs/example/coarse.npy \
  --candidates runs/example/candidates/candidates.json \
  --coordinate-frame scene0000_00_raw \
  --metric voxel --voxel-size 0.01 \
  --output runs/example/selection.json
```

Use the exact mesh vertex order used by SAI3D when importing binary masks. The frames manifest must use the same `coordinate_frame` identifier. Output files are not overwritten automatically. `selection.json` records the selected candidate, matching rule, fallback status, and tight AABB; it is an inference artifact, not an accuracy report.

## Repository layout

```text
scene_r1/
  inference.py     # single-example VLM interface
  parsing.py       # shared, strict answer parsing
  geometry.py      # projection, lifting, and matching
  sai3d.py         # upstream generation + mask import adapter
  io.py           # explicit non-pickle formats
  __main__.py      # command-line interface
docs/
  SAI3D.md         # installation boundary and matching protocol
  DATA_FORMATS.md  # coordinates, manifests, and outputs
  RELEASE_SCOPE.md # included / external / omitted components
tests/            # synthetic CPU unit tests, not benchmarks
```

## Paper and acknowledgements

Please refer to [the Scene-R1 arXiv record](https://arxiv.org/abs/2506.17545) for the paper, current author metadata, and citation. The repository title follows the revised manuscript; the arXiv metadata may change when that revision is uploaded.

This release builds on the Scene-R1 baseline and acknowledges [Time-R1 / TimeZero](https://github.com/xiaomi-research/Time-R1), Hugging Face Transformers, and Qwen2.5-VL. The refinement interface is designed for [SAI3D](https://github.com/yd-yin/SAI3D); target-mask propagation can be supplied by [SAM2](https://github.com/facebookresearch/sam2). See [NOTICE](NOTICE) for provenance.

## License and external assets

The code in this distribution retains the baseline's [Apache-2.0 license](LICENSE.txt). External models, datasets, SAI3D, Semantic-SAM, and SAM2 are **not bundled** and remain subject to their respective terms. The repository license does not grant rights to those assets. Obtain them from their maintainers and review their licenses before use or redistribution.
