# GRPO training

This release restores the **temporal grounding, image grounding and QA training flows** from the supplied `Scene-R1-main.zip`. Their common image/video trainer has been consolidated into `scene_r1/training/trainer.py`; the original Hugging Face / Time-R1 attribution is retained. It is a maintained release adaptation, not an untouched historical experiment snapshot.

## Install

Use a separate Python 3.10-3.12 environment, preferably Python 3.11 on Linux/CUDA:

```bash
python -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -e ".[train]"
python -m scene_r1.training --help
python -m unittest discover -s tests -v
python -m unittest discover -s training_tests -v
```

The training extra pins the core Torch/Transformers/TRL/Accelerate versions and PyAV 14.2.0 (newer PyAV versions can break this TorchVision video reader). It does not require FlashAttention, vLLM, W&B, an API key or a remote logging service. The model and processor must already be available in a **local Qwen2.5-VL-compatible checkpoint directory**. Checkpoints saved by this training entrypoint can be passed to `scene-r1 infer --model ...`.

Full-parameter 7B GRPO holds an initial frozen reference model as well as the policy and optimizer states. A single-GPU run may exceed memory. For multi-GPU sharding, optionally install `deepspeed==0.16.4` in a compatible Linux/CUDA environment and use `--deepspeed configs/zero3.json`. DDP works without DeepSpeed but replicates these states. The sharded GPU path has not been exercised by the small CPU tests.

## Prepare training annotations

Only pass the dataset's **training split**. The converter does not discover official train/test splits and cannot infer split membership from a filename. Do not relabel validation/test annotations as training data.

The original dataset layouts are supported:

| Task | Original annotation layout |
| --- | --- |
| Temporal | Scene-keyed JSON; parallel `sentences` and `timestamps` lists, plus `duration`. Each `timestamps` entry must be a list of relevant **frame indices**, not a pair of seconds. |
| Image | Scene-keyed JSON; parallel `sentences`, `image_path`, `bbox_2d` lists. Boxes are xyxy pixels in the **original image**. |
| QA | List of records containing `scene_id`, `situation`, `question`, `answers`. All answer references are retained. |

Convert the original files to explicit JSONL manifests:

```bash
# Set frame-index-base to the actual origin of YOUR annotations: 0 or 1.
# --fps describes the annotation frame grid and requested visual sampling rate.
python -m scene_r1.training.data --task temporal \
  --legacy-json data/scanrefer/train.json --media-root data/scannet/videos \
  --fps 2 --frame-index-base 0 --output data/scanrefer/train-temporal.jsonl

python -m scene_r1.training.data --task box \
  --legacy-json data/scanrefer/img_g_train.json --media-root data/scannet/images \
  --output data/scanrefer/train-box.jsonl

python -m scene_r1.training.data --task qa \
  --legacy-json data/sqa3d/SQA_train.json --media-root data/scannet/videos \
  --fps 2 --output data/sqa3d/train-qa.jsonl
```

The media paths above are examples, not bundled data. Video filenames are matched to scene IDs with `.mp4`, `.mkv` or `.webm`. Relative original image paths resolve against `--media-root`; existing absolute paths remain absolute. Missing media, ambiguous video matches or unequal annotation-list lengths produce an error instead of dropping training examples. Existing output manifests are not overwritten.

Alternatively, prepare one JSON object per line directly:

```json
{"id":"scene0:0","split":"train","task":"temporal","query":"the chair","video":"videos/scene0.mp4","fps":2,"duration":5,"frame_index_base":0,"gt_frames":[2,3,4]}
```

```json
{"id":"scene0:0","split":"train","task":"box","query":"the chair","image":"images/frame.png","bbox_2d":[20,10,100,50]}
```

```json
{"id":"scene0:0","split":"train","task":"qa","query":"Facing the door, what is on the left?","video":"videos/scene0.mp4","fps":2,"answers":["chair","seat"]}
```

Each manifest must contain one task only and unique IDs. Relative media paths resolve against the manifest directory. `duration` is the actual duration in seconds; temporal labels must use the declared fps grid. Grounding-label generation from raw dataset annotations is dataset-specific and is not replaced by invented boxes or intervals here.

### Video preprocessing

The trainer decodes local video on demand with `qwen_vl_utils.process_vision_info`, using the declared fps and configured pixel budget. It forwards the processor's **actual sampled timing (`second_per_grid_ts`)** to both the policy and reference likelihood passes. Thus the old `.pt` cache preprocessing is not required. There is no silent fallback to a different clip, black frame, or different target.

Prepare correctly ordered scene videos before training. For ScanNet, the paper uses 640x480 frames; for ARKitScenes it retains native resolution before smart-resize. The default total video budget is `16384 * 28 * 28` pixels. A prompt over `max_prompt_length` is rejected; it is not truncated independently of visual tokens. Adjust the budget explicitly if needed. Temporal GT coordinates remain on the declared annotation timeline, not indices of whatever subset the decoder sampled.

## Validate before training

```bash
python -m scene_r1.training.data --task temporal --data data/scanrefer/train-temporal.jsonl
python -m scene_r1.training --task temporal \
  --train-data data/scanrefer/train-temporal.jsonl \
  --model checkpoints/Qwen2.5-VL-7B-Instruct --output runs/tg \
  --config configs/training.json --dry-run
```

Dry-run validates annotations, media existence and settings and prints the query-batch calculation. It does not decode videos, load model weights or establish GPU memory sufficiency.

## Train the three tasks

The launcher names follow the original repository. Run from the repository root after installation:

```bash
# Stage 1: temporal grounding
NPROC_PER_NODE=4 bash scripts/run_scanrefer.sh \
  --train-data data/scanrefer/train-temporal.jsonl \
  --model checkpoints/Qwen2.5-VL-7B-Instruct --output runs/tg \
  --config configs/training.json --deepspeed configs/zero3.json

# Stage 2: image grounding
NPROC_PER_NODE=4 bash scripts/run_scanrefer_image.sh \
  --train-data data/scanrefer/train-box.jsonl \
  --model checkpoints/Qwen2.5-VL-7B-Instruct --output runs/ig \
  --config configs/training.json --deepspeed configs/zero3.json

# Video QA
NPROC_PER_NODE=4 bash scripts/run_sqa3d.sh \
  --train-data data/sqa3d/train-qa.jsonl \
  --model checkpoints/Qwen2.5-VL-7B-Instruct --output runs/qa \
  --config configs/training.json --deepspeed configs/zero3.json
```

The two grounding stages are task-specific training runs initialized from the supplied model; the launcher does not silently substitute a temporal checkpoint as the image model. The image data should contain the appropriate target frames and their 2D annotations. Affordance grounding can use the same temporal/image loaders with the corresponding task descriptions and training annotations.

Remove `--deepspeed` only when DDP memory requirements fit your hardware. Add `--max-steps 1` for a short real-data smoke run before committing to an epoch. Resume only your own trusted checkpoint, for example `--resume runs/tg/checkpoint-500`; optimizer/RNG resume files use the framework's serialization. No checkpoint path is hardcoded or auto-discovered.

The published recipe uses AdamW, learning rate 1e-5, betas (0.9, 0.95), weight decay 0.02, cosine decay without warmup and one epoch. One query/device/microbatch x four devices x eight accumulation steps gives **32 queries per optimizer update**; G=8 gives 256 sampled responses, not 32. Change accumulation when changing the device count. The KL coefficient 0.04 and generation settings are explicit release defaults rather than independently recovered experiment metadata.

## Rewards and source changes

| Task | Release training reward |
| --- | --- |
| Temporal | Frame-set IoU + 0.1 x valid think/answer format |
| Image | Pixel box IoU + 0.1 x valid JSON schema + 0.1 x valid think/answer format |
| QA | Normalized EM + LEO-style refined EM + 0.1 x valid think/answer format |

These rewards follow the revised manuscript, and are **training rewards, not benchmark reporting code**. LEO-style refined EM here is the original whitespace-insensitive substring rule, not a semantic similarity model; its normalization functions and MIT notice are retained. Format rewards do not establish the correctness of a reasoning trace.

Material differences from the original archive are explicit:

- The original video trainer had a global QA prompt even when called by temporal training; task-specific prompts now match the requested task.
- Original entrypoints summed unweighted format rewards; image training lacked the separate JSON term, and QA computed refined EM but used EM alone. The release implements the manuscript equations above.
- Original temporal code used an exclusive end index and a fixed 2 fps. The release uses the manuscript's inclusive frame interval, an explicit index origin and fps, and bounds the last frame to the clip. Invalid/out-of-duration predictions receive zero IoU.
- The image reward uses the actual processor grid/patch size to map predictions to original-image pixels. The original hardcoded patch-size assumption is removed.
- The image/video trainer common core keeps **single-update on-policy GRPO** with group-normalized advantages and the frozen-reference KL estimator. It does not add unverified multiple rollout-reuse iterations. Constant-reward groups have zero advantage.
- Removed the ineffective Dataset `__getitem__` monkey patch, mandatory local tensor caches, vLLM imports, hardcoded CUDA IDs, local model paths and forced resume path.
- The original two-epoch/DeepSpeed launcher values differ from the manuscript. The release uses one epoch and explicit settings; optional DeepSpeed is an execution alternative, **not a claim to reproduce the paper's reported FSDP run or GPU-hours**.

## Validation and remaining external components

`tests/test_training.py` checks conversion, schemas, reward values, resizing, frame boundaries, task prompts and dry-run behavior without loading a model. `training_tests/test_trainer.py` exercises tiny random Qwen2.5-VL image/video forward-backward paths and a synthetic optimizer/checkpoint step without downloading weights. The synthetic rollout text is deliberately controlled to test gradients; it is not evidence of answer quality or a completed 7B run.

Datasets, label-generation outputs and trained weights are not bundled. SAM2 propagation and upstream SAI3D remain external. No benchmark result files or evaluators were added. A new training run can produce usable checkpoints, but publishing these scripts alone does not establish reproduction of the paper's scores.
