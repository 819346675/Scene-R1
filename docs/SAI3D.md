# SAI3D integration

## Why an external dependency?

The extended source archive contains a vendored SAI3D/Semantic-SAM tree and several environment-specific evaluation scripts. This distribution integrates the **candidate-generation and GT-free matching interface**, not those evaluation scripts or an unverified redistribution of all third-party source.

SAI3D and Semantic-SAM are intentionally external. A corresponding license file was not available in the supplied vendored trees, so we do not assume the repository's Apache license can cover them. Obtain permission/terms from the upstream maintainers as appropriate. Do not infer that public GitHub access alone grants redistribution rights.

## Prepare the external environment

Upstream: <https://github.com/yd-yin/SAI3D>

The CLI/export interface was inspected at this reference revision:

```text
1d9a6a2393c5005276c28cb9eb95391d799d1279
```

If permitted by the upstream terms, obtain that revision in `third_party/SAI3D` and follow **its README** for Semantic-SAM, checkpoint installation, ScanNet data preparation, and superpoint generation. Keep the SAI3D environment separate from the current Qwen/Transformers environment; their CUDA/PyTorch requirements can differ. No automatic installer runs third-party code in this repository.

The adapter expects upstream layout:

```text
data/ScanNet/
  posed_images/<scene>/       # RGB, aligned depth, intrinsics and poses
  scans/<scene>/              # original mesh and required geometric primitives
  2D_masks/<scene>/semantic-sam/maskraw_*.png
  Tasks/Benchmark/scannetv2_train.txt
  Tasks/Benchmark/scannetv2_val.txt
```

These split-list files are required because the inspected upstream CLI loads them before applying `--scene_id`; this adapter does not evaluate a split or synthesize lists. Dataset preparation and any restrictions remain the user's responsibility.

For ScanNet, the upstream primitive file is `<scene>_vh_clean_2.0.010000.segs.json`, a geometric oversegmentation. The adapter does not load `<scene>.aggregation.json` or use GT object IDs. If the external checkout already has `points.pts`, ensure it was generated from the same mesh and has not been reordered. Upstream may create cache files such as `points.pts` and a `results/` directory under the supplied data root; use a prepared working copy, not a read-only canonical dataset.

Generate per-frame Semantic-SAM labels following upstream instructions. At the reference revision its script accepts **`--data_dir`**, unlike the modified full archive's `--base_dir` argument. Do not mix those CLIs. From the upstream Semantic-SAM environment/directory, the documented shape is:

```bash
# Run in your external SAI3D/Semantic-SAM environment after installing its checkpoint.
python ../sam_auto_generation.py --data_dir /path/to/ScanNet --view_freq 5
```

Inspect the upstream script's split selection before running it: this generates the scenes in its configured split, not a single query. Target-specific SAM2 masks for Scene-R1 lifting are a separate input; Semantic-SAM's scene-wide candidate masks are not a substitute for that temporal propagation step.

## Candidate generation

`python -m scene_r1 sai3d ...` prints a structured plan with the executable, argument list, source-file checksum and output directory. It does nothing until `--execute` is supplied. Use `--python` to select the external environment's Python.

The wrapper calls `sai3d.py` with a single explicit scene, a fresh output directory, and the full archive's merging configuration: view frequency 5, connection thresholds `0.9,0.5,5`, neighbor distance 2, distance decay 0.5, small-region threshold 200, and `2-norm` similarity. These are disclosed integration defaults, not a claim about the unpublished experiment configuration. The original flag `--eval_dir` is the upstream **mask-export path**; no evaluator is invoked.

The exported `<scene>.txt` index lists the relative mask paths. The wrapper does not assume a `<scene>/` mask directory: the inspected exporter actually writes `<scene>_pred_mask/`. `import-candidates` reads the index and validates row counts, mask values, unique IDs and path containment. It does not load ScanNet aggregation labels or ground-truth boxes.

## Matching protocol

The extended archive contains different post-processing branches:

| Branch | Selection rule | Public integration |
| --- | --- | --- |
| `stage3_evaluate_141scenes_fixed.py` | Voxel overlap of fused coarse points and candidate instance points | Default `--metric voxel`, geometry-only, with validation and explicit fallback |
| `match_and_evaluate_3d.py` | Project candidates to 2D and match image boxes | Not combined with the default; omitted to avoid silently changing protocols |
| Optional alternative matching mode | Match candidate geometry at box level | Explicit `--metric aabb`; an adapter option, not the default voxel matching described in the revised manuscript |

Voxel mode uses occupied voxel sets with default side length **0.01 m** in the same world coordinate frame. Repeated points do not increase overlap. AABB mode computes box intersection-over-union for selection. This selection score must **not** be reported as a ScanRefer box benchmark score or a SceneFun3D segmentation metric.

The matcher reads only query-derived coarse geometry and independently generated candidate geometry. It does not accept a ground-truth target ID. A strict positive-overlap requirement avoids choosing an arbitrary first candidate on a zero-overlap tie; equal positive scores are resolved by sorted candidate ID. Empty coarse geometry is explicit, and absent/empty/nonoverlapping candidates retain the coarse AABB rather than silently dropping the query.

## Validation boundary

Synthetic CPU tests cover mask import, projection, coordinate contracts, deterministic matching and fallback behavior. No real SAI3D checkpoint, CUDA scene run, or full paper reproduction has been executed as part of assembling this release. Compare the upstream interface and dependencies if using a different revision. The runner records the actual source checksum but does not claim to certify the external checkout.
