# Release scope and provenance

This release is built from the simpler Scene-R1 baseline and selectively integrates the extended archive's SAI3D interface. It includes **training, inference and refinement components**, not a complete experiment archive.

## Included, external, omitted

| Component | Status |
| --- | --- |
| Single-example temporal / image / QA inference | Included; requires a user-supplied compatible model |
| Strict shared parsing and coordinate conversion | Included; synthetic CPU tests |
| RGB-D mask lifting and scene-vertex selection | Included; explicit camera/frame contract |
| SAI3D generation command and export-format adapter | Included; uses an external SAI3D installation |
| GT-free candidate matching and AABB extraction | Included; voxel default, explicit AABB option |
| Temporal frame extraction and target-mask propagation | User integration; the paper's SAM2 pipeline is not included |
| Semantic-SAM / SAI3D model code and checkpoints | External, separately licensed |
| Temporal/image/QA GRPO training | Included as a baseline-derived adaptation; see [training guide](TRAINING.md) |
| Benchmark scripts, metrics, predictions, result tables | Not included |
| Datasets, label-generation notebook outputs, model weights | Not included |

Training is included with explicit data adapters, rewards and launchers. Benchmark evaluation is outside the release scope. The commands are not a one-command end-to-end experiment reproduction script.

## Changes from the baseline

- Retain the Qwen2.5-VL inference flow as a small lazy-loaded interface, without dataset evaluation loops or hardcoded model locations.
- Use task-specific prompts and one strict answer parser; preserve raw responses and explicit invalid-prediction status.
- Isolate metric geometry from training/evaluation. Require poses, scale and frame identity instead of guessing dataset-specific conventions.
- Integrate the extended archive's voxel-overlap selection idea, but import SAI3D's actual exported index, validate masks and implement deterministic non-overlap fallback.
- Do not retain the extended evaluator's default target-ID substitution, GT-dependent reporting, rectangle-as-target-mask construction, or skipped-case statistics.
- Exclude cache files, developer symlinks, API helper scripts, environment dumps, copied third-party reports, bundled annotation JSONs and notebook outputs.
- Keep SAI3D/Semantic-SAM external; no new claim of permission to redistribute their code or weights.

The implementation here is a cleaned release adaptation. Prompt, parser, and fallback changes can affect outputs, so it should not be represented as an unchanged archive of the scripts used to produce the paper tables.

## Input identities

Baseline GitHub archive comment: `22ed5fb1bed32d39c4ee866bed65d11a588db957`.

Source archive SHA-256 values:

```text
Scene-R1-main.zip
af33f1a6c91e3b1d154fa96808d04f1a3fe3044be505515e6f1858434fd6cb35

Scene-R1-open-source.tar.gz
bcc674740951b3a1d9e300980da16860517e8f0e44329301fb47bc4be4a02f51
```

## Runtime validation

Synthetic CPU tests cover parsing, projection, candidate import, deterministic matching, training data schemas and rewards. Optional training smoke tests use tiny randomly initialized Qwen2.5-VL models. Full-scale GPU training, real-checkpoint GPU inference and real-scene SAI3D execution have not been validated as part of assembling this release. Use the environments and data preparation documented by the respective upstream projects. Models, datasets, SAI3D, Semantic-SAM and SAM2 are external and remain subject to their own terms.
