# Data formats and coordinate contract

All examples describe inputs you prepare locally. No dataset or experiment output is included in the repository.

## Frames manifest

`frames.json` contains a single query's already-selected target masks:

```json
{
  "schema_version": 1,
  "coordinate_frame": "scene0000_00_raw",
  "depth_scale": 1000.0,
  "frames": [
    {
      "depth": "depth/00000.npy",
      "mask": "masks/00000.npy",
      "intrinsic": [[577.87, 0, 319.5], [0, 577.87, 239.5], [0, 0, 1]],
      "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    }
  ]
}
```

The numeric camera values above are illustrative, not universal ScanNet calibration. Supply the actual per-frame matrices. An identity pose is only correct when the world and that camera frame coincide.

- Paths are relative to the manifest directory and cannot escape it (including through symlinks).
- Depth is an H x W numeric `.npy` array. `depth_scale` means **raw units per metre**: 1000 for millimetres, 1 for metres. Nonfinite/nonpositive depth pixels are ignored.
- The mask is an equally sized binary 0/1 `.npy` array, after selecting a single query's target. It is not a 0/255 visualization or an integer instance-label map.
- Depth and mask must already be registered to the same camera/image grid. Intrinsics are for that grid. No implicit resizing or color-depth registration is performed.
- Camera coordinates use the optical convention x-right, y-down, z-forward. `camera_to_world` maps camera-space column vectors into the world frame. It must be a rigid transform, not world-to-camera.
- `--scene-points` selects existing world-space mesh vertices with nearest-pixel projection and an absolute depth-consistency tolerance (default 0.05 m). Specify `--depth-tolerance` explicitly if another tolerance is intended.
- Without `--scene-points`, masked depth pixels are back-projected into world-space points. Sample occupancy may differ from the scene mesh; voxel matching is sensitive to that difference. Prefer mesh projection when matching mesh-indexed SAI3D candidates.
- Output is `coarse.npy` (N x 3, metres) plus `coarse.json` recording frame, units, mode and a checksum. Neither is a benchmark result.

## SAI3D candidate import

The upstream export is a text **index**, not a directory guessed from an object ID:

```text
scene0000_00_pred_mask/scene0000_00_1.txt 1 1.000000
scene0000_00_pred_mask/scene0000_00_2.txt 1 1.000000
```

Each referenced file has exactly one 0/1 value per scene vertex. Pass the identical mesh, in the identical vertex order, used by SAI3D. A matching length alone cannot establish that a separately reordered mesh is correct.

`import-candidates` validates all masks before producing:

```text
candidates/
  candidates.json
  scene_points.npy
  masks/000000.npy
  masks/000001.npy
```

The `.npy` mask files contain vertex **indices**, not pickle objects. The adapter stores a checksum of the copied scene vertices and verifies it on load. The class field in the upstream text index is parsed only to validate the format; it is never used to select the queried instance. Empty candidates remain empty and cannot win a match.

## Refinement output

`selection.json` records:

- `status`: `matched`, `coarse_fallback`, or `empty_geometry`;
- `candidate_id`: an ID from the imported candidate index, or null;
- `matching_metric` / `matching_overlap`: geometric selection diagnostics, **not benchmark accuracy**;
- `aabb`: `min`, `max`, `center`, `size` in the declared world frame and metres, or null;
- input checksums and the voxel size when applicable.

For a selected point-level instance, use the selected ID's indices in `candidates.json` against `scene_points.npy`. A coarse fallback deliberately has no fabricated candidate ID. If using coarse points from another implementation, pass their correct `--coordinate-frame`; this is an explicit user assertion, not automatic coordinate registration. When a same-name JSON sidecar is present, its checksum, frame and units must match.

## Single-example VLM output

The inference JSON retains the raw response, actual prompt, model identifier, and a parsed prediction. Box coordinates are original-image pixels; temporal intervals are seconds relative to the supplied video. A cropped clip needs an explicit timestamp offset when mapping back to a full video. The CLI does not infer that offset.

The parser accepts one JSON object or a one-element object list for a box, optionally fenced or inside answer tags. It rejects malformed, nonfinite, inverted, multiple-target, and out-of-bounds boxes. It never uses `eval`, never defaults an invalid answer to a candidate, and never searches trailing reasoning for a replacement prediction.

All output paths must be new. Keep runtime artifacts under ignored `runs/` and data under ignored `data/`; do not commit private input or model-generated logs by accident.
