import argparse
import json
import math
from pathlib import Path
import sys
import subprocess

import numpy as np

from .geometry import choose_candidate, lift_depth_mask, select_scene_points
from .io import contained_path, digest, read_json, read_points, write_array, write_json
from .sai3d import execute_generation, generation_plan, import_candidates, load_candidates


def lift_frames(args):
    manifest_path = Path(args.frames).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1 or not manifest.get("coordinate_frame"):
        raise ValueError("Frames require schema_version=1 and coordinate_frame")
    scale = float(manifest["depth_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale must be positive and finite")
    if not manifest.get("frames"):
        raise ValueError("No frames supplied")
    points = read_points(args.scene_points) if args.scene_points else None
    chunks = []
    for frame in manifest["frames"]:
        depth = np.load(contained_path(manifest_path.parent, frame["depth"]), allow_pickle=False)
        mask = np.load(contained_path(manifest_path.parent, frame["mask"]), allow_pickle=False)
        kwargs = dict(depth=depth, mask=mask, intrinsic=frame["intrinsic"],
                      camera_to_world=frame["camera_to_world"], depth_scale=scale)
        chunk = lift_depth_mask(**kwargs) if points is None else select_scene_points(
            points, **kwargs, depth_tolerance=args.depth_tolerance)
        chunks.append(chunk)
    fused = np.unique(np.concatenate(chunks, axis=0), axis=0)
    output = Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Choose new output paths")
    write_array(output, fused)
    write_json(output.with_suffix(".json"), {"schema_version": 1,
        "coordinate_frame": manifest["coordinate_frame"], "units": "metres",
        "points_sha256": digest(output), "num_points": len(fused),
        "mode": "mesh_projection" if points is not None else "depth_backprojection",
        "frames_manifest_sha256": digest(manifest_path)})


def parser():
    ap = argparse.ArgumentParser(description="Scene-R1 inference components and SAI3D refinement (no benchmark evaluation)")
    commands = ap.add_subparsers(dest="command", required=True)
    p = commands.add_parser("infer", help="Single image/video inference using your checkpoint")
    p.add_argument("--task", choices=["temporal", "box", "qa"], required=True)
    media = p.add_mutually_exclusive_group(required=True)
    media.add_argument("--image")
    media.add_argument("--video")
    p.add_argument("--query", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow-download", action="store_true", help="Explicitly allow Hugging Face model downloads")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--total-pixels", type=int, default=3584 * 28 * 28)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--duration", type=float, help="Optional video duration to validate the temporal interval")
    p.add_argument("--output", required=True)
    p = commands.add_parser("lift", help="Fuse aligned depth + supplied target masks in a common world frame")
    p.add_argument("--frames", required=True)
    p.add_argument("--scene-points", help="Optional original ScanNet mesh .ply/.npy, for vertex-aligned projection")
    p.add_argument("--depth-tolerance", type=float, default=0.05)
    p.add_argument("--output", required=True)
    p = commands.add_parser("sai3d", help="Plan upstream candidate generation; add --execute to run it")
    p.add_argument("--sai3d-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--scene-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--python", default=sys.executable, help="Python executable in your separate SAI3D environment")
    p.add_argument("--view-freq", type=int, default=5)
    p.add_argument("--use-torch", action="store_true")
    p.add_argument("--execute", action="store_true")
    p = commands.add_parser("import-candidates", help="Import SAI3D index and vertex-aligned binary masks")
    p.add_argument("--index", required=True)
    p.add_argument("--scene-points", required=True)
    p.add_argument("--coordinate-frame", required=True)
    p.add_argument("--output-dir", required=True)
    p = commands.add_parser("refine", help="Select a candidate without any GT input")
    p.add_argument("--coarse", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--coordinate-frame", required=True)
    p.add_argument("--metric", choices=["voxel", "aabb"], default="voxel")
    p.add_argument("--voxel-size", type=float, default=0.01)
    p.add_argument("--min-overlap", type=float, default=0.0)
    p.add_argument("--output", required=True)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "infer":
            if not math.isfinite(args.fps) or args.fps <= 0 or args.total_pixels <= 0 or args.max_new_tokens <= 0:
                raise ValueError("FPS and generation budgets must be positive")
            if Path(args.output).exists():
                raise FileExistsError("Choose a new output file")
            from .inference import infer
            result = infer(args)
            write_json(args.output, result)
            return 0 if result["status"] == "ok" else 2
        if args.command == "lift":
            if Path(args.output).suffix != ".npy":
                raise ValueError("lift output must end in .npy")
            lift_frames(args)
        elif args.command == "sai3d":
            plan = generation_plan(args.sai3d_root, args.data_root, args.scene_id, args.output_dir,
                                   args.python, args.view_freq, args.use_torch)
            print(json.dumps(plan, indent=2))
            if args.execute:
                execute_generation(plan)
        elif args.command == "import-candidates":
            print(import_candidates(args.index, args.scene_points, args.output_dir, args.coordinate_frame))
        elif args.command == "refine":
            candidates = load_candidates(args.candidates, args.coordinate_frame)
            coarse = read_points(args.coarse)
            sidecar = Path(args.coarse).with_suffix(".json")
            if sidecar.exists():
                info = read_json(sidecar)
                if info.get("coordinate_frame") != args.coordinate_frame or info.get("units") != "metres" or info.get("points_sha256") != digest(args.coarse):
                    raise ValueError("Coarse-point sidecar does not match its data or coordinate frame")
            result = choose_candidate(coarse, candidates, args.metric, args.voxel_size, args.min_overlap)
            result.update(schema_version=1, coordinate_frame=args.coordinate_frame, units="metres",
                          coarse_sha256=digest(args.coarse), candidates_sha256=digest(args.candidates))
            write_json(args.output, result)
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"scene-r1: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
