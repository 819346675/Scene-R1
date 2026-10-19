"""GT-free adapter for the actual SAI3D exported-mask index format."""
import math
from pathlib import Path
import re
import subprocess

import numpy as np

from .io import contained_path, digest, read_json, read_points, write_array, write_json

UPSTREAM = "https://github.com/yd-yin/SAI3D"
REFERENCE_COMMIT = "1d9a6a2393c5005276c28cb9eb95391d799d1279"


def import_candidates(index_path, scene_points_path, output_dir, coordinate_frame):
    """Read mask-path / class / confidence rows, NOT a guessed scene subfolder."""
    if not coordinate_frame or not coordinate_frame.strip():
        raise ValueError("A coordinate-frame identifier is required")
    index = Path(index_path).resolve()
    points = read_points(scene_points_path)
    entries, ids = [], set()
    for line_no, line in enumerate(index.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"Invalid SAI3D index row {line_no}: expected path class confidence")
        mask_file, label, confidence = fields
        int(label)  # Validate format only; this field is NEVER used to select a target.
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence at row {line_no}")
        mask_path = contained_path(index.parent, mask_file)
        candidate_id = Path(mask_file).stem
        if candidate_id in ids:
            raise ValueError(f"Duplicate candidate ID: {candidate_id}")
        ids.add(candidate_id)
        mask = np.loadtxt(mask_path, ndmin=1)
        if mask.ndim != 1 or len(mask) != len(points) or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Mask must be binary with exactly {len(points)} rows: {mask_file}")
        entries.append((candidate_id, np.flatnonzero(mask == 1), confidence))
    # Validate all inputs before creating a new bundle; never overwrite a prior run.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_array(output / "scene_points.npy", points)
    candidates = []
    for n, (candidate_id, indices, confidence) in enumerate(entries):
        name = f"masks/{n:06d}.npy"
        write_array(output / name, indices)
        candidates.append({"id": candidate_id, "indices": name, "confidence": confidence})
    manifest = {"schema_version": 1, "coordinate_frame": coordinate_frame, "units": "metres",
                "scene_points": "scene_points.npy", "scene_points_sha256": digest(output / "scene_points.npy"),
                "source_index_sha256": digest(index), "candidates": candidates}
    write_json(output / "candidates.json", manifest)
    return output / "candidates.json"


def load_candidates(manifest_path, coordinate_frame):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("units") != "metres":
        raise ValueError("Unsupported candidate schema or units")
    if not coordinate_frame or manifest.get("coordinate_frame") != coordinate_frame:
        raise ValueError("Coarse and candidate coordinate-frame identifiers must match")
    points_path = contained_path(manifest_path.parent, manifest["scene_points"])
    if digest(points_path) != manifest["scene_points_sha256"]:
        raise ValueError("Scene vertex file changed after candidate import")
    points = read_points(points_path)
    candidates, seen = [], set()
    for entry in manifest["candidates"]:
        candidate_id = entry["id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError("Candidate IDs must be nonempty unique strings")
        seen.add(candidate_id)
        indices = np.load(contained_path(manifest_path.parent, entry["indices"]), allow_pickle=False)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or (len(indices) and (indices.min() < 0 or indices.max() >= len(points))):
            raise ValueError("Invalid candidate vertex indices")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Duplicate vertex indices in candidate")
        candidates.append((candidate_id, points[indices]))
    return candidates


def generation_plan(sai3d_root, data_root, scene_id, output_dir, python="python",
                    view_freq=5, use_torch=False):
    """Build argv for upstream primitive merging; no shell strings or evaluation."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", scene_id):
        raise ValueError("Use one safe scene identifier, not a comma-separated batch")
    if view_freq <= 0:
        raise ValueError("view_freq must be positive")
    root = Path(sai3d_root).resolve()
    data = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    if not (root / "sai3d.py").is_file():
        raise ValueError("sai3d_root must contain the upstream sai3d.py")
    if output.exists():
        raise FileExistsError("Use a new output directory to avoid replacing candidate masks")
    command = [python, str(root / "sai3d.py"), "--base_dir", str(data),
               "--scene_id", scene_id, "--mask_name", "semantic-sam", "--view_freq", str(view_freq),
               "--thres_connect", "0.9,0.5,5", "--max_neighbor_distance", "2",
               "--dis_decay", "0.5", "--thres_merge", "200", "--similar_metric", "2-norm",
               "--eval_dir", str(output)]
    if use_torch:
        command.append("--use_torch")
    return {"cwd": str(root), "argv": command, "source_sha256": digest(root / "sai3d.py"),
            "reference_commit": REFERENCE_COMMIT, "output_dir": str(output),
            "note": "Upstream calls its export argument eval_dir; this command only generates masks."}


def execute_generation(plan):
    command = plan["argv"]
    if digest(command[1]) != plan["source_sha256"]:
        raise ValueError("Upstream source changed after planning")
    data = Path(command[command.index("--base_dir") + 1])
    scene_id = command[command.index("--scene_id") + 1]
    for relative in [f"posed_images/{scene_id}", f"scans/{scene_id}",
                     f"2D_masks/{scene_id}/semantic-sam", "Tasks/Benchmark/scannetv2_train.txt",
                     "Tasks/Benchmark/scannetv2_val.txt"]:
        if not (data / relative).exists():
            raise FileNotFoundError(f"Missing upstream input: {relative}")
    if not list((data / "2D_masks" / scene_id / "semantic-sam").glob("maskraw_*.png")):
        raise ValueError("No pre-generated Semantic-SAM masks found")
    if Path(plan["output_dir"]).exists():
        raise FileExistsError("Candidate output appeared after planning; refusing to overwrite")
    subprocess.run(command, cwd=plan["cwd"], check=True)
    index = Path(plan["output_dir"]) / f"{scene_id}.txt"
    if not index.is_file():
        raise RuntimeError("Upstream exited without producing the expected candidate index")
    write_json(Path(plan["output_dir"]) / "generation_manifest.json", plan)
