"""Metric geometry only: no ground-truth labels, target IDs, or benchmark scoring."""
import math

import numpy as np


def points_array(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite N x 3 points in metres")
    return points


def aabb(points):
    points = points_array(points)
    if len(points) == 0:
        return None
    low, high = points.min(axis=0), points.max(axis=0)
    return {"min": low.tolist(), "max": high.tolist(),
            "center": ((low + high) / 2).tolist(), "size": (high - low).tolist()}


def aabb_overlap(a, b):
    """Geometric candidate matching score, not a benchmark result."""
    if a is None or b is None:
        return 0.0
    amin, amax = np.asarray(a["min"]), np.asarray(a["max"])
    bmin, bmax = np.asarray(b["min"]), np.asarray(b["max"])
    intersection = float(np.prod(np.maximum(0, np.minimum(amax, bmax) - np.maximum(amin, bmin))))
    union = float(np.prod(amax - amin) + np.prod(bmax - bmin) - intersection)
    return intersection / union if union > 0 else 0.0


def voxel_keys(points, voxel_size):
    points = points_array(points)
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive and finite")
    scaled = np.floor(points / voxel_size)
    if scaled.size and np.abs(scaled).max() >= 2**62:
        raise ValueError("Coordinates exceed safe voxel indexing range")
    return {tuple(row) for row in scaled.astype(np.int64)}


def voxel_overlap(a, b, voxel_size=0.01):
    a, b = voxel_keys(a, voxel_size), voxel_keys(b, voxel_size)
    return len(a & b) / len(a | b) if a and b else 0.0


def camera_inputs(depth, mask, intrinsic, camera_to_world, depth_scale):
    depth, mask = np.asarray(depth), np.asarray(mask)
    k, pose = np.asarray(intrinsic, dtype=float), np.asarray(camera_to_world, dtype=float)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Mask and aligned depth must be equal-sized 2D arrays")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Mask must be binary; select the intended instance explicitly")
    if k.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
        raise ValueError("Expected finite 3x3 intrinsics and 4x4 camera-to-world pose")
    if not np.allclose(k[2], [0, 0, 1]) or k[0, 0] <= 0 or k[1, 1] <= 0 or abs(np.linalg.det(k)) < 1e-12:
        raise ValueError("Invalid pinhole camera intrinsics")
    r = pose[:3, :3]
    if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1, atol=1e-4):
        raise ValueError("Pose must be a rigid camera-to-world transform")
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale is raw depth units per metre and must be positive")
    depth_m = depth.astype(float) / depth_scale
    valid = np.isfinite(depth_m) & (depth_m > 0)
    return depth_m, mask.astype(bool), k, pose, valid


def lift_depth_mask(depth, mask, intrinsic, camera_to_world, depth_scale=1000.0):
    depth_m, mask, k, pose, valid = camera_inputs(depth, mask, intrinsic, camera_to_world, depth_scale)
    y, x = np.nonzero(mask & valid)
    rays = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(k).T
    camera = rays * depth_m[y, x, None]
    return camera @ pose[:3, :3].T + pose[:3, 3]


def select_scene_points(points, depth, mask, intrinsic, camera_to_world,
                        depth_scale=1000.0, depth_tolerance=0.05):
    """Visibility-filtered projection onto existing mesh vertices, preserving order."""
    points = points_array(points)
    depth_m, mask, k, pose, valid_depth = camera_inputs(depth, mask, intrinsic, camera_to_world, depth_scale)
    if not math.isfinite(depth_tolerance) or depth_tolerance <= 0:
        raise ValueError("depth_tolerance must be positive and finite")
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    visible = camera[:, 2] > 0
    ids = np.flatnonzero(visible)
    projected = camera[ids] @ k.T
    xy = projected[:, :2] / projected[:, 2:3]
    h, w = depth_m.shape
    inside = (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
    ids = ids[inside]
    pixels = np.rint(xy[inside]).astype(int)
    x, y = pixels[:, 0], pixels[:, 1]
    keep = mask[y, x] & valid_depth[y, x] & (np.abs(camera[ids, 2] - depth_m[y, x]) <= depth_tolerance)
    return points[ids[keep]]


def choose_candidate(coarse, candidates, metric="voxel", voxel_size=0.01, min_overlap=0.0):
    coarse = points_array(coarse)
    if metric not in {"voxel", "aabb"}:
        raise ValueError("metric must be voxel or aabb")
    if not math.isfinite(min_overlap) or not 0 <= min_overlap <= 1:
        raise ValueError("min_overlap must be in [0, 1]")
    # Validate even when input/candidate sets are empty.
    voxel_keys(np.empty((0, 3)), voxel_size)
    if len({c[0] for c in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique")
    best_id, best_score, best_points = None, 0.0, coarse
    if len(coarse):
        for candidate_id, points in sorted(candidates, key=lambda c: c[0]):
            points = points_array(points)
            score = voxel_overlap(coarse, points, voxel_size) if metric == "voxel" else aabb_overlap(aabb(coarse), aabb(points))
            # Strict > makes zero overlap a fallback, not an arbitrary first candidate.
            if score > best_score and score >= min_overlap:
                best_id, best_score, best_points = candidate_id, score, points
    return {"status": "empty_geometry" if not len(coarse) else ("matched" if best_id is not None else "coarse_fallback"),
            "candidate_id": best_id, "matching_metric": metric, "matching_overlap": best_score,
            "voxel_size_m": voxel_size if metric == "voxel" else None,
            "aabb": aabb(best_points)}
