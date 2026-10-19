"""Explicit, non-pickle input formats and conservative output handling."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    """Never silently replace an existing artifact."""
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def write_array(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)


def contained_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("Manifest paths must be relative")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Manifest path escapes its directory")
    return resolved


def read_points(path):
    from .geometry import points_array
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return points_array(np.load(path, allow_pickle=False))
    if path.suffix.lower() == ".ply":
        try:
            from plyfile import PlyData
        except ImportError as exc:
            raise ImportError('PLY input needs: pip install -e ".[ply]"') from exc
        vertices = PlyData.read(str(path))["vertex"]
        # Preserve vertex order: SAI3D binary masks index these exact rows.
        return points_array(np.column_stack([vertices[k] for k in ("x", "y", "z")]))
    raise ValueError("Point clouds must be .npy (N x 3) or .ply")
