"""Explicit training schemas and adapters for the original Scene-R1 JSON files."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image

from ..inference import build_prompt


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def validate(example, root):
    row = dict(example)
    if row.get("split") != "train":
        raise ValueError("Only explicitly labelled split=train records may enter this training loader")
    if row.get("task") not in ("temporal", "box", "qa"):
        raise ValueError("task must be temporal, box or qa")
    if not isinstance(row.get("query"), str) or not row["query"].strip():
        raise ValueError("query must be a nonempty string")
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        raise ValueError("Each training record needs a nonempty id")
    kind = "image" if row["task"] == "box" else "video"
    if not isinstance(row.get(kind), str) or not row[kind]:
        raise ValueError(f"Missing local {kind} path")
    media = (Path(root) / row[kind]).resolve()
    if not media.is_file():
        raise FileNotFoundError(media)
    row[kind] = str(media)
    if kind == "video":
        positive(row.get("fps"), "fps")
    if row["task"] == "temporal":
        duration = positive(row.get("duration"), "duration")
        base = row.get("frame_index_base")
        if type(base) is not int or base not in (0, 1):
            raise ValueError("Declare frame_index_base as 0 or 1; indices are never guessed")
        frames = row.get("gt_frames")
        limit = math.ceil(duration * row["fps"])
        if not isinstance(frames, list) or not frames or any(
            type(x) is not int or not base <= x < limit + base for x in frames
        ):
            raise ValueError("gt_frames must be nonempty integer indices on the declared fps grid")
        row["gt_frames"] = sorted(set(frames))
    elif row["task"] == "box":
        with Image.open(media) as image:
            width, height = image.size
        box = row.get("bbox_2d")
        if not isinstance(box, list) or len(box) != 4 or any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in box
        ):
            raise ValueError("bbox_2d must contain four finite pixel coordinates")
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError("GT box must be ordered and within original-image pixel bounds")
        row.update(width=width, height=height)
    else:
        answers = row.get("answers")
        if not isinstance(answers, list) or not answers or any(not isinstance(x, str) or not x.strip() for x in answers):
            raise ValueError("answers must be a nonempty list of nonempty strings")
    return row


def load_training_data(path, task):
    path = Path(path).resolve()
    rows, identities = [], set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = validate(json.loads(line), path.parent)
                if row["task"] != task or row["id"] in identities:
                    raise ValueError("Mixed tasks or duplicate training id")
            except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
                raise ValueError(f"Training row {number}: {exc}") from exc
            identities.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("Empty training split")
    return rows


def conversation(example, total_pixels, min_pixels):
    kind = "image" if example["task"] == "box" else "video"
    media = {"type": kind, kind: example[kind], "min_pixels": min_pixels}
    if kind == "video":
        media.update(fps=example["fps"], total_pixels=total_pixels)
    else:
        media["max_pixels"] = total_pixels
    return [{"role": "user", "content": [
        {"type": "text", "text": build_prompt(example["task"], example["query"])}, media]}]


def convert_legacy(source, task, media_root, fps, frame_index_base=None):
    """Read original TG/IG/QA annotation layouts; never silently drop missing media."""
    source, root = Path(source), Path(media_root).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    rows = []

    def video(scene):
        matches = [root / f"{scene}.{ext}" for ext in ("mp4", "mkv", "webm") if (root / f"{scene}.{ext}").is_file()]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one video for scene {scene}, found {len(matches)}")
        return str(matches[0])

    if task == "qa":
        if not isinstance(raw, list):
            raise ValueError("Legacy QA annotations must be a list")
        for i, item in enumerate(raw):
            rows.append(dict(id=f"{item['scene_id']}:{i}", task=task, split="train", fps=fps,
                             video=video(item["scene_id"]),
                             query=f"{item['situation']} {item['question']}".strip(), answers=item["answers"]))
    else:
        if not isinstance(raw, dict):
            raise ValueError("Legacy grounding annotations must be a scene-keyed object")
        for scene, item in raw.items():
            fields = ["sentences", "timestamps"] if task == "temporal" else ["sentences", "image_path", "bbox_2d"]
            if any(not isinstance(item.get(key), list) for key in fields) or len({len(item[key]) for key in fields}) != 1:
                raise ValueError(f"Parallel annotation lists differ in scene {scene}")
            for i, query in enumerate(item["sentences"]):
                row = dict(id=f"{scene}:{i}", task=task, split="train", query=query)
                if task == "temporal":
                    row.update(video=video(scene), fps=fps, duration=item["duration"],
                               gt_frames=item["timestamps"][i], frame_index_base=frame_index_base)
                else:
                    row.update(image=str((root / item["image_path"][i]).resolve()), bbox_2d=item["bbox_2d"][i])
                rows.append(row)
    return [validate(row, root) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["temporal", "box", "qa"], required=True)
    parser.add_argument("--data", help="Validate an existing JSONL training manifest")
    parser.add_argument("--legacy-json", help="Convert the original training annotation JSON")
    parser.add_argument("--media-root")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frame-index-base", type=int, choices=[0, 1])
    parser.add_argument("--output")
    args = parser.parse_args()
    if bool(args.data) == bool(args.legacy_json):
        parser.error("Choose exactly one of --data and --legacy-json")
    if args.data:
        rows = load_training_data(args.data, args.task)
    else:
        if not args.media_root or not args.output:
            parser.error("Conversion requires --media-root and --output")
        if args.task == "temporal" and args.frame_index_base is None:
            parser.error("Temporal conversion requires --frame-index-base")
        rows = convert_legacy(args.legacy_json, args.task, args.media_root, args.fps, args.frame_index_base)
        if not rows:
            raise ValueError("Empty training split")
        with Path(args.output).open("x", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Validated {len(rows)} {args.task} training records. Media existence checked; videos not decoded.")


if __name__ == "__main__":
    main()
