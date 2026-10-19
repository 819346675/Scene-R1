# Adapted from the supplied Scene-R1 open_r1 training entrypoints.
# Copyright 2025 The HuggingFace Team. Licensed under Apache-2.0.
"""Paper-aligned training rewards, not benchmark evaluators."""
import json
import math
import re

from ..parsing import answer_text, parse_box, parse_interval
from .leo_answer import clean_answer, answer_match


def format_reward(text):
    return float(re.fullmatch(r"<think>(?:(?!</?think>|</?answer>).)*</think>\s*"
                              r"<answer>(?:(?!</?think>|</?answer>).)+</answer>",
                              text.strip(), re.S) is not None)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def temporal_iou(text, example):
    """Inclusive frame set in the revised manuscript, with explicit index origin.

    Compute intersection counts without allocating a set for the predicted span.
    An out-of-video interval is invalid, not silently clipped to a good answer.
    """
    try:
        start, end = parse_interval(text, example["duration"])
    except (ValueError, TypeError):
        return 0.0
    fps, base = example["fps"], example["frame_index_base"]
    lo = math.floor(start * fps) + base
    hi = min(math.ceil(end * fps) + base, math.ceil(example["duration"] * fps) - 1 + base)
    gt = set(example["gt_frames"])
    count = max(0, hi - lo + 1)
    overlap = sum(lo <= frame <= hi for frame in gt)
    union = count + len(gt) - overlap
    return overlap / union if union else 0.0


def reward_components(text, example, input_size=None):
    result = {"accuracy": 0.0, "format": format_reward(text), "json": 0.0}
    task = example["task"]
    if task == "temporal":
        result["accuracy"] = temporal_iou(text, example)
    elif task == "box":
        if input_size is None:
            raise ValueError("Image rewards need the processor's resized pixel dimensions")
        try:
            payload = answer_text(text)
            fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", payload, re.S)
            value = json.loads(fenced.group(1) if fenced else payload)
            # The paper schema requires one object with a nonempty class label.
            if not isinstance(value, dict) or not isinstance(value.get("label"), str) or not value["label"].strip():
                raise ValueError("Missing label")
            pred = parse_box(payload, input_size, (example["width"], example["height"]))
            result["json"] = 1.0
            result["accuracy"] = box_iou(pred, example["bbox_2d"])
        except (ValueError, TypeError, KeyError):
            pass
    elif task == "qa":
        # Do not reward the reasoning block as if it were a final answer.
        matches = re.findall(r"<answer>(.*?)</answer>", text, re.S)
        if matches:
            prediction = clean_answer(matches[-1].strip())
            targets = [clean_answer(x) for x in example["answers"]]
            em, refined = answer_match(prediction, targets)
            result["accuracy"] = float(em + refined)
    else:
        raise ValueError(f"Unknown training task: {task}")
    return result


def total_reward(text, example, input_size=None, format_weight=0.1):
    if not math.isfinite(format_weight) or format_weight < 0:
        raise ValueError("format_weight must be finite and nonnegative")
    parts = reward_components(text, example, input_size)
    return parts["accuracy"] + format_weight * (parts["format"] + parts["json"])
