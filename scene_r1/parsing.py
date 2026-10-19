"""One answer protocol for inference; invalid answers are never made into predictions."""
import json
import math
import re


def answer_text(response):
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    if "<answer>" in response or "</answer>" in response:
        raise ValueError("Unbalanced answer tags")
    return response.strip()


def parse_interval(response, duration=None):
    text = answer_text(response)
    number = r"(?:\d+(?:\.\d*)?|\.\d+)"
    match = re.fullmatch(rf"({number})\s+(?:to|and)\s+({number})", text, re.I)
    if not match:
        raise ValueError("Expected <answer>start to end</answer> in seconds")
    start, end = map(float, match.groups())
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError("Time interval must be finite, non-negative and ordered")
    if duration is not None and (not math.isfinite(duration) or duration <= 0 or end > duration):
        raise ValueError("Interval exceeds the supplied video duration")
    return [start, end]


def parse_box(response, input_size, original_size):
    text = answer_text(response)
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Exactly one target box is required")
        value = value[0]
    if not isinstance(value, dict) or "bbox_2d" not in value:
        raise ValueError("Expected a JSON object with bbox_2d")
    coords = value["bbox_2d"]
    if not isinstance(coords, list) or len(coords) != 4:
        raise ValueError("bbox_2d must have four coordinates")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in coords):
        raise ValueError("Coordinates must be finite numbers")
    iw, ih = input_size
    ow, oh = original_size
    if min(iw, ih, ow, oh) <= 0:
        raise ValueError("Image sizes must be positive")
    x1, y1, x2, y2 = coords
    if not (0 <= x1 < x2 <= iw and 0 <= y1 < y2 <= ih):
        raise ValueError("Box must be ordered and within model input pixel bounds")
    return [x1 * ow / iw, y1 * oh / ih, x2 * ow / iw, y2 * oh / ih]
