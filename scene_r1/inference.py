"""Single-example Qwen2.5-VL inference, adapted from the Scene-R1 baseline.

Baseline provenance and upstream acknowledgements are in NOTICE.
Model imports are lazy: geometry tools do not require torch or transformers.
"""
from pathlib import Path

from .parsing import answer_text, parse_box, parse_interval


def build_prompt(task, query):
    if not query.strip():
        raise ValueError("query cannot be empty")
    if task == "temporal":
        return (f'Locate the object described as "{query}" in the video. '
                'Output reasoning in <think></think> and the time interval in seconds '
                'as <answer>start to end</answer>.')
    if task == "box":
        return (f'Outline the object described as "{query}" in the image. '
                'Output reasoning in <think></think> and exactly one JSON object in <answer></answer>: '
                '{"bbox_2d": [x1, y1, x2, y2], "label": "object"}. '
                'Use pixel coordinates of the image you see, not 0-1 or 0-1000 normalized coordinates.')
    if task == "qa":
        return f'{query} Output reasoning in <think></think> and the short final answer in <answer></answer>.'
    raise ValueError("Unknown task")


def infer(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    media_path = Path(args.image or args.video).resolve()
    if not media_path.is_file():
        raise FileNotFoundError(media_path)
    if args.task == "box" and not args.image:
        raise ValueError("box inference requires --image")
    if args.task == "temporal" and not args.video:
        raise ValueError("temporal inference requires --video")
    prompt = build_prompt(args.task, args.query)
    local_only = not args.allow_download
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=local_only, trust_remote_code=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=local_only, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16,
        attn_implementation="sdpa", device_map=args.device).eval()
    kind = "image" if args.image else "video"
    media = {"type": kind, kind: str(media_path)}
    if args.video:
        media.update(fps=args.fps, total_pixels=args.total_pixels, min_pixels=16 * 28 * 28)
    else:
        media.update(max_pixels=args.total_pixels)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, media]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=images, videos=videos, padding=True,
                       return_tensors="pt", **video_kwargs).to(args.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    response = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
    result = {"schema_version": 1, "task": args.task, "query": args.query,
              "model": args.model, "prompt": prompt, "response": response, "status": "ok", "prediction": None,
              "release_note": "Single-example inference, not a benchmark result."}
    try:
        if args.task == "temporal":
            result["prediction"] = {"interval_seconds": parse_interval(response, args.duration)}
        elif args.task == "box":
            with Image.open(media_path) as img:
                original = img.size
            patch_size = processor.image_processor.patch_size
            grid = inputs["image_grid_thw"][0].tolist()
            input_size = (int(grid[2]) * patch_size, int(grid[1]) * patch_size)
            result["prediction"] = {"bbox_xyxy_original_pixels": parse_box(response, input_size, original),
                                    "original_size_wh": list(original), "model_input_size_wh": list(input_size)}
        else:
            answer = answer_text(response)
            if not answer or ("<think>" in answer):
                raise ValueError("Missing a clean final answer")
            result["prediction"] = {"answer": answer}
    except (ValueError, TypeError, KeyError) as exc:
        result.update(status="invalid_prediction", error=str(exc))
    return result
