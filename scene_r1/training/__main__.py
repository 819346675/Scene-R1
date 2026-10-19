"""Launch the baseline-derived Scene-R1 GRPO training flow."""
import argparse
import json
import math
import os
from pathlib import Path

from .data import load_training_data

DEFAULTS = dict(learning_rate=1e-5, adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.02,
                num_train_epochs=1.0, gradient_accumulation_steps=8, num_generations=8,
                beta=0.04, format_weight=0.1, temperature=1.0, max_prompt_length=8192,
                max_completion_length=1024, total_pixels=16384 * 28 * 28, min_pixels=16 * 28 * 28,
                seed=42, save_steps=500, logging_steps=1)


def settings_from_file(path):
    settings = dict(DEFAULTS)
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(supplied) - set(settings)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        settings.update(supplied)
    integers = {key for key, value in DEFAULTS.items() if type(value) is int}
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid setting {key}")
        if key in integers and type(value) is not int:
            raise ValueError(f"{key} must be an integer")
        if value < 0 or (key not in ("beta", "format_weight", "weight_decay", "seed") and value == 0):
            raise ValueError(f"Invalid setting {key}")
    if settings["num_generations"] < 2 or settings["min_pixels"] > settings["total_pixels"]:
        raise ValueError("Invalid generation group or pixel budget")
    if not 0 < settings["adam_beta1"] < 1 or not 0 < settings["adam_beta2"] < 1:
        raise ValueError("Adam betas must be strictly between zero and one")
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["temporal", "box", "qa"], required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--model", required=True, help="Local Qwen2.5-VL checkpoint directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--deepspeed", help="Optional local DeepSpeed configuration JSON")
    parser.add_argument("--resume", help="Explicit trusted local checkpoint; never inferred")
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive override for a short smoke run")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/settings without importing torch or loading weights")
    parser.add_argument("--cpu", action="store_true", help="Float32 CPU smoke runs with a small local model")
    args = parser.parse_args(argv)
    settings = settings_from_file(args.config)
    rows = load_training_data(args.train_data, args.task)
    if not Path(args.model).is_dir():
        parser.error("--model must point to an existing local checkpoint directory")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or positive")
    if args.resume and not (Path(args.resume) / "trainer_state.json").is_file():
        parser.error("--resume must contain trainer_state.json")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Output directory is not empty; choose a new directory or explicitly resume")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    plan = dict(task=args.task, examples=len(rows), world_size=world_size, settings=settings,
                effective_query_batch=world_size * settings["gradient_accumulation_steps"],
                completions_per_query=settings["num_generations"])
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    import torch
    from transformers import TrainingArguments, set_seed
    from .trainer import SceneR1GRPOTrainer
    if not args.cpu and not torch.cuda.is_available():
        parser.error("GPU training requires CUDA; use --cpu only with a small smoke-test model")
    set_seed(settings["seed"])
    training_args = TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=1,
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"], adam_beta1=settings["adam_beta1"],
        adam_beta2=settings["adam_beta2"], weight_decay=settings["weight_decay"],
        lr_scheduler_type="cosine", warmup_ratio=0.0, optim="adamw_torch",
        num_train_epochs=settings["num_train_epochs"], max_steps=args.max_steps,
        bf16=not args.cpu, use_cpu=args.cpu, gradient_checkpointing=not args.cpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, remove_unused_columns=False,
        report_to=[], push_to_hub=False, save_safetensors=True,
        save_steps=settings["save_steps"], logging_steps=settings["logging_steps"],
        seed=settings["seed"], data_seed=settings["seed"], deepspeed=args.deepspeed,
        eval_strategy="no", dataloader_num_workers=0)
    trainer = SceneR1GRPOTrainer(args.model, training_args, rows, settings)
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output))
    trainer.save_state()
    if trainer.is_world_process_zero():
        trainer.processing_class.save_pretrained(str(output))
        (output / "training-settings.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
