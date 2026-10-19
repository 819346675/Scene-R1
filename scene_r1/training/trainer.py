# Copyright 2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE.txt.
# Adapted from Scene-R1-main/src/open_r1/trainer/grpo_trainer.py and
# grpo_trainer_video.py (the baseline acknowledges Time-R1 / TimeZero).
# Changes: consolidated image/video path, explicit task prompts and rewards,
# raw-media loading, pixel/timing alignment, validated batch size and safe resume.
"""Single-on-policy-update GRPO, preserving the supplied trainer's objective.

One query per device/microbatch, G responses per query. Each rollout is used
once, so the old-policy ratio has value one and retains its policy gradient.
This is not a multi-iteration PPO trainer. See docs/TRAINING.md.
"""
from collections import defaultdict

import torch
from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration, Trainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation

from .data import conversation
from .rewards import total_reward


def identity_collator(features):
    return features


def group_advantages(rewards, group_size):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("GRPO requires complete groups of at least two responses")
    groups = rewards.view(-1, group_size)
    return ((groups - groups.mean(1, keepdim=True)) / (groups.std(1, keepdim=True) + 1e-4)).flatten()


def completion_mask(token_ids, eos_token_id):
    ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    is_eos = torch.zeros_like(token_ids, dtype=torch.bool)
    for token in ids:
        is_eos |= token_ids == token
    # The EOS token itself contributes to the sequence likelihood.
    return (is_eos.cumsum(1) - is_eos.long() == 0).long()


def repeat_vision_inputs(inputs, count):
    """Repeat one query's flattened visual patches and timing in rollout order."""
    result = {}
    for key, value in inputs.items():
        if key in ("input_ids", "attention_mask"):
            continue
        if key not in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"):
            raise ValueError(f"Unsupported processor field: {key}")
        if isinstance(value, torch.Tensor):
            result[key] = value.repeat((count,) + (1,) * (value.ndim - 1))
        elif isinstance(value, (list, tuple)):
            result[key] = list(value) * count
        else:
            raise TypeError(f"Unexpected {key} type: {type(value)}")
    return result


class SceneR1GRPOTrainer(Trainer):
    def __init__(self, model, args, train_dataset, settings, processing_class=None, **kwargs):
        if args.per_device_train_batch_size != 1:
            raise ValueError("This baseline-derived trainer requires one query per device; use gradient accumulation")
        if args.fsdp:
            raise ValueError("FSDP generation is not validated here; use DDP or the optional DeepSpeed configuration")
        if args.eval_strategy != "no":
            raise ValueError("This training release does not provide a benchmark evaluation loop")
        self.num_generations = settings["num_generations"]
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least two")
        self.settings = settings
        self.beta = settings["beta"]
        load_kwargs = dict(torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
                           attn_implementation="sdpa", local_files_only=True,
                           trust_remote_code=False, use_safetensors=True)
        if isinstance(model, str):
            model_path = model
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
            if processing_class is None:
                processing_class = AutoProcessor.from_pretrained(model_path, local_files_only=True,
                                                                trust_remote_code=False)
            self.ref_model = (Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
                              if is_deepspeed_zero3_enabled() else create_reference_model(model))
        else:
            if processing_class is None:
                raise ValueError("An instantiated model needs a processing_class")
            self.ref_model = create_reference_model(model)
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()
        # Disable dropout so rollout and likelihood policies are consistent.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.config.use_cache = False
        tokenizer = processing_class.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("The processor must define EOS and padding token IDs")
        self.generation_config = GenerationConfig(
            max_new_tokens=settings["max_completion_length"], do_sample=True,
            temperature=settings["temperature"], top_p=1.0, top_k=0,
            num_return_sequences=self.num_generations,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        model.warnings_issued["estimate_tokens"] = True
        self._metrics = defaultdict(list)
        args.remove_unused_columns = False
        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         data_collator=identity_collator, processing_class=processing_class, **kwargs)
        self.model_accepts_loss_kwargs = False
        if self.is_deepspeed_enabled:
            self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
        else:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_inputs(self, inputs):
        return inputs

    def _encode_example(self, example):
        from qwen_vl_utils import process_vision_info
        messages = conversation(example, self.settings["total_pixels"], self.settings["min_pixels"])
        text = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        values = self.processing_class(text=[text], images=images, videos=videos,
                                       padding=True, return_tensors="pt", **video_kwargs)
        # Never truncate visual placeholder tokens separately from their patches.
        if values["input_ids"].shape[1] > self.settings["max_prompt_length"]:
            raise ValueError("Prompt exceeds max_prompt_length; explicitly reduce the media budget or increase the limit")
        return super()._prepare_inputs(values)

    @staticmethod
    def _get_per_token_logps(model, input_ids, attention_mask, vision, prompt_length):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **vision)
        logits = outputs.logits[:, prompt_length - 1:-1, :]
        targets = input_ids[:, prompt_length:]
        # Only materialize log-softmax for completion tokens, in float32.
        return torch.stack([row.float().log_softmax(-1).gather(1, ids.unsqueeze(1)).squeeze(1)
                            for row, ids in zip(logits, targets)])

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs or len(inputs) != 1:
            raise ValueError("Expected one query and scalar training loss")
        example = inputs[0]
        prompt_inputs = self._encode_example(example)
        prompt_length = prompt_inputs["input_ids"].shape[1]
        with torch.no_grad(), unwrap_model_for_generation(model, self.accelerator) as unwrapped:
            sequences = unwrapped.generate(**prompt_inputs, generation_config=self.generation_config)
        completion_ids = sequences[:, prompt_length:]
        if sequences.shape[0] != self.num_generations or completion_ids.shape[1] == 0:
            raise ValueError("Generation returned an incomplete rollout group")
        mask = completion_mask(completion_ids, self.generation_config.eos_token_id)
        attention_mask = torch.cat([prompt_inputs["attention_mask"].repeat_interleave(self.num_generations, 0), mask], 1)
        vision = repeat_vision_inputs(prompt_inputs, self.num_generations)
        per_token_logps = self._get_per_token_logps(model, sequences, attention_mask, vision, prompt_length)
        with torch.no_grad():
            reference_logps = self._get_per_token_logps(self.ref_model, sequences, attention_mask, vision, prompt_length)
        delta = reference_logps - per_token_logps
        per_token_kl = delta.exp() - delta - 1
        responses = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)
        input_size = None
        if example["task"] == "box":
            grid = prompt_inputs["image_grid_thw"][0]
            patch = self.processing_class.image_processor.patch_size
            input_size = (int(grid[2]) * patch, int(grid[1]) * patch)
        rewards = torch.tensor([total_reward(response, example, input_size, self.settings["format_weight"])
                                for response in responses], dtype=torch.float32, device=per_token_logps.device)
        advantages = group_advantages(rewards, self.num_generations)
        # Retained single-update GRPO objective from the original trainers.
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * mask).sum(1) / mask.sum(1)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite GRPO loss")
        for key, value in {"reward": rewards.mean(), "reward_std": rewards.std(),
                           "kl": ((per_token_kl * mask).sum(1) / mask.sum(1)).mean(),
                           "completion_length": mask.sum(1).float().mean()}.items():
            self._metrics[key].append(self.accelerator.gather_for_metrics(value.detach()).float().mean().item())
        return loss

    def log(self, logs, start_time=None):
        logs = {**logs, **{key: sum(values) / len(values) for key, values in self._metrics.items() if values}}
        super().log(logs, start_time)
        self._metrics.clear()
