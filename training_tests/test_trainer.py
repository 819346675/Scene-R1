"""Optional torch/transformers smoke tests, with no network or checkpoints.

Tiny randomly initialized Qwen2.5-VL: real visual forward/backward and optimizer
steps, deterministic synthetic rollouts. Not a real-data quality benchmark.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, TrainingArguments

from scene_r1.training.__main__ import DEFAULTS
from scene_r1.training.trainer import SceneR1GRPOTrainer, group_advantages, completion_mask, repeat_vision_inputs


class TinyQwen(Qwen2_5_VLForConditionalGeneration):
    def generate(self, input_ids, generation_config, **kwargs):
        ends = torch.tensor([[5, 2], [6, 2]], device=input_ids.device)
        return torch.cat([input_ids.repeat(2, 1), ends], 1)


class SyntheticProcessor:
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, padding_side="left")
    image_processor = SimpleNamespace(patch_size=14)

    def __init__(self, task="qa"):
        self.task = task

    def batch_decode(self, tokens, **kwargs):
        answers = {"qa": ["chair", "table"], "temporal": ["1 to 2", "4 to 5"],
                   "box": ['{"bbox_2d":[0,0,14,14],"label":"chair"}',
                           '{"bbox_2d":[14,14,28,28],"label":"chair"}']}[self.task]
        return [f"<think>x</think><answer>{answer}</answer>" for answer in answers]

    def save_pretrained(self, path):
        Path(path, "synthetic-processor.txt").write_text("Synthetic test processor, not a usable checkpoint processor.")


class SyntheticTrainer(SceneR1GRPOTrainer):
    def _encode_example(self, example):
        video = example.get("video_input", False)
        values = {"input_ids": torch.tensor([[1, 29, 31 if video else 30, 28, 3]]),
                  "attention_mask": torch.ones(1, 5, dtype=torch.long)}
        values["pixel_values_videos" if video else "pixel_values"] = torch.ones(4, 3 * 2 * 14 * 14)
        values["video_grid_thw" if video else "image_grid_thw"] = torch.tensor([[1, 2, 2]])
        if video:
            values["second_per_grid_ts"] = [1.0]
        return {key: value.to(self.accelerator.device) if isinstance(value, torch.Tensor) else value
                for key, value in values.items()}


def tiny_model():
    config = Qwen2_5_VLConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
        image_token_id=30, video_token_id=31, vision_start_token_id=29, vision_end_token_id=28,
        pad_token_id=0, eos_token_id=2,
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           in_channels=3, patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
                           out_hidden_size=32, fullatt_block_indexes=[0], window_size=28))
    config._attn_implementation = "eager"
    return TinyQwen(config)


class TrainerTests(unittest.TestCase):
    def test_raw_video_decode_and_fps(self):
        import av
        import numpy as np
        from qwen_vl_utils import process_vision_info
        from scene_r1.training.data import conversation
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "synthetic.mp4")
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream("mpeg4", rate=2)
                stream.width = stream.height = 112
                stream.pix_fmt = "yuv420p"
                for value in range(8):
                    frame = av.VideoFrame.from_ndarray(np.full((112, 112, 3), value * 20, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            messages = conversation(dict(task="temporal", query="chair", video=str(path), fps=2),
                                    12845056, 12544)
            images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            self.assertIsNone(images)
            self.assertEqual(len(videos), 1)
            self.assertEqual(videos[0].shape[0], 8)
            self.assertAlmostEqual(video_kwargs["fps"][0], 2)

    def test_constant_reward_groups_are_finite_zero(self):
        values = group_advantages(torch.tensor([1., 1., 2., 3.]), 2)
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(values[:2].tolist(), [0, 0])
        self.assertLess(values[2], 0)
        self.assertGreater(values[3], 0)

    def test_eos_mask_includes_first_eos_only(self):
        values = completion_mask(torch.tensor([[5, 2, 0, 0], [4, 5, 6, 7]]), 2)
        self.assertEqual(values.tolist(), [[1, 1, 0, 0], [1, 1, 1, 1]])

    def test_timing_and_visual_repeat(self):
        values = repeat_vision_inputs(dict(pixel_values_videos=torch.arange(8).view(4, 2),
                                          video_grid_thw=torch.tensor([[1, 2, 2]]), second_per_grid_ts=[.5]), 2)
        self.assertEqual(values["second_per_grid_ts"], [.5, .5])
        self.assertTrue(torch.equal(values["pixel_values_videos"][:4], values["pixel_values_videos"][4:]))

    def test_real_qwen_image_and_video_forward_backward(self):
        for task in ("box", "temporal", "qa"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as folder:
                model = tiny_model()
                args = TrainingArguments(output_dir=folder, use_cpu=True, report_to=[],
                                         per_device_train_batch_size=1, remove_unused_columns=False)
                settings = dict(DEFAULTS, num_generations=2, max_completion_length=2)
                example = dict(task=task, answers=["chair"], video_input=task != "box", fps=2, duration=5,
                               frame_index_base=0, gt_frames=[2,3,4], width=28, height=28, bbox_2d=[0,0,14,14])
                trainer = SyntheticTrainer(model, args, [example], settings, processing_class=SyntheticProcessor(task))
                loss = trainer.compute_loss(model, [example])
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(any(torch.count_nonzero(g).item() for g in grads))
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertTrue(all(p.grad is None for p in trainer.ref_model.parameters()))

    def test_optimizer_step_and_checkpoint_save(self):
        with tempfile.TemporaryDirectory() as folder:
            model = tiny_model()
            before = copy.deepcopy(model.state_dict())
            args = TrainingArguments(output_dir=folder, use_cpu=True, report_to=[],
                                     max_steps=1, per_device_train_batch_size=1,
                                     save_steps=1, logging_steps=1, disable_tqdm=True)
            trainer = SyntheticTrainer(model, args, [dict(task="qa", answers=["chair"])],
                                       dict(DEFAULTS, num_generations=2, max_completion_length=2),
                                       processing_class=SyntheticProcessor())
            trainer.train()
            self.assertEqual(trainer.state.global_step, 1)
            self.assertTrue(any(not torch.equal(before[k], value) for k, value in model.state_dict().items()))
            self.assertTrue(Path(folder, "checkpoint-1", "model.safetensors").is_file())
            self.assertTrue(Path(folder, "checkpoint-1", "trainer_state.json").is_file())

    def test_microbatch_larger_than_one_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = TrainingArguments(output_dir=folder, use_cpu=True, report_to=[], per_device_train_batch_size=2)
            with self.assertRaises(ValueError):
                SyntheticTrainer(tiny_model(), args, [], dict(DEFAULTS), processing_class=SyntheticProcessor())


if __name__ == "__main__":
    unittest.main()
