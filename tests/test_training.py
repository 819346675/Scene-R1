"""CPU-only reward/schema tests. No weights, datasets or torch required."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scene_r1.training.data import validate, conversation, convert_legacy, load_training_data
from scene_r1.training.rewards import total_reward, temporal_iou, format_reward, reward_components
from scene_r1.training.__main__ import settings_from_file, main


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.temporal = dict(task="temporal", fps=2.0, duration=5.0, frame_index_base=0, gt_frames=[2, 3, 4])
        self.box = dict(task="box", width=200, height=100, bbox_2d=[20, 10, 100, 50])
        self.qa = dict(task="qa", answers=["two chairs", "2 chairs"])

    def test_temporal_inclusive_endpoint(self):
        self.assertEqual(temporal_iou("1 to 2", self.temporal), 1)
        self.assertAlmostEqual(total_reward("<think>x</think><answer>1 to 2</answer>", self.temporal), 1.1)

    def test_temporal_origin_is_explicit(self):
        one = dict(self.temporal, frame_index_base=1, gt_frames=[3, 4, 5])
        self.assertEqual(temporal_iou("1 to 2", one), 1)

    def test_temporal_sparse_gt_and_invalid_output(self):
        self.assertEqual(temporal_iou("1 to 2", dict(self.temporal, gt_frames=[2, 4])), 2/3)
        for value in ("-1 to 2", "2 to 1", "1 to 1000000000000", "nan to 2", "garbage"):
            self.assertEqual(temporal_iou(value, self.temporal), 0)

    def test_temporal_final_frame_clip(self):
        self.assertEqual(temporal_iou("4 to 5", dict(self.temporal, gt_frames=[8, 9])), 1)

    def test_box_resize_and_weights(self):
        text = '<think>x</think><answer>{"bbox_2d":[10,5,50,25],"label":"chair"}</answer>'
        self.assertAlmostEqual(total_reward(text, self.box, (100, 50)), 1.2)

    def test_box_schema_missing_label_or_invalid_coords(self):
        for text in ('{"bbox_2d":[10,5,50,25]}', '{"bbox_2d":[50,5,10,25],"label":"chair"}',
                     '{"bbox_2d":[true,5,50,25],"label":"chair"}'):
            self.assertEqual(total_reward(text, self.box, (100, 50)), 0)

    def test_format_is_independent(self):
        self.assertEqual(total_reward("<think>x</think><answer>wrong</answer>", self.box, (100, 50)), .1)
        for text in ("<think>x</think><answer></answer>", "<think>x</think><answer>1</answer>extra",
                     "<think>x</think><answer>1</answer><answer>2</answer>"):
            self.assertEqual(format_reward(text), 0)

    def test_qa_uses_em_and_refined_em(self):
        self.assertEqual(total_reward("<think>x</think><answer>2 chairs</answer>", self.qa), 2.1)
        self.assertEqual(reward_components("<answer>chairs</answer>", self.qa)["accuracy"], 1)
        self.assertEqual(reward_components("<think>two chairs</think>", self.qa)["accuracy"], 0)

    def test_qa_all_references_and_empty_normalization(self):
        self.assertEqual(total_reward("<answer>sofa</answer>", dict(self.qa, answers=["couch", "sofa"])), 2)
        self.assertEqual(total_reward("<answer>chair</answer>", dict(self.qa, answers=["!!!"])), 0)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scene.mp4").touch()
        Image.new("RGB", (200, 100)).save(self.root / "frame.png")
        self.row = dict(id="a", split="train", task="temporal", video="scene.mp4", query="chair", fps=2,
                        duration=5, gt_frames=[2, 3, 4], frame_index_base=0)

    def test_manifest_validation(self):
        row = validate(self.row, self.root)
        self.assertEqual(row["video"], str((self.root / "scene.mp4").resolve()))

    def test_reject_eval_missing_media_and_bad_frame_base(self):
        for update in (dict(split="test"), dict(frame_index_base=True), dict(gt_frames=[100]),
                       dict(fps=0), dict(gt_frames=[1.5]), dict(query="")):
            with self.assertRaises(ValueError):
                validate(dict(self.row, **update), self.root)
        with self.assertRaises(FileNotFoundError):
            validate(dict(self.row, video="missing.mp4"), self.root)

    def test_duplicate_id(self):
        path = self.root / "train.jsonl"
        path.write_text((json.dumps(self.row) + "\n") * 2)
        with self.assertRaises(ValueError):
            load_training_data(path, "temporal")

    def test_original_temporal_layout(self):
        path = self.root / "legacy.json"
        path.write_text(json.dumps({"scene": {"sentences": ["chair"], "timestamps": [[2, 3, 4]], "duration": 5}}))
        rows = convert_legacy(path, "temporal", self.root, 2, 0)
        self.assertEqual(rows[0]["gt_frames"], [2, 3, 4])

    def test_original_image_layout(self):
        path = self.root / "legacy.json"
        path.write_text(json.dumps({"scene": {"sentences": ["chair"], "image_path": ["frame.png"],
                                             "bbox_2d": [[0, 0, 50, 50]]}}))
        self.assertEqual(convert_legacy(path, "box", self.root, 2)[0]["width"], 200)

    def test_original_qa_layout(self):
        path = self.root / "legacy.json"
        path.write_text(json.dumps([dict(scene_id="scene", situation="Facing the door.",
                                         question="What is left?", answers=["chair", "seat"])]))
        self.assertEqual(convert_legacy(path, "qa", self.root, 2)[0]["answers"], ["chair", "seat"])

    def test_parallel_lists_may_not_be_silently_truncated(self):
        path = self.root / "legacy.json"
        path.write_text(json.dumps({"scene": dict(sentences=["chair", "sofa"], timestamps=[[2]], duration=5)}))
        with self.assertRaises(ValueError):
            convert_legacy(path, "temporal", self.root, 2, 0)

    def test_prompts_do_not_leak_labels(self):
        prompt = conversation(self.row, 12845056, 12544)
        text = prompt[0]["content"][0]["text"]
        self.assertIn("time interval", text)
        self.assertNotIn("gt_frames", text)
        self.assertEqual(prompt[0]["content"][1]["fps"], 2)

    def test_dry_run(self):
        manifest = self.root / "train.jsonl"
        manifest.write_text(json.dumps(self.row) + "\n")
        output = self.root / "output"
        main(["--task", "temporal", "--train-data", str(manifest), "--model", str(self.root),
              "--output", str(output), "--dry-run"])
        self.assertFalse(output.exists())

    def test_invalid_config(self):
        config = self.root / "config.json"
        for value in (dict(num_generations=1), dict(gradient_accumulation_steps=1.5), dict(unknown=2)):
            config.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                settings_from_file(config)


if __name__ == "__main__":
    unittest.main()
