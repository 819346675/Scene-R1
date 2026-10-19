"""Synthetic unit tests only; no benchmark data, scores, weights, or network."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scene_r1.__main__ import main
from scene_r1.geometry import (aabb, aabb_overlap, choose_candidate, lift_depth_mask,
                               points_array, select_scene_points, voxel_overlap)
from scene_r1.inference import build_prompt
from scene_r1.io import contained_path, read_json, write_array, write_json
from scene_r1.parsing import answer_text, parse_box, parse_interval
from scene_r1.sai3d import execute_generation, generation_plan, import_candidates, load_candidates


class ParsingTests(unittest.TestCase):
    def test_last_answer_not_trailing_text(self):
        self.assertEqual(parse_interval('<answer>1 to 2</answer> extra 8 to 9'), [1, 2])

    def test_interval_errors(self):
        for value in ['nonsense', '<answer>1 to 2', '-1 to 2', '2 to 1', '1 to 1', '1e309 to 2']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_interval(value)
        with self.assertRaises(ValueError):
            parse_interval('1 to 3', duration=2)

    def test_object_list_and_fence(self):
        body = '{"bbox_2d": [10, 20, 30, 40], "label": "chair"}'
        for text in [body, '['+body+']', '<answer>'+body+'</answer>', '```json\n'+body+'\n```']:
            self.assertEqual(parse_box(text, (100, 100), (200, 300)), [20, 60, 60, 120])

    def test_invalid_box(self):
        for coords in [[1, 1, 1, 3], [-1, 0, 3, 4], [0, 0, 101, 3], [True, 0, 3, 4], [0, 0, float('nan'), 3]]:
            with self.subTest(coords=coords), self.assertRaises(ValueError):
                parse_box(json.dumps({'bbox_2d': coords}), (100, 100), (100, 100))
        with self.assertRaises(ValueError):
            parse_box('[{"bbox_2d":[0,0,2,2]},{"bbox_2d":[1,1,3,3]}]', (100,100), (100,100))

    def test_distinct_prompts(self):
        self.assertIn('time interval', build_prompt('temporal', 'chair'))
        self.assertIn('bbox_2d', build_prompt('box', 'chair'))
        self.assertNotIn('bbox_2d', build_prompt('qa', 'What is visible?'))
        with self.assertRaises(ValueError):
            build_prompt('box', '')


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((2, 2), 1000.0)
        self.mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        self.k = np.eye(3)
        self.pose = np.eye(4)
        self.pose[0, 3] = 10

    def test_metric_backprojection(self):
        actual = lift_depth_mask(self.depth, self.mask, self.k, self.pose)
        np.testing.assert_allclose(actual, [[10, 0, 1], [11, 1, 1]])

    def test_visibility_and_camera_inverse(self):
        points = np.array([[10,0,1],[11,1,1],[10,0,2],[10,0,-1],[11,0,1]])
        actual = select_scene_points(points, self.depth, self.mask, self.k, self.pose)
        np.testing.assert_allclose(actual, [[10,0,1],[11,1,1]])

    def test_invalid_depth_is_not_lifted(self):
        self.depth[0,0] = np.nan
        self.depth[1,1] = 0
        self.assertEqual(lift_depth_mask(self.depth, self.mask, self.k, self.pose).shape, (0,3))

    def test_bad_camera_contract(self):
        with self.assertRaises(ValueError):
            lift_depth_mask(self.depth, self.mask*255, self.k, self.pose)
        with self.assertRaises(ValueError):
            lift_depth_mask(self.depth, self.mask, self.k, self.pose, 0)
        bad_pose = self.pose.copy()
        bad_pose[0,0] = 2
        with self.assertRaises(ValueError):
            lift_depth_mask(self.depth, self.mask, self.k, bad_pose)

    def test_voxel_deduplication(self):
        a = np.array([[0,0,0],[0,0,0],[1,1,1]])
        b = np.array([[0,0,0],[1,1,1]])
        self.assertEqual(voxel_overlap(a,b), 1)
        self.assertEqual(voxel_overlap(np.empty((0,3)),np.empty((0,3))), 0)

    def test_two_matching_modes_are_distinct(self):
        a = np.array([[0,0,0],[1,1,0],[1,0,1],[0,1,1]])
        b = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1]])
        self.assertEqual(aabb_overlap(aabb(a), aabb(b)), 1)
        self.assertEqual(voxel_overlap(a,b), 0)

    def test_deterministic_match_and_fallback(self):
        points = np.array([[0,0,0],[1,1,1]])
        result = choose_candidate(points, [('z',points),('a',points)])
        self.assertEqual(result['candidate_id'], 'a')
        self.assertEqual(choose_candidate(points,[('b',points+10)])['status'], 'coarse_fallback')
        self.assertEqual(choose_candidate(points,[])['aabb'], aabb(points))
        self.assertEqual(choose_candidate(np.empty((0,3)),[])['status'], 'empty_geometry')

    def test_bad_points_and_options(self):
        with self.assertRaises(ValueError):
            points_array([[0,0,np.nan]])
        with self.assertRaises(ValueError):
            choose_candidate(np.empty((0,3)), [], min_overlap=-1)
        with self.assertRaises(ValueError):
            choose_candidate(np.empty((0,3)), [], voxel_size=0)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.points = np.array([[0,0,1],[1,1,1],[1,0,1]],dtype=float)
        write_array(self.root/'scene.npy',self.points)
        (self.root/'scene_pred_mask').mkdir()
        (self.root/'scene_pred_mask/one.txt').write_text('1\n1\n0\n')
        (self.root/'scene.txt').write_text('scene_pred_mask/one.txt 1 1.0\n')

    def imported(self):
        return import_candidates(self.root/'scene.txt',self.root/'scene.npy',self.root/'bundle','scene_raw')

    def test_real_export_index_layout(self):
        manifest = self.imported()
        candidates = load_candidates(manifest,'scene_raw')
        self.assertEqual(candidates[0][0],'one')
        np.testing.assert_equal(candidates[0][1],self.points[:2])

    def test_bad_mask_is_rejected_before_output(self):
        (self.root/'scene_pred_mask/one.txt').write_text('1\n0\n')
        with self.assertRaises(ValueError):
            self.imported()
        self.assertFalse((self.root/'bundle').exists())

    def test_path_traversal(self):
        (self.root/'scene.txt').write_text('../outside.txt 1 1.0\n')
        with self.assertRaises(ValueError):
            self.imported()
        with self.assertRaises(ValueError):
            contained_path(self.root,'/tmp/outside.txt')

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as other:
            (self.root/'link').symlink_to(other, target_is_directory=True)
            with self.assertRaises(ValueError):
                contained_path(self.root,'link/mask.txt')

    def test_frame_and_checksum(self):
        manifest = self.imported()
        with self.assertRaises(ValueError):
            load_candidates(manifest,'different_frame')
        np.save(self.root/'bundle/scene_points.npy',self.points+1)
        with self.assertRaises(ValueError):
            load_candidates(manifest,'scene_raw')

    def test_non_overwrite(self):
        self.imported()
        with self.assertRaises(FileExistsError):
            self.imported()
        write_json(self.root/'out.json', {'ok':True})
        with self.assertRaises(FileExistsError):
            write_json(self.root/'out.json', {'ok':False})

    def test_empty_index_fallback(self):
        (self.root/'scene.txt').write_text('')
        self.assertEqual(load_candidates(self.imported(),'scene_raw'), [])

    def test_lift_import_refine_cli(self):
        write_array(self.root/'depth.npy',np.ones((2,2)))
        write_array(self.root/'mask.npy',np.array([[1,0],[0,1]]))
        write_json(self.root/'frames.json', {'schema_version':1, 'coordinate_frame':'scene_raw','depth_scale':1,
            'frames':[{'depth':'depth.npy','mask':'mask.npy','intrinsic':np.eye(3).tolist(),'camera_to_world':np.eye(4).tolist()}]})
        self.assertEqual(main(['lift','--frames',str(self.root/'frames.json'),'--scene-points',str(self.root/'scene.npy'),
                               '--output',str(self.root/'coarse.npy')]),0)
        manifest = self.imported()
        command = ['refine','--coarse',str(self.root/'coarse.npy'),'--candidates',str(manifest),
                   '--coordinate-frame','scene_raw','--output',str(self.root/'selection.json')]
        self.assertEqual(main(command),0)
        self.assertEqual(read_json(self.root/'selection.json')['candidate_id'],'one')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(command),2)

    def test_generation_dry_run_and_argument_paths(self):
        upstream = self.root/'upstream with spaces'
        upstream.mkdir()
        (upstream/'sai3d.py').write_text('# synthetic upstream stub, never executed\n')
        plan = generation_plan(upstream,self.root/'data','scene0000_00',self.root/'new masks')
        self.assertEqual(plan['argv'][1],str((upstream/'sai3d.py').resolve()))
        self.assertIn('--eval_dir',plan['argv'])
        with patch('subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['sai3d','--sai3d-root',str(upstream),'--data-root',str(self.root/'data'),
                                  '--scene-id','scene0000_00','--output-dir',str(self.root/'new masks')]),0)
            run.assert_not_called()
        with self.assertRaises(ValueError):
            generation_plan(upstream,self.root/'data','../scene',self.root/'other')

    def test_generation_runner_with_synthetic_upstream(self):
        import sys
        upstream, data = self.root/'upstream', self.root/'data'
        upstream.mkdir()
        # This tiny authored fixture exercises subprocess wiring, not SAI3D/CUDA.
        (upstream/'sai3d.py').write_text(
            'import pathlib, sys\n'
            'out = pathlib.Path(sys.argv[sys.argv.index("--eval_dir")+1])\n'
            'scene = sys.argv[sys.argv.index("--scene_id")+1]\n'
            'out.mkdir(parents=True, exist_ok=False)\n'
            '(out/(scene+".txt")).write_text("")\n')
        for name in ['posed_images/scene', 'scans/scene', '2D_masks/scene/semantic-sam','Tasks/Benchmark']:
            (data/name).mkdir(parents=True)
        for name in ['Tasks/Benchmark/scannetv2_train.txt','Tasks/Benchmark/scannetv2_val.txt',
                     '2D_masks/scene/semantic-sam/maskraw_00000.png']:
            (data/name).touch()
        plan = generation_plan(upstream,data,'scene',self.root/'generated',python=sys.executable)
        execute_generation(plan)
        self.assertTrue((self.root/'generated/scene.txt').is_file())
        self.assertEqual(read_json(self.root/'generated/generation_manifest.json')['argv'],plan['argv'])

    def test_inference_cli_preserves_invalid_status(self):
        output = self.root/'inference.json'
        with patch('scene_r1.inference.infer',return_value={'status':'invalid_prediction','prediction':None}):
            self.assertEqual(main(['infer','--task','box','--image','unused.jpg','--query','chair',
                                   '--model','unused','--output',str(output)]),2)
        self.assertIsNone(read_json(output)['prediction'])


if __name__ == '__main__':
    unittest.main()
