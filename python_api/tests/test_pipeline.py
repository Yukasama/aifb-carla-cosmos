"""CPU tests for offline preflight, orchestration and dataset boundaries."""
import ast
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import common
import run_cosmos
import run_detector
from run_detector import PipelineMixin


def put(path, value=b'x'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def js(path, value):
    return put(path, json.dumps(value).encode())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def config(self, extra=''):
        text = '[paths]\n' + '\n'.join(f'{key} = {json.dumps(str(value))}' for key, value in {
            'clips': self.root / 'clips', 'datasets': self.root / 'datasets', 'runs': self.root / 'runs',
        }.items()) + '\n' + extra
        return common.Config(put(self.root / 'pipeline.toml', text.encode()))

    def raw(self):
        root = self.root / 'raw'
        entries = []
        for sid in ('000001', '000002'):
            item = {'image_path': f'image/{sid}.jpg', 'pointcloud_path': f'velodyne/{sid}.pcd',
                    'calib_camera_intrinsic_path': f'calib/camera_intrinsic/{sid}.json',
                    'calib_virtuallidar_to_camera_path': f'calib/virtuallidar_to_camera/{sid}.json',
                    'label_camera_std_path': f'label/camera/{sid}.json'}
            for value in item.values():
                put(root / value)
            entries.append(item)
        js(root / 'data_info.json', entries)
        split = js(root / 'split.json', {'train': ['000001'], 'val': ['000002'], 'test': []})
        return root, split

    def prepared(self, output, synthetic=True, variant='carla-hd'):
        root, split = self.raw()
        output.mkdir(parents=True)
        for subset in ('train', 'val'):
            put(output / (subset + '.pkl'), pickle.dumps([{'sample_token': 'image/000001.jpg'}]))
        for sid in ('000001', '000002'):
            put(output / 'labels' / (sid + '.txt'), b'')
        manifest = {'raw_root': str(root), 'split': str(split), 'train_info': str(output / 'train.pkl'),
                    'val_info': str(output / 'val.pkl'), 'gt_labels': str(output / 'labels'),
                    'synthetic': synthetic, 'variant': variant, 'counts': {'train': 1, 'val': 1}}
        js(output / 'prepared.json', manifest)
        return manifest

    def test_missing_section_aborts_every_entrypoint_before_other_work(self):
        config = put(self.root / 'missing.toml', b'[paths]\nclips="/path/which/does/not/exist"\n')
        calls = [('cosmos25', 'run_cosmos.py', '2.5'), ('cosmos3', 'run_cosmos.py', '3'),
                 ('bevheight++', 'run_detector.py', 'train', '--name', 'x', '--data', 'carla'),
                 ('cobev', 'run_detector.py', 'evaluate', '--model', 'cobev', '--name', 'x', '--checkpoint', '/missing.ckpt')]
        for section, script, *args in calls:
            with self.subTest(script=script, args=args):
                result = subprocess.run([sys.executable, str(common.ROOT / script), *args, '--config', str(config)],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(f'[{section}]', result.stderr)
                self.assertNotIn('ModuleNotFoundError', result.stderr)
        self.assertEqual(list(self.root.iterdir()), [config])

    def test_offline_environment_without_launcher_venv(self):
        with patch.dict(os.environ, {'VIRTUAL_ENV': '/launcher/.venv'}):
            env = common.offline_env()
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
            self.assertEqual(env[key], '1')
        self.assertNotIn('VIRTUAL_ENV', env)
        self.assertNotIn('PYTHONPATH', env)

    def test_detector_script_parses_in_detector_python(self):
        # _prepare/_detector laufen in der Python-3.8-Umgebung der Detektoren
        for script in ('common.py', 'run_detector.py'):
            ast.parse((common.ROOT / script).read_text(encoding='utf-8'), feature_version=(3, 8))

    def test_splits_and_missing_raw_files_fail(self):
        root, split = self.raw()
        self.assertEqual(run_detector.validate_raw(root, split)['val'], ['000002'])
        js(split, {'train': ['000001'], 'val': ['000001']})
        with self.assertRaisesRegex(ValueError, 'ueberschneiden'):
            run_detector.validate_raw(root, split)
        js(split, {'train': ['000001'], 'val': ['000002']})
        (root / 'label/camera/000002.json').unlink()
        with self.assertRaises(FileNotFoundError):
            run_detector.validate_raw(root, split)

    def test_variants_are_distinct_and_outputs_never_overwritten(self):
        names = {run_detector.dataset_key(data, version, fullhd)
                 for data in ('carla', 'cosmos') for version in ('2.5', '3') for fullhd in (False, True)}
        self.assertEqual(names, {'carla-hd', 'carla-fullhd', 'cosmos25-hd', 'cosmos25-fullhd', 'cosmos3-hd', 'cosmos3-fullhd'})
        with self.assertRaises(FileExistsError):
            common.new_output(self.root)
        with self.assertRaises(ValueError):
            run_detector.safe_name('../elsewhere')

    def test_spec_collection_requires_every_modality_and_matching_padding(self):
        clip = self.root / 'clips' / 'test-clip'
        js(clip / 'clip.json', {'frames': 82, 'fps': 30})
        js(clip / 'cosmos/manifest.json', {'source_frames': 82, 'video_frames': 93, 'fps': 30, 'size': [1280, 720]})
        spec = {'name': clip.name, 'num_frames': 93, 'fps': 30, 'depth': {'control_path': 'depth.mp4'}}
        js(clip / 'cosmos/cosmos3.json', spec)
        for stream in ('rgb', 'seg', 'vis', 'edge'):
            put(clip / 'cosmos' / (stream + '.mp4'))
        with self.assertRaisesRegex(ValueError, '1920x1080'):  # Cosmos bekommt die Full-HD-Videos
            run_cosmos.collect_specs(clip.parent, '3')
        js(clip / 'cosmos/manifest.json', {'source_frames': 82, 'video_frames': 93, 'fps': 30, 'size': [1920, 1080]})
        with self.assertRaises(FileNotFoundError):
            run_cosmos.collect_specs(clip.parent, '3')
        put(clip / 'cosmos/depth.mp4')
        self.assertEqual(len(run_cosmos.collect_specs(clip.parent, '3')), 1)
        js(clip / 'cosmos/cosmos3.json', dict(spec, aspect_ratio='16,9', seed=7))
        put(clip / 'cosmos/prompt.json', b'{"caption": "x"}')
        request = run_cosmos.sim_transfer_request(run_cosmos.collect_specs(clip.parent, '3')[0], {'num_steps': 4}, 'edge')
        self.assertEqual((request['num_frames'], request['max_frames'], request['num_video_frames_per_chunk']), (93, 93, 93))
        self.assertEqual(request['vision_path'], str(clip / 'cosmos/rgb.mp4'))
        self.assertNotIn('depth', request)
        depth = run_cosmos.sim_transfer_request(clip / 'cosmos/cosmos3.json', {}, 'depth')
        self.assertEqual(depth['depth'], {'control_path': str(clip / 'cosmos/depth.mp4')})
        # Holdout-Clips von record_scenes.py: eine Ebene tiefer ignoriert, direkt im Clip-Ordner abgelehnt
        holdout = self.root / 'clips' / 'holdout' / 'test-clip'
        js(holdout / 'clip.json', {'frames': 82, 'fps': 30, 'holdout': True})
        self.assertEqual(run_cosmos.collect_specs(clip.parent, '3'), [str(clip / 'cosmos/cosmos3.json')])
        with self.assertRaisesRegex(ValueError, 'Holdout-Clip'):
            run_cosmos.collect_specs(holdout.parent, '3')
        spec['num_frames'] = 82
        js(clip / 'cosmos/cosmos3.json', spec)
        with self.assertRaisesRegex(ValueError, 'Kontrollvideos'):
            run_cosmos.collect_specs(clip.parent, '3')

    def test_subprocess_failure_is_propagated_with_logs(self):
        output = self.root / 'failed'
        with self.assertRaisesRegex(RuntimeError, 'Exitcode 7'):
            common.execute([sys.executable, '-c', 'print("failure"); raise SystemExit(7)'],
                           self.root, os.environ.copy(), output)
        self.assertIn('failure', (output / 'run.log').read_text())

    def test_training_and_evaluation_preflight_keep_dataset_paths_separate(self):
        experiment = put(self.root / 'repo/exp.py', b'class HeightFormerLightningModel: pass\n')
        for path in ('dataset/nusc_mv_det_dataset.py', 'evaluators/result2kitti.py'):
            put(self.root / 'repo' / path)
        backbone = put(self.root / 'backbone.pth')
        train = self.prepared(self.root / 'datasets/carla-hd')
        real = self.prepared(self.root / 'real-prepared', synthetic=False)
        extra = f'''[heightformer]
repo={json.dumps(str(experiment.parent))}
experiment={json.dumps(str(experiment))}
model_class="HeightFormerLightningModel"
backbone_checkpoint={json.dumps(str(backbone))}
epochs=1
[dair]
root={json.dumps(real['raw_root'])}
split={json.dumps(real['split'])}
prepared={json.dumps(str(self.root / 'real-prepared'))}
[detector_runtime]
gpus=["0"]
batch_size=1
workers=0
precision=32
'''
        config = self.config(extra)
        common_args = ['--config', str(config.path), '--model', 'heightformer', '--check-only']

        def stage(*argv):
            run_detector.run(run_detector.parse(argv))
        with patch.object(run_detector, 'probe'), patch.object(run_detector, 'execute') as run, \
                patch.object(run_detector, 'detector_env', lambda repo, env: ([sys.executable], env)):
            # Real preflight, but no heavy environment or GPU subprocess.
            stage('train', *common_args, '--name', 'carla', '--data', 'carla')
            run.assert_not_called()
            checkpoint = put(self.root / 'trained.ckpt')
            stage('evaluate', *common_args, '--name', 'real', '--checkpoint', str(checkpoint))
            run.assert_not_called()
            js(self.root / 'real-prepared/prepared.json', dict(real, synthetic=True))
            with self.assertRaisesRegex(ValueError, 'echten DAIR'):
                stage('evaluate', *common_args, '--name', 'bad', '--checkpoint', str(checkpoint))
            # BEVHeight++ is the default detector
            with self.assertRaisesRegex(ValueError, r'\[bevheight\+\+\]'):
                stage('train', '--config', str(config.path), '--name', 'x', '--data', 'carla', '--check-only')
        self.assertNotEqual(train['train_info'], real['train_info'])
        self.assertFalse((self.root / 'runs').exists())

    def test_missing_dataset_is_built_but_aborted_build_is_not_reused(self):
        self.assertIsNone(run_detector.existing_dataset(self.root / 'datasets/carla-hd'))
        (self.root / 'datasets/carla-hd/raw').mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, 'prepared.json'):
            run_detector.existing_dataset(self.root / 'datasets/carla-hd')

    def test_loader_adapter_uses_manifest_not_authors_hardcoded_paths(self):
        captured = []
        class Dataset:
            def __init__(self, **kwargs):
                captured.append(kwargs)
            def __len__(self):
                return 2
        source = NS(NuscMVDetDataset=Dataset, collate_fn=lambda *args, **kwargs: None)
        fake_torch = NS(utils=NS(data=NS(DataLoader=lambda *args, **kwargs: kwargs)))
        class Model(PipelineMixin):
            pass
        model = Model()
        for key, value in {'ida_aug_conf': {'H': 1080, 'W': 1920, 'final_dim': (864, 1536)}, 'class_names': [], 'img_conf': {}, 'num_sweeps': 1,
                           'sweep_idxes': [], 'key_idxes': [], 'batch_size_per_device': 1}.items():
            setattr(model, key, value)
        model._pipeline_request = {'train_dataset': {'raw_root': '/synthetic', 'train_info': '/synthetic/train.pkl',
                                                     'image_size': [1280, 720]},
                                   'val_dataset': {'raw_root': '/real', 'val_info': '/real/val.pkl',
                                                   'image_size': [1920, 1080]},
                                   'runtime': {'workers': 0}}
        package = NS(nusc_mv_det_dataset=source)
        with patch.dict(sys.modules, {'torch': fake_torch, 'dataset': package, 'dataset.nusc_mv_det_dataset': source}):
            training = model.train_dataloader()
            testing = model.test_dataloader()
        self.assertEqual(captured[0]['info_path'], '/synthetic/train.pkl')
        self.assertEqual(captured[1]['data_root'], '/real')
        self.assertFalse(captured[1]['is_train'])
        self.assertEqual([(c['ida_aug_conf']['W'], c['ida_aug_conf']['H']) for c in captured], [(1280, 720), (1920, 1080)])
        self.assertEqual(model.ida_aug_conf['H'], 1080)
        self.assertTrue(training['shuffle'])
        self.assertFalse(testing['drop_last'])


if __name__ == '__main__':
    unittest.main()
