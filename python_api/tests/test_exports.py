"""Portable Exporttests ohne CARLA-Wheel: python -m unittest discover -s tests -p test_exports.py."""

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import cv2
import numpy as np

import record_scenes as rs
from simapi import convert, cosmos, dair, dataset
from simapi.camera import Rig
from simapi.clips import Recorder
from simapi.config import RESOLUTIONS, build_parser
from simapi.geometry import intrinsics, matrix_rotation, ue_matrix, world_from_cv


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(value, f)


def fixture(root, n=82):
    """Analytische Strassenebene mit je einem Auto, Radfahrer und Fussgaenger."""
    size = (192, 108)
    camera = {'width': size[0], 'height': size[1], 'fov': 90,
              'intrinsics': intrinsics(*size, 90).tolist(),
              'transform': {'location': {'x': 0, 'y': 0, 'z': 12},
                            'rotation': {'pitch': -30, 'yaw': 0, 'roll': 0}}}
    meta = {'map': 'test', 'fps': 30, 'frames': n, 'camera': camera,
            'frame_ids': list(range(1000, 1000 + n)), 'labels': 'frames/%04d.json',
            'images': {k: 'frames/%s/%%04d.png' % k for k in ('rgb', 'depth', 'instance')}}
    for k in meta['images']:
        os.makedirs(os.path.join(root, 'frames', k))
    write_json(os.path.join(root, 'clip.json'), meta)
    T, K = world_from_cv(camera), np.array(camera['intrinsics'])
    v, u = np.mgrid[:size[1], :size[0]]
    rays = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones(u.shape)], -1)
    with np.errstate(divide='ignore'):
        distance = -T[2, 3] / (rays @ T[2, :3])
    road = (distance > 0) & (distance < 999)
    distance = np.where(road, distance, 1000)
    tags, ids = np.where(road, 1, 11), np.zeros(u.shape, int)
    objects = []
    for j, (kind, tag) in enumerate((('car', 14), ('bicycle', 19), ('pedestrian', 12))):
        sl = np.s_[55:62, 70 + j * 15:77 + j * 15]
        tags[sl], ids[sl], distance[sl] = tag, 300 + j, 20
        objects.append({'id': 300 + j, 'class': kind,
                        'bbox_3d': {'center': {'x': 20, 'y': (j - 1) * 3, 'z': 0.75},
                                    'extent': {'x': 2, 'y': 1, 'z': 0.75},
                                    'rotation': {'pitch': 0, 'yaw': 30, 'roll': 0}}})
    encoded = np.round(distance / 1000 * (256 ** 3 - 1)).astype(np.int64)
    images = {'depth': np.stack([(encoded >> 16) & 255, (encoded >> 8) & 255, encoded & 255], -1).astype('uint8'),
              'instance': np.stack([ids >> 8, ids & 255, tags], -1).astype('uint8'),
              'rgb': np.full((size[1], size[0], 3), 100, np.uint8)}
    for i in range(n):
        for k, im in images.items():
            assert cv2.imwrite(os.path.join(root, meta['images'][k] % i), im)
        write_json(os.path.join(root, meta['labels'] % i), {'objects': objects})
    return meta


class ExportTests(unittest.TestCase):
    def test_whole_clip_plan(self):
        names = ['p%d' % i for i in range(8)]
        plan = rs.distribute(20000, names)
        self.assertEqual(plan, dict.fromkeys(names, [93] * 27))  # 2511 je Position, 20088 insgesamt
        for n in (1, 93, 94, 999, 20001):
            total = sum(map(sum, rs.distribute(n, names).values()))
            self.assertTrue(n <= total < n + 93)
        with self.assertRaises(ValueError):
            rs.distribute(20000, ['same', 'same'])

    def test_running_carla_needs_use_running(self):
        running = object()
        with patch.object(rs, 'connect', return_value=running):
            with self.assertRaises(SystemExit):
                rs.start_carla(NS(host='127.0.0.1', port=2000, use_running=False), None, [])
            self.assertIs(rs.start_carla(NS(host='127.0.0.1', port=2000, use_running=True), None, []), running)

    @unittest.skipIf(os.name == 'nt', 'Windows beendet per taskkill /F')
    def test_stop_carla_kills_hanging_group(self):
        import subprocess
        # ignoriert SIGTERM wie ein haengender Server, mit Kindprozess wie CarlaUE4.sh
        proc = subprocess.Popen(['sh', '-c', 'trap "" TERM; sleep 60 & wait'], start_new_session=True)
        with patch.object(rs, 'CARLA_STOP_TIMEOUT', 0.5):
            rs.stop_carla(proc, 2000)
        self.assertIsNotNone(proc.wait(5))
        for _ in range(50):  # auch der Kindprozess ist weg (sobald init ihn abgeraeumt hat)
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            self.fail('Kindprozess laeuft noch')

    def test_camera_resolution(self):
        attributes = []
        class BP:
            def set_attribute(self, key, value):
                attributes.append((key, value))
            def has_attribute(self, key):
                return False
        actors = []
        def spawn(bp, transform):
            actor = NS(listen=lambda callback: None, stop=lambda: None, destroy=lambda: None)
            actors.append(actor)
            return actor
        world = NS(get_blueprint_library=lambda: NS(find=lambda key: BP()), spawn_actor=spawn)
        pose = NS(location=NS(x=0, y=0, z=10), rotation=NS(pitch=-30, yaw=0, roll=0))
        rig = Rig(world)
        for quality, expected in [('fullhd', (1920, 1080)), ('hd', (1280, 720))]:
            cfg = build_parser().parse_args(['--quality', quality])
            rig.update({'rgb', 'depth', 'instance'}, pose, RESOLUTIONS[cfg.quality])
            self.assertEqual(attributes[-3:], [('image_size_x', str(expected[0])),
                                               ('image_size_y', str(expected[1])), ('fov', '90.0')])
        self.assertEqual(len(actors), 6)
        self.assertEqual(build_parser().parse_args([]).quality, 'fullhd')

    def test_session_stops_at_budget(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = NS(export=False, clip_frames=93)
            conversions = NS(add=lambda clip: None, poll=lambda: None, pending=lambda: 0, failed=[])
            session = rs.Session({'a': [93, 7], 'b': [93, 6]}, cfg, NS(export=True, clip_frames=93), conversions, rs.Display())
            session.show = lambda *args: None
            class FakeSim:
                recorder = NS(clip=None)
                def apply(self, cfg):
                    self.cfg = cfg
                def goto_camera(self, name):
                    pass
                def advance(self, frame):
                    if self.cfg.export:
                        c = self.recorder.clip or {'n': 0}
                        c['n'] += 1
                        self.recorder.clip = c
                        if c['n'] == self.cfg.clip_frames:
                            path = os.path.join(root, str(frame))
                            os.makedirs(path)
                            write_json(os.path.join(path, 'clip.json'), {'frames': c['n']})
                            self.recorder.clip = None
                            session.on_clip(path)
                    return frame + 1
            sim, frame = FakeSim(), 0
            with patch.object(rs, 'view_problem', return_value=None), patch.object(rs, 'SETTLE_SECONDS', 0):
                frame = session.record(sim, 'a', 1, frame)
                session.record(sim, 'b', 2, frame)
            self.assertEqual(dict(session.frames_done), {'a': 100, 'b': 99})
            self.assertIsNone(sim.recorder.clip)
            # Sensoren liefern immer wieder unvollstaendige Clips: begrenzt abbrechen.
            broken = rs.Session({'a': [93]}, cfg, NS(export=True, clip_frames=93), conversions, rs.Display())
            broken.show = lambda *args: None
            class BrokenSim(FakeSim):
                def advance(self, frame):
                    if self.cfg.export:
                        self.recorder.clip = {'n': frame % 10}
                    return frame + 1
            with patch.object(rs, 'SETTLE_SECONDS', 0), self.assertRaisesRegex(RuntimeError, 'Kein vollstaendiger Clip'):
                broken.record(BrokenSim(), 'a', 1, 0)

    def test_matrix_rotation_composes_local_box_rotation(self):
        zero = dict.fromkeys('xyz', 0)
        for angles in ((-30, 47, 3), (90, 30, 20), (-90, -70, 12)):
            R = ue_matrix(zero, dict(zip(('pitch', 'yaw', 'roll'), angles)))[:3, :3]
            R = R @ ue_matrix(zero, {'pitch': 3, 'yaw': 21, 'roll': -4})[:3, :3]
            np.testing.assert_allclose(ue_matrix(zero, matrix_rotation(R))[:3, :3], R, atol=1e-7)

    def test_recorder_real_png_sizes_and_sensor_frame_check(self):
        for w, h in ((1920, 1080), (1280, 720)):
            with self.subTest(size=(w, h)), tempfile.TemporaryDirectory() as root:
                pose = NS(location=NS(x=0, y=0, z=12), rotation=NS(pitch=-30, yaw=0, roll=0))
                cam = NS(transform=pose, w=w, h=h, fov=90, K=intrinsics(w, h, 90))
                data = np.zeros((h, w, 4), np.uint8).tobytes()
                images = {k: NS(width=w, height=h, frame=100, raw_data=data) for k in ('rgb', 'depth', 'instance')}
                snap = NS(frame=100, timestamp=NS(elapsed_seconds=1))
                rec = Recorder(root, 'test', 30)
                try:
                    path = rec.add(snap, [], images, cam, 1)
                    self.assertIsNotNone(path)
                    for k in images:
                        self.assertEqual(cv2.imread(os.path.join(path, 'frames', k, '0000.png')).shape, (h, w, 3))
                    images['depth'].frame = 101
                    self.assertIsNone(rec.add(snap, [], images, cam, 1))
                    self.assertIsNone(rec.clip)
                finally:
                    rec.close()

    def test_short_clip_videos_labels_and_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            clip = os.path.join(root, 'clip')
            meta = fixture(clip)
            out = os.path.join(clip, 'cosmos')
            self.assertEqual(convert.main([clip, '--size', '96x54']), 0)
            with open(os.path.join(out, 'manifest.json')) as f:
                manifest = json.load(f)
            self.assertEqual((manifest['source_frames'], manifest['video_frames'], manifest['padding_frames']), (82, 93, 11))
            for stream in cosmos.STREAMS:
                frames, size = dataset.frames_of(os.path.join(out, stream + '.mp4'), {0, 81, 82, 92}, 93, 30)
                self.assertEqual(size, (96, 54))
                np.testing.assert_array_equal(frames[81], frames[92])
            labels_out = os.path.join(clip, 'dair')
            with open(os.path.join(labels_out, 'meta.json')) as f:
                info = json.load(f)
            self.assertEqual(info['frames'], list(range(82)))
            self.assertEqual(info['objects'], 82 * 3)
            self.assertAlmostEqual(info['camera_height'], 12, places=2)
            with open(os.path.join(labels_out, 'label', '0081.json')) as f:
                labels = json.load(f)
            self.assertEqual([lb['type'] for lb in labels], ['Car', 'Cyclist', 'Pedestrian'])
            self.assertAlmostEqual(labels[0]['rotation'], -np.pi / 6, places=3)
            self.assertAlmostEqual(labels[0]['3d_location']['y'], 3, places=3)
            ds = dataset.build([clip], os.path.join(root, 'dataset'), val=0)
            self.assertEqual(len(ds), 82)  # keine Padding-Duplikate im Training
            self.assertEqual(ds[-1]['source']['frame'], 81)
            self.assertEqual(cv2.imread(os.path.join(root, 'dataset', ds[-1]['image_path'])).shape, (720, 1280, 3))
            # dair/ ist fuer 1920 x 1080 kalibriert; HD skaliert Pixelkanten um 2/3 (Pixelmitten-Konvention)
            with open(os.path.join(labels_out, 'calib', 'camera_intrinsic.json')) as f:
                full = np.reshape(json.load(f)['cam_K'], (3, 3))
            with open(os.path.join(root, 'dataset', ds[-1]['calib_camera_intrinsic_path'])) as f:
                hd = np.reshape(json.load(f)['cam_K'], (3, 3))
            np.testing.assert_allclose(hd[:2, :2], full[:2, :2] * 2 / 3, atol=1e-5)
            np.testing.assert_allclose(hd[:2, 2], (full[:2, 2] + 0.5) * 2 / 3 - 0.5, atol=1e-5)
            with open(os.path.join(root, 'dataset', ds[-1]['label_camera_std_path'])) as f:
                self.assertAlmostEqual(json.load(f)[0]['2d_box']['xmax'], round(labels[0]['2d_box']['xmax'] * (2 / 3), 2))
            with self.assertRaisesRegex(ValueError, 'Frames'):
                dataset.frames_of(os.path.join(out, 'rgb.mp4'), {0}, 82, 30)
            with self.assertRaisesRegex(ValueError, 'fps'):
                dataset.frames_of(os.path.join(out, 'rgb.mp4'), {0}, 93, 16)
            self.assertEqual(convert.main([clip, '--drop-frames']), 0)
            self.assertFalse(os.path.exists(os.path.join(clip, 'frames')))

    def test_missing_labels_fail(self):
        with tempfile.TemporaryDirectory() as root:
            meta = fixture(root, n=1)
            write_json(os.path.join(root, meta['labels'] % 0), {'objects': []})
            out = os.path.join(root, 'dair')
            os.makedirs(out)
            with self.assertRaisesRegex(ValueError, 'ohne 3D-Label'):
                dair.build(root, meta, out)


if __name__ == '__main__':
    unittest.main()
