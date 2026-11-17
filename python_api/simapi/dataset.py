"""Trainingsdaten fuer BEVHeight, HeightFormer und CoBEV: python -m simapi.dataset <clips|export-ordner> --out ORDNER

Baut aus umgewandelten Clips (mit dair/) einen Datensatz im Rohformat von DAIR-V2X-I. Die
Bilder kommen aus den Ergebnissen von Cosmos (--cosmos AUSGABEORDNER, mehrfach moeglich:
Transfer2.5 legt <clipname>.mp4 ab, Cosmos 3 <clipname>/vision.mp4) oder ohne --cosmos aus dem
CARLA-Kamerabild (cosmos/rgb.mp4), etwa als Vergleich.
Cosmos-Ergebnisse muessen vor dem Training auf geometrische Labeltreue kontrolliert werden.
Mehrere Ausgabeordner (etwa Transfer2.5 und Cosmos 3,
oder verschiedene Prompts) ergeben je Clip mehrere Bilder mit denselben Labels; sie landen
immer zusammen im Training oder in der Validierung.

  <out>/image/000000.jpg                            1280 x 720, mit --fullhd 1920 x 1080
  <out>/velodyne/000000.pcd
  <out>/calib/camera_intrinsic/000000.json
  <out>/calib/virtuallidar_to_camera/000000.json
  <out>/label/camera/000000.json                    (label/virtuallidar/ gleich)
  <out>/data_info.json
  <out>/single-infrastructure-split-data.json       train/val, nach Kameraposition getrennt

Holdout-Clips von record_scenes.py (<export-dir>/holdout/, "holdout": true in clip.json) sind
Testdaten: Ein Export-Ordner findet sie nicht (nur eine Ebene tief), direkt angegeben brechen sie ab.

Kalibrierung und 2D-Boxen sind auf die Bildgroesse umgerechnet. Weiter wie bei DAIR-V2X-I: <out> als
data/dair-v2x-i ins Repo des Detektors legen, die Split-Datei nach data/ kopieren, dann
scripts/data_converter/dair2kitti.py und scripts/gen_info_dair.py. Die Experimente setzen
H = 1080, W = 1920 voraus; bei 1280 x 720 dort H = 720, W = 1280 eintragen (run_detector.py macht das selbst).
"""

import argparse
import collections
import json
import logging
import os
import random
import shutil
import sys

import cv2
import imageio_ffmpeg
import numpy as np

from .convert import find_clips
from .dair import SIZE as FULLHD

log = logging.getLogger('simapi.dataset')
SPLIT_FILE = 'single-infrastructure-split-data.json'
HD = (1280, 720)


def camera_position(meta):
    """Gleiche Kameraposition (auf 10 cm bzw. 0,1 Grad) = gleiche Gruppe beim Aufteilen."""
    t = meta['camera']['transform']
    return (meta['map'],) + tuple(round(t['location'][k], 1) for k in 'xyz') + \
        tuple(round(t['rotation'][k], 1) for k in ('pitch', 'yaw', 'roll'))


def split(groups, val, seed):
    """groups: Liste von ID-Listen, die zusammenbleiben. Ganze Gruppen in die Validierung, bis val erreicht ist."""
    groups = list(groups)
    random.Random(seed).shuffle(groups)
    total = sum(len(g) for g in groups)
    val_ids = []
    for g in groups:
        if len(val_ids) >= val * total or len(val_ids) + len(g) >= total:
            continue
        val_ids += g
    chosen = set(val_ids)
    return [i for g in groups for i in g if i not in chosen], val_ids


def frames_of(video, wanted, expected_frames=None, expected_fps=None):
    """Gewuenschte Frames (Index -> RGB) aus einem Video."""
    reader = imageio_ffmpeg.read_frames(video)
    info = next(reader)
    w, h = info['size']
    out, count = {}, 0
    try:
        if expected_fps is not None and abs(info['fps'] - expected_fps) > 0.01:
            raise ValueError('%s: %.3f fps statt %.3f; zeitliche Zuordnung der Labels ungueltig' %
                             (video, info['fps'], expected_fps))
        for i, buf in enumerate(reader):
            count = i + 1
            if i in wanted:
                out[i] = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        reader.close()
    if expected_frames is not None and count != expected_frames:
        raise ValueError('%s: %d statt %d Frames; zeitliche Zuordnung der Labels ungueltig' %
                         (video, count, expected_frames))
    if set(out) != wanted:
        raise ValueError('%s: beschriftete Frames fehlen' % video)
    return out, (w, h)


def _write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def cosmos_result(cosmos_dir, name):
    """Ergebnis von Cosmos fuer einen Clip: Transfer2.5 <name>.mp4, Cosmos 3 <name>/vision.mp4."""
    for path in (os.path.join(cosmos_dir, name + '.mp4'), os.path.join(cosmos_dir, name, 'vision.mp4')):
        if os.path.isfile(path):
            return path
    return None


def build(clips, out, cosmos_dirs=(), val=0.2, seed=0, quality=95, size=HD):
    if os.path.isdir(out) and os.listdir(out):
        raise FileExistsError('%s existiert und ist nicht leer' % out)
    dirs = ('image', 'velodyne', 'calib/camera_intrinsic', 'calib/virtuallidar_to_camera',
            'label/camera', 'label/virtuallidar')
    for d in dirs:
        os.makedirs(os.path.join(out, d), exist_ok=True)

    info, positions, types, skipped = [], collections.defaultdict(list), collections.Counter(), 0
    sx, sy = size[0] / FULLHD[0], size[1] / FULLHD[1]
    for clip in clips:
        name = os.path.basename(os.path.normpath(clip))
        meta_path = os.path.join(clip, 'dair', 'meta.json')
        if not os.path.isfile(meta_path):
            raise FileNotFoundError('%s fehlt; simapi.convert zuerst erfolgreich abschliessen' % meta_path)
        sources = [(cosmos_result(d, name), os.path.basename(os.path.normpath(d))) for d in cosmos_dirs]
        for d, (video, _) in zip(cosmos_dirs, sources):
            if video is None:
                raise FileNotFoundError('%s: kein Cosmos-Ergebnis in %s' % (name, d))
        sources = [s for s in sources if s[0] is not None]
        if not cosmos_dirs:
            sources.append((os.path.join(clip, 'cosmos', 'rgb.mp4'), 'carla'))
        with open(os.path.join(clip, 'clip.json'), encoding='utf-8') as f:
            meta = json.load(f)
        if meta.get('holdout'):
            raise ValueError('%s ist ein Holdout-Clip (Testdaten aus record_scenes.py) und gehoert in keinen '
                             'Trainingsdatensatz' % clip)
        with open(meta_path, encoding='utf-8') as f:
            labeled = json.load(f)['frames']
        cam = meta['camera']
        manifest_path = os.path.join(clip, 'cosmos', 'manifest.json')
        expected_frames = meta['frames']
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding='utf-8') as f:
                manifest = json.load(f)
            if manifest['source_frames'] != meta['frames']:
                raise ValueError('%s: Cosmos-Manifest passt nicht zur Aufnahme' % name)
            expected_frames = manifest['video_frames']
        calib_dir = os.path.join(clip, 'dair', 'calib')
        # dair/ ist fuer 1920 x 1080 kalibriert; Pixelmitten-Konvention wie simapi.dair
        with open(os.path.join(calib_dir, 'camera_intrinsic.json'), encoding='utf-8') as f:
            intrinsic = json.load(f)
        K = np.array(intrinsic['cam_K'], float).reshape(3, 3)
        K[:2] *= [[sx], [sy]]
        K[:2, 2] += [0.5 * sx - 0.5, 0.5 * sy - 0.5]
        intrinsic['cam_K'] = [round(float(v), 6) for v in K.ravel()]

        group = []
        for video, origin in sources:
            if not os.path.isfile(video):
                raise FileNotFoundError(video)
            images, (w, h) = frames_of(video, set(labeled), expected_frames, meta['fps'])
            # Cosmos-3-Sim-Transfer rendert per direct_resize das volle Sichtfeld auf 832x480 (bis 3 % gestreckt);
            # das Zurueckskalieren auf size stellt die Geometrie wieder her. Sonst muss es exakt passen.
            stretch = 0.03 if os.path.basename(video) == 'vision.mp4' else 0
            if abs(w * cam['height'] / (h * cam['width']) - 1) > stretch:
                raise ValueError('%s: %s ist %dx%d; Seitenverhaeltnis passt nicht zur Kalibrierung' %
                                 (name, video, w, h))
            for i in labeled:
                sid = '%06d' % len(info)
                interp = cv2.INTER_AREA if w > size[0] else cv2.INTER_CUBIC
                img = cv2.cvtColor(cv2.resize(images[i], size, interpolation=interp), cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(os.path.join(out, 'image', sid + '.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    raise OSError('Bild nicht geschrieben: %s' % sid)
                _write_json(os.path.join(out, 'calib', 'camera_intrinsic', sid + '.json'), intrinsic)
                shutil.copyfile(os.path.join(calib_dir, 'virtuallidar_to_camera.json'),
                                os.path.join(out, 'calib', 'virtuallidar_to_camera', sid + '.json'))
                with open(os.path.join(clip, 'dair', 'label', '%04d.json' % i), encoding='utf-8') as f:
                    labels = json.load(f)
                for lb in labels:
                    lb['2d_box'] = {k: round(v * (sx if k[0] == 'x' else sy), 2) for k, v in lb['2d_box'].items()}
                types.update(lb['type'] for lb in labels)
                for kind in ('camera', 'virtuallidar'):
                    _write_json(os.path.join(out, 'label', kind, sid + '.json'), labels)
                shutil.copyfile(os.path.join(clip, 'dair', 'velodyne', '%04d.pcd' % i),
                                os.path.join(out, 'velodyne', sid + '.pcd'))
                info.append({
                    'image_path': 'image/%s.jpg' % sid,
                    'pointcloud_path': 'velodyne/%s.pcd' % sid,
                    'calib_camera_intrinsic_path': 'calib/camera_intrinsic/%s.json' % sid,
                    'calib_virtuallidar_to_camera_path': 'calib/virtuallidar_to_camera/%s.json' % sid,
                    'label_camera_std_path': 'label/camera/%s.json' % sid,
                    'label_lidar_std_path': 'label/virtuallidar/%s.json' % sid,
                    'source': {'clip': name, 'frame': i, 'image': origin},
                })
                group.append(sid)
        if group:
            positions[camera_position(meta)].append(group)
        else:
            skipped += 1

    if not info:
        raise ValueError('keine Bilder: keine umgewandelten Clips%s gefunden' % (' mit Cosmos-Ergebnis' if cosmos_dirs else ''))
    # Bilder einer Kameraposition sehen fast gleich aus: nie auf Training und Validierung verteilen.
    # Nur eine Position: nach Clips trennen (zeitlich getrennt, aber gleiche Ansicht).
    if len(positions) > 1:
        groups = [sum(clips_, []) for clips_ in positions.values()]
    else:
        groups = next(iter(positions.values()))
        if val > 0:
            log.warning('Nur eine Kameraposition: Validierung zeigt dieselbe Ansicht wie das Training')
    train, val_ids = split(groups, val, seed)
    if val > 0 and not val_ids:
        log.warning('Nur ein Clip: keine Validierungsdaten')
    _write_json(os.path.join(out, 'data_info.json'), info)
    with open(os.path.join(out, SPLIT_FILE), 'w', encoding='utf-8') as f:
        json.dump({'train': sorted(train), 'val': sorted(val_ids), 'test': []}, f, indent=1)
    log.info('%d Bilder (%d Training, %d Validierung) aus %d Kamerapositionen nach %s; Objekte: %s',
             len(info), len(train), len(val_ids), len(positions), os.path.abspath(out),
             ', '.join('%s %d' % kv for kv in sorted(types.items())) or 'keine')
    if skipped:
        log.warning('%d Clips uebersprungen', skipped)
    return info


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m simapi.dataset', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('paths', nargs='+', metavar='ORDNER', help='Export-Ordner oder Clip-Ordner')
    p.add_argument('--out', required=True, help='Zielordner (neu oder leer), z.B. .../BEVHeight/data/dair-v2x-i')
    p.add_argument('--cosmos', metavar='ORDNER', action='append', default=[],
                   help='Ausgabeordner von Cosmos (Transfer2.5: <clipname>.mp4, Cosmos 3: <clipname>/vision.mp4), '
                        'mehrfach moeglich; ohne: CARLA-Bilder')
    p.add_argument('--fullhd', action='store_true', help='Bilder in 1920x1080 statt 1280x720')
    p.add_argument('--val', type=float, default=0.2, help='Anteil Validierung (Default 0.2)')
    p.add_argument('--seed', type=int, default=0, help='Zufall beim Aufteilen')
    p.add_argument('--jpeg-quality', type=int, default=95, choices=range(1, 101), metavar='1-100')
    args = p.parse_args(argv)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if not 0 <= args.val < 1:
        p.error('--val zwischen 0 und 1')
    try:
        build(list(find_clips(args.paths)), args.out, args.cosmos, args.val, args.seed, args.jpeg_quality,
              FULLHD if args.fullhd else HD)
    except (OSError, ValueError) as e:
        log.error('%s', e)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
