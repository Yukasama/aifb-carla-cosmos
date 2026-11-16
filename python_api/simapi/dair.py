"""3D-Labels im Rohformat von DAIR-V2X-I fuer BEVHeight, HeightFormer und CoBEV (ohne CARLA testbar).

Pro Clip entsteht <clip>/dair/ fuer jeden every-ten Frame, schon in der Bildgroesse von
DAIR-V2X-I (1920 x 1080; die Skripte und Configs der Detektoren setzen sie fest voraus):

  meta.json                           Frames, Kamerahoehe, Bodenebene, unbeschriftete Objekte
  calib/camera_intrinsic.json         {"cam_K": 3x3 zeilenweise, "cam_D": 5 x 0}
  calib/virtuallidar_to_camera.json   {"rotation": 3x3, "translation": 3x1}
  label/0000.json                     wie DAIR label/camera: Typ, 3D-Box, 2D-Box, Verdeckung, Abschneidung
  velodyne/0000.pcd                   Punktwolke aus dem Tiefenbild (x y z intensity, binaer)

Bezugssystem ist bei DAIR ein "virtuelles LiDAR", dessen Ebene z = 0 die Fahrbahn ist: daraus
leiten BEVHeight und CoBEV die Hoehe ueber Grund ab. Hier: Ursprung auf der Fahrbahn senkrecht
unter der Kamera, z senkrecht zur Fahrbahn nach oben, x in Blickrichtung, y nach links. Die
Fahrbahnebene wird aus den Strassenpixeln von Tiefen- und Segmentierungsbild geschaetzt.

3D-Box wie DAIR: 3d_location = Mittelpunkt, 3d_dimensions h/w/l, rotation = Gierwinkel um z
(0 = Fahrzeug zeigt entlang x). 2D-Box: eng um die sichtbaren Pixel. occluded_state (0 frei,
1 teilweise, 2 ueberwiegend verdeckt) und truncated_state (0 im Bild, 1 links/rechts, 2 oben/unten
abgeschnitten) bestimmen nur die Schwierigkeitsstufen der Auswertung.
"""

import json
import math
import os

import cv2
import numpy as np

from .geometry import (FLIP, backproject, decode_instance, depth_m, fit_plane, ground_frame, transform,
                       ue_matrix, world_from_cv)

SIZE = (1920, 1080)
# CARLA base_type -> DAIR-Typ; BEVHeight fasst spaeter Van/Truck/Bus zu Car und Motorcyclist zu Cyclist zusammen
TYPES = {'car': 'Car', 'van': 'Van', 'truck': 'Truck', 'bus': 'Bus',
         'motorcycle': 'Motorcyclist', 'bicycle': 'Cyclist', 'pedestrian': 'Pedestrian'}
DEFAULT_TYPE = 'Car'  # Fahrzeuge ohne base_type
ROAD = (1, 24)  # CARLA-Tags: Roads, RoadLines
SKY = 11
OBJECT_TAGS = (12, 13, 14, 15, 16, 18, 19)  # Pedestrian, Rider, Car, Truck, Bus, Motorcycle, Bicycle
POINT_STEP = 4         # jeder 4. Pixel in jeder Richtung: etwa 130000 Punkte bei 1920 x 1080
MAX_RANGE = 200.0      # m, weiter weg keine Punkte
NEAR = 0.1             # m vor der Kamera
OCCLUDER_MARGIN = 0.5  # m: naeher als die Box minus dieser Abstand = verdeckt sie
UNLABELED_MIN = 1      # jedes sichtbare Objekt braucht ein Label


def _read(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _vec(d):
    return np.array([d['x'], d['y'], d['z']], float)


def _r(v, n=4):
    return round(float(v), n)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def frame_paths(clip_dir, meta, i):
    return {k: os.path.join(clip_dir, meta['images'][k] % i) for k in ('rgb', 'depth', 'instance')}, \
        os.path.join(clip_dir, meta['labels'] % i)


def fit_ground(meta, depth, tags):
    """Fahrbahnebene (rechtshaendige Welt) aus einem Frame: (Normale, Punkt)."""
    K = np.array(meta['camera']['intrinsics'])
    pts, (v, u) = backproject(depth, K, POINT_STEP)
    road = np.isin(tags[v, u], ROAD) & (pts[:, 2] < MAX_RANGE)
    return fit_plane(transform(world_from_cv(meta['camera']), pts[road]))


class Frame:
    """Umrechnung fuer einen Clip: Bodensystem <-> Kamera, Kalibrierung in DAIR-Bildgroesse."""

    def __init__(self, meta, normal, point):
        cam = meta['camera']
        self.w, self.h = cam['width'], cam['height']
        self.K = np.array(cam['intrinsics'])  # Aufnahmegroesse, Pixelkanten-Konvention
        world_cv = world_from_cv(cam)
        self.world_ground = ground_frame(world_cv, normal, point)
        self.ground_world = np.linalg.inv(self.world_ground)
        self.ground_cv = self.ground_world @ world_cv
        self.cv_ground = np.linalg.inv(self.ground_cv)
        self.scale = (SIZE[0] / self.w, SIZE[1] / self.h)

    def calib(self):
        K = np.diag([*self.scale, 1.0]) @ self.K
        K[:2, 2] -= 0.5  # Pixelmitten-Konvention wie OpenCV und KITTI
        return ({'cam_K': [_r(v, 6) for v in K.ravel()], 'cam_D': [0.0] * 5},
                {'rotation': [[_r(v, 8) for v in row] for row in self.cv_ground[:3, :3]],
                 'translation': [[_r(v, 6)] for v in self.cv_ground[:3, 3]]})

    def box(self, obj):
        """Weltbox aus frames/NNNN.json -> (Mittelpunkt, l, w, h, Gierwinkel) im Bodensystem."""
        b = obj['bbox_3d']
        center = transform(self.ground_world, FLIP @ _vec(b['center']))
        heading = self.ground_world[:3, :3] @ FLIP @ ue_matrix({'x': 0, 'y': 0, 'z': 0}, b['rotation'])[:3, 0]
        e = b['extent']
        return center, 2 * e['x'], 2 * e['y'], 2 * e['z'], math.atan2(heading[1], heading[0])

    def label(self, obj, ids, depth):
        """DAIR-Label eines Objekts, None wenn kein Pixel sichtbar ist."""
        mask = ids == (obj['id'] & 0xFFFF)
        visible = int(mask.sum())
        if not visible:
            return None
        center, l, w, h, yaw = self.box(obj)
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)])
        corners = transform(self.cv_ground, center + local @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]))

        # Abschneidung und Verdeckung in Aufnahmegroesse
        front = corners[:, 2] > NEAR
        uv = corners[front, :2] / corners[front, 2:] @ self.K[:2, :2].T + self.K[:2, 2]
        if not front.all():
            truncated = 1
        elif uv[:, 0].min() < 0 or uv[:, 0].max() > self.w:
            truncated = 1
        elif uv[:, 1].min() < 0 or uv[:, 1].max() > self.h:
            truncated = 2
        else:
            truncated = 0
        occluders = self._occluders(uv, ids, depth, obj['id'] & 0xFFFF, corners[:, 2].min()) if len(uv) >= 3 else 0
        share = visible / (visible + occluders)
        occluded = 0 if share >= 0.9 else 1 if share >= 0.5 else 2

        ys, xs = np.nonzero(mask)
        sx, sy = self.scale
        heading = self.cv_ground[:3, :3] @ np.array([c, s, 0.0])
        bottom = transform(self.cv_ground, center - np.array([0.0, 0.0, h / 2]))
        alpha = _wrap(math.atan2(-heading[2], heading[0]) - math.atan2(bottom[0], bottom[2]))
        return {
            'type': TYPES.get(obj['class'], DEFAULT_TYPE),
            'truncated_state': truncated,
            'occluded_state': occluded,
            'alpha': _r(alpha),
            '2d_box': {'xmin': _r(xs.min() * sx, 2), 'ymin': _r(ys.min() * sy, 2),
                       'xmax': _r((xs.max() + 1) * sx, 2), 'ymax': _r((ys.max() + 1) * sy, 2)},
            '3d_dimensions': {'h': _r(h), 'w': _r(w), 'l': _r(l)},
            '3d_location': {'x': _r(center[0]), 'y': _r(center[1]), 'z': _r(center[2])},
            'rotation': _r(_wrap(yaw)),
        }

    def _occluders(self, uv, ids, depth, id16, nearest):
        """Pixel in der projizierten Box, die etwas Naeheres zeigen als die Box."""
        x0, y0 = (max(0, int(math.floor(v))) for v in uv.min(0))
        x1, y1 = min(self.w, int(math.ceil(uv[:, 0].max()))), min(self.h, int(math.ceil(uv[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            return 0
        hull = np.zeros((y1 - y0, x1 - x0), np.uint8)
        pts = np.clip(np.round(uv - (x0, y0)), -1e6, 1e6).astype(np.int32)
        cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
        roi = (slice(y0, y1), slice(x0, x1))
        return int(np.count_nonzero(hull.astype(bool) & (ids[roi] != id16) & (depth[roi] < nearest - OCCLUDER_MARGIN)))

    def points(self, depth, tags):
        """Punktwolke (N,3) im Bodensystem, ohne Himmel."""
        pts, (v, u) = backproject(depth, self.K, POINT_STEP)
        keep = (tags[v, u] != SKY) & (pts[:, 2] > NEAR) & (pts[:, 2] < MAX_RANGE)
        return transform(self.ground_cv, pts[keep])


def write_pcd(path, points):
    """PCD v0.7, binaer, Felder x y z intensity (float32) wie die DAIR-Punktwolken; intensity = 0."""
    data = np.c_[points, np.zeros(len(points))].astype('<f4')
    n = len(data)
    header = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\n'
              'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\n'
              'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n' % (n, n))
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())


def read_pcd(path):
    """Gegenstueck zu write_pcd: (N,4) float32."""
    with open(path, 'rb') as f:
        n = None
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('POINTS'):
                n = int(line.split()[1])
            if line.startswith('DATA'):
                break
        return np.frombuffer(f.read(), '<f4').reshape(n, 4)


def unlabeled(tags, ids, labeled):
    """Objekte im Bild (Fahrzeug- oder Personen-Pixel) ohne Label, mit mindestens UNLABELED_MIN Pixeln."""
    m = np.isin(tags, OBJECT_TAGS) & ~np.isin(ids, [i & 0xFFFF for i in labeled])
    _, counts = np.unique(ids[m], return_counts=True)
    return int(np.count_nonzero(counts >= UNLABELED_MIN))


def build(clip_dir, meta, out, every=1, allow_unlabeled=False):
    """<clip>/dair/-Inhalt nach out schreiben (Ordner existiert und ist leer)."""
    for sub in ('calib', 'label', 'velodyne'):
        os.makedirs(os.path.join(out, sub))
    images, _ = frame_paths(clip_dir, meta, 0)
    tags0, _ = decode_instance(_read(images['instance']))
    normal, point = fit_ground(meta, depth_m(_read(images['depth'])), tags0)
    fr = Frame(meta, normal, point)

    intrinsic, extrinsic = fr.calib()
    for name, data in (('camera_intrinsic', intrinsic), ('virtuallidar_to_camera', extrinsic)):
        with open(os.path.join(out, 'calib', name + '.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)

    frames, objects, missing = list(range(0, meta['frames'], every)), 0, {}
    for i in frames:
        images, label_path = frame_paths(clip_dir, meta, i)
        depth = depth_m(_read(images['depth']))
        tags, ids = decode_instance(_read(images['instance']))
        with open(label_path, encoding='utf-8') as f:
            objs = json.load(f)['objects']
        labels = [lb for lb in (fr.label(o, ids, depth) for o in objs) if lb is not None]
        objects += len(labels)
        n = unlabeled(tags, ids, [o['id'] for o in objs])
        if n:
            missing['%04d' % i] = n
        with open(os.path.join(out, 'label', '%04d.json' % i), 'w', encoding='utf-8') as f:
            json.dump(labels, f)
        write_pcd(os.path.join(out, 'velodyne', '%04d.pcd' % i), fr.points(depth, tags))

    info = {
        'size': list(SIZE),
        'frames': frames,
        'objects': objects,
        'camera_height': _r(fr.ground_cv[2, 3], 3),
        'ground': {'normal': [_r(v, 6) for v in normal], 'point': [_r(v, 3) for v in point]},  # Welt, rechtshaendig
        'unlabeled': missing,  # Frame -> Fahrzeuge/Personen im Bild ohne Label
    }
    with open(os.path.join(out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1)
    if missing and not allow_unlabeled:
        raise ValueError('Fahrzeuge/Personen ohne 3D-Label: %s. Actors/Kameraposition pruefen; '
                         '--allow-unlabeled erlaubt einen unvollstaendigen Datensatz ausdruecklich.' % missing)
    return info
