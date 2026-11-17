"""3D-Labels im Rohformat von DAIR-V2X-I fuer BEVHeight, HeightFormer und CoBEV (ohne CARLA testbar).

Pro Clip entsteht <clip>/dair/ fuer jeden every-ten Frame, schon in der Bildgroesse von
DAIR-V2X-I (1920 x 1080; die Skripte und Configs der Detektoren setzen sie fest voraus):

  meta.json                           Frames, Kamerahoehe, Bodenebene, verdeckte und unbeschriftete Objekte
  calib/camera_intrinsic.json         {"cam_K": 3x3 zeilenweise, "cam_D": 5 x 0}
  calib/virtuallidar_to_camera.json   {"rotation": 3x3, "translation": 3x1}
  label/0000.json                     wie DAIR label/camera: Typ, 3D-Box, 2D-Box, Verdeckung, Abschneidung
  velodyne/0000.pcd                   Punktwolke aus dem Tiefenbild (x y z intensity, binaer)

Bezugssystem ist bei DAIR ein "virtuelles LiDAR", dessen Ebene z = 0 die Fahrbahn ist: daraus
leiten BEVHeight und CoBEV die Hoehe ueber Grund ab. Hier: Ursprung auf der Fahrbahn senkrecht
unter der Kamera, z senkrecht zur Fahrbahn nach oben, x in Blickrichtung, y nach links. Die
Fahrbahnebene wird aus den Strassenpixeln von Tiefen- und Segmentierungsbild geschaetzt.

Ein Label bekommt nur, was die Kamera wirklich sieht und nah genug ist: Boxmitte hoechstens
MAX_DISTANCE von der Kamera, mindestens MIN_PIXELS eigene Pixel (auf 1920 x 1080 umgerechnet) und
mindestens MIN_SHARE der Box sichtbar, also im Bild und nicht verdeckt (View.visibility).
Aufgezeichnet wird jedes Objekt ab einem Pixel; die ohne Label zaehlt meta.json unter "hidden".
Sie gelten nicht als unbeschriftet.

3D-Box wie DAIR: 3d_location = Mittelpunkt, 3d_dimensions h/w/l, rotation = Gierwinkel um z
(0 = Fahrzeug zeigt entlang x). 2D-Box: eng um die sichtbaren Pixel. occluded_state (0 frei,
1 teilweise, 2 ueberwiegend verdeckt) und truncated_state (0 im Bild, 1 links/rechts, 2 oben/unten
abgeschnitten) bestimmen nur die Schwierigkeitsstufen der Auswertung.
"""

import collections
import json
import math
import os

import cv2
import numpy as np

from .geometry import (FLIP, OBJECT_TAGS, backproject, decode_instance, depth_m, fit_plane, ground_frame,
                       project_box, to_camera, transform, ue_matrix, world_from_cv)

SIZE = (1920, 1080)
# CARLA base_type -> DAIR-Typ; BEVHeight fasst spaeter Van/Truck/Bus zu Car und Motorcyclist zu Cyclist zusammen
TYPES = {'car': 'Car', 'van': 'Van', 'truck': 'Truck', 'bus': 'Bus',
         'motorcycle': 'Motorcyclist', 'bicycle': 'Cyclist', 'pedestrian': 'Pedestrian'}
DEFAULT_TYPE = 'Car'  # Fahrzeuge ohne base_type
ROAD = (1, 24)  # CARLA-Tags: Roads, RoadLines
SKY = 11
POINT_STEP = 4         # jeder 4. Pixel in jeder Richtung: etwa 130000 Punkte bei 1920 x 1080
MAX_RANGE = 200.0      # m, weiter weg keine Punkte
NEAR = 0.1             # m vor der Kamera
OCCLUDER_MARGIN = 0.5  # m: naeher als die Box minus dieser Abstand = verdeckt sie
UNLABELED_MIN = 1      # jedes sichtbare Objekt braucht ein Label
# Wann ein Objekt ein Label bekommt. Darunter (ein paar Pixel durchs Laub, ein Dachrand hinter
# einem Lkw, ein Rest am Bildrand) koennte es kein Detektor finden; weiter weg zaehlt es nicht.
MIN_PIXELS = 10        # eigene Pixel, auf 1920 x 1080 umgerechnet
MIN_SHARE = 0.05       # sichtbarer Anteil der Box: Anteil im Bild x unverdeckter Anteil
MAX_DISTANCE = 150.0   # m von der Kamera bis zur Boxmitte, 0 = unbegrenzt
Limits = collections.namedtuple('Limits', 'min_pixels min_share max_distance',
                                defaults=(MIN_PIXELS, MIN_SHARE, MAX_DISTANCE))
LIMITS = Limits()


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


def world_vertices(b):
    """8 Ecken einer Box aus frames/NNNN.json (Welt, UE-Achsen), mit voller Rotation."""
    signs = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], float)
    rot = ue_matrix({'x': 0, 'y': 0, 'z': 0}, b['rotation'])[:3, :3]
    return _vec(b['center']) + (signs * _vec(b['extent'])) @ rot.T


class Pixels:
    """Fahrzeug- und Personenpixel eines Frames, nach 16-Bit-Actor-ID gruppiert.

    Nur Objektklassen zaehlen: Kartenobjekte ohne Actor haben eigene IDs, die zufaellig zu einer
    16-Bit-Actor-ID passen koennen.
    """

    def __init__(self, ids, tags):
        self.ids = ids
        self.objects = np.isin(tags, OBJECT_TAGS)
        ys, xs = np.nonzero(self.objects)
        own = ids[ys, xs]
        order = np.argsort(own, kind='stable')
        self.ys, self.xs, self.own = ys[order], xs[order], own[order]

    def of(self, actor_id):
        """(ys, xs) der Pixel dieses Actors."""
        id16 = actor_id & 0xFFFF
        lo, hi = np.searchsorted(self.own, id16, 'left'), np.searchsorted(self.own, id16, 'right')
        return self.ys[lo:hi], self.xs[lo:hi]

    def mask(self, actor_id, roi):
        """Maske der Pixel dieses Actors im Ausschnitt roi."""
        return (self.ids[roi] == (actor_id & 0xFFFF)) & self.objects[roi]


class View:
    """Aufnahmekamera aus clip.json: wie gut ist ein aufgezeichnetes Objekt zu sehen?"""

    def __init__(self, camera):
        self.w, self.h = camera['width'], camera['height']
        self.K = np.array(camera['intrinsics'])  # Aufnahmegroesse, Pixelkanten-Konvention
        t = camera['transform']
        self.origin = _vec(t['location'])
        self.w2c = np.linalg.inv(ue_matrix(t['location'], t['rotation']))
        self.rect = np.array([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]], np.float32)

    def visibility(self, obj, px, depth):
        """({'distance', 'pixels', 'in_frame', 'unoccluded', 'share'}, (ys, xs) seiner Pixel) eines Objekts; px: Pixels.

        distance: Kamera bis Boxmitte in m.
        pixels: Fahrzeug-/Personenpixel mit seiner Actor-ID. in_frame: Anteil der projizierten Box im Bild.
        unoccluded: sichtbare / (sichtbare + verdeckende) Pixel im Bildteil der Box; verdeckend ist ein
        fremdes Pixel, das mehr als OCCLUDER_MARGIN naeher ist als die naechste Boxecke.
        share = in_frame * unoccluded: sichtbarer Anteil des ganzen Objekts.
        """
        own = px.of(obj['id'])
        pixels = len(own[0])
        distance = float(np.linalg.norm(_vec(obj['bbox_3d']['center']) - self.origin))
        vis = {'distance': distance, 'pixels': pixels, 'in_frame': 0.0, 'unoccluded': 0.0, 'share': 0.0}
        verts = world_vertices(obj['bbox_3d'])
        proj = project_box(verts, self.w2c, self.K, self.w, self.h)  # an der Nahebene geschnitten
        if not pixels or proj is None:
            return vis, own
        hull = cv2.convexHull(np.clip(proj[0].reshape(-1, 2), -1e6, 1e6).astype(np.float32))
        area = cv2.contourArea(hull)
        in_frame = min(1.0, cv2.intersectConvexConvex(hull, self.rect)[0] / area) if area > 0 else 1.0

        # Pixel, deren Mitte in der Box liegt, im Bildausschnitt um die Box
        pts = hull.reshape(-1, 2)
        x0, y0 = (max(0, int(math.floor(v))) for v in pts.min(0))
        x1, y1 = min(self.w, int(math.ceil(pts[:, 0].max()))), min(self.h, int(math.ceil(pts[:, 1].max())))
        occluders = 0
        if x1 > x0 and y1 > y0:
            inside = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(inside, np.round((pts - (x0 + 0.5, y0 + 0.5)) * 16).astype(np.int32), 1, shift=4)
            nearest = max(NEAR, to_camera(verts, self.w2c)[:, 2].min())
            roi = (slice(y0, y1), slice(x0, x1))
            occluders = int(np.count_nonzero(inside.astype(bool) & ~px.mask(obj['id'], roi) &
                                             (depth[roi] < nearest - OCCLUDER_MARGIN)))
        unoccluded = pixels / (pixels + occluders)
        vis.update(in_frame=in_frame, unoccluded=unoccluded, share=in_frame * unoccluded)
        return vis, own

    def labeled(self, vis, limits=LIMITS):
        """Label ja/nein (limits: Limits): nah genug, genug Pixel (auf 1920 x 1080 umgerechnet) und sichtbarer Anteil."""
        fullhd = vis['pixels'] * SIZE[0] * SIZE[1] / (self.w * self.h)
        near = not limits.max_distance or vis['distance'] <= limits.max_distance
        return near and vis['pixels'] > 0 and fullhd >= limits.min_pixels and vis['share'] >= limits.min_share


def visible_objects(clip_dir, meta, i, limits=LIMITS):
    """Aufgezeichnete Objekte von Frame i, die ein Label bekommen (fuer die Objektzahlen im Cosmos-Prompt)."""
    images, label_path = frame_paths(clip_dir, meta, i)
    tags, ids = decode_instance(_read(images['instance']))
    depth = depth_m(_read(images['depth']))
    with open(label_path, encoding='utf-8') as f:
        objs = json.load(f)['objects']
    view, px = View(meta['camera']), Pixels(ids, tags)
    return [o for o in objs if view.labeled(view.visibility(o, px, depth)[0], limits)]


class Frame:
    """Umrechnung fuer einen Clip: Bodensystem <-> Kamera, Kalibrierung in DAIR-Bildgroesse."""

    def __init__(self, meta, normal, point):
        cam = meta['camera']
        self.view = View(cam)
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

    def label(self, obj, px, depth, limits=LIMITS):
        """DAIR-Label eines Objekts (px: Pixels des Frames), None wenn es zu weit weg oder zu wenig sichtbar
        ist (View.labeled)."""
        vis, (ys, xs) = self.view.visibility(obj, px, depth)
        if not self.view.labeled(vis, limits):
            return None
        center, l, w, h, yaw = self.box(obj)
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)])
        corners = transform(self.cv_ground, center + local @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]))

        # Abschneidung in Aufnahmegroesse; Verdeckung im Bildteil der Box
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
        occluded = 0 if vis['unoccluded'] >= 0.9 else 1 if vis['unoccluded'] >= 0.5 else 2

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
    """Objekte im Bild (Fahrzeug- oder Personen-Pixel) ohne Actor in der Aufzeichnung, also ohne
    3D-Box, mit mindestens UNLABELED_MIN Pixeln. `labeled`: IDs aller aufgezeichneten Objekte."""
    m = np.isin(tags, OBJECT_TAGS) & ~np.isin(ids, [i & 0xFFFF for i in labeled])
    _, counts = np.unique(ids[m], return_counts=True)
    return int(np.count_nonzero(counts >= UNLABELED_MIN))


def build(clip_dir, meta, out, every=1, allow_unlabeled=False, limits=LIMITS, classes=None):
    """<clip>/dair/-Inhalt nach out schreiben (Ordner existiert und ist leer).

    classes: Dict oder None; traegt pro bearbeitetem Frame die Klassen der Objekte mit Label ein (dieselben wie
    visible_objects), damit der Cosmos-Prompt sie zaehlen kann, ohne die Sichtbarkeit noch einmal zu rechnen.
    """
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

    frames, objects, missing, hidden = list(range(0, meta['frames'], every)), 0, {}, {}
    for i in frames:
        images, label_path = frame_paths(clip_dir, meta, i)
        depth = depth_m(_read(images['depth']))
        tags, ids = decode_instance(_read(images['instance']))
        with open(label_path, encoding='utf-8') as f:
            objs = json.load(f)['objects']
        px = Pixels(ids, tags)
        labels, kept = [], []
        for o in objs:
            lb = fr.label(o, px, depth, limits)
            if lb is not None:
                labels.append(lb)
                kept.append(o['class'])
        if classes is not None:
            classes[i] = kept
        objects += len(labels)
        if len(objs) > len(labels):
            hidden['%04d' % i] = len(objs) - len(labels)
        n = unlabeled(tags, ids, [o['id'] for o in objs])  # die ohne Label haben einen Actor, zaehlen nicht
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
        'visibility': dict(limits._asdict()),  # Schwellen fuer ein Label
        'hidden': hidden,      # Frame -> aufgezeichnete Objekte ohne Label: zu weit weg oder zu wenig sichtbar
        'unlabeled': missing,  # Frame -> Fahrzeuge/Personen im Bild ohne Actor (ohne 3D-Box)
    }
    with open(os.path.join(out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1)
    if missing and not allow_unlabeled:
        raise ValueError('Fahrzeuge/Personen ohne 3D-Label: %s. Actors/Kameraposition pruefen; '
                         '--allow-unlabeled erlaubt einen unvollstaendigen Datensatz ausdruecklich.' % missing)
    return info
