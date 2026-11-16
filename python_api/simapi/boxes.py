"""Bounding Boxes: filtern (Distanz, Bildausschnitt, Verdeckung), in der Welt zeichnen, aufnehmen."""

import json
import os
import time

import numpy as np

from .geometry import intrinsics, project_box, visible_pixels, ue_matrix, matrix_rotation

# Farbe pro Typ (Blueprint-Attribut 'base_type'; Fussgaenger -> 'pedestrian').
# Gedeckte Toene: CARLA rendert Debug-Linien selbstleuchtend, sie wirken heller als ihr RGB-Wert.
COLORS = {
    'car':        (40, 110, 60),
    'van':        (40, 90, 120),
    'truck':      (140, 90, 40),
    'bus':        (100, 60, 110),
    'motorcycle': (125, 115, 50),
    'bicycle':    (130, 50, 50),
    'pedestrian': (110, 110, 110),
}
DEFAULT_COLOR = (90, 90, 90)
THICKNESS = 0.1    # m
LIFE_FACTOR = 1.5  # Lebensdauer in Ticks: Flackern -> erhoehen, doppelte Boxen -> verringern


def actor_class(actor):
    if actor.type_id.startswith('walker.'):
        return 'pedestrian'
    return actor.attributes.get('base_type', '').lower() or 'vehicle'


def predicted_transform(actor_snap, dt):
    """Pose zum naechsten Tick: eine jetzt gezeichnete Box erscheint erst im naechsten Frame."""
    import carla
    t = actor_snap.get_transform()
    v = actor_snap.get_velocity()          # m/s
    w = actor_snap.get_angular_velocity()  # deg/s
    rot = carla.Rotation(pitch=t.rotation.pitch, yaw=t.rotation.yaw + w.z * dt, roll=t.rotation.roll)
    return carla.Transform(t.location + v * dt, rot)


class Camera:
    """Projektion der Kamera; ids = Actor-ID pro Pixel (Instanz-Segmentierung) oder None."""

    def __init__(self, transform, w, h, fov, ids=None):
        self.transform, self.w, self.h, self.fov = transform, w, h, fov
        self.K = intrinsics(w, h, fov)
        self.w2c = np.array(transform.get_inverse_matrix())
        self.ids = ids


def detect(actors, snap, cfg, cam, max_distance):
    """Objekte im Kamerabild bis max_distance (0 = unbegrenzt).

    Mit Instanzbild zusaetzlich 'visible_pixels' und 'occluded' (weniger als
    min_visible_pixels sichtbar). 'boxed': bekommt eine Box (nicht verdeckt
    oder hide_occluded aus).
    """
    origin = cam.transform.location
    dets = []
    for actor in actors:
        s = snap.find(actor.id)
        if s is None:
            continue
        t = s.get_transform()
        dist = t.location.distance(origin)
        if max_distance > 0 and dist > max_distance:
            continue
        verts = np.array([[v.x, v.y, v.z] for v in actor.bounding_box.get_world_vertices(t)])
        proj = project_box(verts, cam.w2c, cam.K, cam.w, cam.h)
        if proj is None:
            continue
        det = {'actor': actor, 'snap': s, 'transform': t, 'distance': dist,
               'class': actor_class(actor), 'box2d': proj[1], 'boxed': True}
        if cam.ids is not None:
            det['visible_pixels'] = visible_pixels(cam.ids, actor.id, det['box2d'])
            det['occluded'] = det['visible_pixels'] < cfg.min_visible_pixels
            det['boxed'] = not (cfg.hide_occluded and det['occluded'])
        dets.append(det)
    return dets


def draw(world, dets, dt):
    """Boxen vor dem naechsten Tick in die Welt zeichnen (zeichnen -> tick: kein Flackern)."""
    import carla
    for d in dets:
        t = predicted_transform(d['snap'], dt)
        bb = d['actor'].bounding_box
        world.debug.draw_box(carla.BoundingBox(t.transform(carla.Location(bb.location)), bb.extent), t.rotation,
                             THICKNESS, carla.Color(*COLORS.get(d['class'], DEFAULT_COLOR)), dt * LIFE_FACTOR)


def _xyz(v):
    return {'x': v.x, 'y': v.y, 'z': v.z}


def _rotation(r):
    return {'pitch': r.pitch, 'yaw': r.yaw, 'roll': r.roll}


def _object(d):
    import carla
    actor, t = d['actor'], d['transform']
    bb = actor.bounding_box
    return {
        'id': actor.id,
        'class': d['class'],
        'blueprint_id': actor.type_id,
        'bbox_3d': {  # Welt, UE-Achsen
            # Kopie: Transform.transform() aendert sein Argument
            'center': _xyz(t.transform(carla.Location(bb.location))),
            'extent': _xyz(bb.extent),  # halbe Masse
            'rotation': matrix_rotation(
                ue_matrix(_xyz(t.location), _rotation(t.rotation))[:3, :3] @
                ue_matrix({'x': 0, 'y': 0, 'z': 0}, _rotation(bb.rotation))[:3, :3]),
        },
        'visible_pixels': d.get('visible_pixels'),
    }


def _color(actor):
    """Lackfarbe [r, g, b] aus dem Blueprint-Attribut 'color' (Fussgaenger und manche Fahrzeuge: None)."""
    try:
        rgb = [int(c) for c in actor.attributes['color'].split(',')]
    except (KeyError, ValueError):
        return None
    return rgb if len(rgb) == 3 else None


def write_status(path, snap, dets):
    """Objekte im Bild fuer Sim Control, nach Distanz sortiert. Atomar: Leser sehen nie eine halbe Datei."""
    data = {
        'frame': snap.frame,
        'written_at': time.time(),
        'objects': [{
            'id': d['actor'].id,
            'class': d['class'],
            'blueprint_id': d['actor'].type_id,
            'color': _color(d['actor']),
            'distance': round(d['distance'], 1),
            'visible_pixels': d.get('visible_pixels'),  # None: kein Instanzbild in diesem Frame
            'occluded': d.get('occluded'),
            'boxed': d['boxed'],
        } for d in sorted(dets, key=lambda d: d['distance'])],
    }
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    os.replace(tmp, path)


def camera_record(cam):
    """Kalibrierung fuer den Export: Bildgroesse, Intrinsik (Pixel), Pose (Welt, UE-Achsen)."""
    return {
        'width': cam.w, 'height': cam.h, 'fov': cam.fov,
        'intrinsics': cam.K.tolist(),
        'transform': {'location': _xyz(cam.transform.location), 'rotation': _rotation(cam.transform.rotation)},
    }


def frame_record(index, snap, dets):
    """Objekte eines Frames fuer die Umwandlung (simapi.dair rechnet daraus die Labels)."""
    return {
        'index': index,
        'frame_id': snap.frame,
        'timestamp': snap.timestamp.elapsed_seconds,
        'objects': [_object(d) for d in dets],
    }
