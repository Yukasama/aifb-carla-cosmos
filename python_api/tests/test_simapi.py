"""Selbsttest ohne CARLA: python tests/test_simapi.py (oder pytest)."""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simapi import presets  # noqa: E402
from simapi.config import START_ONLY, build_parser  # noqa: E402
from simapi.geometry import decode_instance, intrinsics, project_box, visible_pixels  # noqa: E402

W, H = 800, 600
K = intrinsics(W, H, 90)
I = np.eye(4)  # Kamera im Ursprung, Blick entlang +x (UE)


def box(cx, cy=0.0, cz=0.0, e=1.0):
    """8 Eckpunkte eines Wuerfels in carla-Reihenfolge (x,y,z jeweils -/+)."""
    return np.array([[cx + sx * e, cy + sy * e, cz + sz * e]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def test_projection_center():
    edges, (x0, y0, x1, y1) = project_box(box(10), I, K, W, H)
    assert edges.shape == (12, 2, 2)
    assert x0 < W / 2 < x1 and y0 < H / 2 < y1
    # Wuerfel 2 m breit, Vorderseite in 9 m: f = 400 -> ca. 89 px breit
    assert abs((x1 - x0) - 2 * 400 / 9) < 1


def test_projection_axes():
    _, right = project_box(box(10, cy=3), I, K, W, H)
    _, up = project_box(box(10, cz=3), I, K, W, H)
    assert right[0] > W / 2   # +y (UE rechts) -> rechts im Bild
    assert up[3] < H / 2      # +z (oben) -> oben im Bild


def test_behind_and_outside():
    assert project_box(box(-10), I, K, W, H) is None
    assert project_box(box(10, cy=100), I, K, W, H) is None


def test_straddling_camera_is_clipped():
    _, (x0, y0, x1, y1) = project_box(box(0.5, e=2), I, K, W, H)
    assert (x0, y0, x1, y1) == (0, 0, W - 1, H - 1)


def test_instance_decode_and_visibility():
    bgra = np.zeros((4, 4, 4), np.uint8)
    bgra[1:3, 1:3] = (0x01, 0x02, 14, 255)  # B, G, R(=Tag car)
    tags, ids = decode_instance(bgra)
    assert tags[1, 1] == 14 and ids[1, 1] == 0x0102
    assert visible_pixels(ids, 0x0102, (0, 0, 3, 3)) == 4
    assert visible_pixels(ids, 0x0102, (2, 2, 3, 3)) == 1
    assert visible_pixels(ids, 7, (0, 0, 3, 3)) == 0


def test_update_is_full_state():
    p = build_parser()
    cfg = p.parse_args(['--hide-occluded', '--number-of-vehicles', '5', '--max-distance', '0'])
    assert cfg.hide_occluded and cfg.number_of_vehicles == 5 and cfg.max_distance == 0
    assert not p.parse_args([]).hide_occluded  # fehlendes Flag = Default
    assert all(hasattr(cfg, k) for k in START_ONLY)
    for bad in (['--clip-frames', '0'], ['--min-visible-pixels', '0'], ['--number-of-vehicles', '-1']):
        try:
            p.parse_args(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)


def test_walkers_start_after_tick():
    """controller.start() erst, wenn der Fussgaenger im Snapshot steht; sonst traegt CARLA ihn bei (0, 0, 0) ein."""
    from types import SimpleNamespace as NS

    from simapi.traffic import Traffic

    started, known = [], set()

    class Controller:
        def __init__(self, i):
            self.id = i

        def start(self):
            started.append(self.id)

        def go_to_location(self, loc):
            pass

        def set_max_speed(self, v):
            pass

    world = NS(get_snapshot=lambda: NS(find=lambda i: object() if i in known else None),
               get_actors=lambda ids: [Controller(i) for i in ids],
               get_random_location_from_navigation=lambda: None)
    t = Traffic.__new__(Traffic)
    t.world, t.to_start = world, [(21, 11, 1.4), (22, 12, 1.4)]
    t.after_tick()  # gerade gespawnt, noch kein Tick: niemand startet
    assert started == [] and len(t.to_start) == 2
    known.add(11)
    t.after_tick()
    assert started == [21] and t.to_start == [(22, 12, 1.4)]
    known.add(12)
    t.after_tick()
    t.after_tick()
    assert started == [21, 22] and t.to_start == []


def test_refill_replaces_deleted_walkers():
    """CARLA loescht angefahrene Fussgaenger; refill ersetzt sie, gerade gespawnte zaehlen nicht als geloescht."""
    from types import SimpleNamespace as NS

    from simapi.traffic import Traffic

    alive, spawned, destroyed = {11, 13}, [], []
    t = Traffic.__new__(Traffic)
    t.world = NS(get_snapshot=lambda: NS(find=lambda i: object() if i in alive else None))
    t.walkers, t.walker_target = [(11, 21), (12, 22), (13, 23), (14, 24)], 4
    t.to_start = [(24, 14, 1.0)]  # 14 gerade gespawnt, noch nicht im Snapshot
    t._destroy = lambda ids, quiet=False: destroyed.append((ids, quiet))
    t._spawn_walkers = lambda k, where=None: spawned.append((k, where))
    where = object()
    assert t.refill(where) == 1
    assert t.walkers == [(11, 21), (13, 23), (14, 24)] and spawned == [(1, where)] and destroyed == [([22], True)]
    alive.update({12, 14})
    assert t.refill() == 0 and spawned == [(1, where)]


def test_detect_counts_pixels_outside_box():
    """Ein Arm am Bildrand zaehlt auch, wenn die Box knapp ausserhalb liegt (sonst fehlt ein Label)."""
    import carla
    from types import SimpleNamespace as NS

    from simapi import boxes
    from simapi.geometry import object_pixels

    tags = np.array([[12, 12, 1], [14, 1, 1]])
    ids = np.array([[0x0105, 0x0105, 0x0105], [7, 0, 0]])
    assert object_pixels(tags, ids) == {0x0105: 2, 7: 1}  # Strasse mit derselben ID zaehlt nicht

    cam = make_camera()
    cam.seen = {0x0105: 2}
    behind = carla.Transform(carla.Location(-20, 0, 0))  # Box hinter der Kamera, nicht im Bild
    actors = [NS(id=0x10105, type_id='walker.pedestrian.0001', attributes={}, bounding_box=car_box()),
              NS(id=8, type_id='vehicle.test', attributes={'base_type': 'car'}, bounding_box=car_box())]
    snap = NS(find=lambda i: NS(get_transform=lambda: behind))
    cfg = NS(min_visible_pixels=50, hide_occluded=False)
    (d,) = boxes.detect(actors, snap, cfg, cam, 0)  # 8: nicht im Bild und kein Pixel
    assert d['actor'].id == 0x10105 and d['box2d'] is None and d['visible_pixels'] == 2 and d['occluded']


def test_status_file():
    from types import SimpleNamespace as NS

    from simapi.boxes import write_status

    def det(id, dist, attrs, **extra):
        actor = NS(id=id, type_id='vehicle.tesla.model3', attributes=attrs)
        return {'actor': actor, 'distance': dist, 'class': 'car', 'boxed': True, **extra}

    dets = [det(2, 30.04, {'color': '255,0,0'}, visible_pixels=10, occluded=True, boxed=False),
            det(1, 12.0, {}),                     # ohne Instanzbild, ohne Lackfarbe
            det(3, 5.0, {'color': 'kaputt'}, visible_pixels=900, occluded=False)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'status.json')
        write_status(path, NS(frame=42), dets, {'fov': 60.0})
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        assert not os.path.exists(path + '.tmp')
    assert data['frame'] == 42 and isinstance(data['written_at'], float) and data['camera'] == {'fov': 60.0}
    objs = data['objects']
    assert [o['id'] for o in objs] == [3, 1, 2]  # nach Distanz
    assert objs[2] == {'id': 2, 'class': 'car', 'blueprint_id': 'vehicle.tesla.model3', 'color': [255, 0, 0],
                       'distance': 30.0, 'visible_pixels': 10, 'occluded': True, 'boxed': False}
    assert objs[1]['occluded'] is None and objs[1]['visible_pixels'] is None and objs[1]['color'] is None
    assert objs[0]['color'] is None


# ---------- Testszene: Kamera ueber einer Strasse, ein Auto ----------

CW, CH = 192, 108  # kleine Aufnahme, Cosmos-Video halb so gross


class FakeImage:
    """carla.Image-Ersatz: BGRA-Rohdaten wie vom Sensor."""

    def __init__(self, bgr):
        self.height, self.width = bgr.shape[:2]
        self.raw_data = np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)]).tobytes()


def carla_depth(meters):
    """Meter -> CARLA-Tiefenbild (BGR)."""
    v = np.round(np.asarray(meters, np.float64) / 1000 * (256 ** 3 - 1)).astype(np.int64)
    return np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], axis=-1).astype(np.uint8)


def carla_instance(tags, ids):
    """Klasse + Actor-ID -> CARLA-Instanzbild (BGR: B = ID-High, G = ID-Low, R = Klasse)."""
    return np.stack([np.asarray(ids) >> 8, np.asarray(ids) & 255, tags], axis=-1).astype(np.uint8)


def make_camera(x=0.0, fov=90.0):
    """12 m hoch, 30 Grad nach unten, Blick entlang +x."""
    import carla

    from simapi.boxes import Camera
    return Camera(carla.Transform(carla.Location(x, 0, 12), carla.Rotation(pitch=-30)), CW, CH, fov)


def car_box():
    import carla
    return carla.BoundingBox(carla.Location(0, 0, 0.75), carla.Vector3D(2, 1, 0.75))


def car_at(i, y=3.0, yaw=30.0):
    """Auto faehrt mit 0,5 m pro Frame entlang x, 20 m vor der Kamera."""
    import carla
    return carla.Transform(carla.Location(20 + 0.5 * i, y, 0), carla.Rotation(yaw=yaw))


def car_det(i):
    """Erkennung wie aus boxes.detect fuer den Recorder."""
    from types import SimpleNamespace as NS
    return {'actor': NS(id=300, type_id='vehicle.test', bounding_box=car_box()),
            'transform': car_at(i), 'class': 'car', 'visible_pixels': 500}


def scene(cam, objects=(), stripe=None):
    """Strasse (Tag 1) als Ebene z = 0, Himmel (11) ab 999 m, Objekte (id, tag, Transform, Box) als
    gefuellte Huelle ihrer CARLA-Box, optional ein Mast (Tag 6) in 3 m ueber den Spalten stripe.
    -> Tiefe in m, Tags, IDs"""
    import cv2

    from simapi.boxes import camera_record
    from simapi.geometry import to_camera, world_from_cv

    T, K = world_from_cv(camera_record(cam)), cam.K
    v, u = np.mgrid[0:cam.h, 0:cam.w]
    rays = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones(u.shape)], axis=-1)
    with np.errstate(divide='ignore'):
        t = -T[2, 3] / (rays @ T[2, :3])
    sky = ~((t > 0) & (t < 999))
    depth, tags, ids = np.where(sky, 1000.0, t), np.where(sky, 11, 1), np.zeros(u.shape, np.int64)
    for oid, tag, tr, bb in objects:
        c = to_camera(np.array([[p.x, p.y, p.z] for p in bb.get_world_vertices(tr)]), cam.w2c)
        uv = c[:, :2] / c[:, 2:] @ K[:2, :2].T + K[:2, 2]
        mask = np.zeros(u.shape, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        m = mask.astype(bool)
        tags[m], ids[m], depth[m] = tag, oid, c[:, 2].min() + 0.2
    if stripe:
        sl = np.s_[:, stripe[0]:stripe[1]]
        tags[sl], ids[sl], depth[sl] = 6, 0, 3.0
    return depth, tags, ids


def fake_images(depth, tags, ids):
    rgb = np.full(tags.shape + (3,), (90, 90, 90), np.uint8)
    rgb[tags == 11] = (235, 200, 120)  # BGR, hellblau
    rgb[tags == 14] = (30, 30, 200)    # rot
    return {'rgb': FakeImage(rgb), 'depth': FakeImage(carla_depth(depth)),
            'instance': FakeImage(carla_instance(tags, ids))}


def synthetic_frame(i):
    images = fake_images(*scene(make_camera(), [(300, 14, car_at(i), car_box())]))
    for image in images.values():
        image.frame = snap_of(i).frame
    return images


def snap_of(i):
    from types import SimpleNamespace as NS
    return NS(frame=1000 + i, timestamp=NS(elapsed_seconds=i / 30))


def cv2_read(clip, kind, i):
    import cv2
    return cv2.imread(os.path.join(clip, 'frames', kind, '%04d.png' % i), cv2.IMREAD_COLOR)


# ---------- Aufnahme ----------

def test_recorder_clips():
    from simapi.clips import Recorder

    with tempfile.TemporaryDirectory() as d:
        rec = Recorder(d, 'Town10HD_Opt', 30)
        done = [rec.add(snap_of(i), [], synthetic_frame(i), make_camera(), 4) for i in range(6)]
        assert done[:3] == [None] * 3 and done[3] and done[4] is None
        clip = done[3]
        assert clip.endswith('_000') and os.path.isdir(rec.clip['tmp'])  # zweiter Clip laeuft
        with open(os.path.join(clip, 'clip.json')) as f:
            meta = json.load(f)
        assert meta['frames'] == 4 and meta['fps'] == 30 and meta['frame_ids'] == [1000, 1001, 1002, 1003]
        assert meta['camera']['fov'] == 90 and meta['camera']['width'] == CW
        assert os.path.isfile(os.path.join(clip, meta['images']['depth'] % 3))
        with open(os.path.join(clip, meta['labels'] % 2)) as f:
            assert json.load(f) == {'index': 2, 'frame_id': 1002, 'timestamp': 2 / 30, 'objects': []}

        # Kamera bewegt: laufenden Clip verwerfen, neuer Take
        partial, take = rec.clip['tmp'], rec.take[0]
        assert rec.add(snap_of(6), [], synthetic_frame(6), make_camera(x=1.0), 4) is None
        assert not os.path.exists(partial) and rec.index == 1
        assert rec.take[0] != take and rec.clip['name'] == rec.take[0] + '_000'
        # fehlendes Bild: verwerfen
        frame = synthetic_frame(7)
        frame['depth'] = None
        rec.add(snap_of(7), [], frame, make_camera(x=1.0), 4)
        assert rec.clip is None
        assert sorted(os.listdir(d)) == [os.path.basename(clip)]
        rec.close()


def test_frame_record_box():
    import carla
    from types import SimpleNamespace as NS

    from simapi import boxes

    # Dasselbe BoundingBox-Objekt bei jedem Zugriff: faellt auf, wenn es unterwegs veraendert wird
    actor = NS(id=7, type_id='vehicle.test',
               bounding_box=carla.BoundingBox(carla.Location(0.5, 0, 0.7), carla.Vector3D(2, 1, 0.7)))
    d = {'actor': actor, 'transform': carla.Transform(carla.Location(100, 50, 0), carla.Rotation(yaw=90)),
         'class': 'car', 'visible_pixels': 12}
    for _ in range(2):
        o = boxes._object(d)
        assert np.allclose([o['bbox_3d']['center'][k] for k in 'xyz'], [100, 50.5, 0.7], atol=1e-4)
    assert o['visible_pixels'] == 12 and o['bbox_3d']['rotation']['yaw'] == 90
    assert np.allclose([o['bbox_3d']['extent'][k] for k in 'xyz'], [2, 1, 0.7])


def test_replace_dir_retries():
    from simapi import convert

    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) < 3:
            raise PermissionError(5, 'Zugriff verweigert')
        real(src, dst)

    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, 'a.partial'), os.path.join(d, 'a')
        os.makedirs(src)
        os.replace = flaky
        try:
            convert.replace_dir(src, dst, wait=0)
        finally:
            os.replace = real
        assert len(calls) == 3 and os.path.isdir(dst) and not os.path.exists(src)


# ---------- Cosmos ----------

def test_cosmos_frames():
    from simapi import cosmos
    from simapi.geometry import depth_m

    # Tiefe: CARLA-Kodierung -> Meter, nah = hell, Himmel schwarz
    m = np.array([[1.0, 15.0, 1000.0]])
    assert np.allclose(depth_m(carla_depth(m)), m, rtol=1e-4)
    disp = cosmos.disparity(carla_depth(m), (3, 1))
    g = cosmos.depth_frame(disp, disp.min(), disp.max())[..., 0]
    assert g[0, 0] == 255 and g[0, 2] == 0 and 0 < g[0, 1] < 255

    # Segmente: Autos einzeln, Strasse als Klasse, unbeschriftet schwarz, Farben stabil
    inst = carla_instance(np.array([[14, 14, 1, 1, 0]]), np.array([[300, 301, 5, 6, 0]]))
    keys = cosmos.segment_keys(inst)
    assert keys[0, 0] != keys[0, 1] and keys[0, 2] == keys[0, 3] == 1 and keys[0, 4] == 0
    colors = {}
    seg = cosmos.seg_frame(inst, (5, 1), colors)
    assert (seg[0, 4] == 0).all() and not (seg[0, 0] == seg[0, 1]).all() and (seg[0, 2] == seg[0, 3]).all()
    assert cosmos.segment_color(int(keys[0, 0])) == tuple(seg[0, 0])
    # Groesster moeglicher Schluessel (Fahrrad, Actor-ID 65535) hat noch eine Zeile in der Farbtabelle
    inst = carla_instance(np.array([[19, 1]]), np.array([[65535, 0]]))
    seg = cosmos.seg_frame(inst, (2, 1), {})
    assert tuple(seg[0, 0]) == cosmos.segment_color((19 << 16) | 65535) and tuple(seg[0, 1]) == cosmos.segment_color(1)

    rgb = np.zeros((72, 128, 3), np.uint8)
    rgb[20:50, 40:90] = 255
    e = cosmos.edge_frame(rgb)
    assert e.shape == rgb.shape and e[20:50, 40:90].any() and not e[:10].any()
    assert cosmos.vis_frame(rgb).shape == rgb.shape


# ---------- Detektor-Labels (DAIR-V2X-I) ----------

def test_ue_matrix_matches_carla():
    import carla

    from simapi.geometry import ue_matrix

    loc, rot = {'x': 3.0, 'y': -7.5, 'z': 12.0}, {'pitch': -31.0, 'yaw': 127.0, 'roll': 4.0}
    want = np.array(carla.Transform(carla.Location(**loc), carla.Rotation(**rot)).get_matrix())
    assert np.allclose(ue_matrix(loc, rot), want, atol=1e-5)


def car_record(i, **kw):
    from simapi.boxes import _object
    return _object(dict(car_det(i), transform=car_at(i, **kw)))


def test_dair_frame():
    import math

    from simapi import dair
    from simapi.boxes import camera_record
    from simapi.geometry import to_camera

    cam = make_camera()
    meta = {'camera': camera_record(cam)}
    depth, tags, _ = scene(cam)
    normal, point = dair.fit_ground(meta, depth, tags)
    assert np.allclose(normal, [0, 0, 1], atol=1e-6) and abs(point[2]) < 1e-3
    fr = dair.Frame(meta, normal, point)
    assert np.allclose(fr.ground_cv[:3, 3], [0, 0, 12], atol=1e-3)  # Kamera 12 m senkrecht ueber dem Ursprung

    center, l, w, h, yaw = fr.box(car_record(0))
    # UE: y nach rechts, Gierwinkel im Uhrzeigersinn -> DAIR: y nach links, gegen den Uhrzeigersinn
    assert np.allclose(center, [20, -3, 0.75], atol=1e-3) and np.allclose((l, w, h), (4, 2, 1.5))
    assert math.isclose(yaw, -math.pi / 6, abs_tol=1e-6)

    # Kalibrierung, wie dair2kitti und gen_info_dair sie lesen: Bodensystem -> Pixel in 1920 x 1080
    intrinsic, extrinsic = fr.calib()
    K = np.array(intrinsic['cam_K']).reshape(3, 3)
    R, t = np.array(extrinsic['rotation']), np.array(extrinsic['translation']).reshape(3)
    p = K @ (R @ center + t)
    q = cam.K @ to_camera(np.array([[20.0, 3.0, 0.75]]), cam.w2c)[0]  # ueber CARLA
    assert np.allclose(p[:2] / p[2] + 0.5, q[:2] / q[2] * dair.SIZE[0] / CW, atol=1e-3)


def test_dair_labels():
    from simapi import dair
    from simapi.boxes import camera_record

    cam = make_camera()
    meta = {'camera': camera_record(cam)}
    car = (300, 14, car_at(0), car_box())
    depth, tags, ids = scene(cam, [car])
    fr = dair.Frame(meta, *dair.fit_ground(meta, depth, tags))
    s = dair.SIZE[0] / CW

    lb = fr.label(car_record(0), dair.Pixels(ids, tags), depth)
    ys, xs = np.nonzero(ids == 300)
    assert lb['type'] == 'Car' and lb['truncated_state'] == 0 and lb['occluded_state'] == 0
    assert lb['2d_box'] == {'xmin': round(xs.min() * s, 2), 'ymin': round(ys.min() * s, 2),
                            'xmax': round((xs.max() + 1) * s, 2), 'ymax': round((ys.max() + 1) * s, 2)}
    assert np.allclose([lb['3d_location'][k] for k in 'xyz'], [20, -3, 0.75], atol=1e-3)
    assert lb['3d_dimensions'] == {'h': 1.5, 'w': 2.0, 'l': 4.0} and abs(lb['rotation'] + 0.5236) < 1e-3

    # Mast vor der rechten Haelfte: teilweise verdeckt, 2D-Box nur um den sichtbaren Teil
    x0, x1 = xs.min(), xs.max() + 1
    mid = (x0 + x1) // 2
    depth2, tags2, ids2 = scene(cam, [car], stripe=(mid, x1))
    lb2 = fr.label(car_record(0), dair.Pixels(ids2, tags2), depth2)
    assert lb2['occluded_state'] >= 1 and lb2['2d_box']['xmax'] <= mid * s
    depth4, tags4, ids4 = scene(cam, [car], stripe=(x0, x1))
    assert fr.label(car_record(0), dair.Pixels(ids4, tags4), depth4) is None  # komplett verdeckt: kein Label
    # bis auf einen schmalen Streifen verdeckt: aufgezeichnet, aber zu wenig sichtbar fuer ein Label
    depth5, tags5, ids5 = scene(cam, [car], stripe=(x0 + 1, x1))
    px5 = dair.Pixels(ids5, tags5)
    vis, _ = fr.view.visibility(car_record(0), px5, depth5)
    assert 0 < vis['pixels'] and vis['in_frame'] > 0.999 and vis['share'] < dair.MIN_SHARE
    assert fr.label(car_record(0), px5, depth5) is None
    assert fr.label(car_record(0), px5, depth5, dair.Limits(min_share=0.0)) is not None  # Schwelle einstellbar

    # links aus dem Bild ragend: gut die Haelfte im Bild bekommt ein Label, ein Rest am Rand nicht
    edge = (300, 14, car_at(0, y=-23, yaw=0), car_box())
    depth3, tags3, ids3 = scene(cam, [edge])
    lb3 = fr.label(car_record(0, y=-23, yaw=0), dair.Pixels(ids3, tags3), depth3)
    assert lb3['truncated_state'] == 1 and lb3['2d_box']['xmin'] == 0 and lb3['occluded_state'] == 0
    depth6, tags6, ids6 = scene(cam, [(300, 14, car_at(0, y=-25.5, yaw=0), car_box())])
    px6 = dair.Pixels(ids6, tags6)
    vis6, _ = fr.view.visibility(car_record(0, y=-25.5, yaw=0), px6, depth6)
    assert vis6['pixels'] > 0 and vis6['unoccluded'] == 1.0 and vis6['in_frame'] < dair.MIN_SHARE
    assert fr.label(car_record(0, y=-25.5, yaw=0), px6, depth6) is None

    # 16-Bit-ID eines Kartenobjekts (Strasse) gleich der des Autos: zaehlt nicht als Auto
    ids7 = np.where(tags == 1, 300, ids)
    px7 = dair.Pixels(ids7, tags)
    vis7, (ys7, xs7) = fr.view.visibility(car_record(0), px7, depth)
    assert vis7['pixels'] == np.count_nonzero(ids == 300) and (tags[ys7, xs7] == 14).all()
    assert fr.label(car_record(0), px7, depth)['2d_box'] == lb['2d_box']

    # Pixelschwelle gilt fuer 1920 x 1080: ein Pixel dieser Aufnahme (192 x 108) zaehlt wie 100
    seen = {'distance': 20.0, 'pixels': 1, 'share': 1.0}
    assert fr.view.labeled(seen, dair.Limits(min_pixels=100))
    assert not fr.view.labeled(seen, dair.Limits(min_pixels=101))
    assert not fr.view.labeled(dict(seen, pixels=0), dair.Limits(min_pixels=0))
    # Default: ab 10 Pixeln und 5 % bis 150 m (Kamera bis Boxmitte), weiter weg nie
    assert fr.view.labeled(dict(seen, distance=150.0, share=0.05))
    assert not fr.view.labeled(dict(seen, distance=150.1))
    assert fr.view.labeled(dict(seen, distance=500.0), dair.Limits(max_distance=0))  # 0 = unbegrenzt
    assert np.isclose(vis['distance'], np.linalg.norm([20, 3, 0.75 - 12]))  # Kamera (0, 0, 12), Boxmitte (20, 3, 0.75)

    # Punktwolke: Strasse auf z = 0, Auto (hier eine flache Attrappe) darueber
    pts = fr.points(depth, tags)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'p.pcd')
        dair.write_pcd(path, pts)
        back = dair.read_pcd(path)
    assert back.shape == (len(pts), 4) and np.allclose(back[:, :3], pts, atol=1e-4)
    assert np.mean(np.abs(pts[:, 2]) < 1e-3) > 0.95 and pts[:, 2].min() > -1e-3


def test_unlabeled_objects():
    from simapi.dair import UNLABELED_MIN, unlabeled

    tags, ids = np.ones((40, 40), int), np.zeros((40, 40), int)
    tags[:20, :15], ids[:20, :15] = 14, 301     # 300 Pixel Auto
    tags[30:35, :10], ids[30:35, :10] = 12, 302
    assert UNLABELED_MIN == 1
    assert unlabeled(tags, ids, []) == 2 and unlabeled(tags, ids, [301]) == 1
    assert unlabeled(tags, ids, [301 + 65536, 302]) == 0  # Instanzbild hat nur 16 Bit


def test_split_keeps_groups():
    from simapi.dataset import split

    groups = [['a1', 'a2', 'a3'], ['b1', 'b2'], ['c1', 'c2', 'c3', 'c4', 'c5']]
    train, val = split(groups, 0.2, 0)
    assert sorted(train + val) == sorted(sum(groups, [])) and val and train
    assert all(set(g) <= set(train) or set(g) <= set(val) for g in groups)
    assert split(groups, 0.0, 0)[1] == [] and split([['x', 'y']], 0.5, 0) == (['x', 'y'], [])


def test_convert_and_dataset():
    import shutil

    import cv2
    import imageio_ffmpeg

    from simapi import convert, cosmos, dataset
    from simapi.clips import Recorder

    n, size = 6, (CW // 2, CH // 2)
    with tempfile.TemporaryDirectory() as d:
        export = os.path.join(d, '_out')
        rec = Recorder(export, 'Town10HD_Opt', 30)
        for i in range(n):
            clip = rec.add(snap_of(i), [car_det(i)], synthetic_frame(i), make_camera(), n)
        rec.close()
        assert convert.main([export, '--size', '%dx%d' % size, '--every', '2']) == 0

        # Cosmos: fuenf Videos und je eine Spec fuer Transfer2.5 und Cosmos 3, deckungsgleich mit der Aufnahme
        out = os.path.join(clip, 'cosmos')
        assert sorted(os.listdir(out)) == sorted([s + '.mp4' for s in cosmos.STREAMS] +
                                                 ['transfer2.5.json', 'cosmos3.json', 'prompt.json', 'manifest.json'])
        with open(os.path.join(out, 'transfer2.5.json')) as f:
            spec = json.load(f)
        assert spec['name'] == os.path.basename(clip) and spec['video_path'] == 'rgb.mp4'
        assert {k: spec.get(k, {}).get('control_weight') for k in ('edge', 'vis', 'depth', 'seg')} == \
            {'edge': 1.0, 'vis': None, 'depth': 0.5, 'seg': 1.0}
        assert all(spec[k]['control_path'] == k + '.mp4' for k in ('edge', 'depth', 'seg'))
        with open(os.path.join(out, 'cosmos3.json')) as f:
            spec3 = json.load(f)
        assert spec3['name'] == os.path.basename(clip) and spec3['num_frames'] == 93 and spec3['fps'] == 30
        assert spec3['resolution'] == '720' and spec3['aspect_ratio'] == '16,9'
        assert all(os.path.isfile(os.path.join(out, spec3[k]['control_path'])) for k in ('edge', 'blur', 'depth', 'seg'))
        with open(os.path.join(out, spec3['prompt_path'])) as f:
            prompt = json.load(f)
        assert [s['number_of_subjects'] for s in prompt['subjects']] == [1]  # ein Auto, keine Personen
        assert prompt['cinematography']['camera_motion'] == 'Static' and prompt['fps'] == 30
        for s in cosmos.STREAMS:
            reader = imageio_ffmpeg.read_frames(os.path.join(out, s + '.mp4'))
            info = next(reader)
            assert tuple(info['size']) == size and info['fps'] == 30
            frames = [np.frombuffer(b, np.uint8).reshape(size[1], size[0], 3) for b in reader]
            assert len(frames) == 93, s
            if s == 'seg':  # verlustfrei genug: Segmentgrenzen exakt, Farben bis auf Rundung
                for i in range(n):
                    want = cosmos.seg_frame(cv2_read(clip, 'instance', i), size, {})
                    assert np.abs(frames[i].astype(int) - want).max() <= 3

        # Detektoren: jeder 2. Frame
        with open(os.path.join(clip, 'dair', 'meta.json')) as f:
            meta = json.load(f)
        assert meta['frames'] == [0, 2, 4] and meta['objects'] == 3 and meta['unlabeled'] == {}
        assert abs(meta['camera_height'] - 12) < 0.01
        with open(os.path.join(clip, 'dair', 'label', '0002.json')) as f:
            (lb,) = json.load(f)
        assert lb['type'] == 'Car' and np.allclose([lb['3d_location'][k] for k in 'xy'], [21, -3], atol=1e-3)
        assert convert.main([export]) == 0  # schon umgewandelt: ueberspringen

        # Datensatz aus den CARLA-Bildern
        ds = os.path.join(d, 'dair-v2x-i')
        assert dataset.main([export, '--out', ds]) == 0
        with open(os.path.join(ds, 'data_info.json')) as f:
            info = json.load(f)
        assert [e['image_path'] for e in info] == ['image/00000%d.jpg' % i for i in range(3)]
        assert all(os.path.isfile(os.path.join(ds, v)) for e in info for k, v in e.items() if k.endswith('_path'))
        assert info[1]['source'] == {'clip': os.path.basename(clip), 'frame': 2, 'image': 'carla'}
        assert cv2.imread(os.path.join(ds, 'image', '000001.jpg')).shape == (720, 1280, 3)  # Default HD
        with open(os.path.join(ds, dataset.SPLIT_FILE)) as f:
            sp = json.load(f)
        assert sorted(sp['train'] + sp['val']) == ['000000', '000001', '000002'] and sp['test'] == []
        with open(os.path.join(ds, 'label', 'camera', '000001.json')) as f:
            assert json.load(f) == [dict(lb, **{'2d_box': {k: round(v * (2 / 3), 2) for k, v in lb['2d_box'].items()}})]
        assert dataset.main([export, '--out', ds]) == 1  # Ziel nicht leer

        # Ergebnisse von Transfer2.5 (<name>.mp4) und Cosmos 3 (<name>/vision.mp4), hier einfach das Eingabevideo
        name = os.path.basename(clip)
        res25, res3 = os.path.join(d, 'transfer25'), os.path.join(d, 'cosmos3', name)
        os.makedirs(res25)
        os.makedirs(res3)
        shutil.copyfile(os.path.join(out, 'rgb.mp4'), os.path.join(res25, name + '.mp4'))
        shutil.copyfile(os.path.join(out, 'rgb.mp4'), os.path.join(res3, 'vision.mp4'))
        ds2 = os.path.join(d, 'ds2')
        assert dataset.main([export, '--out', ds2, '--cosmos', res25, '--cosmos', os.path.dirname(res3)]) == 0
        with open(os.path.join(ds2, 'data_info.json')) as f:
            info2 = json.load(f)
        assert [e['source']['image'] for e in info2] == ['transfer25'] * 3 + ['cosmos3'] * 3
        assert [e['source']['frame'] for e in info2] == [0, 2, 4] * 2
        ds3 = os.path.join(d, 'ds3')
        assert dataset.main([export, '--out', ds3, '--fullhd']) == 0
        assert cv2.imread(os.path.join(ds3, 'image', '000001.jpg')).shape == (1080, 1920, 3)
        with open(os.path.join(ds3, 'label', 'camera', '000001.json')) as f:
            assert json.load(f) == [lb]  # dair/ ist schon 1920 x 1080

        # Rohbilder loeschen; danach kein Neuaufbau mehr
        assert convert.main([clip, '--drop-frames']) == 0 and not os.path.exists(os.path.join(clip, 'frames'))
        assert convert.main([clip, '--force']) == 1


def test_cosmos3_spec_fields():
    """Cosmos 3 lehnt unbekannte Felder ab: nur Felder aus den Transfer-Beispielen von cosmos-framework."""
    from simapi.cosmos import cosmos3_spec

    allowed = {'name', 'model_mode', 'resolution', 'aspect_ratio', 'num_frames', 'fps', 'shift', 'num_steps', 'seed',
               'num_video_frames_per_chunk', 'num_conditional_frames', 'num_first_chunk_conditional_frames',
               'share_vision_temporal_positions', 'negative_metadata_mode', 'negative_prompt_keep_metadata',
               'guidance', 'control_guidance', 'negative_prompt_file', 'prompt_path', 'emphasize_control_in_prompt',
               'edge', 'blur', 'depth', 'seg'}
    hint = {'control_path', 'weight'}
    spec = cosmos3_spec('clip', 93, 30, (1280, 720))
    assert set(spec) <= allowed and 'vis' not in spec and 'video_path' not in spec
    assert set(spec['edge']) <= hint | {'preset_edge_threshold'} and set(spec['blur']) <= hint | {'preset_blur_strength'}
    assert set(spec['depth']) <= hint and set(spec['seg']) <= hint
    assert (spec['resolution'], spec['aspect_ratio'], spec['model_mode']) == ('720', '16,9', 'video2video')


def test_record_scenes_plan():
    from types import SimpleNamespace as NS

    import record_scenes as rs

    assert rs.dur(45) == '45 s' and rs.dur(600) == '10 min' and rs.dur(3 * 3600 + 5 * 60) == '3 h 05 min'
    assert rs.num(20088) == '20.088'
    names = ['p%d' % i for i in range(8)]
    assert rs.distribute(20000, names) == dict.fromkeys(names, [93] * 27)
    assert rs.distribute(93, names) == {'p0': [93]}
    assert rs.distribute(1000, names) == {'p0': [93] * 2, 'p1': [93] * 2, 'p2': [93] * 2,
                                          **dict.fromkeys(names[3:], [93])}
    with tempfile.TemporaryDirectory() as d:
        args = NS(drop_frames=True, export_dir=os.path.join(d, 'gibt', 'es', 'noch', 'nicht'), every=1, quality='fullhd')
        rs.check_plan({'p': [93]}, args)
        try:
            rs.check_plan({'p': [10 ** 9]}, args)
        except SystemExit as e:
            assert 'Zu wenig Platz' in str(e)
        else:
            raise AssertionError('Plattenplatz nicht geprueft')


def test_record_scenes_view_check():
    import record_scenes as rs
    from simapi.clips import Recorder

    def wall(i):
        """Kamera steckt in einem Mesh: Tiefe und Segmentierung sehen nur eine Wand 0,3 m vor der Linse."""
        shape = (CH, CW)
        images = fake_images(np.full(shape, 0.3), np.full(shape, 3), np.full(shape, 37104))
        for image in images.values():
            image.frame = snap_of(i).frame
        return images

    with tempfile.TemporaryDirectory() as d:
        rec = Recorder(d, 'Town10HD_Opt', 30)
        good = [rec.add(snap_of(i), [], synthetic_frame(i), make_camera(), 2) for i in range(2)][-1]
        bad = [rec.add(snap_of(i), [], wall(i), make_camera(x=5.0), 2) for i in range(2, 4)][-1]
        rec.close()
        assert rs.view_problem(good) is None
        assert 'direkt vor der Linse' in rs.view_problem(bad)


def test_record_scenes_conversion_queue():
    import record_scenes as rs
    from simapi import clips

    class Proc:
        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

    started, real = [], clips.start_conversion
    clips.start_conversion = lambda clip, drop, extra: started.append((clip, Proc())) or started[-1][1]
    try:
        conv = rs.Conversions(2, False)
        for c in 'abcd':
            conv.add(c)
        assert [c for c, _ in started] == ['a', 'b'] and len(conv.waiting) == 2  # hoechstens zwei gleichzeitig
        started[0][1].returncode = 0
        started[1][1].returncode = 1
        conv.poll()
        assert [c for c, _ in started] == ['a', 'b', 'c', 'd'] and conv.failed == ['b'] and conv.pending() == 2
    finally:
        clips.start_conversion = real


def test_presets_roundtrip():
    pose = {'map': 'Town10HD_Opt', 'x': 1.5, 'y': -2, 'z': 8, 'pitch': -20, 'yaw': 90, 'roll': 0}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'cameras.json')
        assert presets.load(path) == {}
        presets.put('Kreuzung Süd', pose, path)
        presets.put('Nord', dict(pose, x=3), path)
        presets.put('Nord', dict(pose, x=4), path)  # gleicher Name ueberschreibt
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
        raw['kaputt'] = {'map': 'Town10HD_Opt', 'x': 'a'}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(raw, f)
        loaded = presets.load(path)
        assert set(loaded) == {'Kreuzung Süd', 'Nord'} and loaded['Nord']['x'] == 4
        presets.put('West', pose, path)
        with open(path, encoding='utf-8') as f:
            assert 'kaputt' in json.load(f)  # ungueltige Eintraege bleiben in der Datei
        with open(path, 'w') as f:
            f.write('{kaputt')
        assert presets.load(path) == {}
        try:
            presets.put('Ost', pose, path)  # kaputte Datei nicht ueberschreiben
        except ValueError:
            pass
        else:
            raise AssertionError('kaputte Datei ueberschrieben')



def test_presets_fov():
    from types import SimpleNamespace as NS

    from simapi.camera import focal_length, preset_fov, to_preset

    pose = {'map': 'Town10HD_Opt', 'x': 1.5, 'y': -2, 'z': 8, 'pitch': -20, 'yaw': 90, 'roll': 0}
    assert presets.valid(pose) and preset_fov(pose) == 90  # alte Eintraege ohne Sichtfeld
    assert presets.valid(dict(pose, fov=45.5)) and preset_fov(dict(pose, fov=45.5)) == 45.5
    for bad in (5, 170, True, '60', None):
        assert not presets.valid(dict(pose, fov=bad)), bad
    t = NS(location=NS(x=1.23456, y=0, z=8), rotation=NS(pitch=-20, yaw=90, roll=0))
    assert to_preset('Town10HD_Opt', t, 73.739795)['fov'] == 73.74
    assert to_preset('Town10HD_Opt', t)['fov'] == 90
    assert abs(focal_length(90) - 18) < 1e-9 and abs(focal_length(2 * np.degrees(np.arctan(18 / 50))) - 50) < 1e-9


def test_stdin_commands():
    try:
        from simapi.app import _command
    except ImportError:  # carla-Wheel fehlt (macOS)
        return
    assert _command({'set_fov': 60}) == ('set_fov', 60)
    assert _command({'goto_camera': 'Nord'}) == ('goto_camera', 'Nord')
    for bad in ({'set_fov': '60'}, {'set_fov': True}, {'goto_camera': 3}, {'set_fov': 60, 'x': 1}, {'quit': 1}, []):
        assert _command(bad) is None, bad


def test_caption_lens():
    from simapi.cosmos import caption, lens

    assert [lens(f) for f in (90, 54, 40, 29, 20)] == ['Wide-angle lens', 'Wide-angle lens', 'Standard lens',
                                                       'Standard lens', 'Telephoto lens']
    counts = {'vehicles': 1, 'two_wheelers': 0, 'pedestrians': 0}
    assert caption(counts, 93, 16, (1280, 720))['cinematography']['lens_focal_length'] == 'Wide-angle lens'
    assert caption(counts, 93, 16, (1280, 720), 20)['cinematography']['lens_focal_length'] == 'Telephoto lens'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('ok')
