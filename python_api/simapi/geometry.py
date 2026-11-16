"""Bildgeometrie, nur numpy (ohne CARLA testbar)."""

import numpy as np

# Kanten einer carla.BoundingBox in der Reihenfolge von get_world_vertices()
EDGES = ((0, 1), (1, 3), (3, 2), (2, 0), (0, 4), (4, 5), (5, 1), (5, 7), (7, 6), (6, 4), (6, 2), (7, 3))
NEAR = 0.1  # m vor der Kamera; Kanten dahinter werden hier abgeschnitten


def intrinsics(w, h, fov):
    f = w / (2.0 * np.tan(np.radians(fov) / 2))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])


def to_camera(points, world_to_camera):
    """Weltpunkte (N,3) -> Kamerakoordinaten (x rechts, y unten, z vorne)."""
    p = np.c_[points, np.ones(len(points))] @ world_to_camera.T
    return np.stack([p[:, 1], -p[:, 2], p[:, 0]], axis=1)  # UE: x vorne, y rechts, z oben


def project_box(verts, world_to_camera, K, w, h):
    """8 Eckpunkte (Welt) ins Bild projizieren.

    Gibt (kanten, box2d) zurueck: kanten (E,2,2) in Pixeln, an der Nahebene
    geclippt; box2d (xmin, ymin, xmax, ymax) auf das Bild begrenzt.
    None, wenn die Box nicht im Bild liegt.
    """
    c = to_camera(verts, world_to_camera)
    segs = []
    for i, j in EDGES:
        a, b = c[i], c[j]
        if a[2] < NEAR and b[2] < NEAR:
            continue
        if a[2] < NEAR:
            a = a + (b - a) * (NEAR - a[2]) / (b[2] - a[2])
        elif b[2] < NEAR:
            b = b + (a - b) * (NEAR - b[2]) / (a[2] - b[2])
        segs.append((a, b))
    if not segs:
        return None
    p = np.array(segs).reshape(-1, 3) @ K.T
    uv = p[:, :2] / p[:, 2:]
    (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
    if x1 < 0 or y1 < 0 or x0 > w - 1 or y0 > h - 1:
        return None
    return uv.reshape(-1, 2, 2), (max(x0, 0.0), max(y0, 0.0), min(x1, w - 1.0), min(y1, h - 1.0))


def decode_instance(bgra):
    """Instanz-Segmentierung (H,W,4 BGRA) -> (semantische Klasse, Actor-ID) pro Pixel.

    Das Bild hat nur 16 Bit fuer die ID: Actor-IDs ab 65536 erscheinen als id & 0xFFFF.
    """
    return bgra[..., 2], bgra[..., 1].astype(np.uint16) | (bgra[..., 0].astype(np.uint16) << 8)


def visible_pixels(ids, actor_id, box2d):
    """Pixel dieses Actors in seiner 2D-Box (0 = komplett verdeckt)."""
    x0, y0, x1, y1 = (int(v) for v in box2d)
    return int(np.count_nonzero(ids[y0:y1 + 1, x0:x1 + 1] == (actor_id & 0xFFFF)))


def depth_m(bgr):
    """CARLA-Tiefenbild wie von cv2.imread (BGR) -> Abstand entlang der Blickachse in Metern."""
    b, g, r = (bgr[..., i].astype(np.float32) for i in range(3))
    return 1000.0 * (r + 256 * g + 65536 * b) / (256 ** 3 - 1)


# ---------- Koordinatensysteme ----------
# UE/CARLA: x vorne, y rechts, z oben (linkshaendig), Winkel in Grad.
# Rechtshaendig (DAIR, KITTI): y an der Welt spiegeln. OpenCV-Kamera: x rechts, y unten, z vorne.

FLIP = np.diag([1.0, -1.0, 1.0])  # UE <-> rechtshaendig (selbstinvers)
CV_FROM_UE = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # Kameraachsen UE -> OpenCV


def ue_matrix(location, rotation):
    """4x4 lokal -> Welt wie carla.Transform.get_matrix(); location/rotation als Dicts (x/y/z, Grad)."""
    p, y, r = (np.radians(rotation[k]) for k in ('pitch', 'yaw', 'roll'))
    cp, sp, cy, sy, cr, sr = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, location['x']],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, location['y']],
        [sp, -cp * sr, cp * cr, location['z']],
        [0.0, 0.0, 0.0, 1.0],
    ])


def homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def matrix_rotation(R):
    """Inverse von ue_matrix fuer die Rotation, einschliesslich lokal gedrehter Actor-Boxen."""
    pitch = np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-8:
        yaw, roll = np.arctan2(R[1, 0], R[0, 0]), np.arctan2(-R[2, 1], R[2, 2])
    else:
        yaw, roll = np.arctan2(-R[0, 1], R[1, 1]), 0.0
    return dict(zip(('pitch', 'yaw', 'roll'), map(float, np.degrees([pitch, yaw, roll]))))


def world_from_cv(camera):
    """4x4: OpenCV-Kamerakoordinaten -> rechtshaendige Welt, fuer eine Kamera aus clip.json."""
    t = camera['transform']
    return homogeneous(FLIP, 0) @ ue_matrix(t['location'], t['rotation']) @ homogeneous(CV_FROM_UE.T, 0)


def backproject(depth, K, step=1):
    """Tiefenbild (Meter entlang der Blickachse) -> OpenCV-Kamerapunkte (N,3) und Pixel (v, u).

    Jeder step-te Pixel, Pixelmitten; K in Pixelkanten-Konvention (Hauptpunkt w/2, h/2).
    """
    h, w = depth.shape
    v, u = np.mgrid[step // 2:h:step, step // 2:w:step]
    z = depth[v, u]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1).reshape(-1, 3), (v.ravel(), u.ravel())


def transform(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def fit_plane(points, min_points=100):
    """Ebene z = a*x + b*y + c robust durch Punkte (N,3): (Normale nach oben, Punkt auf der Ebene)."""
    keep = np.ones(len(points), bool)
    coef = np.zeros(3)
    for tol in (None, 0.5, 0.2, 0.1):  # Ausreisser (Bordstein, Fahrzeuge an den Raendern) schrittweise weg
        if tol is not None:
            keep = np.abs(points[:, 2] - np.c_[points[:, :2], np.ones(len(points))] @ coef) < tol
        if keep.sum() < min_points:
            raise ValueError('zu wenig Strasse im Bild (%d Punkte) fuer die Bodenebene' % keep.sum())
        coef = np.linalg.lstsq(np.c_[points[keep, :2], np.ones(keep.sum())], points[keep, 2], rcond=None)[0]
    n = np.array([-coef[0], -coef[1], 1.0])
    return n / np.linalg.norm(n), np.array([0.0, 0.0, coef[2]])


def ground_frame(cam_world, normal, point):
    """4x4 Boden -> Welt: Ursprung auf der Ebene senkrecht unter der Kamera, z = Normale,
    x = Blickrichtung in die Ebene gelegt, y nach links (rechtshaendig)."""
    pos = cam_world[:3, 3]
    origin = pos - np.dot(pos - point, normal) * normal
    for axis in (cam_world[:3, 2], -cam_world[:3, 1]):  # Blickrichtung; senkrecht nach unten: Bild-oben
        x = axis - np.dot(axis, normal) * normal
        if np.linalg.norm(x) > 1e-6:
            break
    x /= np.linalg.norm(x)
    return homogeneous(np.stack([x, np.cross(normal, x), normal], axis=1), origin)
