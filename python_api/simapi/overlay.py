"""Labels eines Clips ins Bild zeichnen: python -m simapi.overlay <clip|export-ordner> [--cosmos ORDNER]

Zeigt, worauf BEVHeight, HeightFormer und CoBEV trainieren: die Labels aus <clip>/dair/ (dieselben,
die simapi.dataset in den Datensatz schreibt), projiziert mit dair/calib in das Trainingsbild. Ohne
--cosmos ist das das CARLA-Bild (cosmos/rgb.mp4), mit --cosmos das Ergebnis von Cosmos
(Transfer2.5 <clipname>.mp4, Cosmos 3 <clipname>/vision.mp4). Damit laesst sich vor dem Training
pruefen, ob die generierten Objekte noch unter den CARLA-Boxen liegen.

  3D-Box        Farbe nach Auswerteklasse: Car (Car/Van/Truck/Bus), Cyclist (Cyclist/Motorcyclist),
                Pedestrian; der Strich am Boden zeigt die Fahrtrichtung
  grau          Label ausserhalb des BEV-Rasters der Detektoren (Configs *_102: x 0..102,4 m nach
                vorn, y +-51,2 m): steht im Datensatz, liegt aber nicht im Raster des Detektors
  gelb          Rand dieses Rasters auf der Fahrbahnebene (zeigt auch, ob die Bodenebene passt)
  --2d          zusaetzlich die 2D-Box (eng um die sichtbaren Pixel)
  --hidden      aufgezeichnete Objekte ohne Label rot gestrichelt (zu weit weg oder zu wenig
                sichtbar, siehe simapi.convert --max-distance/--min-pixels/--min-share);
                braucht die Rohdaten frames/*.json

Text an der Box: Typ, Entfernung Kamera-Boxmitte, o1/o2 = occluded_state, t1/t2 = truncated_state.

Ausgabe: <clip>/overlay-<quelle>.mp4 (H.264, yuv420p, laeuft in jedem Player) mit einer Infozeile
unter dem Bild; --out DATEI.mp4 legt das Video woanders ab, --out ORDNER schreibt ein JPG pro Frame.
"""

import argparse
import json
import logging
import math
import os
import sys

import cv2
import imageio_ffmpeg
import numpy as np

from .convert import find_clips
from .dair import SIZE as FULLHD, Frame
from .dataset import cosmos_result

log = logging.getLogger('simapi.overlay')
BEV_X = (0.0, 102.4)  # m, Bodensystem von dair/: x in Blickrichtung
BEV_Y = (-51.2, 51.2)  # m, y nach links
NEAR = 0.1  # m vor der Kamera; Kanten dahinter werden abgeschnitten
# Auswerteklassen wie run_detector.py; BEVHeight fasst Van/Truck/Bus zu Car und Motorcyclist zu Cyclist zusammen
GROUP = {'Car': 'Car', 'Van': 'Car', 'Truck': 'Car', 'Bus': 'Car',
         'Cyclist': 'Cyclist', 'Motorcyclist': 'Cyclist', 'Pedestrian': 'Pedestrian'}
COLORS = {'Car': (0, 220, 255), 'Cyclist': (255, 160, 0), 'Pedestrian': (255, 40, 210)}  # RGB
OUTSIDE = (165, 165, 165)
HIDDEN = (255, 50, 50)
BORDER = (255, 230, 0)
# Ecken: 0-3 unten, 4-7 oben; 0 und 1 vorne (in Fahrtrichtung)
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def corners(location, l, w, h, yaw):
    """8 Ecken einer DAIR-Box (Mittelpunkt, Masse, Gierwinkel um z) im Bodensystem."""
    x = np.array([1, 1, -1, -1]) * l / 2
    y = np.array([1, -1, -1, 1]) * w / 2
    z = np.full(4, h / 2)
    local = np.r_[np.c_[x, y, -z], np.c_[x, y, z]]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(location, float) + local @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T


def box_of(label):
    d = label['3d_dimensions']
    loc = [label['3d_location'][k] for k in 'xyz']
    return loc, corners(loc, d['l'], d['w'], d['h'], label['rotation'])


def in_bev(location):
    return BEV_X[0] <= location[0] <= BEV_X[1] and BEV_Y[0] <= location[1] <= BEV_Y[1]


class Camera:
    """dair/calib (1920 x 1080, Pixelmitten-Konvention), umgerechnet auf die Ausgabegroesse wie simapi.dataset."""

    def __init__(self, calib_dir, size):
        K = np.array(_load(os.path.join(calib_dir, 'camera_intrinsic.json'))['cam_K'], float).reshape(3, 3)
        sx, sy = size[0] / FULLHD[0], size[1] / FULLHD[1]
        K[:2] *= [[sx], [sy]]
        K[:2, 2] += [0.5 * sx - 0.5, 0.5 * sy - 0.5]
        ext = _load(os.path.join(calib_dir, 'virtuallidar_to_camera.json'))
        self.K, self.scale = K, (sx, sy)
        self.R = np.array(ext['rotation'], float)
        self.t = np.array(ext['translation'], float).reshape(3)

    def to_camera(self, points):
        return np.asarray(points, float) @ self.R.T + self.t

    def _uv(self, p):
        q = self.K @ p
        return q[:2] / q[2]

    def segments(self, points, edges):
        """Kanten (Bodensystem) -> Liste von Pixelstrecken, an der Nahebene abgeschnitten."""
        c = self.to_camera(points)
        out = []
        for i, j in edges:
            a, b = c[i], c[j]
            if a[2] < NEAR and b[2] < NEAR:
                continue
            if a[2] < NEAR:
                a = a + (b - a) * (NEAR - a[2]) / (b[2] - a[2])
            elif b[2] < NEAR:
                b = b + (a - b) * (NEAR - b[2]) / (a[2] - b[2])
            out.append((self._uv(a), self._uv(b)))
        return out


class Painter:
    def __init__(self, cam, size, show_2d):
        self.cam, self.size, self.show_2d = cam, size, show_2d
        self.s = size[1] / FULLHD[1]
        self.thick = max(1, round(2 * self.s))
        self.font = 0.45 * max(self.s, 0.6)
        # Rand des BEV-Rasters auf der Fahrbahn (z = 0), fein abgetastet, damit die Nahebene sauber schneidet
        (x0, x1), (y0, y1) = BEV_X, BEV_Y
        n = 200
        ring = np.r_[np.c_[np.linspace(x0, x1, n), np.full(n, y0)], np.c_[np.full(n, x1), np.linspace(y0, y1, n)],
                     np.c_[np.linspace(x1, x0, n), np.full(n, y1)]]
        self.border = cam.segments(np.c_[ring, np.zeros(len(ring))], [(i, i + 1) for i in range(len(ring) - 1)])
        self.status_scale = 0.55 * max(self.s, 0.6)
        self.line_h = cv2.getTextSize('Ag', FONT, self.status_scale, 1)[0][1] + 10
        self.status_h = (2 * self.line_h + 10 + 1) // 2 * 2  # zwei Zeilen, gerade Hoehe fuer yuv420p

    def line(self, img, a, b, color, thick, dash=0):
        if dash:
            ok, p, q = cv2.clipLine((0, 0, self.size[0], self.size[1]),
                                    tuple(int(round(v)) for v in np.clip(a, -1e6, 1e6)),
                                    tuple(int(round(v)) for v in np.clip(b, -1e6, 1e6)))
            if not ok:
                return
            p, q = np.array(p, float), np.array(q, float)
            k = max(1, int(np.hypot(*(q - p)) // dash))
            for j in range(0, k, 2):
                u, v = p + (q - p) * j / k, p + (q - p) * min(j + 1, k) / k
                cv2.line(img, tuple(int(x) for x in u), tuple(int(x) for x in v), color, thick, cv2.LINE_AA)
        else:
            f = lambda v: tuple(int(round(x * 16)) for x in np.clip(v, -1e6, 1e6))
            cv2.line(img, f(a), f(b), color, thick, cv2.LINE_AA, shift=4)

    def text(self, img, xy, s, color, scale=None):
        scale = scale or self.font
        x, y = int(xy[0]), int(xy[1])
        # Kontur aus versetzten Kopien: mit groesserer Dicke laeuft der Text in OpenCV 5 breiter
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            cv2.putText(img, s, (x + dx, y + dy), FONT, scale, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, s, (x, y), FONT, scale, color, 1, cv2.LINE_AA)

    def box(self, img, pts, color, thick, dash=0, caption=None):
        for a, b in self.cam.segments(pts, EDGES):
            self.line(img, a, b, color, thick, dash)
        bottom, front = pts[:4].mean(0), pts[:2].mean(0)
        for a, b in self.cam.segments(np.array([bottom, front]), [(0, 1)]):
            self.line(img, a, b, color, thick, dash)
        if caption:
            uv = [p for seg in self.cam.segments(pts, EDGES) for p in seg]
            if uv:
                uv = np.array(uv)
                (tw, th), _ = cv2.getTextSize(caption, FONT, self.font, 1)
                x = min(max(uv[:, 0].min(), 2), self.size[0] - tw - 2)
                y = min(max(uv[:, 1].min() - 4, th + 2), self.size[1] - 4)
                self.text(img, (x, y), caption, color)

    def draw(self, img, labels, hidden=()):
        for a, b in self.border:
            self.line(img, a, b, BORDER, 1)
        for pts in hidden:
            self.box(img, pts, HIDDEN, 1, dash=max(4, round(6 * self.s)))
        # fern zuerst, damit nahe Boxen oben liegen
        items = sorted(labels, key=lambda lb: -np.linalg.norm(self.cam.to_camera([box_of(lb)[0]])[0]))
        for lb in items:
            loc, pts = box_of(lb)
            inside = in_bev(loc)
            color = COLORS[GROUP.get(lb['type'], 'Car')] if inside else OUTSIDE
            caption = '%s %.0fm' % (lb['type'], np.linalg.norm(self.cam.to_camera([loc])[0]))
            if lb.get('occluded_state'):
                caption += ' o%d' % lb['occluded_state']
            if lb.get('truncated_state'):
                caption += ' t%d' % lb['truncated_state']
            if self.show_2d:
                sx, sy = self.cam.scale
                b2 = lb['2d_box']
                cv2.rectangle(img, (int(b2['xmin'] * sx), int(b2['ymin'] * sy)),
                              (int(b2['xmax'] * sx), int(b2['ymax'] * sy)), color, 1, cv2.LINE_AA)
            self.box(img, pts, color, self.thick if inside else 1, caption=caption)
        return img

    def status(self, lines, legend):
        """Infozeile unter dem Bild: zwei Zeilen links, legend [(text, farbe)] rechts oben."""
        scale = self.status_scale
        strip = np.full((self.status_h, self.size[0], 3), 25, np.uint8)
        for k, s in enumerate(lines[:2]):
            self.text(strip, (10, (k + 1) * self.line_h), s, (235, 235, 235), scale)
        x = self.size[0] - 10
        for s, color in reversed(legend):
            x -= cv2.getTextSize(s, FONT, scale, 1)[0][0]
            self.text(strip, (x, self.line_h), s, color, scale)
            x -= 18
        return strip


class VideoOut:
    def __init__(self, path, size, fps):
        self.gen = imageio_ffmpeg.write_frames(path, size, fps=fps, codec='libx264', pix_fmt_in='rgb24',
                                               pix_fmt_out='yuv420p', quality=None, macro_block_size=1,
                                               output_params=['-crf', '18', '-preset', 'veryfast'],
                                               ffmpeg_log_level='error')
        self.gen.send(None)

    def write(self, frame, i):
        self.gen.send(np.ascontiguousarray(frame))

    def close(self):
        self.gen.close()


class JpgOut:
    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        self.path = path

    def write(self, frame, i):
        if not cv2.imwrite(os.path.join(self.path, '%04d.jpg' % i), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError('Bild nicht geschrieben: %s' % os.path.join(self.path, '%04d.jpg' % i))


def hidden_boxes(clip, meta, fr, labels, i):
    """Aufgezeichnete Objekte von Frame i ohne Label (Ecken im Bodensystem)."""
    path = os.path.join(clip, meta['labels'] % i)
    if not os.path.isfile(path):
        raise FileNotFoundError('%s fehlt; --hidden braucht die Rohdaten (frames/, nicht mit --drop-frames geloescht)'
                                % path)
    centers = np.array([box_of(lb)[0] for lb in labels]).reshape(-1, 3)
    out = []
    for obj in _load(path)['objects']:
        center, l, w, h, yaw = fr.box(obj)
        if len(centers) and np.linalg.norm(centers - center, axis=1).min() < 0.05:
            continue  # hat ein Label (dair rundet auf 0,1 mm)
        out.append(corners(center, l, w, h, yaw))
    return out


def render(clip, out=None, cosmos_dir=None, size=FULLHD, frames=None, show_2d=False, show_hidden=False):
    """Overlay eines Clips schreiben; gibt den Ausgabepfad zurueck."""
    name = os.path.basename(os.path.normpath(clip))
    meta = _load(os.path.join(clip, 'clip.json'))
    dair_dir = os.path.join(clip, 'dair')
    if not os.path.isfile(os.path.join(dair_dir, 'meta.json')):
        raise FileNotFoundError('%s fehlt; simapi.convert zuerst erfolgreich abschliessen'
                                % os.path.join(dair_dir, 'meta.json'))
    info = _load(os.path.join(dair_dir, 'meta.json'))
    labeled = info['frames']
    if frames:
        missing = sorted(set(frames) - set(labeled))
        if missing:
            raise ValueError('%s: Frames %s haben kein Label (beschriftet: %d..%d, Schritt %d)' % (
                name, missing, labeled[0], labeled[-1], labeled[1] - labeled[0] if len(labeled) > 1 else 1))
        wanted = sorted(set(frames))
    else:
        wanted = list(labeled)
    if not wanted:
        raise ValueError('%s: keine beschrifteten Frames' % name)

    if cosmos_dir:
        video, origin = cosmos_result(cosmos_dir, name), os.path.basename(os.path.normpath(cosmos_dir))
        if video is None:
            raise FileNotFoundError('%s: kein Cosmos-Ergebnis in %s' % (name, cosmos_dir))
    else:
        video, origin = os.path.join(clip, 'cosmos', 'rgb.mp4'), 'carla'
        if not os.path.isfile(video):
            raise FileNotFoundError(video)
    expected = meta['frames']
    manifest = os.path.join(clip, 'cosmos', 'manifest.json')
    if os.path.isfile(manifest):
        expected = _load(manifest)['video_frames']
    out = out or os.path.join(clip, 'overlay-%s.mp4' % origin)

    fr = Frame(meta, np.array(info['ground']['normal']), np.array(info['ground']['point'])) if show_hidden else None
    cam = Camera(os.path.join(dair_dir, 'calib'), size)
    painter = Painter(cam, size, show_2d)
    legend = [('Car', COLORS['Car']), ('Cyclist', COLORS['Cyclist']), ('Pedestrian', COLORS['Pedestrian']),
              ('ausserhalb BEV', OUTSIDE), ('BEV-Rand', BORDER)] + ([('ohne Label', HIDDEN)] if show_hidden else [])

    reader = imageio_ffmpeg.read_frames(video)
    meta_v = next(reader)
    vw, vh = meta_v['size']
    writer = None
    count, todo = 0, set(wanted)
    try:
        if abs(meta_v['fps'] - meta['fps']) > 0.01:
            raise ValueError('%s: %.3f fps statt %.3f; Labels passen zeitlich nicht' % (video, meta_v['fps'], meta['fps']))
        stretch = 0.03 if os.path.basename(video) == 'vision.mp4' else 0  # wie simapi.dataset
        if abs(vw * meta['camera']['height'] / (vh * meta['camera']['width']) - 1) > stretch:
            raise ValueError('%s ist %dx%d; Seitenverhaeltnis passt nicht zur Kalibrierung' % (video, vw, vh))
        if out.lower().endswith('.mp4'):
            # Echtzeit, wenn jeder N-te Frame beschriftet ist
            step = min((b - a for a, b in zip(wanted, wanted[1:])), default=1)
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            writer = VideoOut(out, (size[0], size[1] + painter.status_h), meta['fps'] / step)
        else:
            writer = JpgOut(out)
        for i, buf in enumerate(reader):
            count = i + 1
            if i not in todo:
                continue
            todo.discard(i)
            img = np.frombuffer(buf, np.uint8).reshape(vh, vw, 3)
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if vw > size[0] else cv2.INTER_CUBIC)
            labels = _load(os.path.join(dair_dir, 'label', '%04d.json' % i))
            hidden = hidden_boxes(clip, meta, fr, labels, i) if show_hidden else ()
            painter.draw(img, labels, hidden)
            n_in = sum(in_bev(box_of(lb)[0]) for lb in labels)
            n_hidden = info.get('hidden', {}).get('%04d' % i, 0)
            strip = painter.status([
                '%s  Frame %04d/%d  Bild: %s (%dx%d)' % (name, i, meta['frames'] - 1, origin, vw, vh),
                'Labels: %d  (im BEV-Raster %d, ausserhalb %d)  |  aufgezeichnet ohne Label: %d'
                % (len(labels), n_in, len(labels) - n_in, n_hidden)], legend)
            writer.write(np.vstack([img, strip]), i)
    finally:
        reader.close()
        if isinstance(writer, VideoOut):
            writer.close()
    if count != expected:
        raise ValueError('%s: %d statt %d Frames; Labels passen zeitlich nicht' % (video, count, expected))
    if todo:
        raise ValueError('%s: Frames %s fehlen im Video' % (video, sorted(todo)))
    return out


def _size(s):
    try:
        w, h = (int(v) for v in s.lower().split('x'))
    except ValueError:
        raise argparse.ArgumentTypeError('BREITExHOEHE, z.B. 1280x720')
    if w < 16 or h < 16 or w % 2 or h % 2:
        raise argparse.ArgumentTypeError('mindestens 16x16, gerade Zahlen (H.264)')
    return w, h


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m simapi.overlay', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('paths', nargs='+', metavar='ORDNER', help='Clip-Ordner oder Export-Ordner mit Clips')
    p.add_argument('--cosmos', metavar='ORDNER',
                   help='Ausgabeordner von Cosmos (Transfer2.5: <clipname>.mp4, Cosmos 3: <clipname>/vision.mp4); '
                        'ohne: CARLA-Bild')
    p.add_argument('--out', metavar='DATEI.mp4|ORDNER',
                   help='Video (.mp4) oder Ordner fuer JPGs; Default <clip>/overlay-<quelle>.mp4 (nur bei einem Clip)')
    p.add_argument('--frames', type=int, nargs='+', metavar='N', help='nur diese Frames (Default: alle beschrifteten)')
    p.add_argument('--size', type=_size, default=FULLHD, metavar='BxH',
                   help='Bildgroesse (Default 1920x1080; Training ohne --fullhd: 1280x720)')
    p.add_argument('--2d', dest='show_2d', action='store_true', help='2D-Boxen mit einzeichnen')
    p.add_argument('--hidden', action='store_true',
                   help='aufgezeichnete Objekte ohne Label rot gestrichelt zeigen (braucht frames/*.json)')
    args = p.parse_args(argv)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    clips = list(find_clips(args.paths))
    if not clips:
        log.error('Keine Clips (Ordner mit clip.json) gefunden')
        return 1
    if args.out and len(clips) > 1:
        p.error('--out geht nur mit einem Clip; ohne --out landet jedes Video in seinem Clip-Ordner')
    failed = 0
    for clip in clips:
        try:
            out = render(clip, args.out, args.cosmos, args.size, args.frames, args.show_2d, args.hidden)
            log.info('%s: %s', clip, out)
        except (OSError, ValueError, KeyError) as e:
            log.error('%s: %s', clip, e)
            failed += 1
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
