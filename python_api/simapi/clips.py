"""Clips aufnehmen: feste Kamera, fester Takt, feste Laenge.

Ein Clip entsteht in <export-dir>/<take>_<nnn>.partial/ und wird erst nach dem
letzten Frame umbenannt; abgebrochene Clips (Kamera bewegt, Frame fehlt,
Export aus) werden verworfen. Ein Take ist eine Folge von Clips mit derselben
Kamera, benannt nach seiner Startzeit. Fertige Clips wandelt simapi.convert um.

<clip>/clip.json                  Map, fps, Frames, Kamera (fest fuer den ganzen Clip)
<clip>/frames/0000.json           Objekte dieses Frames: 3D-Box in Weltkoordinaten (UE), sichtbare Pixel
<clip>/frames/{rgb,depth,instance}/0000.png   Rohbilder, verlustfrei
"""

import collections
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import boxes
from .camera import pixels
from .convert import replace_dir

log = logging.getLogger(__name__)
KINDS = ('rgb', 'depth', 'instance')  # Sensoren, die jeder Frame braucht
PARTIAL = '.partial'
# PNG schreiben kostet mit carla.Image.save_to_disk 2,5 s pro Frame (3 x 1080p), so in Threads 0,07 s
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]
WRITERS = 6
MAX_PENDING = 30  # Bilder in der Warteschlange, danach wartet die Simulation


def _write_png(path, bgr):
    if not cv2.imwrite(path, bgr, PNG):
        raise OSError('PNG nicht geschrieben: %s' % path)


def camera_key(cam):
    """Gleicher Schluessel = gleiche Kamera (Pose auf mm bzw. 1/1000 Grad, Sichtfeld, Bildgroesse)."""
    l, r = cam.transform.location, cam.transform.rotation
    return tuple(round(v, 3) for v in (l.x, l.y, l.z, r.pitch, r.yaw, r.roll, cam.fov)) + (cam.w, cam.h)


class Recorder:
    def __init__(self, export_dir, map_name, fps):
        self.export_dir, self.map_name, self.fps = export_dir, map_name, fps
        self.take = None  # (Name, Kameraschluessel)
        self.index = 0    # naechste Clip-Nummer im Take
        self.clip = None  # laufender Clip
        self.pool = ThreadPoolExecutor(WRITERS, thread_name_prefix='png')

    def add(self, snap, dets, images, cam, length):
        """Einen Frame anhaengen (images: Sensor -> carla.Image oder None).

        Gibt den Ordner zurueck, wenn der Clip damit fertig ist, sonst None.
        """
        if any(images.get(k) is None for k in KINDS):
            self.drop('Frame %d unvollstaendig' % snap.frame)
            return None
        if any(images[k].frame != snap.frame or (images[k].width, images[k].height) != (cam.w, cam.h) for k in KINDS):
            self.drop('Sensoren sind nicht frame- und aufloesungsgleich')
            return None
        if self.clip is not None and snap.frame != self.clip['frame_ids'][-1] + 1:
            self.drop('nicht aufeinanderfolgende CARLA-Frames')
        key = camera_key(cam)
        if self.clip is not None and self.clip['key'] != key:
            self.drop('Kamera bewegt')
        if self.clip is None:
            self._start(key, length, boxes.camera_record(cam))

        c = self.clip
        name = '%04d' % c['n']
        for kind in KINDS:
            bgr = np.ascontiguousarray(pixels(images[kind])[..., :3])  # Kopie: der Sensorpuffer lebt nicht ewig
            c['pending'].append(self.pool.submit(_write_png, os.path.join(c['tmp'], 'frames', kind, name + '.png'), bgr))
        with open(os.path.join(c['tmp'], 'frames', name + '.json'), 'w', encoding='utf-8') as f:
            json.dump(boxes.frame_record(c['n'], snap, dets), f)
        c['n'] += 1
        c['frame_ids'].append(snap.frame)
        try:
            while len(c['pending']) > MAX_PENDING:
                c['pending'].popleft().result()
        except OSError as e:
            self.drop(str(e))
            return None
        return self._finish() if c['n'] >= c['length'] else None

    def drop(self, reason):
        """Laufenden Clip verwerfen."""
        c, self.clip = self.clip, None
        if c is None:
            return
        for f in c['pending']:
            f.exception()  # warten, sonst schreibt ein Thread in den geloeschten Ordner
        shutil.rmtree(c['tmp'], ignore_errors=True)
        log.warning('Clip %s verworfen nach %d von %d Frames: %s', c['name'], c['n'], c['length'], reason)

    def close(self):
        self.drop('Simulation beendet')
        self.pool.shutdown()

    def _new_take(self):
        """Name nach Startzeit; zwei Takes in derselben Sekunde bekommen -2, -3, ..."""
        base = name = time.strftime('%Y%m%d-%H%M%S')
        taken = {e.split('_')[0] for e in os.listdir(self.export_dir)} if os.path.isdir(self.export_dir) else set()
        if self.take is not None:
            taken.add(self.take[0])
        k = 1
        while name in taken:
            k += 1
            name = '%s-%d' % (base, k)
        return name

    def _start(self, key, length, camera):
        if self.take is None or self.take[1] != key:
            self.take, self.index = (self._new_take(), key), 0
        while True:
            name = '%s_%03d' % (self.take[0], self.index)
            self.index += 1
            final = os.path.join(self.export_dir, name)
            if not os.path.exists(final) and not os.path.exists(final + PARTIAL):
                break
        tmp = final + PARTIAL
        for kind in KINDS:
            os.makedirs(os.path.join(tmp, 'frames', kind))
        self.clip = {'name': name, 'dir': final, 'tmp': tmp, 'key': key, 'length': length,
                     'camera': camera, 'n': 0, 'frame_ids': [], 'pending': collections.deque()}

    def _finish(self):
        c = self.clip
        try:
            while c['pending']:
                c['pending'].popleft().result()
        except OSError as e:
            self.drop(str(e))
            return None
        self.clip = None
        meta = {
            'map': self.map_name,
            'fps': self.fps,
            'frames': c['n'],
            'frame_ids': c['frame_ids'],
            'camera': c['camera'],
            'images': {kind: 'frames/%s/%%04d.png' % kind for kind in KINDS},
            'labels': 'frames/%04d.json',
        }
        with open(os.path.join(c['tmp'], 'clip.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=1)
        try:
            replace_dir(c['tmp'], c['dir'])
        except OSError as e:  # Simulation laeuft weiter; der Clip ist vollstaendig, nur nicht umbenannt
            log.error('Clip %s fertig, aber nicht umbenannt (%s); liegt vollstaendig in %s', c['name'], e, c['tmp'])
            return None
        log.info('Clip %s fertig (%d Frames)', c['name'], c['n'])
        return c['dir']


def start_conversion(clip_dir, drop_frames=False, extra=()):
    """Cosmos-Eingabe und Detektor-Labels im Hintergrund bauen (eigener Prozess, ueberlebt das Ende der Simulation)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kwargs = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt'
              else {'start_new_session': True})
    argv = [sys.executable, '-m', 'simapi.convert', os.path.abspath(clip_dir)] + \
        (['--drop-frames'] if drop_frames else []) + list(extra)
    with open(os.path.join(clip_dir, 'convert.log'), 'w', encoding='utf-8') as out:
        return subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, **kwargs)
