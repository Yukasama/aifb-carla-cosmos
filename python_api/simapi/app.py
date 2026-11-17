"""Hauptschleife: Welt synchron ticken, Verkehr, Roadside-Kamera, Boxen, Clips fuer Cosmos und die Detektoren."""

import json
import logging
import os
import queue
import signal
import sys
import threading
import time

import carla
import cv2
import numpy as np

from . import boxes, clips, presets
from .camera import FOV, Rig, draw_frame, focal_length, from_preset, pixels, preset_fov, to_preset
from .config import FPS, RESOLUTIONS, START_ONLY, build_parser
from .geometry import decode_instance, object_pixels, to_camera
from .traffic import Traffic

log = logging.getLogger('simapi')
DT = 1.0 / FPS       # s pro Tick
ACTOR_REFRESH = 20   # Ticks zwischen zwei Abfragen aller Actors (auch fremd gespawnte)
STATUS_EVERY = 0.2   # s Wanduhr zwischen zwei Statusdateien
STATUS_FAILURES = 10  # Fehlschlaege in Folge, ab denen gewarnt wird
SPAWN_MARGIN = 100   # px um das Kamerabild, in denen keine Ersatz-Fussgaenger erscheinen
COMMANDS = {'save_camera': str, 'goto_camera': str, 'set_fov': (int, float)}  # Kommando -> Typ des Werts


def _command(msg):
    """{"Kommando": Wert} -> (Kommando, Wert), sonst None."""
    if not isinstance(msg, dict) or len(msg) != 1:
        return None
    kind, value = next(iter(msg.items()))
    if kind not in COMMANDS or not isinstance(value, COMMANDS[kind]) or isinstance(value, bool):
        return None
    return kind, value


def _lens(fov):
    return '%.0f mm, %.1f Grad' % (focal_length(fov), fov)


def read_control(parser, updates, stop):
    """Pro stdin-Zeile Flags (JSON-Liste) oder ein Kommando (JSON-Objekt). EOF -> beenden."""
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            if isinstance(msg, list) and all(isinstance(a, str) for a in msg):
                updates.put(('cfg', parser.parse_args(msg)))
            elif _command(msg):
                updates.put(_command(msg))
            else:
                raise ValueError('JSON-Liste aus Strings, {"save_camera"|"goto_camera": "Name"} '
                                 'oder {"set_fov": Grad} erwartet')
        except SystemExit:
            pass  # argparse hat den Fehler schon ausgegeben
        except ValueError as e:
            log.error('Ungueltiges Update: %s', e)
    stop.set()


class Sim:
    def __init__(self, cfg):
        self.client = carla.Client(cfg.host, cfg.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map_name = self.world.get_map().name.split('/')[-1]
        self.spectator = self.world.get_spectator()
        self.tm = self.client.get_trafficmanager(cfg.tm_port)
        self.traffic = Traffic(self.client, self.world, self.tm)
        self.rig = Rig(self.world)
        self.recorder = None  # beim ersten apply(), Export-Ordner steht dann fest
        self.record_from = None  # erster Frame ohne Debug-Boxen im Bild
        self.cfg = None
        self.fov = FOV  # horizontales Sichtfeld der Kamera; gehoert wie die Pose zur Kameraposition
        self.actors = []
        self.cam = None  # Kamera des letzten Frames
        self.status_due = 0.0
        self.status_failures = 0  # in Folge
        # Fertiger Clip -> Umwandlung im Hintergrund; record_scenes.py haengt hier seine Warteschlange ein
        self.on_clip = lambda clip: clips.start_conversion(clip, self.cfg.drop_frames)

        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.global_percentage_speed_difference(30.0)
        # Immer selbst ticken, auch wenn ein abgestuerzter Lauf die Welt synchron hinterlassen hat
        self.tm.set_synchronous_mode(True)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        self.world.apply_settings(settings)

    def apply(self, new):
        old, self.cfg = self.cfg, new
        if old is not None:
            for k in START_ONLY:
                setattr(new, k, getattr(old, k))

        def changed(k):
            return old is None or getattr(old, k) != getattr(new, k)

        if changed('number_of_vehicles'):
            self.traffic.set_vehicles(new.number_of_vehicles)
        if changed('number_of_walkers'):
            self.traffic.set_walkers(new.number_of_walkers)
        if changed('export_dir'):
            if self.recorder is not None:
                self.recorder.drop('Export-Ordner geaendert')
                self.recorder.close()
            self.recorder = clips.Recorder(new.export_dir, self.map_name, FPS)
        if old is not None and changed('quality'):
            self.recorder.drop('Aufloesung geaendert')
            self.rig.destroy()
        if new.export and (changed('export') or changed('export_dir')):
            os.makedirs(new.export_dir, exist_ok=True)
            self.record_from = None
            log.info('Clips (%d Frames, %d fps) nach %s; Boxen werden solange nicht gezeichnet',
                     new.clip_frames, FPS, os.path.abspath(new.export_dir))
        elif not new.export:
            self.recorder.drop('Export aus')

    def save_camera(self, name):
        try:
            presets.put(name, to_preset(self.map_name, self.spectator.get_transform(), self.fov))
        except (OSError, ValueError) as e:
            log.error('Position "%s" nicht gespeichert: %s', name, e)
            return
        log.info('Position "%s" gespeichert (%s)', name, _lens(self.fov))

    def goto_camera(self, name):
        p = presets.load().get(name)
        if p is None or p['map'] != self.map_name:
            log.warning('Keine gespeicherte Position "%s" fuer %s', name, self.map_name)
            return
        self.spectator.set_transform(from_preset(p))
        self.fov = preset_fov(p)
        log.info('Kamera: %s (%s)', name, _lens(self.fov))

    def set_fov(self, fov):
        if not presets.valid_fov(fov):
            log.error('Sichtfeld %s nicht zwischen %g und %g Grad', fov, *presets.FOV_RANGE)
            return
        self.fov = float(fov)

    def run(self, updates, stop):
        log.info('Laeuft, Ctrl+C beendet.')
        frame = self.world.tick()
        while not stop.is_set():
            new = None
            while not updates.empty():
                kind, value = updates.get_nowait()
                if kind == 'cfg':
                    new = value  # nur der neueste Stand zaehlt
                else:
                    getattr(self, kind)(value)
            if new is not None:
                self.apply(new)
            frame = self.advance(frame)

    def advance(self, frame):
        """Frame verarbeiten (Verkehr, Kamera, Boxen, Aufnahme), dann weiterticken; gibt den neuen Frame zurueck."""
        self.traffic.after_tick()
        self.step(frame)
        return self.world.tick()

    def step(self, frame):
        cfg, world = self.cfg, self.world
        snap = world.get_snapshot()
        if frame % ACTOR_REFRESH == 0:
            self.traffic.refill(self.out_of_view)  # von CARLA geloeschte (angefahrene) Fussgaenger ersetzen
        if self.traffic.dirty or frame % ACTOR_REFRESH == 0:
            self.actors = [a for a in world.get_actors() if a.type_id.startswith(('vehicle.', 'walker.pedestrian.'))]
            self.traffic.dirty = False

        # Bilder dieses Frames; ohne Sensor wird mit der Spectator-Pose projiziert
        images = self.rig.read(frame)
        rgb, inst = images.get('rgb'), images.get('instance')
        view = self.spectator.get_transform()
        img = inst if inst is not None else rgb
        if img is not None:
            seen = object_pixels(*decode_instance(pixels(inst))) if inst is not None else None
            cam = boxes.Camera(img.transform, img.width, img.height, img.fov, seen)
        else:
            cam = boxes.Camera(view, *RESOLUTIONS[cfg.quality], self.fov)
        self.cam = cam

        # Aufnahmen zeichnen alles ab einem sichtbaren Pixel auf, egal wie weit weg; ob es genug zu sehen
        # ist fuer ein Label (nicht verdeckt, nicht nur ein Rest am Bildrand), entscheidet simapi.dair
        dets = boxes.detect(self.actors, snap, cfg, cam, 0 if cfg.export else cfg.max_distance)
        if not cfg.export:
            boxes.draw(world, [d for d in dets if d['boxed']], DT)
            if self.fov < FOV:  # enger als das CARLA-Fenster: Bildausschnitt einzeichnen
                draw_frame(world, view, self.fov, RESOLUTIONS[cfg.quality], DT * boxes.LIFE_FACTOR)
        else:
            # Debug-Boxen sind Teil der Welt und landen im Kamerabild: waehrend der Aufnahme nicht
            # zeichnen und warten, bis die zuletzt gezeichneten (Lebensdauer 1,5 Ticks) weg sind
            if self.record_from is None:
                self.record_from = frame + 2
            if frame >= self.record_from:
                visible = [d for d in dets if d.get('visible_pixels', 0) > 0]
                done = self.recorder.add(snap, visible, images, cam, cfg.clip_frames)
                if done:
                    self.on_clip(done)
        if cfg.status_file and time.monotonic() >= self.status_due:
            self.status_due = time.monotonic() + STATUS_EVERY
            self.write_status(snap, [d for d in dets if not cfg.max_distance or d['distance'] <= cfg.max_distance])
            if cfg.preview_file:
                preview = images.get('preview')
                self.write_preview(preview if preview is not None else images.get('rgb'))

        # Sensoren fuer den naechsten Frame; die Statusdatei zeigt auch die Verdeckung
        occlusion = cfg.hide_occluded or bool(cfg.status_file)
        kinds = (set(clips.KINDS) if cfg.export else set()) | ({'instance'} if occlusion else set())
        if cfg.preview_file and cfg.status_file and not cfg.export:
            kinds.add('preview')  # waehrend der Aufnahme reicht die RGB-Kamera
        self.rig.update(kinds, view, RESOLUTIONS[cfg.quality], self.fov)

    def out_of_view(self, loc):
        """Ein neuer Fussgaenger bei loc taucht nicht im Kamerabild auf (Fuesse und Kopf, mit Rand)."""
        if self.cam is None:
            return True
        c = to_camera(np.array([[loc.x, loc.y, loc.z], [loc.x, loc.y, loc.z + 2.0]]), self.cam.w2c)
        c = c[c[:, 2] > 0.1]
        if not len(c):
            return True  # hinter der Kamera
        u, v = (c[:, :2] / c[:, 2:] @ self.cam.K[:2, :2].T + self.cam.K[:2, 2]).T
        m = SPAWN_MARGIN
        return not ((u > -m) & (u < self.cam.w + m) & (v > -m) & (v < self.cam.h + m)).any()

    def write_status(self, snap, dets):
        try:
            boxes.write_status(self.cfg.status_file, snap, dets, {'fov': self.fov})
            self.status_failures = 0
        except OSError as e:
            # Unter Windows scheitert das Ersetzen gelegentlich, waehrend Sim Control die Datei
            # gerade liest; erst melden, wenn es dauerhaft nicht klappt
            self.status_failures += 1
            if self.status_failures == STATUS_FAILURES:
                log.warning('Statusdatei: %s', e)

    def write_preview(self, image):
        """Kamerabild als JPEG fuer Sim Control; ohne Bild in diesem Frame bleibt das letzte stehen."""
        if image is None:
            return
        bgr = pixels(image)[..., :3]
        w, h = Rig.sensor_size('preview', (image.width, image.height))
        if w != image.width:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode('.jpg', np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        tmp = self.cfg.preview_file + '.tmp'
        try:
            with open(tmp, 'wb') as f:
                f.write(jpg.tobytes())
            os.replace(tmp, self.cfg.preview_file)
        except OSError:
            pass  # Sim Control liest gerade; der naechste Takt klappt

    def close(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # zweites Ctrl+C darf das Aufraeumen nicht abbrechen

        def unsync():
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.tm.set_synchronous_mode(False)

        def close_recorder():
            if self.recorder is not None:
                self.recorder.close()

        for step in (close_recorder, self.rig.destroy, unsync, self.traffic.destroy):
            try:
                step()
            except Exception as e:  # Rest trotzdem aufraeumen
                log.error('Aufraeumen: %s', e)


def main(argv=None):
    parser = build_parser()
    cfg = parser.parse_args(argv)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    updates, stop = queue.Queue(), threading.Event()
    if cfg.stdin_control:
        threading.Thread(target=read_control, args=(parser, updates, stop), daemon=True).start()

    try:
        sim = Sim(cfg)
    except RuntimeError as e:  # Timeout: CARLA laedt noch, haengt, oder zwei Instanzen teilen sich den Port
        log.error('CARLA unter %s:%d antwortet nicht: %s', cfg.host, cfg.port, e)
        sys.exit(1)
    try:
        sim.apply(cfg)
        sim.run(updates, stop)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()
        print('done.')
