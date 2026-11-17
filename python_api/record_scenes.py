"""Szenen ohne Fenster aufnehmen: startet CARLA headless und nimmt Clips aus den gespeicherten
Kamerapositionen (cameras.json) auf, gleichmaessig verteilt.

    uv run python record_scenes.py                                  # 20000 Frames auf alle Positionen, dazu Holdout
    uv run python record_scenes.py --frames 93                      # Testlauf: ein Clip, dazu ein Holdout-Clip
    uv run python record_scenes.py --frames 4000 --positions Museum "Kreuzung 1"

Gezaehlt werden aufgenommene Frames (16 fps), immer in ganzen Clips zu 93 Frames: 20000 Frames
auf 8 Positionen sind 216 Clips, je Position 27 Clips = 2511 Frames, insgesamt 20088.
Die Holdout-Position (--holdout, Default "Holdout") zaehlt nicht dazu: Sie bekommt zusaetzlich so
viele Clips wie eine Trainingsposition (hier 27 = 2511 Frames, zusammen 22599) und landet in
<export-dir>/holdout/. Das sind reine Testdaten: run_cosmos.py und der Trainingsdatensatz sehen nur
die Clips direkt im Export-Ordner, und clip.json traegt "holdout": true, womit simapi.dataset und
run_cosmos.py solche Clips ablehnen. --no-holdout nimmt sie nicht auf.
--quality fullhd (Default) nimmt 1920x1080 auf, --quality hd nimmt 1280x720 auf.
Jeder fertige Clip wird wie in Sim Control umgewandelt: cosmos/ fuer Cosmos-Transfer2.5 und
Cosmos 3, dair/ fuer die Detektoren mit jedem LABEL_EVERY-ten Frame als Trainingsbild.

Laeuft unter HOST:PORT schon ein CARLA-Server, bricht das Skript ab, damit nicht versehentlich eins
mit Fenster aufnimmt. Mit --use-running wird er benutzt; eine Simulation, die ihn gerade steuert
(Sim Control), vorher trennen, zwei Programme duerfen die Welt nicht gleichzeitig ticken.
Ctrl+C (oder EOF auf stdin mit --stop-on-eof) bricht sauber ab: fertige Clips bleiben, laufende
Umwandlungen laufen zu Ende, ein selbst gestartetes CARLA wird beendet.
"""

import _thread
import argparse
import collections
import copy
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from simapi import clips, cosmos, dair, presets
from simapi.config import FPS, RESOLUTIONS, build_parser
from simapi.convert import replace_dir
from simapi.geometry import decode_instance, depth_m

# ---------- Einstellungen ----------
# Der erste vorhandene wird gestartet; unter Linux heisst der Starter normalerweise CarlaUE4.sh
CARLA_EXES = [r'C:\Users\yukas\Downloads\CARLA_0.9.16\CarlaUE4.exe',
              '/opt/carla-simulator/0.9.16/CARLAUE4.exe',
              '/opt/carla-simulator/0.9.16/CarlaUE4.sh']
CARLA_ARGS = ['-RenderOffScreen', '-nosound']  # ohne Fenster; die Kameras rendern trotzdem
HOST, PORT, TM_PORT = '127.0.0.1', 2000, 8000
TOTAL_FRAMES = 20_000         # aufgenommene Frames ueber alle Positionen
CLIP_FRAMES = 93              # Frames pro Clip = ein Abschnitt bei Cosmos
LABEL_EVERY = 1               # jeder Frame bekommt Detektor-Labels
VEHICLES, WALKERS = 90, 75
EXPORT_DIR = '_out'           # relativ zu diesem Ordner
HOLDOUT = 'Holdout'           # Kameraposition fuer Testdaten, nie fuers Training (--holdout)
HOLDOUT_DIR = 'holdout'       # deren Clips: <export-dir>/holdout/
DROP_FRAMES = False           # Rohbilder nach der Umwandlung loeschen (spart rund 520 MB pro Clip)
PARALLEL_CONVERSIONS = 2      # Umwandlungen gleichzeitig neben der Aufnahme
WARMUP_SECONDS = 10.0         # Simulationszeit nach dem Spawnen, bis der Verkehr fliesst
SETTLE_SECONDS = 2.0          # nach jedem Kamerawechsel: Texturen der neuen Ansicht nachladen lassen
CARLA_START_TIMEOUT = 300     # s
CARLA_STOP_TIMEOUT = 30       # s nach SIGTERM (Linux), dann SIGKILL
# Schaetzung vor der ersten Messung (gemessen headless auf RTX 2070 Super, 3 Kameras in 1920 x 1080,
# 90 Fahrzeuge und 75 Fussgaenger, jeder Frame beschriftet)
GUESS_FPS = 3.0
GUESS_CONVERT_SECONDS = 80    # pro Clip, neben der Aufnahme
RAW_MB_PER_FRAME = 5.6

ROOT = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger('record_scenes')


# ---------- Anzeige ----------

def num(n):
    return '{:,}'.format(n).replace(',', '.')


def dur(seconds):
    s = int(round(seconds))
    if s < 90:
        return '%d s' % s
    m = int(round(s / 60))
    return '%d min' % m if m < 90 else '%d h %02d min' % (m // 60, m % 60)


class Display:
    """Eine Statuszeile, die sich selbst ueberschreibt; Meldungen erscheinen darueber.
    Ohne Terminal (Log-Datei, Sim Control) alle paar Sekunden eine eigene Zeile."""

    def __init__(self):
        self.tty = sys.stdout.isatty()
        self.line = ''
        self.shown = 0.0

    def _width(self):
        return shutil.get_terminal_size().columns - 1

    def status(self, text, force=False):
        now = time.monotonic()
        if not force and now - self.shown < (0.25 if self.tty else 5):
            return
        self.line, self.shown = text, now
        if self.tty:
            w = self._width()
            sys.stdout.write('\r' + text[:w].ljust(w))
            sys.stdout.flush()
        else:
            print(text, flush=True)

    def message(self, text):
        if self.tty and self.line:
            sys.stdout.write('\r' + ' ' * self._width() + '\r')
        print(text, flush=True)
        if self.tty and self.line:
            self.status(self.line, force=True)

    def end(self):
        if self.tty and self.line:
            sys.stdout.write('\n')
        self.line = ''


class Handler(logging.Handler):
    def __init__(self, display):
        super().__init__()
        self.display = display

    def emit(self, record):
        prefix = '' if record.levelno < logging.WARNING else 'Warnung: '
        self.display.message(prefix + record.getMessage())


# ---------- CARLA ----------

def connect(host, port, timeout=2.0):
    import carla
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        client.get_server_version()
        return client
    except RuntimeError:
        return None


def start_carla(args, display, launched):
    """Client; ein selbst gestartetes CARLA landet in `launched` (wird am Ende beendet)."""
    client = connect(args.host, args.port)
    if client is not None:
        if not args.use_running:
            raise SystemExit('Auf %s:%d laeuft schon ein CARLA (womoeglich mit Fenster). Fuer die Aufnahme ohne '
                             'Fenster erst beenden, oder es mit --use-running benutzen' % (args.host, args.port))
        log.info('CARLA laeuft schon auf %s:%d und wird benutzt', args.host, args.port)
        return client
    exe = next((p for p in CARLA_EXES if os.path.isfile(p)), None)
    if exe is None:
        raise SystemExit('CARLA nicht gefunden: %s (CARLA_EXES oben in record_scenes.py anpassen)' % ', '.join(CARLA_EXES))
    kwargs = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt'
              else {'start_new_session': True})  # Ctrl+C geht nicht an CARLA, wir beenden es selbst
    proc = subprocess.Popen([exe, *CARLA_ARGS, '-carla-rpc-port=%d' % args.port], cwd=os.path.dirname(exe),
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
    launched.append(proc)
    t0 = time.monotonic()
    while True:
        display.status('CARLA startet ohne Fenster ... %s' % dur(time.monotonic() - t0))
        client = connect(args.host, args.port)
        if client is not None:
            client.set_timeout(120.0)
            client.get_world()  # erst nach dem Laden der Karte bereit
            log.info('CARLA laeuft (headless, Start %s)', dur(time.monotonic() - t0))
            return client
        if time.monotonic() - t0 > CARLA_START_TIMEOUT:
            raise SystemExit('CARLA antwortet nach %s nicht auf Port %d' % (dur(CARLA_START_TIMEOUT), args.port))
        time.sleep(2)


def _listening_pids(port):
    """Windows: Prozesse, die auf dem Port lauschen (CarlaUE4.exe startet den eigentlichen Server als Kindprozess)."""
    out = subprocess.run(['netstat', '-ano', '-p', 'TCP'], capture_output=True, text=True, errors='replace').stdout
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        # Lauschende Sockets haben keine Gegenstelle; der Zustand selbst ist uebersetzt (LISTENING, ABHOEREN, ...)
        if len(parts) >= 5 and parts[1].endswith(':%d' % port) and parts[2] in ('0.0.0.0:0', '[::]:0') \
                and parts[4].isdigit():
            pids.add(int(parts[4]))
    return pids


def idle(client):
    """Selbst gestartetes CARLA bis zum Beenden anhalten: synchron und ohne Tick rechnet es nichts."""
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 1.0 / FPS
    world.apply_settings(settings)


def stop_carla(proc, port):
    if os.name == 'nt':
        for pid in {proc.pid} | _listening_pids(port):
            subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
    else:  # CarlaUE4.sh startet den Server als Kindprozess: ganze Prozessgruppe beenden
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(CARLA_STOP_TIMEOUT)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:  # haengt beim Beenden: sonst bliebe es verwaist auf der GPU
            os.killpg(proc.pid, signal.SIGKILL)
    print('CARLA beendet', flush=True)


def ensure_map(client, name, display):
    world = client.get_world()
    if world.get_map().name.split('/')[-1] == name:
        return
    settings = world.get_settings()  # Laden im synchronen Modus haengt ohne jemanden, der tickt
    settings.synchronous_mode, settings.fixed_delta_seconds = False, None
    world.apply_settings(settings)
    display.status('Lade Karte %s ...' % name, force=True)
    client.set_timeout(300.0)
    client.load_world(name)
    client.set_timeout(120.0)


def close_sim(sim):
    """Actors weg, Sync-Modus aus, Traffic Manager beenden. Der laeuft in diesem Prozess und reisst ihn
    sonst mit, sobald CARLA beendet wird. Sim.close() ignoriert danach Ctrl+C fuer immer (gedacht fuers
    Prozessende); hier wiederherstellen."""
    handler = signal.getsignal(signal.SIGINT)
    try:
        sim.close()
        sim.tm.shut_down()
    except RuntimeError as e:
        log.warning('Aufraeumen: %s', e)
    finally:
        signal.signal(signal.SIGINT, handler)


# ---------- Umwandlung ----------

class Conversions:
    """Warteschlange: hoechstens `parallel` Umwandlungen gleichzeitig, damit die Aufnahme nicht ausgebremst wird."""

    def __init__(self, parallel, drop_frames, every=LABEL_EVERY):
        self.parallel, self.drop_frames = parallel, drop_frames
        self.every = every
        self.waiting, self.running = collections.deque(), {}
        self.durations, self.failed = [], []

    def add(self, clip):
        self.waiting.append(clip)
        self.poll()

    def poll(self):
        for proc, (clip, t0) in list(self.running.items()):
            if proc.poll() is not None:
                del self.running[proc]
                self.durations.append(time.monotonic() - t0)
                if proc.returncode:
                    self.failed.append(clip)
                    log.warning('Umwandlung fehlgeschlagen: %s (siehe convert.log)', os.path.basename(clip))
        while self.waiting and len(self.running) < self.parallel:
            clip = self.waiting.popleft()
            proc = clips.start_conversion(clip, self.drop_frames, ['--every', str(self.every)])
            self.running[proc] = (clip, time.monotonic())

    def pending(self):
        return len(self.waiting) + len(self.running)

    def seconds_each(self):
        return sum(self.durations) / len(self.durations) if self.durations else None


# ---------- Aufnahme ----------

def view_problem(clip):
    """Warum die Kamera dieser Position unbrauchbar ist, oder None; prueft den ersten Frame eines Clips."""
    with open(os.path.join(clip, 'clip.json'), encoding='utf-8') as f:
        meta = json.load(f)
    images, _ = dair.frame_paths(clip, meta, 0)
    depth = depth_m(cv2.imread(images['depth'], cv2.IMREAD_COLOR))
    tags, _ = decode_instance(cv2.imread(images['instance'], cv2.IMREAD_COLOR))
    near = float(np.median(depth))
    if near < 3.0:
        return ('Tiefen- und Segmentierungskamera sehen ein Objekt direkt vor der Linse (meist %.1f m entfernt), '
                'das im Kamerabild fehlt; die Position etwas verschieben und neu speichern' % near)
    try:
        dair.fit_ground(meta, depth, tags)
    except ValueError as e:
        return str(e)
    return None


def distribute(frames, names):
    """Nur ganze Clips, reihum verteilt (die ersten Positionen bekommen den Rest): lieber ein paar Frames
    mehr als Restclips, die Cosmos mit Standbildern auf 93 Frames auffuellen muesste."""
    if frames < 1 or not names or len(set(names)) != len(names):
        raise ValueError('positive Framezahl und eindeutige Kamerapositionen erforderlich')
    base, extra = divmod(math.ceil(frames / CLIP_FRAMES), len(names))
    plan = {name: [CLIP_FRAMES] * (base + (i < extra)) for i, name in enumerate(names)}
    return {name: group for name, group in plan.items() if group}


def with_holdout(plan, holdout):
    """Holdout zusaetzlich und zuletzt, mit so vielen Clips wie die groesste Trainingsposition."""
    if holdout in plan:
        raise ValueError('Holdout-Position %s steht schon im Trainingsplan' % holdout)
    return {**plan, holdout: [CLIP_FRAMES] * max(map(len, plan.values()))}


def mark_holdout(clip):
    """clip.json als Holdout kennzeichnen; simapi.dataset und run_cosmos.py lehnen solche Clips ab."""
    path = os.path.join(clip, 'clip.json')
    with open(path, encoding='utf-8') as f:
        meta = json.load(f)
    meta['holdout'] = True
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    replace_dir(path + '.tmp', path)  # wiederholt, falls ein Virenscanner die Datei kurz offen haelt


class Session:
    def __init__(self, plan, cfg_off, cfg_on, conversions, display, holdout=None):
        self.plan = plan                    # Position -> Liste von Clip-Laengen
        self.cfg_off, self.cfg_on = cfg_off, cfg_on
        self.holdout = holdout              # diese Position nimmt nach <export-dir>/holdout/ auf
        self.conversions, self.display = conversions, display
        self.total_clips = sum(map(len, plan.values()))
        self.total_frames = sum(map(sum, plan.values()))
        self.done = collections.Counter()  # Position -> fertige Clips
        self.frames_done = collections.Counter()
        self.clips_done = collections.defaultdict(list)
        self.skipped = {}                   # Position -> Grund
        self.position = None
        self.started = None                 # Wanduhr bei der ersten Aufnahme

    def configs(self, name):
        """Einstellungen ohne und mit Aufnahme fuer eine Position; Holdout-Clips in einen eigenen Ordner."""
        if name != self.holdout:
            return self.cfg_off, self.cfg_on
        off, on = copy.copy(self.cfg_off), copy.copy(self.cfg_on)
        off.export_dir = on.export_dir = os.path.join(self.cfg_on.export_dir, HOLDOUT_DIR)
        return off, on

    def on_clip(self, clip):
        if self.position == self.holdout:
            mark_holdout(clip)
        if not self.done[self.position]:  # erster Clip: taugt die Kamera ueberhaupt?
            problem = view_problem(clip)
            if problem:
                self.skipped[self.position] = problem
                log.warning('%s uebersprungen: %s (Clip %s bleibt unumgewandelt liegen)',
                            self.position, problem, os.path.basename(clip))
                return
        self.done[self.position] += 1
        self.clips_done[self.position].append(os.path.basename(clip))
        with open(os.path.join(clip, 'clip.json'), encoding='utf-8') as f:
            self.frames_done[self.position] += json.load(f)['frames']
        self.conversions.add(clip)

    def remaining_clips(self):
        return sum(len(lengths) - self.done[p] for p, lengths in self.plan.items() if p not in self.skipped)

    def remaining_frames(self):
        return sum(sum(lengths) - self.frames_done[p] for p, lengths in self.plan.items() if p not in self.skipped)

    def show(self, sim, index, note=None):
        running = sim.recorder.clip['n'] if sim.recorder is not None and sim.recorder.clip else 0
        frames = sum(self.frames_done.values()) + running
        total = self.total_frames
        wanted, conv = len(self.plan[self.position]), self.conversions
        parts = ['[%d/%d %s%s]' % (index, len(self.plan), self.position,
                                   ', Holdout-Testdaten' if self.position == self.holdout else ''),
                 note or 'Clip %d/%d' % (min(self.done[self.position] + 1, wanted), wanted),
                 '%s/%s Frames (%.1f %%)' % (num(frames), num(total), 100.0 * frames / total)]
        if self.started is not None and frames:
            rate = frames / (time.monotonic() - self.started)
            rest = (self.remaining_frames() - running) / rate
            each = conv.seconds_each() or GUESS_CONVERT_SECONDS  # Umwandlungen koennen hinterherlaufen
            rest = max(rest + each, (conv.pending() + self.remaining_clips()) * each / conv.parallel)
            parts.append('%.1f fps' % rate)
            parts.append('noch ~%s' % dur(rest))
        parts.append('Umwandlung: %d laufen, %d warten' % (len(conv.running), len(conv.waiting)))
        self.display.status(' | '.join(parts))

    def record(self, sim, name, index, frame):
        """Alle Clips einer Position aufnehmen; gibt den aktuellen Frame zurueck."""
        self.position = name
        cfg_off, cfg_on = self.configs(name)
        sim.apply(cfg_off)  # Kameras aus, waehrend der Spectator springt
        sim.goto_camera(name)
        until, ticks = time.monotonic() + SETTLE_SECONDS, 0
        while ticks < 30 or time.monotonic() < until:
            frame = sim.advance(frame)
            ticks += 1
            self.show(sim, index, 'Kamera ausrichten')
        cfg = copy.copy(cfg_on)
        cfg.clip_frames = self.plan[name][0]
        sim.apply(cfg)
        if self.started is None:
            self.started = time.monotonic()
        stalled, completed = 0, self.done[name]
        while self.done[name] < len(self.plan[name]) and name not in self.skipped:
            sim.cfg.clip_frames = self.plan[name][self.done[name]]
            frame = sim.advance(frame)
            self.conversions.poll()
            self.show(sim, index)
            while self.conversions.pending() >= PARALLEL_CONVERSIONS * 4:
                self.conversions.poll()
                self.show(sim, index, 'Warte auf Umwandlungen')
                time.sleep(0.2)
            if self.conversions.failed:
                raise RuntimeError('Umwandlung fehlgeschlagen; siehe convert.log. Rohbilder bleiben erhalten.')
            stalled = stalled + 1 if self.done[name] == completed else 0
            completed = self.done[name]
            if stalled >= max(300, 3 * sim.cfg.clip_frames):
                raise RuntimeError('Kein vollstaendiger Clip nach %d Ticks an %s; Sensoren/Schreibfehler pruefen' %
                                   (stalled, name))
        return frame


def check_plan(plan, args, holdout=None):
    """Plan, Platzbedarf und grobe Dauer ausgeben; bricht ab, wenn der Platz nicht reicht."""
    lengths = [n for group in plan.values() for n in group]
    n_clips, frames = len(lengths), sum(lengths)
    images = sum(len(range(0, n, args.every)) for n in lengths)
    test_images = sum(len(range(0, n, args.every)) for n in plan.get(holdout, []))
    w, h = RESOLUTIONS[args.quality]
    raw_mb = RAW_MB_PER_FRAME * w * h / (1920 * 1080)
    raw_gb = frames * raw_mb / 1024
    # PCD: vier float32-Werte pro Punkt, jeder vierte Pixel in beiden Richtungen.
    out_gb = (n_clips * 70 + images * (w // 4) * (h // 4) * 16 / 1e6) / 1024
    need_gb = out_gb + (PARALLEL_CONVERSIONS * 4 * CLIP_FRAMES * raw_mb / 1024 if args.drop_frames else raw_gb)
    probe = args.export_dir
    while not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    free_gb = shutil.disk_usage(probe).free / 2 ** 30
    print('Plan: %d Clips x %d Frames = %s Frames auf %d Positionen%s, Aufnahme %dx%d'
          % (n_clips, CLIP_FRAMES, num(frames), len(plan) - (holdout in plan),
             ' + Holdout' if holdout in plan else '', *RESOLUTIONS[args.quality]), flush=True)
    for name, group in plan.items():
        print('      %s: %s Frames, %d Clips%s' % (name, num(sum(group)), len(group),
                                                 ' -> %s/ (Testdaten: kein Cosmos, kein Training)' % HOLDOUT_DIR
                                                 if name == holdout else ''), flush=True)
    print('      %s Trainingsbilder%s (jeder %d. Frame), %d Fahrzeuge, %d Fussgaenger'
          % (num(images - test_images), ' + %s Holdout-Bilder' % num(test_images) if holdout in plan else '',
             args.every, VEHICLES, WALKERS), flush=True)
    guess = max(frames / GUESS_FPS, n_clips * GUESS_CONVERT_SECONDS / PARALLEL_CONVERSIONS) + GUESS_CONVERT_SECONDS
    print('      Speicher ~%.1f GB%s, frei %.0f GB'
          % (need_gb, '' if args.drop_frames else ' (davon Rohbilder %.1f GB)' % raw_gb, free_gb), flush=True)
    print('      Dauer grob ~%s (Aufnahme ~%.0f fps, Umwandlung ~%s pro Clip, %d parallel; genauer nach den ersten Clips)'
          % (dur(guess + 60), GUESS_FPS, dur(GUESS_CONVERT_SECONDS), PARALLEL_CONVERSIONS), flush=True)
    print('      Ziel: %s' % args.export_dir, flush=True)
    if need_gb > 0.95 * free_gb:
        raise SystemExit('Zu wenig Platz: braucht ~%.0f GB, frei %.0f GB. --drop-frames oder weniger --frames.'
                         % (need_gb, free_gb))


def stop_on_eof():
    """Sim Control beendet den Lauf, indem es stdin schliesst (wie Ctrl+C)."""
    def wait():
        sys.stdin.read()
        _thread.interrupt_main()
    threading.Thread(target=wait, daemon=True).start()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', type=int, default=TOTAL_FRAMES, help='aufgenommene Frames insgesamt (Default %d)' % TOTAL_FRAMES)
    p.add_argument('--positions', nargs='+', metavar='NAME',
                   help='nur diese Kamerapositionen fuers Training (Default: alle aus cameras.json ausser der Holdout-Position)')
    p.add_argument('--holdout', default=HOLDOUT, metavar='NAME',
                   help='Kameraposition fuer Testdaten (Default "%s"): nie im Training, zusaetzlich so viele Clips wie '
                        'eine Trainingsposition, nach <export-dir>/%s/' % (HOLDOUT, HOLDOUT_DIR))
    p.add_argument('--no-holdout', action='store_true',
                   help='Holdout-Position nicht aufnehmen (bleibt trotzdem aus dem Training)')
    p.add_argument('--export-dir', default=EXPORT_DIR)
    p.add_argument('--quality', '--resolution', choices=RESOLUTIONS, default='fullhd',
                   help='fullhd/1920x1080 (Default) oder hd/1280x720')
    p.add_argument('--every', type=int, default=LABEL_EVERY, help='jeden N-ten Frame fuer Detektoren aufbereiten (Default 1)')
    p.add_argument('--dry-run', action='store_true', help='Plan pruefen, ohne CARLA zu starten')
    p.add_argument('--drop-frames', action='store_true', default=DROP_FRAMES,
                   help='Rohbilder nach der Umwandlung loeschen')
    p.add_argument('--host', default=HOST)
    p.add_argument('--port', type=int, default=PORT)
    p.add_argument('--use-running', action='store_true',
                   help='ein schon laufendes CARLA benutzen statt eins ohne Fenster zu starten')
    p.add_argument('--stop-on-eof', action='store_true', help='bei EOF auf stdin sauber abbrechen (fuer Sim Control)')
    args = p.parse_args(argv)
    if args.frames < 1:
        p.error('--frames mindestens 1')
    if args.every < 1:
        p.error('--every mindestens 1')
    args.export_dir = os.path.abspath(os.path.join(ROOT, args.export_dir))

    display = Display()
    logging.basicConfig(level=logging.WARNING, handlers=[Handler(display)])
    log.setLevel(logging.INFO)
    if args.stop_on_eof:
        stop_on_eof()

    saved = presets.load()
    names = args.positions or [n for n in saved if n != args.holdout]
    unknown = [n for n in names if n not in saved]
    if len(set(names)) != len(names):
        p.error('--positions darf keine Namen doppelt enthalten')
    if args.holdout in names:
        p.error('"%s" ist die Holdout-Position und kommt nie ins Training; mit --holdout eine andere waehlen'
                % args.holdout)
    if not names or unknown:
        raise SystemExit('Unbekannte Kamerapositionen: %s (vorhanden: %s)' % (', '.join(unknown) or '-', ', '.join(saved) or 'keine'))
    holdout = None if args.no_holdout else args.holdout
    if holdout is not None and holdout not in saved:
        raise SystemExit('Holdout-Position "%s" fehlt in cameras.json (vorhanden: %s); --holdout NAME waehlt eine '
                         'andere, --no-holdout nimmt ohne auf' % (holdout, ', '.join(saved)))
    plan = distribute(args.frames, names)
    if holdout is not None:
        plan = with_holdout(plan, holdout)
    check_plan(plan, args, holdout)
    if args.dry_run:
        return 0

    base = ['--host', args.host, '--port', str(args.port), '--tm-port', str(TM_PORT),
            '--number-of-vehicles', str(VEHICLES), '--number-of-walkers', str(WALKERS),
            '--export-dir', args.export_dir, '--clip-frames', str(CLIP_FRAMES), '--quality', args.quality] + \
        (['--drop-frames'] if args.drop_frames else [])
    parser = build_parser()
    conversions = Conversions(PARALLEL_CONVERSIONS, args.drop_frames, args.every)
    session = Session(plan, parser.parse_args(base), parser.parse_args(base + ['--export']), conversions, display,
                      holdout)

    launched, code = [], 1
    try:
        code = run(args, saved, session, conversions, display, launched)
    except SystemExit as e:
        display.end()
        print(e.code if isinstance(e.code, str) else 'Abbruch', file=sys.stderr, flush=True)
    except KeyboardInterrupt:  # noch vor der Aufnahme, etwa beim Start von CARLA
        display.end()
        print('Abgebrochen', flush=True)
    finally:
        if launched:
            stop_carla(launched[0], args.port)
            # Die carla-Bibliothek wuerde danach endlos versuchen, neu zu verbinden: sofort beenden
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
    return code


def run(args, saved, session, conversions, display, launched):
    try:
        import carla
    except ImportError:  # das Wheel 0.9.16 gibt es nur fuer Linux x86_64 und Windows
        raise SystemExit('Python-Paket carla fehlt: aufnehmen geht nur unter Linux und Windows, hier nur --dry-run')
    from simapi.app import Sim
    by_map = {}
    for name in session.plan:
        by_map.setdefault(saved[name]['map'], []).append(name)

    t0, client, sim, finished = time.monotonic(), None, None, False
    try:
        client = start_carla(args, display, launched)
        index = 0
        for map_name, group in by_map.items():
            ensure_map(client, map_name, display)
            # Geparkte Autos der _Opt-Karten sind Mesh statt Actor, also ohne 3D-Box; die Umwandlung
            # verwirft sonst jeden Clip, in dem eins zu sehen ist (simapi.dair, unbeschriftete Objekte)
            client.get_world().unload_map_layer(carla.MapLayer.ParkedVehicles)
            sim = Sim(session.cfg_off)
            sim.on_clip = session.on_clip
            sim.apply(session.cfg_off)
            frame = sim.world.tick()
            for i in range(int(WARMUP_SECONDS * FPS)):
                frame = sim.advance(frame)
                display.status('Verkehr anlaufen lassen auf %s ... %d %%' % (map_name, 100 * i // (WARMUP_SECONDS * FPS)))
            for name in group:
                index += 1
                frame = session.record(sim, name, index, frame)
            close_sim(sim)
            sim = None
        finished = True
    except KeyboardInterrupt:
        display.end()
        log.warning('Abgebrochen')
    except RuntimeError as e:  # CARLA haengt oder ist abgestuerzt
        display.end()
        log.warning('Aufnahme fehlgeschlagen: %s', e)
    finally:
        if sim is not None:
            close_sim(sim)
    rec_seconds = time.monotonic() - session.started if session.started else 0.0
    if launched and client is not None:
        try:
            idle(client)  # Rechenzeit fuer die Umwandlungen; beendet wird CARLA ganz am Ende
        except RuntimeError:
            pass

    # Auf die Umwandlungen warten: `uv run` beendet beim Ende dieses Skripts auch dessen Kindprozesse.
    # Nach einem Abbruch nur die laufenden zu Ende bringen, die wartenden lassen sich nachholen.
    unconverted = []
    if not finished:
        unconverted, conversions.waiting = list(conversions.waiting), collections.deque()
    try:
        while conversions.pending():
            conversions.poll()
            each = conversions.seconds_each()
            rest = ' | noch ~%s' % dur(math.ceil(conversions.pending() / conversions.parallel) * each) if each else ''
            display.status('Umwandlung: %d laufen, %d warten%s (Ctrl+C: nicht abwarten)'
                           % (len(conversions.running), len(conversions.waiting), rest))
            time.sleep(1)
    except KeyboardInterrupt:
        unconverted += [clip for clip, _ in conversions.running.values()] + list(conversions.waiting)
    display.end()

    holdout = session.holdout if session.holdout in session.plan else None
    train = [name for name in session.plan if name != holdout]
    n = sum(session.done.values())
    count = sum(session.frames_done.values())
    images = {name: sum(len(range(0, length, args.every)) for length in group[:session.done[name]])
              for name, group in session.plan.items()}
    print('%s nach %s: %d Clips (%s Frames), %s Trainingsbilder%s, %d Umwandlungen fehlgeschlagen.'
          % ('Fertig' if finished else 'Gestoppt', dur(time.monotonic() - t0), n, num(count),
             num(sum(images[name] for name in train)),
             ' + %s Holdout-Bilder' % num(images[holdout]) if holdout else '', len(conversions.failed)), flush=True)
    for name, wanted in session.plan.items():
        print('  %-24s %d/%d Frames, %d/%d Clips%s%s' % (name, session.frames_done[name], sum(wanted), session.done[name], len(wanted),
                                           '  (Holdout, %s/)' % HOLDOUT_DIR if name == holdout else '',
                                           '  uebersprungen: ' + session.skipped[name] if name in session.skipped else ''),
              flush=True)
    if n and rec_seconds:
        each = conversions.seconds_each()
        print('Gemessen: Aufnahme %.1f fps%s (GUESS_FPS / GUESS_CONVERT_SECONDS oben im Skript)'
              % (count / rec_seconds, ', Umwandlung %s pro Clip' % dur(each) if each else ''), flush=True)
    if unconverted:
        folders = ' '.join('"%s"' % d for d in sorted({os.path.dirname(c) for c in unconverted}))
        print('Nicht umgewandelt: %d Clips. Nachholen (ueberspringt schon umgewandelte):\n'
              '  uv run python -m simapi.convert %s --every %d' % (len(unconverted), folders, args.every),
              flush=True)
    print('Ordner: %s' % args.export_dir, flush=True)
    if holdout:
        print('Holdout: %s (nicht fuer Cosmos und Training)' % os.path.join(args.export_dir, HOLDOUT_DIR), flush=True)
    budgets_met = all(session.frames_done[name] == sum(group) for name, group in session.plan.items())
    complete = finished and budgets_met and not conversions.failed and not unconverted and not session.skipped

    def entry(name):
        return {'requested_frames': sum(session.plan[name]), 'recorded_frames': session.frames_done[name],
                'clips': session.clips_done[name], 'error': session.skipped.get(name)}
    # Oben nur die Trainingsdaten; die Holdout-Position steht getrennt darunter
    report = {'complete': bool(complete), 'requested_frames': sum(sum(session.plan[name]) for name in train),
              'recorded_frames': sum(session.frames_done[name] for name in train),
              'capture_size': list(RESOLUTIONS[args.quality]), 'cosmos_size': list(cosmos.SIZE), 'fps': FPS,
              'label_every': args.every, 'conversion_failures': conversions.failed, 'unconverted': unconverted,
              'positions': {name: entry(name) for name in train},
              'holdout': dict(entry(holdout), position=holdout, dir=HOLDOUT_DIR) if holdout else None}
    os.makedirs(args.export_dir, exist_ok=True)
    path = os.path.join(args.export_dir, 'recording-%d.json' % time.time_ns())
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('Pruefbericht: %s' % path, flush=True)
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
