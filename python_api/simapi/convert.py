"""Aufgenommene Clips umwandeln: python -m simapi.convert <clip|export-ordner> ...

Pro Clip zwei Ausgaben, jeweils erst in <name>.partial gebaut und dann umbenannt:

  cosmos/   Eingabe fuer Cosmos-Transfer2.5: fuenf Videos und transfer2.5.json (siehe simapi.cosmos)
  dair/     3D-Labels fuer BEVHeight, HeightFormer und CoBEV im Format von DAIR-V2X-I (siehe simapi.dair)

Laeuft nach jedem fertigen Clip automatisch (Ausgabe in <clip>/convert.log). Schon umgewandelte
Teile werden uebersprungen, --force baut sie neu. Den Datensatz baut danach python -m simapi.dataset.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time

import cv2

from . import cosmos, dair

log = logging.getLogger('simapi.convert')
OUTPUTS = ('cosmos', 'dair')
# Threads pro OpenCV-Aufruf. cosmos.build rechnet schon in eigenen Threads (cosmos.WORKERS); mit dem Default
# (ein Thread je Kern) kostete ein Frame gemessen 75 % mehr CPU-Zeit fuer 25 % weniger Laufzeit, und die
# fehlt der Aufnahme, neben der die Umwandlung laeuft. Mit 2 statt 1 rund 3 % mehr CPU, 8 % schneller.
OPENCV_THREADS = 2


def replace_dir(src, dst, tries=40, wait=0.25):
    """os.replace mit Wiederholung: unter Windows halten Virenscanner, Indexer oder ein Editor frisch
    geschriebene Dateien kurz offen, dann scheitert das Umbenennen des Ordners mit Zugriff verweigert."""
    for i in range(tries):
        try:
            return os.replace(src, dst)
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def find_clips(paths):
    """Clip-Ordner (mit clip.json) direkt oder eine Ebene darunter; laufende Aufnahmen (.partial) nicht."""
    for p in paths:
        if os.path.isfile(os.path.join(p, 'clip.json')):
            yield p
        elif os.path.isdir(p):
            for sub in sorted(os.listdir(p)):
                if not sub.endswith('.partial') and os.path.isfile(os.path.join(p, sub, 'clip.json')):
                    yield os.path.join(p, sub)


def _build(clip, name, fn, force):
    """Ausgabe `name` per fn(tmp_ordner) bauen; False, wenn sie schon da ist."""
    out = os.path.join(clip, name)
    if os.path.isdir(out) and not force:
        return False
    tmp = out + '.partial'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    fn(tmp)
    shutil.rmtree(out, ignore_errors=True)
    replace_dir(tmp, out)
    return True


def convert(clip, args):
    with open(os.path.join(clip, 'clip.json'), encoding='utf-8') as f:
        meta = json.load(f)
    have_frames = os.path.isdir(os.path.join(clip, 'frames'))
    todo = [n for n in OUTPUTS if args.force or not os.path.isdir(os.path.join(clip, n))]
    if todo and not have_frames:
        raise FileNotFoundError('Rohbilder (frames/) geloescht, %s laesst sich nicht mehr bauen' % ', '.join(todo))
    if not todo:
        log.info('%s: schon umgewandelt (--force baut neu)', clip)

    limits = dair.Limits(args.min_pixels, args.min_share, args.max_distance)
    classes = {}  # Frame -> Klassen der Objekte mit Label: sammelt dair.build, zaehlt der Cosmos-Prompt

    def videos(out):
        n = cosmos.build(clip, meta, out, args.size, args.edge_preset, args.blur_preset, args.crf, args.prompt,
                         limits, classes)
        log.info('%s: Cosmos-Eingabe, %d Frames, %dx%d, %d Segmente', clip, meta['frames'], *args.size, n)

    def labels(out):
        info = dair.build(clip, meta, out, args.every, args.allow_unlabeled, limits, classes)
        log.info('%s: DAIR-Labels fuer %d Frames, %d Objekte (%d ohne Label: zu weit weg oder zu wenig sichtbar), '
                 'Kamera %.1f m ueber der Fahrbahn', clip, len(info['frames']), info['objects'],
                 sum(info['hidden'].values()), info['camera_height'])
        if info['unlabeled']:
            log.warning('%s: Fahrzeuge oder Personen ohne Actor (ohne 3D-Box) im Bild (Frame: Anzahl) %s',
                        clip, info['unlabeled'])

    # DAIR zuerst: Es rechnet ohnehin fuer jedes Objekt die Sichtbarkeit, der Cosmos-Prompt uebernimmt daraus die
    # Objektzahlen (sonst liest count_objects Tiefe und Instanzbild jedes Frames noch einmal)
    _build(clip, 'dair', labels, args.force)
    _build(clip, 'cosmos', videos, args.force)
    if args.drop_frames and have_frames:
        shutil.rmtree(os.path.join(clip, 'frames'))
        log.info('%s: Rohbilder geloescht', clip)


def _size(s):
    try:
        w, h = (int(v) for v in s.lower().split('x'))
    except ValueError:
        raise argparse.ArgumentTypeError('BREITExHOEHE, z.B. 1280x720')
    if w < 16 or h < 16:
        raise argparse.ArgumentTypeError('mindestens 16x16')
    return w, h


def _positive(s):
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError('mindestens 1')
    return v


def _nonnegative(kind):
    def convert(s):
        v = kind(s)
        if not v >= 0:
            raise argparse.ArgumentTypeError('mindestens 0')
        return v
    return convert


def _share(s):
    v = float(s)
    if not 0 <= v <= 1:
        raise argparse.ArgumentTypeError('zwischen 0 und 1')
    return v


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m simapi.convert', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('paths', nargs='+', metavar='ORDNER', help='Clip-Ordner oder Export-Ordner mit Clips')
    p.add_argument('--force', action='store_true', help='vorhandene cosmos/ und dair/ neu bauen')
    p.add_argument('--drop-frames', action='store_true',
                   help='Rohbilder (frames/) danach loeschen; ein Neuaufbau ist dann nicht mehr moeglich')
    g = p.add_argument_group('Cosmos')
    g.add_argument('--size', type=_size, default=cosmos.SIZE, metavar='BxH', help='Videogroesse (Default 1920x1080)')
    g.add_argument('--edge-preset', choices=cosmos.EDGE_PRESETS, default='medium')
    g.add_argument('--blur-preset', choices=cosmos.BLUR_PRESETS, default='medium')
    g.add_argument('--crf', type=int, choices=range(0, 52), default=0, metavar='0-51',
                   help='H.264-Qualitaet, 0 = verlustfrei (Default)')
    g.add_argument('--prompt', default=cosmos.PROMPT, help='Prompt in transfer2.5.json')
    g = p.add_argument_group('Detektoren')
    g.add_argument('--every', type=_positive, default=1,
                   help='jeden N-ten Frame beschriften (Default 1: alle Frames)')
    g.add_argument('--allow-unlabeled', action='store_true',
                   help='Export trotz sichtbarer Fahrzeuge/Personen ohne Actor (ohne 3D-Box) erlauben')
    g.add_argument('--min-pixels', type=_nonnegative(float), default=dair.MIN_PIXELS, metavar='PIXEL',
                   help='Label erst ab so vielen sichtbaren Pixeln, auf 1920x1080 umgerechnet (Default %(default)g); '
                        'gilt auch fuer die Objektzahlen im Cosmos-Prompt')
    g.add_argument('--min-share', type=_share, default=dair.MIN_SHARE, metavar='0-1',
                   help='Label erst ab diesem sichtbaren Anteil der Box: im Bild und nicht verdeckt (Default %(default)g)')
    g.add_argument('--max-distance', type=_nonnegative(float), default=dair.MAX_DISTANCE, metavar='M',
                   help='Label nur bis zu dieser Entfernung Kamera-Boxmitte in m, 0 = unbegrenzt (Default %(default)g)')
    args = p.parse_args(argv)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    cv2.setNumThreads(OPENCV_THREADS)

    clips = list(find_clips(args.paths))
    if not clips:
        log.error('Keine Clips (Ordner mit clip.json) gefunden')
        return 1
    failed = 0
    for clip in clips:
        try:
            convert(clip, args)
        except Exception as e:  # naechsten Clip trotzdem versuchen
            log.error('%s: %s', clip, e)
            failed += 1
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
