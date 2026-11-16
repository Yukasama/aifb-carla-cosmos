"""CARLA-Verkehr mit Roadside-Kamera, Bounding Boxes und Clips fuer Cosmos und die Detektoren.

Die Kamera nimmt auf, was der Spectator (das CARLA-Fenster) sieht. Die Welt
laeuft synchron mit festem Tick (1/16 s), ein Tick ist ein Frame der Clips.

Mit --stdin-control liest der laufende Prozess pro stdin-Zeile entweder
- eine JSON-Liste mit denselben Flags, z.B. ["--number-of-vehicles", "50"]:
  der komplette Sollzustand, fehlende Flags bekommen ihren Default, oder
- {"save_camera": "Name"}: aktuelle Spectator-Pose in cameras.json speichern
  (gleicher Name ueberschreibt), oder
- {"goto_camera": "Name"}: Spectator an die gespeicherte Pose setzen.
EOF auf stdin beendet den Prozess sauber (Actors weg, Sync-Modus aus); unter Windows
ist das der einzige saubere Weg, ein Signal wuerde den Prozess hart beenden.

Mit --status-file schreibt er etwa fuenfmal pro Sekunde die Objekte im Bild
(Modell, Distanz, sichtbare Pixel, verdeckt) als JSON in diese Datei.
"""

import argparse

FPS = 16  # Trainings-fps von Cosmos-Transfer2.5, im empfohlenen Bereich von Cosmos 3 (10-30)
RESOLUTIONS = {'fullhd': (1920, 1080), 'hd': (1280, 720),
               '1920x1080': (1920, 1080), '1280x720': (1280, 720)}

# Wirken nur beim Start; im laufenden Betrieb werden Aenderungen ignoriert.
START_ONLY = ('host', 'port', 'tm_port', 'stdin_control', 'status_file')


def at_least(lo, kind=int):
    def convert(s):
        v = kind(s)
        if v < lo:
            raise argparse.ArgumentTypeError(f'mindestens {lo}')
        return v
    return convert


def build_parser():
    p = argparse.ArgumentParser(prog='python -m simapi', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group('Verbindung')
    g.add_argument('--host', default='127.0.0.1')
    g.add_argument('-p', '--port', type=int, default=2000)
    g.add_argument('--tm-port', type=int, default=8000, help='Traffic-Manager-Port')
    g.add_argument('--stdin-control', action='store_true', help='Live-Updates von stdin lesen (siehe oben)')
    g.add_argument('--status-file', metavar='PFAD',
                   help='Objekte im Bild laufend als JSON hierhin schreiben (misst dafuer immer die Verdeckung)')

    g = p.add_argument_group('Verkehr')
    g.add_argument('-n', '--number-of-vehicles', type=at_least(0), default=30)
    g.add_argument('-w', '--number-of-walkers', type=at_least(0), default=10)

    g = p.add_argument_group('Bounding Boxes')
    g.add_argument('--max-distance', type=at_least(0.0, float), default=50.0, help='in m, 0 = unbegrenzt')
    g.add_argument('--hide-occluded', action='store_true', help='verdeckte Objekte bekommen keine Box')
    g.add_argument('--min-visible-pixels', type=at_least(1), default=50,
                   help='mit weniger sichtbaren Pixeln gilt ein Objekt als verdeckt')

    g = p.add_argument_group('Export')
    g.add_argument('--quality', '--resolution', choices=RESOLUTIONS, default='fullhd',
                   help='Aufnahmeaufloesung: fullhd (1920x1080, Default) oder hd (1280x720)')
    g.add_argument('--export', action='store_true',
                   help='Clips mit fester Kamera aufnehmen und fertige Clips umwandeln (python -m simapi.convert): '
                        'Eingabe fuer Cosmos-Transfer2.5 und 3D-Labels fuer BEVHeight, HeightFormer und CoBEV. '
                        'Beschriftet wird alles Sichtbare, --max-distance gilt nur fuer Anzeige und Statusdatei')
    g.add_argument('--export-dir', default='_out')
    g.add_argument('--clip-frames', type=at_least(1), default=93,
                   help='Aufgenommene Frames pro Clip (Default 93); Cosmos-Export fuellt kurze Restclips auf')
    g.add_argument('--drop-frames', action='store_true',
                   help='Rohbilder (frames/, etwa 500 MB pro Clip) nach erfolgreicher Umwandlung loeschen')
    return p
