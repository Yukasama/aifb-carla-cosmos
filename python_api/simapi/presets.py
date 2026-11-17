"""Gespeicherte Kamerapositionen in cameras.json (ohne CARLA testbar).

{"Kreuzung Nord": {"map": "Town10HD_Opt", "x": .., "y": .., "z": .., "pitch": .., "yaw": .., "roll": .., "fov": ..}}

fov: horizontales Sichtfeld der Kamera in Grad (optional, fehlt bei aelteren Eintraegen: 90).
"""

import json
import logging
import os

log = logging.getLogger(__name__)

FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cameras.json')
NUMBERS = ('x', 'y', 'z', 'pitch', 'yaw', 'roll')
FOV_RANGE = (10.0, 130.0)  # erlaubtes Sichtfeld in Grad (Sim Control stellt es als Brennweite ein)


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def valid_fov(fov):
    return _number(fov) and FOV_RANGE[0] <= fov <= FOV_RANGE[1]


def valid(p):
    return (isinstance(p, dict) and isinstance(p.get('map'), str)
            and all(_number(p.get(k)) for k in NUMBERS)
            and ('fov' not in p or valid_fov(p['fov'])))


def _read(path):
    try:
        with open(path, encoding='utf-8-sig') as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}


def load(path=FILE):
    """Name -> Position, nur gueltige Eintraege. Fehlende oder kaputte Datei -> leer."""
    try:
        data = _read(path)
    except (OSError, ValueError) as e:
        log.error('%s: %s', path, e)
        return {}
    return {name: p for name, p in data.items() if valid(p)}


def put(name, preset, path=FILE):
    """Eintrag setzen; der Rest der Datei bleibt, wie er ist. Kaputte Datei -> Exception."""
    data = _read(path)
    data[name] = preset
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')
    os.replace(tmp, path)
