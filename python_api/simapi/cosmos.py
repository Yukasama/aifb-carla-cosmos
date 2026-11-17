"""Eingabe fuer Cosmos-Transfer2.5 und Cosmos 3 aus einem aufgenommenen Clip (ohne CARLA testbar).

<clip>/cosmos/ enthaelt fuenf Videos mit gleicher Laenge, Groesse und fps, Frame fuer Frame
pixelgenau deckungsgleich; beide Modelle lesen dieselben Dateien:

  rgb.mp4            Kamerabild (Transfer2.5: video_path; Cosmos 3 braucht es nicht)
  depth.mp4          inverse Tiefe, ueber den ganzen Clip auf 0..255 normiert (nah = hell),
                     wie Cosmos Video Depth Anything aufbereitet
  seg.mp4            jedes Segment eine feste Farbe, Hintergrund (unbeschriftet) schwarz.
                     Einzeln: Fahrzeuge, Personen, Masten, Ampeln, Schilder; der Rest pro Klasse
  edge.mp4           Canny-Kanten des Kamerabilds wie Cosmos (Preset medium: Schwellen 100/200)
  vis.mp4            weichgezeichnetes Kamerabild wie Cosmos (Bilateralfilter, Preset medium);
                     heisst bei Transfer2.5 "vis", bei Cosmos 3 "blur"

Dazu je Modell eine Spec, Pfade relativ zur Datei, name = Clipname:

  transfer2.5.json   fuer examples/inference.py von cosmos-transfer2.5: edge 1.0, depth 0.5, seg 1.0,
                     ohne vis (siehe transfer25_spec); Ergebnis: <ausgabeordner>/<clipname>.mp4
  cosmos3.json       fuer cosmos_framework.scripts.inference (Cosmos3-Nano/-Super, model_mode
                     video2video); verlangt Bildgroesse, Laenge und fps ausdruecklich und lehnt
                     unbekannte Felder ab. Ergebnis: <ausgabeordner>/<clipname>/vision.mp4
  prompt.json        strukturierte Szenenbeschreibung fuer Cosmos 3 (Schema der Cosmos-3-Beispiele),
                     mit der Zahl der Fahrzeuge und Personen aus den Labels des Clips

1920 x 1080 wie die Aufnahme (16:9): Beide Modelle skalieren selbst auf ihre Arbeitsgroesse, ohne
zu beschneiden (Transfer2.5: 1280 x 704 und zurueck auf die Eingabegroesse, Cosmos-3-Sim-Transfer:
832 x 480), die Labels bleiben deckungsgleich. Kodiert wird verlustfrei (H.264, QP 0, yuv444p),
damit Kanten und Segmentgrenzen nicht verschmieren.
"""

import colorsys
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import imageio_ffmpeg
import numpy as np

from . import dair
from .geometry import decode_instance, depth_m

SIZE = (1920, 1080)
STREAMS = ('rgb', 'depth', 'seg', 'edge', 'vis')
WORKERS = 4  # Threads pro Umwandlung
# Wie Cosmos-Transfer2.5 (datasets/augmentors/control_input.py), ohne Zufall:
EDGE_PRESETS = {'very_low': (20, 50), 'low': (50, 100), 'medium': (100, 200),
                'high': (200, 300), 'very_high': (300, 400)}           # Canny-Schwellen
BLUR_PRESETS = {'very_low': (1, 4), 'low': (4, 4), 'medium': (2, 10),
                'high': (1, 16), 'very_high': (4, 16)}                 # (Verkleinern vor Filter, Herunter-/Hochskalieren danach)
BILATERAL = (30, 150, 100)  # d, sigma_color, sigma_space
# Klassen, deren Objekte einzeln gefaerbt werden (CARLA-Tags): Mast, Ampel, Schild, Fussgaenger, Fahrer, Auto..Fahrrad
THINGS = np.array([6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19])
PROMPT = ('A fixed traffic camera mounted high on a building looks down at an urban street with cars and '
          'pedestrians. Static camera, no camera motion. Realistic cars, real asphalt and sidewalk textures, '
          'natural lighting and shadows. Photorealistic, high detail.')


# ---------- Einzelbilder (numpy, RGB) ----------

def disparity(bgr, size):
    """Inverse Tiefe (1/m), auf size verkleinert."""
    return cv2.resize(1.0 / np.maximum(depth_m(bgr), 1e-3), size, interpolation=cv2.INTER_AREA)


def depth_frame(disp, lo, hi):
    """Min-max-normiert ueber den Clip (lo/hi), nah = hell; grau als RGB."""
    g = np.clip((disp - lo) / (hi - lo + 1e-12) * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def segment_keys(inst_bgr):
    """Instanzbild (BGR) -> Segment pro Pixel: Dinge (Klasse, Actor-ID) einzeln, sonst die Klasse; 0 = unbeschriftet."""
    tags, ids = decode_instance(inst_bgr)
    tags = tags.astype(np.int64)
    return np.where(np.isin(tags, THINGS), (tags << 16) | ids, tags)


def segment_color(key):
    """Feste Farbe pro Segment, verteilt wie bei Cosmos (HSV: Ton 0..1, Saettigung 0.1..1, Helligkeit 0.2..1)."""
    if key == 0:
        return (0, 0, 0)
    rng = np.random.default_rng(key)
    rgb = colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.1, 1), rng.uniform(0.2, 1))
    return tuple(int(c * 255) for c in rgb)


def seg_frame(inst_bgr, size, colors):
    """Instanzbild -> Farbsegmente in size; Pixel exakt (naechster Nachbar), colors: Cache Schluessel -> Farbe."""
    keys = segment_keys(cv2.resize(inst_bgr, size, interpolation=cv2.INTER_NEAREST_EXACT))
    uniq, inv = np.unique(keys, return_inverse=True)
    for k in uniq.tolist():
        if k not in colors:
            colors[k] = segment_color(k)
    palette = np.array([colors[k] for k in uniq.tolist()], np.uint8)
    return palette[inv.reshape(keys.shape)]


def edge_frame(rgb, preset='medium'):
    lo, hi = EDGE_PRESETS[preset]
    return np.repeat(cv2.Canny(rgb, lo, hi)[..., None], 3, axis=2)


def vis_frame(rgb, preset='medium'):
    k, f = BLUR_PRESETS[preset]
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (w // k, h // k), interpolation=cv2.INTER_AREA)
    x = cv2.bilateralFilter(x, *BILATERAL)
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
    x = cv2.resize(x, (int(w / f), int(h / f)), interpolation=cv2.INTER_AREA)
    return cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)


# ---------- Clip ----------

class Writer:
    def __init__(self, path, size, fps, crf):
        # veryfast: verlustfrei bleibt verlustfrei, nur etwa 10 % groesser und doppelt so schnell wie medium
        quality = (['-qp', '0'] if crf == 0 else ['-crf', str(crf)]) + ['-preset', 'veryfast']
        self.gen = imageio_ffmpeg.write_frames(path, size, fps=fps, codec='libx264', pix_fmt_in='rgb24',
                                               pix_fmt_out='yuv444p', quality=None, macro_block_size=1,
                                               output_params=quality, ffmpeg_log_level='error')
        self.gen.send(None)

    def write(self, frame):
        self.gen.send(np.ascontiguousarray(frame))

    def close(self):
        self.gen.close()


def _read(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def transfer25_spec(name, prompt):
    """Gewichte wie das Multicontrol-Beispiel von Transfer2.5 (docs/inference.md), aber ohne vis: vis gibt Farben
    und Licht des CARLA-Bilds vor (Transfer2.5 behandelt es als Stilvorlage), das Ergebnis saehe nach Simulation aus."""
    spec = {'name': name, 'prompt': prompt, 'video_path': 'rgb.mp4'}
    for key, weight in (('edge', 1.0), ('depth', 0.5), ('seg', 1.0)):
        spec[key] = {'control_path': key + '.mp4', 'control_weight': weight}
    return spec


def cosmos3_spec(name, frames, fps, size):
    """Wie die Transfer-Specs der Cosmos-3-Beispiele (cosmos-framework, cosmos3-inference-transfer),
    mit allen vier Steuerungen gleich gewichtet. Ohne negative_prompt_file nimmt Cosmos 3 sein
    mitgeliefertes neg_prompts.json."""
    if size[0] * 9 != size[1] * 16:
        raise ValueError('Roadside-Cosmos-Export braucht 16:9, damit die Kalibrierung ohne Beschnitt passt')
    return {
        'name': name,
        'model_mode': 'video2video',
        'resolution': '720',  # Modell-Tier, keine frei waehlbare Pixelhoehe (1080 ist ungueltig)
        'aspect_ratio': _aspect(size),
        'num_frames': frames,  # = Laenge der Steuervideos
        'fps': fps,
        'shift': 10.0,
        'num_steps': 50,
        'seed': 2026,
        'num_video_frames_per_chunk': 121,
        'num_conditional_frames': 1,
        'num_first_chunk_conditional_frames': 0,
        'share_vision_temporal_positions': True,
        'negative_metadata_mode': 'none',
        'negative_prompt_keep_metadata': False,
        'guidance': 3.0,
        'control_guidance': 1.5,
        'prompt_path': 'prompt.json',
        'edge': {'control_path': 'edge.mp4', 'preset_edge_threshold': 'medium', 'weight': 1.0},
        'blur': {'control_path': 'vis.mp4', 'preset_blur_strength': 'medium', 'weight': 1.0},
        'depth': {'control_path': 'depth.mp4', 'weight': 1.0},
        'seg': {'control_path': 'seg.mp4', 'weight': 1.0},
        'emphasize_control_in_prompt': False,
    }


def _aspect(size):
    g = math.gcd(*size)
    return '%d,%d' % (size[0] // g, size[1] // g)


def count_objects(clip_dir, meta, min_pixels=dair.MIN_PIXELS, min_share=dair.MIN_SHARE):
    """Hoechste Zahl gleichzeitig beschrifteter Fahrzeuge, Zweiraeder und Personen im Clip.

    Gezaehlt wird, was ein Label bekommt (simapi.dair: genug sichtbar), nicht jedes Objekt mit einem Pixel.
    """
    def frame_classes(i):
        return [o['class'] for o in dair.visible_objects(clip_dir, meta, i, min_pixels, min_share)]

    most = {'vehicles': 0, 'two_wheelers': 0, 'pedestrians': 0}
    with ThreadPoolExecutor(WORKERS) as pool:
        per_frame = list(pool.map(frame_classes, range(meta['frames'])))
    for classes in per_frame:
        n = {'pedestrians': classes.count('pedestrian'),
             'two_wheelers': sum(c in ('bicycle', 'motorcycle') for c in classes)}
        n['vehicles'] = len(classes) - n['pedestrians'] - n['two_wheelers']
        most = {k: max(most[k], n[k]) for k in most}
    return most


def lens(fov):
    """Objektiv fuer die Bildbeschreibung nach horizontalem Sichtfeld (Kleinbild: 35 mm = 54 Grad, 70 mm = 29 Grad)."""
    if fov >= 54:
        return 'Wide-angle lens'
    return 'Standard lens' if fov >= 29 else 'Telephoto lens'


def caption(counts, frames, fps, size, fov=90.0):
    """Strukturierte Beschreibung fuer Cosmos 3 im Schema seiner Beispiel-Prompts (prompt.json)."""
    subjects = []
    if counts['vehicles']:
        subjects.append({
            'description': 'Cars, vans, trucks and buses driving along the road lanes',
            'appearance_details': 'Ordinary passenger cars and commercial vehicles in a variety of colors and models',
            'relationship': 'Regular city traffic passing through the scene',
            'location': 'On the road lanes and through the intersection',
            'relative_size': 'Small within frame',
            'orientation': 'Following the direction of their lanes',
            'action': 'Driving, turning and stopping in traffic',
            'number_of_subjects': counts['vehicles'],
        })
    if counts['two_wheelers']:
        subjects.append({
            'description': 'Riders on motorcycles and bicycles',
            'relationship': 'Part of the road traffic',
            'location': 'On the road lanes',
            'relative_size': 'Small within frame',
            'action': 'Riding along the road',
            'number_of_subjects': counts['two_wheelers'],
        })
    if counts['pedestrians']:
        subjects.append({
            'description': 'Pedestrians in everyday clothing',
            'relationship': 'People walking through the street scene',
            'location': 'On the sidewalks and crosswalks',
            'relative_size': 'Small within frame',
            'action': 'Walking',
            'number_of_subjects': counts['pedestrians'],
        })
    return {
        'subjects': subjects,
        'background_setting': ('An urban street with asphalt roads, lane markings, crosswalks, sidewalks, buildings, '
                               'trees, street lights, poles and traffic lights, seen from above.'),
        'lighting': {
            'conditions': 'Natural daylight',
            'shadows': 'Natural shadows cast by buildings, trees and vehicles onto the road',
        },
        'aesthetics': {
            'composition': 'Wide high-angle view of the street from a fixed camera',
            'mood_atmosphere': 'Everyday, calm city traffic',
        },
        'cinematography': {
            'camera_motion': 'Static',
            'framing': 'Wide shot',
            'camera_angle': 'High angle, looking down at the street',
            'depth_of_field': 'Deep',
            'focus': 'Sharp focus across the whole scene',
            'lens_focal_length': lens(fov),
        },
        'style_medium': 'Live-action video',
        'artistic_style': 'Photorealistic traffic camera footage',
        'context': 'A traffic camera mounted high on a building monitors the street',
        'resolution': {'W': size[0], 'H': size[1]},
        'aspect_ratio': _aspect(size),
        'duration': '%gs' % round(frames / fps, 1),
        'fps': fps,
    }


def build(clip_dir, meta, out, size=SIZE, edge='medium', blur='medium', crf=0, prompt=PROMPT,
          min_pixels=dair.MIN_PIXELS, min_share=dair.MIN_SHARE):
    """<clip>/cosmos/-Inhalt nach out schreiben (Ordner existiert und ist leer)."""
    n, fps = meta['frames'], meta['fps']
    # Beide Tokenizer komprimieren zeitlich. Kurze Restclips bis mindestens 93 bzw. 4k+1
    # auffuellen; nur die originalen n Frames bekommen Labels und zaehlen zur Aufnahme.
    video_frames = max(93, 1 + 4 * math.ceil((n - 1) / 4))

    def frame(kind, i):
        return _read(os.path.join(clip_dir, meta['images'][kind] % i))

    def depth_range(i):
        d = disparity(frame('depth', i), size)
        return float(d.min()), float(d.max())

    colors = {}  # Segment -> Farbe; Threads schreiben hoechstens denselben Wert doppelt

    def render(i):
        rgb = cv2.resize(cv2.cvtColor(frame('rgb', i), cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA)
        return {'rgb': rgb, 'edge': edge_frame(rgb, edge), 'vis': vis_frame(rgb, blur),
                'depth': depth_frame(disparity(frame('depth', i), size), lo, hi),
                'seg': seg_frame(frame('instance', i), size, colors)}

    writers = {s: Writer(os.path.join(out, s + '.mp4'), size, fps, crf) for s in STREAMS}
    # OpenCV gibt die GIL frei: Frames parallel rechnen, in Bloecken, damit nicht der ganze Clip im Speicher liegt
    with ThreadPoolExecutor(WORKERS) as pool:
        try:
            # Tiefenbereich ueber den ganzen Clip, wie Cosmos (eine Normierung pro Video)
            ranges = list(pool.map(depth_range, range(n)))
            lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
            for start in range(0, n, 2 * WORKERS):
                for frames in pool.map(render, range(start, min(n, start + 2 * WORKERS))):
                    for s in STREAMS:
                        writers[s].write(frames[s])
            for _ in range(video_frames - n):
                for s in STREAMS:
                    writers[s].write(frames[s])
        finally:
            for w in writers.values():
                w.close()

    name = os.path.basename(os.path.normpath(clip_dir))
    specs = {
        'transfer2.5.json': transfer25_spec(name, prompt),
        'cosmos3.json': cosmos3_spec(name, video_frames, fps, size),
        'prompt.json': caption(count_objects(clip_dir, meta, min_pixels, min_share), video_frames, fps, size,
                               meta['camera'].get('fov', 90.0)),
        'manifest.json': {'source_frames': n, 'video_frames': video_frames, 'padding_frames': video_frames - n,
                          'padding': 'repeat_last', 'fps': fps, 'size': list(size)},
    }
    for file, data in specs.items():
        with open(os.path.join(out, file), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    return len(colors)
