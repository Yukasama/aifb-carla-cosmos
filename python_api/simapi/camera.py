"""Roadside-Kamera am Spectator: RGB und Tiefe fuer den Export, Instanz-Segmentierung fuer Verdeckung und Export."""

import logging
import queue

import numpy as np

log = logging.getLogger(__name__)

# 16:9 mit 90 Grad horizontal, wie der Spectator im CARLA-Fenster (dessen Sichtfeld laesst sich nicht aendern)
WIDTH, HEIGHT, FOV = 1920, 1080, 90.0
# Kleinbild-Aequivalent: 36 mm Sensorbreite, 90 Grad = 18 mm
SENSOR_WIDTH_MM = 36.0

FRAME_COLOR = (150, 120, 30)  # gedeckt, Debug-Linien leuchten selbst
FRAME_THICKNESS = 0.004  # m, in 1 m Abstand etwa 2-4 Pixel
PREVIEW_WIDTH = 640  # px

SENSORS = {
    'rgb': 'sensor.camera.rgb',
    'preview': 'sensor.camera.rgb',  # kleine Vorschau fuer Sim Control: dasselbe Bild wie die Aufnahme
    'depth': 'sensor.camera.depth',
    'instance': 'sensor.camera.instance_segmentation',
}


def pixels(image):
    """carla.Image -> (H, W, 4) BGRA."""
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)


def focal_length(fov):
    """Horizontales Sichtfeld (Grad) -> Brennweite in mm (Kleinbild-Aequivalent)."""
    return SENSOR_WIDTH_MM / 2 / np.tan(np.radians(fov) / 2)


def to_preset(map_name, transform, fov=FOV):
    l, r = transform.location, transform.rotation
    return {'map': map_name,
            'x': round(l.x, 3), 'y': round(l.y, 3), 'z': round(l.z, 3),
            'pitch': round(r.pitch, 3), 'yaw': round(r.yaw, 3), 'roll': round(r.roll, 3),
            'fov': round(float(fov), 3)}


def preset_fov(p):
    """Sichtfeld einer gespeicherten Position; aeltere Eintraege ohne 'fov' haben 90 Grad."""
    return float(p.get('fov', FOV))


def from_preset(p):
    import carla
    return carla.Transform(carla.Location(x=p['x'], y=p['y'], z=p['z']),
                           carla.Rotation(pitch=p['pitch'], yaw=p['yaw'], roll=p['roll']))


def draw_frame(world, transform, fov, size, life_time):
    """Bildrand der Kamera in die Welt zeichnen, wenn sie enger als das CARLA-Fenster sieht (Ausschnitt sichtbar)."""
    import carla
    d = 1.0  # m vor der Kamera: nichts dazwischen, und die Linie wird nicht verdeckt
    half_w = d * np.tan(np.radians(fov) / 2)
    half_h = half_w * size[1] / size[0]
    # UE-Kameraachsen: x nach vorn, y nach rechts, z nach oben
    corners = [carla.Location(d, -half_w, half_h), carla.Location(d, half_w, half_h),
               carla.Location(d, half_w, -half_h), carla.Location(d, -half_w, -half_h)]
    corners = [transform.transform(c) for c in corners]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        world.debug.draw_line(a, b, FRAME_THICKNESS, carla.Color(*FRAME_COLOR), life_time)


def _pose(t):
    l, r = t.location, t.rotation
    return (l.x, l.y, l.z, r.pitch, r.yaw, r.roll)


class Rig:
    def __init__(self, world):
        self.world = world
        self.pose = None
        self.size = (WIDTH, HEIGHT)
        self.fov = FOV
        self.sensors = {}  # kind -> [actor, queue, Ticks ohne Wartezeit]

    @staticmethod
    def sensor_size(kind, size):
        """Vorschau mit hoechstens PREVIEW_WIDTH Pixeln Breite, gleiches Seitenverhaeltnis; der Rest in `size`."""
        if kind != 'preview' or size[0] <= PREVIEW_WIDTH:
            return size
        return PREVIEW_WIDTH, round(size[1] * PREVIEW_WIDTH / size[0])

    def update(self, kinds, transform, size=(WIDTH, HEIGHT), fov=FOV):
        """Sensoren auf `kinds` bringen und an `transform` setzen (Groesse und Sichtfeld gelten ab dem Spawn)."""
        if size != self.size or fov != self.fov:
            self.destroy()
            self.size, self.fov = size, fov
        for kind in set(self.sensors) - kinds:
            self._remove(kind)
        pose = _pose(transform)
        if pose != self.pose:
            for actor, _, _ in self.sensors.values():
                actor.set_transform(transform)
            self.pose = pose
        for kind in kinds - set(self.sensors):
            bp = self.world.get_blueprint_library().find(SENSORS[kind])
            w, h = self.sensor_size(kind, size)
            bp.set_attribute('image_size_x', str(w))
            bp.set_attribute('image_size_y', str(h))
            bp.set_attribute('fov', str(float(fov)))
            if bp.has_attribute('motion_blur_intensity'):
                # Bewegte Objekte sonst verwischt: Kanten (Edge-Video) laegen nicht mehr auf Tiefe und Segmentierung
                bp.set_attribute('motion_blur_intensity', '0')
            actor = self.world.spawn_actor(bp, transform)
            q = queue.Queue()
            actor.listen(q.put)
            # Frisch gespawnte Sensoren liefern fuer die naechsten Frames evtl. noch nichts
            self.sensors[kind] = [actor, q, 2]

    def read(self, frame):
        """Bild genau dieses Frames pro Sensor (None, wenn keins kam)."""
        out = {}
        for kind, s in self.sensors.items():
            out[kind] = None
            fresh = s[2] > 0
            s[2] = max(0, s[2] - 1)
            try:
                while True:
                    img = s[1].get(timeout=0.0 if fresh else 2.0)
                    if img.frame >= frame:
                        out[kind] = img if img.frame == frame else None
                        break
            except queue.Empty:
                if not fresh:
                    log.warning('%s: kein Bild fuer Frame %d', kind, frame)
        return out

    def _remove(self, kind):
        actor = self.sensors.pop(kind)[0]
        actor.stop()
        actor.destroy()

    def destroy(self):
        for kind in list(self.sensors):
            self._remove(kind)
