"""Roadside-Kamera am Spectator: RGB und Tiefe fuer den Export, Instanz-Segmentierung fuer Verdeckung und Export."""

import logging
import queue

import numpy as np

log = logging.getLogger(__name__)

# 16:9 mit 90 Grad horizontal, wie der Spectator im CARLA-Fenster
WIDTH, HEIGHT, FOV = 1920, 1080, 90.0

SENSORS = {
    'rgb': 'sensor.camera.rgb',
    'depth': 'sensor.camera.depth',
    'instance': 'sensor.camera.instance_segmentation',
}


def pixels(image):
    """carla.Image -> (H, W, 4) BGRA."""
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)


def to_preset(map_name, transform):
    l, r = transform.location, transform.rotation
    return {'map': map_name,
            'x': round(l.x, 3), 'y': round(l.y, 3), 'z': round(l.z, 3),
            'pitch': round(r.pitch, 3), 'yaw': round(r.yaw, 3), 'roll': round(r.roll, 3)}


def from_preset(p):
    import carla
    return carla.Transform(carla.Location(x=p['x'], y=p['y'], z=p['z']),
                           carla.Rotation(pitch=p['pitch'], yaw=p['yaw'], roll=p['roll']))


def _pose(t):
    l, r = t.location, t.rotation
    return (l.x, l.y, l.z, r.pitch, r.yaw, r.roll)


class Rig:
    def __init__(self, world):
        self.world = world
        self.pose = None
        self.size = (WIDTH, HEIGHT)
        self.sensors = {}  # kind -> [actor, queue, Ticks ohne Wartezeit]

    def update(self, kinds, transform, size=(WIDTH, HEIGHT)):
        """Sensoren auf `kinds` bringen und an `transform` setzen."""
        if size != self.size:
            self.destroy()
            self.size = size
        for kind in set(self.sensors) - kinds:
            self._remove(kind)
        pose = _pose(transform)
        if pose != self.pose:
            for actor, _, _ in self.sensors.values():
                actor.set_transform(transform)
            self.pose = pose
        for kind in kinds - set(self.sensors):
            bp = self.world.get_blueprint_library().find(SENSORS[kind])
            bp.set_attribute('image_size_x', str(size[0]))
            bp.set_attribute('image_size_y', str(size[1]))
            bp.set_attribute('fov', str(FOV))
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
