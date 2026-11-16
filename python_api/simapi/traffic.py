"""Fahrzeuge (Autopilot) und Fussgaenger spawnen und ihre Anzahl live anpassen."""

import logging
import random

import carla
from carla.command import DestroyActor, FutureActor, SetAutopilot, SpawnActor

log = logging.getLogger(__name__)


def walker_blueprints(world):
    bps = list(world.get_blueprint_library().filter('walker.pedestrian.*'))
    gen2 = [bp for bp in bps if bp.get_attribute('generation').as_int() == 2]
    return gen2 or bps


class Traffic:
    def __init__(self, client, world, tm):
        self.client, self.world, self.tm = client, world, tm
        self.vehicle_bps = list(world.get_blueprint_library().filter('vehicle.*'))
        self.walker_bps = walker_blueprints(world)
        self.controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        self.spawn_points = world.get_map().get_spawn_points()

        self.vehicles = []   # Actor-IDs
        self.walkers = []    # (walker_id, controller_id)
        self.to_start = []   # (controller_id, max_speed): gehen erst, wenn die Actors einen Tick lang existieren
        self.dirty = True    # Actor-Liste hat sich geaendert

        world.set_pedestrians_cross_factor(0.0)

    def _batch(self, commands):
        """Befehle ohne Tick ausfuehren; Actor-ID pro Befehl, None bei Fehler."""
        ids, errors = [], set()
        for r in self.client.apply_batch_sync(commands, False):
            ids.append(None if r.error else r.actor_id)
            if r.error:
                errors.add(r.error)
        for e in errors:
            log.warning(e)
        return ids

    def _destroy(self, ids):
        if ids:
            self._batch([DestroyActor(i) for i in ids])
            self.dirty = True

    def set_vehicles(self, n):
        if n < len(self.vehicles):
            self._destroy(self.vehicles[n:])
            del self.vehicles[n:]
        elif n > len(self.vehicles):
            snap = self.world.get_snapshot()
            taken = [s.get_transform().location for s in map(snap.find, self.vehicles) if s]
            free = [sp for sp in self.spawn_points if all(sp.location.distance(t) > 5.0 for t in taken)]
            random.shuffle(free)
            want = n - len(self.vehicles)
            if want > len(free):
                log.warning('%d Fahrzeuge gewuenscht, aber nur %d freie Spawnpunkte', want, len(free))
            batch = []
            for sp in free[:want]:
                bp = random.choice(self.vehicle_bps)
                for attr in ('color', 'driver_id'):
                    if bp.has_attribute(attr):
                        bp.set_attribute(attr, random.choice(bp.get_attribute(attr).recommended_values))
                bp.set_attribute('role_name', 'autopilot')
                batch.append(SpawnActor(bp, sp).then(SetAutopilot(FutureActor, True, self.tm.get_port())))
            self.vehicles += [i for i in self._batch(batch) if i is not None]
            self.dirty = True
        log.info('Fahrzeuge: %d', len(self.vehicles))

    def set_walkers(self, n):
        if n < len(self.walkers):
            drop = self.walkers[n:]
            del self.walkers[n:]
            for c in self.world.get_actors([c for _, c in drop]):
                c.stop()
            self._destroy([i for pair in drop for i in pair])
        elif n > len(self.walkers):
            batch, speeds = [], []
            for _ in range(n - len(self.walkers)):
                loc = self.world.get_random_location_from_navigation()
                if loc is None:
                    continue
                bp = random.choice(self.walker_bps)
                if bp.has_attribute('is_invincible'):
                    bp.set_attribute('is_invincible', 'false')
                if bp.has_attribute('can_use_wheelchair') and bp.get_attribute('can_use_wheelchair').as_bool():
                    bp.set_attribute('use_wheelchair', 'true' if random.random() < 0.1 else 'false')
                speeds.append(float(bp.get_attribute('speed').recommended_values[1]) if bp.has_attribute('speed') else 0.0)
                batch.append(SpawnActor(bp, carla.Transform(loc)))
            spawned = [(w, s) for w, s in zip(self._batch(batch), speeds) if w is not None]
            ctrls = self._batch([SpawnActor(self.controller_bp, carla.Transform(), w) for w, _ in spawned])
            for (w, speed), c in zip(spawned, ctrls):
                if c is None:
                    self._destroy([w])
                    continue
                self.walkers.append((w, c))
                self.to_start.append((c, speed))
            self.dirty = True
        log.info('Fussgaenger: %d', len(self.walkers))

    def after_tick(self):
        if self.to_start:
            speeds = dict(self.to_start)
            self.to_start = []
            for c in self.world.get_actors(list(speeds)):
                c.start()
                c.go_to_location(self.world.get_random_location_from_navigation())
                c.set_max_speed(speeds[c.id])

    def destroy(self):
        for c in self.world.get_actors([c for _, c in self.walkers]) if self.walkers else []:
            c.stop()
        log.info('Zerstoere %d Fahrzeuge und %d Fussgaenger', len(self.vehicles), len(self.walkers))
        self._destroy(self.vehicles + [i for pair in self.walkers for i in pair])
        self.vehicles, self.walkers = [], []
