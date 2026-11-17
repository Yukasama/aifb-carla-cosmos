"""Fahrzeuge (Autopilot) und Fussgaenger spawnen und ihre Anzahl live anpassen."""

import logging
import random

import carla
from carla.command import DestroyActor, FutureActor, SetAutopilot, SpawnActor

log = logging.getLogger(__name__)
SPAWN_TRIES = 20  # zufaellige Orte pro Fussgaenger, bis einer passt


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
        self.walker_target = 0  # gewuenschte Zahl; refill() ersetzt geloeschte
        # (controller_id, walker_id, max_speed): starten erst, wenn der Client den Fussgaenger in einem
        # Snapshot kennt. Vorher liest controller.start() dessen Position als (0, 0, 0), traegt ihn dort in
        # die Navigation ein, und er steht fuer immer im Kartenursprung.
        self.to_start = []
        self.dirty = True    # Actor-Liste hat sich geaendert

        world.set_pedestrians_cross_factor(0.0)

    def _batch(self, commands, quiet=False):
        """Befehle ohne Tick ausfuehren; Actor-ID pro Befehl, None bei Fehler."""
        ids, errors = [], set()
        for r in self.client.apply_batch_sync(commands, False):
            ids.append(None if r.error else r.actor_id)
            if r.error:
                errors.add(r.error)
        for e in errors if not quiet else ():
            log.warning(e)
        return ids

    def _destroy(self, ids, quiet=False):
        """quiet: schon geloeschte Actors (etwa von CARLA entfernte Fussgaenger) nicht melden."""
        if ids:
            self._batch([DestroyActor(i) for i in ids], quiet)
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
        self.walker_target = n
        if n < len(self.walkers):
            drop = self.walkers[n:]
            del self.walkers[n:]
            gone = {w for w, _ in drop}
            self.to_start = [entry for entry in self.to_start if entry[1] not in gone]
            for c in self.world.get_actors([c for _, c in drop]):
                c.stop()
            self._destroy([i for pair in drop for i in pair])
        elif n > len(self.walkers):
            self._spawn_walkers(n - len(self.walkers))
        log.info('Fussgaenger: %d', len(self.walkers))

    def _spawn_walkers(self, k, where=None):
        """k Fussgaenger an zufaelligen Orten der Navigation; where(loc) -> False verwirft einen Ort."""
        batch, speeds = [], []
        for _ in range(k):
            loc = next((loc for loc in (self.world.get_random_location_from_navigation() for _ in range(SPAWN_TRIES))
                        if loc is not None and (where is None or where(loc))), None)
            if loc is None:
                continue
            bp = random.choice(self.walker_bps)
            if bp.has_attribute('is_invincible'):
                # Ueberfahrene fallen sonst als Ragdoll um, und ihr Label stuende trotzdem aufrecht
                bp.set_attribute('is_invincible', 'true')
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
            self.to_start.append((c, w, speed))
        self.dirty = True

    def refill(self, where=None):
        """Fussgaenger ersetzen, die CARLA geloescht hat; gibt deren Zahl zurueck.

        Die Fussgaenger-Navigation von CARLA erklaert angefahrene Fussgaenger fuer tot und loescht sie
        samt Controller (auch unverwundbare); bei 100 Fahrzeugen etwa jeden zehnten pro Minute. Ohne
        Ersatz waeren nach einer Viertelstunde kaum noch welche unterwegs. Neue nur, wo where(loc)
        zustimmt (nicht mitten im Kamerabild).
        """
        if not self.walkers:
            return 0
        snap = self.world.get_snapshot()
        pending = {w for _, w, _ in self.to_start}  # gerade gespawnt, noch nicht im Snapshot
        gone = [(w, c) for w, c in self.walkers if w not in pending and snap.find(w) is None]
        if not gone:
            return 0
        self.walkers = [pair for pair in self.walkers if pair not in gone]
        self._destroy([c for _, c in gone], quiet=True)  # Controller loescht CARLA meist schon mit
        self._spawn_walkers(self.walker_target - len(self.walkers), where)
        log.debug('%d Fussgaenger ersetzt, jetzt %d', len(gone), len(self.walkers))
        return len(gone)

    def after_tick(self):
        """Neue Fussgaenger loslaufen lassen, sobald sie im aktuellen Snapshot stehen (mindestens ein Tick)."""
        if not self.to_start:
            return
        snap = self.world.get_snapshot()
        ready = [entry for entry in self.to_start if snap.find(entry[1]) is not None]
        if not ready:
            return
        self.to_start = [entry for entry in self.to_start if entry not in ready]
        speeds = {c: speed for c, _, speed in ready}
        for c in self.world.get_actors(list(speeds)):
            c.start()
            c.go_to_location(self.world.get_random_location_from_navigation())
            c.set_max_speed(speeds[c.id])

    def destroy(self):
        for c in self.world.get_actors([c for _, c in self.walkers]) if self.walkers else []:
            c.stop()
        log.info('Zerstoere %d Fahrzeuge und %d Fussgaenger', len(self.vehicles), len(self.walkers))
        self._destroy(self.vehicles)
        # Seit dem letzten refill() kann CARLA einzelne Fussgaenger schon geloescht haben
        self._destroy([i for pair in self.walkers for i in pair], quiet=True)
        self.vehicles, self.walkers = [], []
