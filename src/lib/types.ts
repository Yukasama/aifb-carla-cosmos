import type { Connection, ModuleId, ParamValues } from "./modules";

export type ModuleState = "idle" | "running" | "stopping" | "exited";

export interface ModuleStatus {
  id: ModuleId;
  state: ModuleState;
  runId: number;
  pid: number | null;
  startedAt: number | null;
  endedAt: number | null;
  exitCode: number | null;
  signal: string | null;
  /** läuft und die Welt tickt (bei Modulen mit `objects`: Statusdatei ist frisch) */
  ready: boolean;
  /** zuletzt übernommene Werte */
  values: ParamValues | null;
  conn: Connection | null;
  stopReason: string | null;
}

export interface LogChunk {
  runId: number;
  offset: number;
  text: string;
  reset: boolean;
}

/** Testlauf von python_api/record_scenes.py */
export interface TestRunStatus {
  /** starting: trennt erst die Simulation */
  state: "idle" | "starting" | "running" | "stopping" | "exited";
  runId: number;
  /** aufgenommene Frames */
  frames: number;
  startedAt: number | null;
  endedAt: number | null;
  /** null: abgebrochen, bevor das Skript lief, oder Startfehler */
  exitCode: number | null;
  /** gesetzt, wenn wir abgebrochen haben */
  stopReason: string | null;
}

export interface StateResponse {
  modules: Record<ModuleId, ModuleStatus>;
  testRun: TestRunStatus;
  now: number;
}

/** Gespeicherte Kameraposition aus python_api/cameras.json (die Pose braucht nur der Python-Prozess) */
export interface CameraPreset {
  name: string;
  map: string;
  /** horizontales Sichtfeld in Grad (ältere Einträge ohne: 90) */
  fov: number;
}

/** Objekt im Kamerabild, aus der Statusdatei des laufenden Python-Prozesses */
export interface FrameObject {
  id: number;
  /** base_type des Blueprints (car, truck, …) oder "pedestrian" */
  class: string;
  /** z.B. vehicle.tesla.model3 */
  blueprintId: string;
  /** Lackfarbe, falls der Blueprint eine hat */
  color: [number, number, number] | null;
  /** m zur Kamera */
  distance: number;
  /** null: für diesen Frame kein Instanzbild, Verdeckung unbekannt */
  visiblePixels: number | null;
  occluded: boolean | null;
  /** bekommt eine Box (verdeckte nicht, wenn „Verdeckte ausblenden“ an ist) */
  boxed: boolean;
}

export interface FrameObjects {
  frame: number;
  /** aktuelles Sichtfeld der Kamera in Grad (null bei älteren Statusdateien) */
  fov: number | null;
  /** Alter der Statusdatei */
  ageMs: number;
  /** nach Distanz sortiert */
  objects: FrameObject[];
}

export interface CarlaProbe {
  reachable: boolean;
  latencyMs: number | null;
  error: string | null;
  /** Laufende CARLA-Server auf diesem Rechner (nur Windows und lokaler Host, sonst null) */
  instances: CarlaInstance[] | null;
}

export interface CarlaInstance {
  pid: number;
  /** Prozessname, z.B. CarlaUE4-Win64-Shipping.exe */
  image: string;
  /** Arbeitsspeicher in MB */
  memoryMb: number;
  /** Details nur, wenn mehrere laufen (sonst null) */
  startedAt: number | null;
  /** hat das normale CARLA-Fenster; ohne ist es meist ein hängengebliebener Rest */
  hasWindow: boolean | null;
  /** nicht der jüngste mit Fenster: wird von „beenden“ erfasst */
  stale: boolean;
}
