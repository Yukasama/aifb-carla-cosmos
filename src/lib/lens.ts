// Brennweite <-> horizontales Sichtfeld der Kamera. CARLA kennt nur das Sichtfeld (fov, Grad);
// angezeigt wird die Brennweite als Kleinbild-Aequivalent (36 mm Sensorbreite, 90° = 18 mm).
// Gleiche Werte wie python_api/simapi/camera.py und presets.py.

import { ArgError } from "./modules";

export const DEFAULT_FOV = 90;
/** erlaubtes Sichtfeld in Grad (Python prueft dasselbe) */
export const FOV_RANGE = [10, 130] as const;
/** Bereich des Schiebereglers in mm */
export const FOCAL_RANGE = [10, 100] as const;
const SENSOR_WIDTH_MM = 36;

const rad = (deg: number) => (deg * Math.PI) / 180;
const deg = (rad: number) => (rad * 180) / Math.PI;

export const focalFromFov = (fov: number) => SENSOR_WIDTH_MM / 2 / Math.tan(rad(fov) / 2);
export const fovFromFocal = (mm: number) => 2 * deg(Math.atan(SENSOR_WIDTH_MM / 2 / mm));

/** „24 mm · 73,7°“ */
export function formatLens(fov: number) {
  const mm = focalFromFov(fov);
  const f = (v: number, digits: number) => v.toLocaleString("de-DE", { maximumFractionDigits: digits });
  return `${f(mm, mm < 20 ? 1 : 0)} mm · ${f(fov, 1)}°`;
}

export function validateFov(raw: unknown): number {
  const fov = typeof raw === "number" ? raw : NaN;
  if (!Number.isFinite(fov) || fov < FOV_RANGE[0] || fov > FOV_RANGE[1]) {
    throw new ArgError(`Sichtfeld: ${FOV_RANGE[0]}–${FOV_RANGE[1]}° erwartet.`);
  }
  return Math.round(fov * 1000) / 1000;
}
