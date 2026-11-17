import "server-only";

import fs from "node:fs";
import path from "node:path";

import { DEFAULT_FOV, FOV_RANGE } from "./lens";
import { SCRIPTS_DIR } from "./process-manager";
import type { CameraPreset } from "./types";

// Schreibt das Python-Package (simapi/presets.py); hier wird nur gelesen
const FILE = path.join(SCRIPTS_DIR, "cameras.json");
const NUMBERS = ["x", "y", "z", "pitch", "yaw", "roll"] as const;

export function readCameraPresets(): CameraPreset[] {
  let data: unknown;
  try {
    const text = fs.readFileSync(FILE, "utf8");
    data = JSON.parse(text.charCodeAt(0) === 0xfeff ? text.slice(1) : text); // BOM vom Windows-Editor
  } catch {
    return []; // fehlt oder kaputt (Python meldet Details im Log)
  }
  if (!data || typeof data !== "object" || Array.isArray(data)) return [];

  const out: CameraPreset[] = [];
  for (const [name, raw] of Object.entries(data)) {
    const p = raw as Record<string, unknown> | null;
    if (!p || typeof p !== "object" || typeof p.map !== "string") continue;
    if (!NUMBERS.every((k) => typeof p[k] === "number")) continue;
    const fov = p.fov ?? DEFAULT_FOV;
    if (typeof fov !== "number" || fov < FOV_RANGE[0] || fov > FOV_RANGE[1]) continue; // wie Python: ungültig
    out.push({ name, map: p.map, fov });
  }
  return out.sort((a, b) => a.map.localeCompare(b.map) || a.name.localeCompare(b.name, "de"));
}
