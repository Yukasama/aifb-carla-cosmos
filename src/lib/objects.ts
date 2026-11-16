import "server-only";

import fs from "node:fs";

import type { FrameObject, FrameObjects } from "./types";

// Schreibt das Python-Package (simapi/boxes.py: write_status); hier wird nur gelesen

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

function parseObject(raw: unknown): FrameObject | null {
  const o = raw as Record<string, unknown> | null;
  if (!o || typeof o !== "object") return null;
  if (!isNum(o.id) || typeof o.class !== "string" || typeof o.blueprint_id !== "string" || !isNum(o.distance)) {
    return null;
  }
  const color =
    Array.isArray(o.color) && o.color.length === 3 && o.color.every(isNum)
      ? (o.color as [number, number, number])
      : null;
  return {
    id: o.id,
    class: o.class,
    blueprintId: o.blueprint_id,
    color,
    distance: o.distance,
    visiblePixels: isNum(o.visible_pixels) ? o.visible_pixels : null,
    occluded: typeof o.occluded === "boolean" ? o.occluded : null,
    boxed: o.boxed !== false,
  };
}

/** Jüngster Stand aus der Statusdatei; null, solange keine (lesbare) da ist. */
export function readFrameObjects(file: string): FrameObjects | null {
  let data: Record<string, unknown>;
  try {
    data = JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return null; // noch nicht geschrieben
  }
  if (!data || typeof data !== "object" || !isNum(data.frame) || !isNum(data.written_at)) return null;
  const objects = Array.isArray(data.objects) ? data.objects.map(parseObject) : [];
  return {
    frame: data.frame,
    ageMs: Math.max(0, Math.round(Date.now() - data.written_at * 1000)),
    objects: objects.filter((o): o is FrameObject => o !== null),
  };
}
