import "server-only";

import { execFile } from "node:child_process";

import type { CarlaInstance } from "./types";

// Laufen zwei CARLA-Server, lauschen beide auf Port 2000 und Windows verteilt die Verbindungen.
// Typisch: Ein Fenster wurde geschlossen, der Prozess lief aber ohne Fenster weiter und
// antwortet nicht mehr. Der Python-Client haengt dann bis zum Timeout.
//
// tasklist /V (mit Fenstertitel) braucht aus dem Server heraus bis zu 30 s, solange ein
// CARLA-Fenster haengt. Darum: tasklist ohne /V (schnell) bei jedem Poll, und nur wenn mehr als
// ein Server laeuft, Startzeit und Fenster per PowerShell (~2 s, gecacht).

const IS_WINDOWS = process.platform === "win32";
const LOCAL_HOST = /^(127\.\d{1,3}\.\d{1,3}\.\d{1,3}|localhost)$/i;
/** Der eigentliche Server; CarlaUE4.exe davor ist nur der Starter */
const SERVER_IMAGE = /^Carla\w*-Win64-Shipping\.exe$/i;
const LIST_CACHE_MS = 2000;
const DETAILS_CACHE_MS = 15_000;

type Proc = { pid: number; image: string; memoryMb: number };
type Details = Map<number, { startedAt: number | null; hasWindow: boolean }>;

let listCache: { at: number; result: Promise<Proc[] | null> } | null = null;
let detailsCache: { key: string; at: number; result: Promise<Details | null> } | null = null;

function run(file: string, args: string[], timeout: number): Promise<string | null> {
  return new Promise((resolve) =>
    execFile(file, args, { windowsHide: true, timeout }, (err, stdout) => resolve(err ? null : stdout)),
  );
}

/** Felder einer CSV-Zeile von tasklist (alle in "", "" steht fuer ein Anfuehrungszeichen) */
function csvFields(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quoted && c === '"' && line[i + 1] === '"') {
      cur += '"';
      i++;
    } else if (c === '"') {
      quoted = !quoted;
    } else if (c === "," && !quoted) {
      out.push(cur);
      cur = "";
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out;
}

async function tasklist(): Promise<Proc[] | null> {
  const out = await run("tasklist", ["/FI", "IMAGENAME eq Carla*", "/FO", "CSV", "/NH"], 5000);
  if (out === null) return null;
  const list: Proc[] = [];
  for (const line of out.split(/\r?\n/)) {
    const f = csvFields(line.trim());
    if (f.length < 5 || !SERVER_IMAGE.test(f[0])) continue; // auch "INFO: Keine Tasks …"
    list.push({
      pid: Number(f[1]),
      image: f[0],
      memoryMb: Math.round(Number(f[4].replace(/\D/g, "")) / 1024), // "1.062.892 K"
    });
  }
  return list;
}

/** Startzeit und ob das Spielfenster da ist (es heisst wie der Prozess, z.B. „CarlaUE4“) */
async function details(procs: Proc[]): Promise<Details | null> {
  const script =
    `ConvertTo-Json -Compress -InputObject @(Get-Process -Id ${procs.map((p) => p.pid).join(",")} ` +
    "-ErrorAction SilentlyContinue | ForEach-Object { [pscustomobject]@{ " +
    "pid = $_.Id; start = $_.StartTime.ToString('o'); title = $_.MainWindowTitle } })";
  const out = await run("powershell", ["-NoProfile", "-NonInteractive", "-Command", script], 15_000);
  if (out === null) return null;
  try {
    const rows = JSON.parse(out) as { pid: number; start: string | null; title: string | null }[];
    const map: Details = new Map();
    for (const r of rows) {
      const image = procs.find((p) => p.pid === r.pid)?.image ?? "";
      const base = image.split("-")[0].toLowerCase();
      const startedAt = r.start ? Date.parse(r.start) : NaN;
      map.set(r.pid, {
        startedAt: Number.isFinite(startedAt) ? startedAt : null,
        hasWindow: !!base && (r.title ?? "").trim().toLowerCase().startsWith(base),
      });
    }
    return map;
  } catch {
    return null;
  }
}

/**
 * Behalten wird der juengste Server mit Fenster (ohne Fenster: der juengste ueberhaupt),
 * die anderen gelten als haengengeblieben. Ohne Details wird nichts als solcher markiert.
 */
async function describe(procs: Proc[], fresh: boolean): Promise<CarlaInstance[]> {
  if (procs.length < 2) return procs.map((p) => ({ ...p, startedAt: null, hasWindow: null, stale: false }));

  const key = procs.map((p) => p.pid).sort().join(",");
  if (fresh || !detailsCache || detailsCache.key !== key || Date.now() - detailsCache.at > DETAILS_CACHE_MS) {
    detailsCache = { key, at: Date.now(), result: details(procs) };
  }
  const d = await detailsCache.result;
  const list = procs.map((p) => ({
    ...p,
    startedAt: d?.get(p.pid)?.startedAt ?? null,
    hasWindow: d?.get(p.pid)?.hasWindow ?? null,
  }));
  if (!d || list.some((i) => i.startedAt === null)) return list.map((i) => ({ ...i, stale: false }));

  const newest = (xs: typeof list) => xs.reduce((a, b) => ((b.startedAt ?? 0) > (a.startedAt ?? 0) ? b : a));
  const withWindow = list.filter((i) => i.hasWindow);
  const keep = newest(withWindow.length ? withWindow : list);
  return list.map((i) => ({ ...i, stale: i.pid !== keep.pid }));
}

/** CARLA-Server auf diesem Rechner; null, wenn das nicht geht (kein Windows, entfernter Host). */
export async function listCarlaInstances(host: string): Promise<CarlaInstance[] | null> {
  if (!IS_WINDOWS || !LOCAL_HOST.test(host)) return null;
  // Mehrere Tabs pollen: hoechstens ein tasklist alle paar Sekunden
  if (!listCache || Date.now() - listCache.at > LIST_CACHE_MS) listCache = { at: Date.now(), result: tasklist() };
  const procs = await listCache.result;
  return procs && describe(procs, false);
}

/** Beendet die haengengebliebenen Server (siehe describe). Gibt deren PIDs zurueck. */
export async function killStaleInstances(): Promise<number[]> {
  if (!IS_WINDOWS) return [];
  const procs = await tasklist();
  if (!procs) throw new Error("tasklist fehlgeschlagen.");
  const stale = (await describe(procs, true)).filter((i) => i.stale);
  await Promise.all(stale.map((i) => run("taskkill", ["/PID", String(i.pid), "/T", "/F"], 10_000)));
  listCache = detailsCache = null;
  return stale.map((i) => i.pid);
}
