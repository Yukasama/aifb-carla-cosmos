// Registry der steuerbaren Python-Skripte. Reine Daten + Validierung, wird von
// Client (Formulare) und Server (Argumente bauen) gemeinsam genutzt.
// Neues Skript hinzufuegen = neuen Eintrag in MODULES anlegen.

export type ModuleId = "sim";

type ParamBase = {
  key: string;
  label: string;
  flag: string;
  help?: string;
  /** Ueberschrift, unter der das Feld erscheint */
  group: string;
  /** Key eines Schalters: Feld erscheint direkt darunter, solange er an ist */
  under?: string;
};

/** Zahl, immer als Schieberegler */
export type NumberParam = ParamBase & {
  type: "int" | "float";
  default: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
};

export type BoolParam = ParamBase & { type: "bool"; default: boolean };

export type TextParam = ParamBase & {
  type: "text";
  default: string;
  /** nur relative Pfade innerhalb des Skript-Ordners */
  relativePath?: boolean;
};

export type SelectParam = ParamBase & {
  type: "select";
  default: string;
  options: { value: string; label: string }[];
};

export type ParamDef = NumberParam | BoolParam | TextParam | SelectParam;
export type ParamValue = number | boolean | string | null;
export type ParamValues = Record<string, ParamValue>;

export type Connection = { host: string; port: number };

export interface ModuleDef {
  id: ModuleId;
  step: number;
  title: string;
  description: string;
  /** Python-Package, gestartet mit `python -m <script>` */
  script: string;
  /** muessen verbunden sein, bevor dieses Modul sich verbinden kann */
  requires: ModuleId[];
  /** Werte beim automatischen Verbinden (Rest: Defaults); erst „Erzeugen“ schickt die Einstellungen */
  connect: ParamValues;
  /** nach EOF auf stdin so lange auf sauberes Beenden warten, dann hart beenden */
  stopTimeoutMs: number;
  params: ParamDef[];
  /** Abschnitt mit gespeicherten Kamerapositionen (python_api/cameras.json), nach dieser Gruppe */
  cameras?: { after: string };
  /** Abschnitt mit den Objekten im Kamerabild, nach dieser Gruppe. Das Skript bekommt dafür `--status-file`. */
  objects?: { after: string };
}

const TRAFFIC = "Verkehr";
const BOXES = "Bounding Boxes";
const EXPORT = "Export";

export const MODULES: Record<ModuleId, ModuleDef> = {
  sim: {
    id: "sim",
    step: 1,
    title: "Simulation",
    description: "Verkehr, Roadside-Kamera und Bounding Boxes in CARLA. „Erzeugen“ oben rechts übernimmt die Einstellungen.",
    script: "simapi",
    requires: [],
    // Verbinden legt nichts an: kein Verkehr, keine Aufnahme
    connect: { vehicles: 0, walkers: 0, export: false },
    stopTimeoutMs: 20_000,
    cameras: { after: TRAFFIC },
    objects: { after: BOXES },
    params: [
      { group: TRAFFIC, key: "vehicles", label: "Fahrzeuge", type: "int", flag: "--number-of-vehicles", default: 30, min: 0, max: 200 },
      { group: TRAFFIC, key: "walkers", label: "Fußgänger", type: "int", flag: "--number-of-walkers", default: 10, min: 0, max: 300 },

      { group: BOXES, key: "maxDistance", label: "Max. Distanz", type: "float", flag: "--max-distance", default: 50, min: 0, max: 200, step: 5, unit: "m", help: "0 = unbegrenzt. Nur für Anzeige und Liste; Aufnahmen beschriften alles Sichtbare." },
      { group: BOXES, key: "minVisiblePixels", label: "Min. sichtbare Pixel", type: "int", flag: "--min-visible-pixels", default: 50, min: 1, max: 2000, help: "Darunter gilt ein Objekt als verdeckt." },
      { group: BOXES, key: "hideOccluded", label: "Verdeckte ausblenden", type: "bool", flag: "--hide-occluded", default: false, help: "Verdeckte Objekte bekommen keine Box." },

      { group: EXPORT, key: "export", label: "Clips aufnehmen", type: "bool", flag: "--export", default: false, help: "16 fps, Kamera fest. Jeder fertige Clip wird umgewandelt: Eingabe für Cosmos-Transfer2.5 und Cosmos 3 (Videos und Specs) und 3D-Labels für BEVHeight, HeightFormer und CoBEV. Kamera bewegen verwirft den laufenden Clip; Boxen werden solange nicht gezeichnet." },
      {
        group: EXPORT, key: "clipFrames", label: "Clip-Länge", type: "select", flag: "--clip-frames", default: "93", under: "export",
        options: [
          { value: "93", label: "93 Frames (5,8 s)" },
          { value: "186", label: "186 Frames (11,6 s)" },
          { value: "279", label: "279 Frames (17,4 s)" },
        ],
        help: "93 Frames sind ein Abschnitt bei Cosmos-Transfer2.5 und passen in einen Abschnitt von Cosmos 3 (bis 121). Längere Clips rechnen beide in mehreren Abschnitten.",
      },
      { group: EXPORT, key: "exportDir", label: "Ordner", type: "text", flag: "--export-dir", default: "_out", relativePath: true, under: "export" },
      { group: EXPORT, key: "dropFrames", label: "Rohbilder danach löschen", type: "bool", flag: "--drop-frames", default: false, under: "export", help: "Spart rund 500 MB pro Clip. Danach lässt sich der Clip nicht mehr mit anderen Einstellungen neu umwandeln." },
    ],
  },
};

export const MODULE_ORDER: ModuleId[] = (Object.keys(MODULES) as ModuleId[]).sort(
  (a, b) => MODULES[a].step - MODULES[b].step,
);

export function isModuleId(id: string): id is ModuleId {
  return Object.prototype.hasOwnProperty.call(MODULES, id);
}

export function defaultValues(id: ModuleId): ParamValues {
  return Object.fromEntries(MODULES[id].params.map((p) => [p.key, p.default]));
}

export function connectValues(id: ModuleId): ParamValues {
  return { ...defaultValues(id), ...MODULES[id].connect };
}

/** Gleiche Werte fuer alle Parameter des Moduls? (z.B. Entwurf gegen den uebernommenen Stand) */
export function sameValues(id: ModuleId, a: ParamValues, b: ParamValues) {
  return MODULES[id].params.every((p) => a[p.key] === b[p.key]);
}

/** Module, die `id` voraussetzen (werden beim Stoppen von `id` zuerst gestoppt). */
export function dependentsOf(id: ModuleId): ModuleId[] {
  return MODULE_ORDER.filter((m) => MODULES[m].requires.includes(id));
}

export const DEFAULT_CONNECTION: Connection = { host: "127.0.0.1", port: 2000 };

const HOST_RE = /^[A-Za-z0-9.-]{1,253}$/;

export function validateConnection(conn: unknown): Connection {
  const c = (conn ?? {}) as Partial<Connection>;
  const host = typeof c.host === "string" ? c.host.trim() : "";
  if (!HOST_RE.test(host)) throw new ArgError("Ungültiger Host.");
  const port = Number(c.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new ArgError("Ungültiger Port.");
  return { host, port };
}

export class ArgError extends Error {}

const CAMERA_NAME_RE = /^[\p{L}\p{N}][\p{L}\p{N} _.()-]{0,39}$/u;

/** Name einer Kameraposition: 1–40 Zeichen, beginnt mit Buchstabe oder Ziffer. */
export function validateCameraName(raw: unknown): string {
  const name = typeof raw === "string" ? raw.trim() : "";
  if (!CAMERA_NAME_RE.test(name)) {
    throw new ArgError("Name: bis 40 Zeichen aus Buchstaben, Ziffern, Leerzeichen und _ . ( ) -");
  }
  return name;
}

function checkText(p: TextParam, v: string): string {
  if (v.length > 200) throw new ArgError(`${p.label}: zu lang.`);
  if (/[\x00-\x1f]/.test(v)) throw new ArgError(`${p.label}: Steuerzeichen nicht erlaubt.`);
  if (v.startsWith("-")) throw new ArgError(`${p.label}: darf nicht mit „-“ beginnen.`);
  if (p.relativePath) {
    if (v.startsWith("/") || v.startsWith("~")) throw new ArgError(`${p.label}: nur relative Pfade.`);
    if (v.split(/[\\/]/).includes("..")) throw new ArgError(`${p.label}: „..“ ist nicht erlaubt.`);
  }
  return v;
}

/**
 * Validiert die Werte gegen die Registry und baut die argv-Liste (ohne Shell).
 * Unbekannte Keys werden ignoriert, fehlende bekommen den Default.
 */
export function buildArgs(
  id: ModuleId,
  values: ParamValues,
  conn: Connection,
): { args: string[]; values: ParamValues } {
  const def = MODULES[id];
  const args = ["--host", conn.host, "--port", String(conn.port)];
  const clean: ParamValues = {};

  for (const p of def.params) {
    const raw = values?.[p.key] === undefined ? p.default : values[p.key];

    switch (p.type) {
      case "bool": {
        const v = raw === true;
        clean[p.key] = v;
        if (v) args.push(p.flag);
        break;
      }
      case "int":
      case "float": {
        if (raw === null || raw === "") throw new ArgError(`${p.label}: Wert fehlt.`);
        const n = Number(raw);
        if (!Number.isFinite(n)) throw new ArgError(`${p.label}: keine Zahl.`);
        if (p.type === "int" && !Number.isInteger(n)) throw new ArgError(`${p.label}: ganze Zahl erwartet.`);
        if (n < p.min) throw new ArgError(`${p.label}: mindestens ${p.min}.`);
        if (n > p.max) throw new ArgError(`${p.label}: höchstens ${p.max}.`);
        clean[p.key] = n;
        args.push(p.flag, String(n));
        break;
      }
      case "text": {
        const v = typeof raw === "string" ? raw.trim() : "";
        if (v === "") throw new ArgError(`${p.label}: Wert fehlt.`);
        clean[p.key] = checkText(p, v);
        args.push(p.flag, v);
        break;
      }
      case "select": {
        const v = String(raw);
        if (!p.options.some((o) => o.value === v)) throw new ArgError(`${p.label}: ungültige Auswahl.`);
        clean[p.key] = v;
        args.push(p.flag, v);
        break;
      }
    }
  }

  return { args, values: clean };
}
