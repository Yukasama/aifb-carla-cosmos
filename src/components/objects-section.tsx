"use client";

import type { FrameObject, FrameObjects } from "@/lib/types";

/** Abstand, in dem die Karte die Statusdatei abfragt (Objekte und Sichtfeld) */
export const OBJECTS_POLL_MS = 500;
/** Älter heißt: Python schreibt nicht mehr (CARLA hängt oder ist pausiert) */
const STALE_MS = 2000;

const CLASSES: Record<string, string> = {
  car: "Pkw",
  van: "Transporter",
  truck: "Lkw",
  bus: "Bus",
  motorcycle: "Motorrad",
  bicycle: "Fahrrad",
  pedestrian: "Fußgänger",
  vehicle: "Fahrzeug",
};
const MAKES: Record<string, string> = { bmw: "BMW", bh: "BH" };

const cap = (w: string) => w.charAt(0).toUpperCase() + w.slice(1);
const words = (s: string) =>
  s
    .split("_")
    .filter(Boolean)
    .map((w) => w.split("-").map(cap).join("-"))
    .join(" ");

/** vehicle.tesla.model3 → „Tesla Model3“, walker.pedestrian.0012 → „Fußgänger 0012“ */
function modelName(blueprintId: string) {
  const [kind, make = "", ...rest] = blueprintId.split(".");
  if (kind === "walker") return `Fußgänger ${rest.join(".")}`.trim();
  return `${MAKES[make] ?? words(make)} ${words(rest.join("."))}`.trim() || blueprintId;
}

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

function Swatch({ color }: { color: FrameObject["color"] }) {
  return color ? (
    <span
      aria-hidden
      className="h-3 w-3 shrink-0 rounded-full border border-white/15"
      style={{ backgroundColor: `rgb(${color.join(" ")})` }}
    />
  ) : (
    <span aria-hidden className="h-3 w-3 shrink-0 rounded-full border border-dashed border-ink-700" />
  );
}

function Visibility({ o }: { o: FrameObject }) {
  if (o.occluded === null) {
    return (
      <span className="text-ink-500" title="Für diesen Frame kam kein Instanzbild, Verdeckung unbekannt.">
        –
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-2 whitespace-nowrap">
      <span
        className={`rounded-full border px-2 py-0.5 text-xs ${
          o.occluded ? "border-warn/40 bg-warn/10 text-warn" : "border-go/30 bg-go/10 text-go"
        }`}
      >
        {o.occluded ? "verdeckt" : "sichtbar"}
      </span>
      <span className="font-mono text-xs tabular-nums text-ink-500">
        {o.visiblePixels} px{!o.boxed && " · keine Box"}
      </span>
    </span>
  );
}

/** Fahrzeuge und Fußgänger im Kamerabild des laufenden Prozesses, nach Distanz sortiert. */
export function ObjectsSection({ running, data }: { running: boolean; data: FrameObjects | null }) {
  if (!running) {
    return (
      <p className="text-xs leading-snug text-ink-500">
        Sobald CARLA verbunden ist, stehen hier die Fahrzeuge und Fußgänger im Kamerabild – mit Modell, Distanz und
        ob sie verdeckt sind.
      </p>
    );
  }

  const objects = data?.objects ?? [];
  const walkers = objects.filter((o) => o.class === "pedestrian").length;
  const occluded = objects.filter((o) => o.occluded).length;
  const stale = !!data && data.ageMs > STALE_MS;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 text-xs">
        <span className="text-ink-300">
          {data
            ? [
                plural(objects.length - walkers, "Fahrzeug", "Fahrzeuge"),
                plural(walkers, "Fußgänger", "Fußgänger"),
                `${occluded} verdeckt`,
              ].join(" · ")
            : "Warte auf den ersten Frame …"}
        </span>
        {data && (
          <span className={`font-mono tabular-nums ${stale ? "text-warn" : "text-ink-500"}`}>
            {stale ? `seit ${Math.round(data.ageMs / 1000)} s keine neuen Daten` : `Frame ${data.frame}`}
          </span>
        )}
      </div>

      {/* feste Höhe: wechselnde Zeilenzahl soll nichts darunter verschieben */}
      <div
        className={`h-72 overflow-auto rounded-md border border-ink-800 bg-ink-950/40 transition-opacity ${
          stale ? "opacity-50" : ""
        }`}
      >
        {objects.length === 0 ? (
          <p className="px-3 py-2 text-sm text-ink-500">{data ? "Nichts im Bild." : ""}</p>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="sticky top-0 bg-ink-900 font-mono text-[11px] uppercase tracking-wider text-ink-500">
              <tr>
                <th className="px-3 py-2 font-normal">Modell</th>
                <th className="hidden px-3 py-2 font-normal sm:table-cell">Typ</th>
                <th className="px-3 py-2 text-right font-normal">Distanz</th>
                <th className="px-3 py-2 font-normal">Sichtbarkeit</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ink-800/70">
              {objects.map((o) => (
                <tr
                  key={o.id}
                  className={o.boxed ? "" : "opacity-60"}
                  title={o.boxed ? undefined : "Verdeckt, bekommt wegen „Verdeckte ausblenden“ keine Box."}
                >
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-2.5">
                      <Swatch color={o.color} />
                      <div className="min-w-0">
                        <div className="text-ink-100">{modelName(o.blueprintId)}</div>
                        <div className="font-mono text-[11px] text-ink-500">
                          {o.blueprintId} · #{o.id}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="hidden px-3 py-1.5 text-ink-300 sm:table-cell">{CLASSES[o.class] ?? o.class}</td>
                  <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono tabular-nums text-ink-300">
                    {o.distance.toFixed(1)} m
                  </td>
                  <td className="px-3 py-1.5">
                    <Visibility o={o} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
