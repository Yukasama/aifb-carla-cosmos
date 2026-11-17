"use client";

import { useEffect, useId, useRef, useState } from "react";

import { DEFAULT_FOV, FOCAL_RANGE, focalFromFov, formatLens, fovFromFocal } from "@/lib/lens";
import { validateCameraName } from "@/lib/modules";
import type { CameraPreset } from "@/lib/types";

import { inputClass } from "./param-field";

/** So lange nach dem letzten Ziehen den eigenen Wert zeigen, bis die Statusdatei ihn bestätigt hat */
const HOLD_MS = 2000;
/** Abstand zwischen zwei Vorschaubildern (Python schreibt etwa fünf pro Sekunde) */
const PREVIEW_MS = 250;

/** Das Bild der Aufnahmekamera, mit eingestellter Brennweite (das CARLA-Fenster bleibt immer bei 90°). */
function Preview({ url }: { url: string | null }) {
  const [shot, setShot] = useState<{ url: string; src: string } | null>(null);

  useEffect(() => {
    if (!url) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let current: string | null = null;
    const ctrl = new AbortController();

    const run = async () => {
      try {
        const res = await fetch(url, { cache: "no-store", signal: ctrl.signal });
        if (res.status === 200) {
          const next = URL.createObjectURL(await res.blob());
          if (cancelled) {
            URL.revokeObjectURL(next);
            return;
          }
          setShot({ url, src: next });
          if (current) URL.revokeObjectURL(current);
          current = next;
        }
      } catch {
        // abgebrochen oder Server weg: beim nächsten Mal wieder
      }
      if (!cancelled) timer = setTimeout(run, PREVIEW_MS);
    };
    void run();

    return () => {
      cancelled = true;
      ctrl.abort();
      clearTimeout(timer);
      if (current) URL.revokeObjectURL(current);
    };
  }, [url]);

  const src = shot && shot.url === url ? shot.src : null;
  return (
    <div className="relative aspect-video overflow-hidden rounded-md border border-ink-800 bg-ink-950/40">
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element -- Blob-URL, next/image hilft hier nicht
        <img src={src} alt="Bild der Aufnahmekamera" className="h-full w-full object-contain" />
      ) : (
        <p className="absolute inset-0 grid place-items-center px-4 text-center text-xs text-ink-500">
          {url ? "Warte auf das erste Kamerabild …" : "Vorschau, sobald CARLA verbunden ist."}
        </p>
      )}
    </div>
  );
}

type Props = {
  presets: CameraPreset[];
  running: boolean;
  busy: boolean;
  /** aktuelles Sichtfeld laut Python (null: noch keine Statusdatei) */
  fov: number | null;
  /** JPEG der Aufnahmekamera (null: nicht verbunden) */
  previewUrl: string | null;
  now: number;
  onCamera: (op: "save" | "goto", name: string) => Promise<boolean>;
  onFov: (fov: number) => Promise<boolean>;
};

const isDefault = (fov: number) => Math.abs(fov - DEFAULT_FOV) < 0.05;

/** Die Kamera ist der Spectator: gespeicherte Positionen anspringen, Brennweite einstellen, aktuelle Ansicht speichern. */
export function CameraSection({ presets, running, busy, fov, previewUrl, now, onCamera, onFov }: Props) {
  const [name, setName] = useState("");
  const maps = [...new Set(presets.map((p) => p.map))];

  const trimmed = name.trim();
  let nameError: string | null = null;
  try {
    if (trimmed) validateCameraName(trimmed);
  } catch (err) {
    nameError = err instanceof Error ? err.message : String(err);
  }
  const canSave = running && !busy && !!trimmed && !nameError;

  async function save() {
    if (canSave && (await onCamera("save", trimmed))) setName("");
  }

  // Brennweite: beim Ziehen sofort senden, aber nie mehr als ein Request gleichzeitig (der neueste Wert gewinnt)
  const [dragged, setDragged] = useState<{ fov: number; at: number } | null>(null);
  const inflight = useRef(false);
  const queued = useRef<number | null>(null);

  async function sendFov(value: number) {
    if (inflight.current) {
      queued.current = value;
      return;
    }
    inflight.current = true;
    try {
      await onFov(value);
    } finally {
      inflight.current = false;
    }
    const next = queued.current;
    queued.current = null;
    if (next !== null) void sendFov(next);
  }

  function changeFov(value: number) {
    setDragged({ fov: value, at: Date.now() });
    void sendFov(value);
  }

  const shownFov = dragged && (fov === null || now - dragged.at < HOLD_MS) ? dragged.fov : (fov ?? DEFAULT_FOV);
  const sliderId = useId();

  return (
    <div className="space-y-3">
      {maps.map((m) => (
        <div key={m}>
          <p className="mb-1.5 font-mono text-[11px] text-ink-500">{m}</p>
          <div className="flex flex-wrap gap-2">
            {presets
              .filter((p) => p.map === m)
              .map((p) => (
                <button
                  key={p.name}
                  type="button"
                  disabled={!running || busy}
                  title={`Brennweite ${formatLens(p.fov)}`}
                  onClick={() => {
                    setDragged(null); // gleich das Sichtfeld der Position zeigen
                    void onCamera("goto", p.name);
                  }}
                  className="rounded-md border border-ink-700 bg-ink-850 px-3 py-1.5 text-sm text-ink-100 transition-colors hover:border-signal hover:text-signal disabled:cursor-not-allowed disabled:text-ink-500 disabled:hover:border-ink-700"
                >
                  {p.name}
                  {!isDefault(p.fov) && (
                    <span className="ml-1.5 font-mono text-[11px] text-ink-500">
                      {Math.round(focalFromFov(p.fov))} mm
                    </span>
                  )}
                </button>
              ))}
          </div>
        </div>
      ))}

      <Preview url={previewUrl} />

      <div className="min-w-0">
        <div className="mb-1 flex items-baseline justify-between gap-2 text-xs text-ink-300">
          <label htmlFor={sliderId}>Brennweite</label>
          <span className="flex items-baseline gap-3">
            {!isDefault(shownFov) && (
              <button
                type="button"
                disabled={!running}
                title={`Zurück auf ${formatLens(DEFAULT_FOV)}`}
                onClick={() => changeFov(DEFAULT_FOV)}
                className="text-ink-500 transition-colors hover:text-ink-300 disabled:cursor-not-allowed"
              >
                Zurücksetzen
              </button>
            )}
            <span className="font-mono tabular-nums text-ink-100">{formatLens(shownFov)}</span>
          </span>
        </div>
        <input
          id={sliderId}
          type="range"
          min={FOCAL_RANGE[0]}
          max={FOCAL_RANGE[1]}
          step={1}
          value={Math.min(FOCAL_RANGE[1], Math.max(FOCAL_RANGE[0], focalFromFov(shownFov)))}
          disabled={!running}
          onChange={(e) => changeFov(fovFromFocal(Number(e.target.value)))}
          className="h-5 w-full cursor-pointer accent-signal disabled:cursor-not-allowed disabled:opacity-50"
        />
        <p className="mt-1 text-xs leading-snug text-ink-500">
          Kleinbild-Äquivalent, 18 mm = 90°. Das CARLA-Fenster bleibt immer bei 90° (länger: Ausschnitt als Rahmen),
          die Vorschau oben zeigt das echte Kamerabild. Wird mit der Position gespeichert und landet als Intrinsik in
          den Clips.
        </p>
      </div>

      <div>
        <div className="flex gap-2">
          <input
            type="text"
            aria-label="Name der Position"
            placeholder="Name der Position"
            spellCheck={false}
            autoComplete="off"
            maxLength={40}
            className={inputClass}
            disabled={!running}
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void save();
            }}
          />
          <button
            type="button"
            disabled={!canSave}
            onClick={() => void save()}
            className="shrink-0 rounded-md border border-signal/50 bg-signal/10 px-3 py-1.5 text-sm font-medium text-signal transition-colors hover:bg-signal/20 disabled:cursor-not-allowed disabled:border-ink-800 disabled:bg-transparent disabled:text-ink-500"
          >
            {presets.some((p) => p.name === trimmed) ? "Überschreiben" : "Ansicht speichern"}
          </button>
        </div>
        <p className={`mt-1 text-xs leading-snug ${nameError ? "text-warn" : "text-ink-500"}`}>
          {nameError ??
            (running
              ? "Die Kamera nimmt auf, was das CARLA-Fenster zeigt, mit der eingestellten Brennweite. Klick auf eine Position springt dorthin."
              : "Springen, Brennweite und Speichern gehen, sobald CARLA verbunden ist.")}
        </p>
      </div>
    </div>
  );
}
