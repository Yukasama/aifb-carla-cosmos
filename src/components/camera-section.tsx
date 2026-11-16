"use client";

import { useState } from "react";

import { validateCameraName } from "@/lib/modules";
import type { CameraPreset } from "@/lib/types";

import { inputClass } from "./param-field";

type Props = {
  presets: CameraPreset[];
  running: boolean;
  busy: boolean;
  onCamera: (op: "save" | "goto", name: string) => Promise<boolean>;
};

/** Die Kamera ist der Spectator: gespeicherte Positionen anspringen, aktuelle Ansicht speichern. */
export function CameraSection({ presets, running, busy, onCamera }: Props) {
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
                  onClick={() => void onCamera("goto", p.name)}
                  className="rounded-md border border-ink-700 bg-ink-850 px-3 py-1.5 text-sm text-ink-100 transition-colors hover:border-signal hover:text-signal disabled:cursor-not-allowed disabled:text-ink-500 disabled:hover:border-ink-700"
                >
                  {p.name}
                </button>
              ))}
          </div>
        </div>
      ))}

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
              ? "Die Kamera nimmt auf, was das CARLA-Fenster zeigt. Klick auf eine Position springt dorthin."
              : "Springen und Speichern gehen, sobald CARLA verbunden ist.")}
        </p>
      </div>
    </div>
  );
}
