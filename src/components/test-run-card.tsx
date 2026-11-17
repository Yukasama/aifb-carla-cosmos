"use client";

import type { TestRunStatus } from "@/lib/types";

import { formatDuration } from "./hooks";
import { LogView } from "./log-view";

type Props = {
  status: TestRunStatus | undefined;
  now: number;
  busy: boolean;
  error: string | undefined;
  onStart: () => void;
  onStop: () => void;
};

function StatusPill({ status, now }: { status: TestRunStatus | undefined; now: number }) {
  const s = status?.state ?? "idle";
  let dot = "bg-ink-500";
  let text = "Bereit";
  let tone = "text-ink-300 border-ink-700";

  if (s === "starting") {
    dot = "bg-warn animate-pulse";
    text = "Trenne die Simulation …";
    tone = "text-warn border-warn/40 bg-warn/10";
  } else if (s === "running") {
    dot = "bg-go animate-pulse";
    text = `Läuft ${formatDuration(now - (status?.startedAt ?? now))}`;
    tone = "text-go border-go/40 bg-go/10";
  } else if (s === "stopping") {
    dot = "bg-warn animate-pulse";
    text = "Bricht ab …";
    tone = "text-warn border-warn/40 bg-warn/10";
  } else if (s === "exited" && status) {
    const took = formatDuration((status.endedAt ?? now) - (status.startedAt ?? now));
    if (status.stopReason) {
      text = "Abgebrochen";
    } else if (status.exitCode === 0) {
      dot = "bg-go";
      text = `Fertig nach ${took}`;
      tone = "text-go border-go/40 bg-go/10";
    } else {
      dot = "bg-fail";
      text = status.exitCode === null ? "Fehlgeschlagen" : `Fehlgeschlagen (Exit-Code ${status.exitCode})`;
      tone = "text-fail border-fail/40 bg-fail/10";
    }
  }

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 font-mono text-xs whitespace-nowrap ${tone}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {text}
    </span>
  );
}

/** Testlauf von record_scenes.py mit einem Clip: prüft den ganzen Weg von CARLA bis zum gespeicherten Clip. */
export function TestRunCard({ status, now, busy, error, onStart, onStop }: Props) {
  const state = status?.state ?? "idle";
  const active = state === "starting" || state === "running" || state === "stopping";
  const frames = status?.frames ?? 93;

  return (
    <section
      className={`rounded-xl border bg-ink-900 shadow-[0_1px_0_0_rgba(255,255,255,0.03)_inset] transition-colors ${
        active ? "border-go/30" : "border-ink-800"
      }`}
    >
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-ink-800 px-5 py-4">
        <span
          className={`grid h-7 w-7 place-items-center rounded-md border font-mono text-xs ${
            active ? "border-go/50 text-go" : "border-ink-700 text-ink-500"
          }`}
          aria-hidden
        >
          ▶
        </span>
        <h2 className="min-w-0 text-base font-semibold leading-tight text-ink-100">Testlauf</h2>
        <div className="ml-auto">
          <StatusPill status={status} now={now} />
        </div>
      </header>

      <div className="space-y-4 px-5 py-4">
        <p className="text-sm text-ink-300">
          Startet <code className="font-mono text-ink-100">record_scenes.py</code> mit {frames} Frames: ein Clip von der
          ersten gespeicherten Kameraposition und einer von der Holdout-Position (Testdaten, landet in{" "}
          <code className="font-mono text-ink-100">holdout/</code>), aufgenommen und umgewandelt (Cosmos-Eingabe und
          Detektor-Labels) wie bei einem großen Lauf. Benutzt das CARLA unter der Adresse oben oder startet eins ohne Fenster. Die Simulation wird
          dafür getrennt und verbindet sich danach wieder.
        </p>

        <div className="flex flex-wrap items-center gap-3 border-t border-ink-800 pt-4">
          {active ? (
            <button
              type="button"
              disabled={busy || state === "stopping"}
              onClick={onStop}
              title="Bricht sauber ab: fertige Clips bleiben, ein selbst gestartetes CARLA wird beendet."
              className="rounded-md border border-fail/50 bg-fail/10 px-3 py-1.5 text-sm font-medium text-fail transition-colors hover:bg-fail/20 disabled:cursor-wait disabled:opacity-60"
            >
              {state === "stopping" ? "Bricht ab …" : "Abbrechen"}
            </button>
          ) : (
            <button
              type="button"
              disabled={busy || !status}
              onClick={onStart}
              className="rounded-md border border-signal/50 bg-signal/10 px-3 py-1.5 text-sm font-medium text-signal transition-colors hover:bg-signal/20 disabled:cursor-not-allowed disabled:border-ink-800 disabled:bg-transparent disabled:text-ink-500"
            >
              Testlauf starten ({frames} Frames)
            </button>
          )}
          {error && <span className="text-sm text-fail">{error}</span>}
        </div>
      </div>

      <LogView
        logKey="test-run"
        url="/api/test-run/logs"
        runId={status?.runId ?? 0}
        live={active}
        emptyText="Noch kein Testlauf."
      />
    </section>
  );
}
