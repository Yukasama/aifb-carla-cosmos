"use client";

import { Fragment, useMemo } from "react";

import { buildArgs, type Connection, type ModuleDef, type ParamDef, type ParamValue, type ParamValues } from "@/lib/modules";
import type { CameraPreset, FrameObjects, ModuleStatus } from "@/lib/types";

import { CameraSection } from "./camera-section";
import { formatDuration, usePoll } from "./hooks";
import { LogView } from "./log-view";
import { OBJECTS_POLL_MS, ObjectsSection } from "./objects-section";
import { ParamField } from "./param-field";

/** Selbst beendet, ohne dass wir getrennt haben: Fehler oder CARLA weg */
export const endedOnItsOwn = (s: ModuleStatus) => s.state === "exited" && s.stopReason === null;

type Props = {
  def: ModuleDef;
  status: ModuleStatus | undefined;
  draft: ParamValues;
  conn: Connection;
  now: number;
  /** CARLA antwortet auf TCP */
  reachable: boolean;
  /** verbunden, aber der Entwurf ist noch nicht übernommen */
  pending: boolean;
  /** Titel der Voraussetzungen, die gerade nicht verbunden sind */
  blockedBy: string[];
  busy: boolean;
  error: string | undefined;
  cameras: CameraPreset[];
  onDraftChange: (values: ParamValues) => void;
  onReset: () => void;
  onKill: () => void;
  onCamera: (op: "save" | "goto", name: string) => Promise<boolean>;
  onFov: (fov: number) => Promise<boolean>;
};

function StatusPill({ status, now, reachable }: { status: ModuleStatus | undefined; now: number; reachable: boolean }) {
  const s = status?.state ?? "idle";
  let dot = "bg-ink-500";
  let text = reachable ? "Nicht verbunden" : "Warte auf CARLA";
  let tone = "text-ink-300 border-ink-700";

  if (s === "running" && status?.ready) {
    dot = "bg-go animate-pulse";
    text = `Verbunden ${formatDuration(now - (status.startedAt ?? now))}`;
    tone = "text-go border-go/40 bg-go/10";
  } else if (s === "running") {
    dot = "bg-warn animate-pulse";
    text = "Verbinde …";
    tone = "text-warn border-warn/40 bg-warn/10";
  } else if (s === "stopping") {
    dot = "bg-warn animate-pulse";
    text = "Trenne …";
    tone = "text-warn border-warn/40 bg-warn/10";
  } else if (status && endedOnItsOwn(status) && reachable) {
    dot = "bg-fail";
    text = "Abgebrochen, verbinde gleich neu";
    tone = "text-fail border-fail/40 bg-fail/10";
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

export function ModuleCard({
  def,
  status,
  draft,
  conn,
  now,
  reachable,
  pending,
  blockedBy,
  busy,
  error,
  cameras,
  onDraftChange,
  onReset,
  onKill,
  onCamera,
  onFov,
}: Props) {
  const state = status?.state ?? "idle";
  const running = state === "running";
  const connected = running && !!status?.ready;
  const groups = [...new Set(def.params.map((p) => p.group))];
  // Statusdatei des Laufs: Objekte im Bild und aktuelles Sichtfeld; neuer Lauf -> erst leer statt der letzten Daten
  const frame = usePoll<FrameObjects>(
    def.objects && running ? `/api/modules/${def.id}/objects?run=${status?.runId ?? 0}` : null,
    OBJECTS_POLL_MS,
  ).data;

  // Validierung des Entwurfs (dieselbe Logik wie auf dem Server)
  const draftError = useMemo(() => {
    try {
      buildArgs(def.id, draft, conn);
      return null;
    } catch (err) {
      return err instanceof Error ? err.message : String(err);
    }
  }, [def.id, draft, conn]);

  const blocked = !running && blockedBy.length > 0;
  const setValue = (key: string, v: ParamValue) => onDraftChange({ ...draft, [key]: v });

  // Felder mit `under` erscheinen direkt unter ihrem Schalter
  const renderFields = (params: ParamDef[]) => {
    const field = (p: ParamDef): React.ReactNode => (
      <ParamField key={p.key} param={p} value={draft[p.key] ?? null} disabled={false} onChange={(v) => setValue(p.key, v)}>
        {params.filter((c) => c.under === p.key).map(field)}
      </ParamField>
    );
    const top = params.filter((p) => !p.under);
    const inputs = top.filter((p) => p.type !== "bool");
    const toggles = top.filter((p) => p.type === "bool");
    return (
      <>
        {inputs.length > 0 && (
          <div className="grid grid-cols-1 gap-x-4 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">{inputs.map(field)}</div>
        )}
        {toggles.length > 0 && (
          <div className="-mx-2 grid grid-cols-1 items-start gap-x-4 gap-y-1 sm:grid-cols-2">{toggles.map(field)}</div>
        )}
      </>
    );
  };

  return (
    <section
      className={`rounded-xl border bg-ink-900 shadow-[0_1px_0_0_rgba(255,255,255,0.03)_inset] transition-colors ${
        connected ? "border-go/30" : "border-ink-800"
      }`}
    >
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-ink-800 px-5 py-4">
        <span
          className={`grid h-7 w-7 place-items-center rounded-md border font-mono text-xs ${
            connected ? "border-go/50 text-go" : "border-ink-700 text-ink-500"
          }`}
        >
          {String(def.step).padStart(2, "0")}
        </span>
        <h2 className="min-w-0 text-base font-semibold leading-tight text-ink-100">{def.title}</h2>
        <div className="ml-auto">
          <StatusPill status={status} now={now} reachable={reachable} />
        </div>
      </header>

      <div className="space-y-4 px-5 py-4">
        <p className="text-sm text-ink-300">{def.description}</p>

        {blocked && (
          <div className="flex items-center gap-2 rounded-md border border-dashed border-ink-700 px-3 py-2 text-sm text-ink-300">
            <span aria-hidden className="text-ink-500">⏸</span>
            Verbindet sich, sobald {blockedBy.map((t) => `„${t}“`).join(", ")} verbunden ist.
          </div>
        )}

        {groups.map((g) => (
          <Fragment key={g}>
            <section className="space-y-3 border-t border-ink-800 pt-4">
              <h3 className="font-mono text-[11px] uppercase tracking-wider text-ink-500">{g}</h3>
              {renderFields(def.params.filter((p) => p.group === g))}
            </section>
            {def.cameras?.after === g && (
              <section className="space-y-3 border-t border-ink-800 pt-4">
                <h3 className="font-mono text-[11px] uppercase tracking-wider text-ink-500">Kamera</h3>
                <CameraSection
                  presets={cameras}
                  running={connected}
                  busy={busy}
                  fov={frame?.fov ?? null}
                  previewUrl={def.objects && connected ? `/api/modules/${def.id}/preview?run=${status?.runId ?? 0}` : null}
                  now={now}
                  onCamera={onCamera}
                  onFov={onFov}
                />
              </section>
            )}
            {def.objects?.after === g && (
              <section className="space-y-3 border-t border-ink-800 pt-4">
                <h3 className="font-mono text-[11px] uppercase tracking-wider text-ink-500">Im Bild</h3>
                <ObjectsSection running={running} data={frame} />
              </section>
            )}
          </Fragment>
        ))}

        <div className="flex flex-wrap items-center gap-3 border-t border-ink-800 pt-4">
          <button
            type="button"
            onClick={onReset}
            className="rounded-md px-2 py-1 text-xs text-ink-500 transition-colors hover:text-ink-300"
          >
            Standardwerte
          </button>
          {state === "stopping" && (
            <button
              type="button"
              onClick={onKill}
              title="Das Skript räumt dann nicht auf (Actors bleiben, CARLA evtl. im Sync-Modus und eingefroren)"
              className="rounded-md px-2 py-1 text-xs text-ink-500 underline-offset-2 hover:text-fail hover:underline"
            >
              Hart beenden
            </button>
          )}
          {pending && !draftError && <span className="text-sm text-warn">Noch nicht übernommen – „Erzeugen“ oben rechts.</span>}
          {draftError && <span className="text-sm text-warn">{draftError}</span>}
          {error && <span className="text-sm text-fail">{error}</span>}
        </div>
      </div>

      <LogView
        logKey={def.id}
        url={`/api/modules/${def.id}/logs`}
        runId={status?.runId ?? 0}
        live={running || state === "stopping"}
      />
    </section>
  );
}
