"use client";

import { Fragment, useEffect, useEffectEvent, useRef, useState } from "react";

import {
  DEFAULT_CONNECTION,
  MODULES,
  MODULE_ORDER,
  buildArgs,
  defaultValues,
  sameValues,
  type Connection,
  type ModuleId,
  type ParamValues,
} from "@/lib/modules";
import type { CameraPreset, CarlaInstance, CarlaProbe, StateResponse, TestRunStatus } from "@/lib/types";

import { useMounted, useNow, usePoll, useStoredState } from "./hooks";
import { ModuleCard, endedOnItsOwn } from "./module-card";
import { TestRunCard } from "./test-run-card";

type Drafts = Record<ModuleId, ParamValues>;
type Action = "connect" | "update" | "saveCamera" | "gotoCamera" | "disconnect" | "kill";

/** Nach einem Abbruch so lange warten, bevor automatisch neu verbunden wird */
const RETRY_MS = 10_000;

function initDrafts(): Drafts {
  return Object.fromEntries(MODULE_ORDER.map((id) => [id, defaultValues(id)])) as Drafts;
}

// Gespeicherte Werte mit aktuellen Defaults mischen (neue/entfernte Parameter)
function reviveDrafts(stored: unknown): Drafts {
  const s = (stored ?? {}) as Partial<Record<string, ParamValues>>;
  return Object.fromEntries(
    MODULE_ORDER.map((id) => {
      const defaults = defaultValues(id);
      const saved = s[id] ?? {};
      const merged = Object.fromEntries(
        Object.keys(defaults).map((k) => [k, k in saved ? saved[k] : defaults[k]]),
      );
      return [id, merged];
    }),
  ) as Drafts;
}

function reviveConn(stored: unknown): Connection {
  const s = (stored ?? {}) as Partial<Connection>;
  return {
    host: typeof s.host === "string" ? s.host : DEFAULT_CONNECTION.host,
    port: typeof s.port === "number" ? s.port : DEFAULT_CONNECTION.port,
  };
}

/** Der Testlauf gehört CARLA, solange er nicht beendet ist */
const testActive = (t: TestRunStatus | undefined) =>
  t?.state === "starting" || t?.state === "running" || t?.state === "stopping";

const isValidConn = (c: Connection) => /^[A-Za-z0-9.-]+$/.test(c.host) && c.port > 0 && c.port < 65536;
const sameConn = (a: Connection | null, b: Connection) => !!a && a.host === b.host && a.port === b.port;

/** Gültige Entwurfswerte, oder null bei Eingabefehlern (die stehen dann an der Karte) */
function cleanDraft(id: ModuleId, draft: ParamValues, conn: Connection) {
  try {
    return buildArgs(id, draft, conn).values;
  } catch {
    return null;
  }
}

export function Dashboard() {
  const mounted = useMounted();
  return (
    <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col px-4 py-8 sm:px-6 sm:py-12">
      {mounted ? <DashboardInner /> : <p className="text-sm text-ink-500">Lade …</p>}
    </div>
  );
}

function CarlaStatus({ probe, error }: { probe: CarlaProbe | null; error: string | null }) {
  let dot = "bg-ink-500";
  let text = "prüfe …";
  if (error) {
    dot = "bg-fail";
    text = error;
  } else if (probe?.reachable) {
    dot = "bg-go";
    text = `erreichbar (${probe.latencyMs} ms)`;
  } else if (probe) {
    dot = "bg-fail";
    text = `nicht erreichbar${probe.error ? ` (${probe.error})` : ""}`;
  }
  return (
    <span className="inline-flex items-center gap-2 font-mono text-xs text-ink-300">
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      {text}
    </span>
  );
}

const connInputClass =
  "rounded-md border border-ink-700 bg-ink-850 px-2.5 py-1.5 font-mono text-sm text-ink-100 focus:border-signal focus:outline-none focus:ring-2 focus:ring-signal/25";

/** Host und Port; übernommen wird erst mit Enter oder beim Verlassen des Felds (dann wird neu verbunden). */
function ConnFields({ conn, onCommit }: { conn: Connection; onCommit: (c: Connection) => void }) {
  const [host, setHost] = useState(conn.host);
  const [port, setPort] = useState(String(conn.port));
  const commit = () => {
    const next = { host: host.trim(), port: Number(port) };
    if (next.host !== conn.host || next.port !== conn.port) onCommit(next);
  };
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") commit();
  };
  const title = "Enter übernimmt; die Simulation verbindet sich dann neu.";
  return (
    <div className="flex items-center gap-2">
      <label className="sr-only" htmlFor="carla-host">
        Host
      </label>
      <input
        id="carla-host"
        value={host}
        spellCheck={false}
        onChange={(e) => setHost(e.target.value)}
        onBlur={commit}
        onKeyDown={onKeyDown}
        className={`w-36 ${connInputClass}`}
        title={`CARLA-Host. ${title}`}
      />
      <span className="text-ink-500">:</span>
      <label className="sr-only" htmlFor="carla-port">
        Port
      </label>
      <input
        id="carla-port"
        type="number"
        min={1}
        max={65535}
        value={port}
        onChange={(e) => setPort(e.target.value)}
        onBlur={commit}
        onKeyDown={onKeyDown}
        className={`w-20 ${connInputClass}`}
        title={`CARLA-Port. ${title}`}
      />
    </div>
  );
}

/** Mehrere CARLA-Server teilen sich den Port: Verbindungen landen zufällig bei einem davon. */
function InstancesWarning({
  instances,
  onKillStale,
}: {
  instances: CarlaInstance[];
  onKillStale: () => Promise<string | null>;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const stale = instances.filter((i) => i.stale);
  const time = (t: number | null) =>
    t === null ? "?" : new Date(t).toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" });

  return (
    <div className="mb-6 space-y-2 rounded-md border border-warn/40 bg-warn/10 px-4 py-3 text-sm text-warn">
      <p>
        {instances.length} CARLA-Server laufen gleichzeitig und teilen sich den Port. Verbindungen landen zufällig bei
        einem davon und hängen dann bis zum Timeout.
      </p>
      <ul className="font-mono text-xs text-ink-300">
        {instances.map((i) => (
          <li key={i.pid} className={i.stale ? "text-warn" : ""}>
            PID {i.pid} · gestartet {time(i.startedAt)} ·{" "}
            {i.hasWindow === null ? "Fenster ?" : i.hasWindow ? "mit Fenster" : "ohne Fenster"} · {i.memoryMb} MB
            {i.stale ? " → wird beendet" : " → bleibt"}
          </li>
        ))}
      </ul>
      <div className="flex flex-wrap items-center gap-3">
        {stale.length > 0 ? (
          <button
            type="button"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              setError(await onKillStale());
              setBusy(false);
            }}
            className="rounded-md border border-warn/50 bg-warn/10 px-3 py-1.5 text-sm font-medium text-warn transition-colors hover:bg-warn/20 disabled:cursor-wait disabled:opacity-60"
          >
            {busy ? "Beende …" : `PID ${stale.map((i) => i.pid).join(", ")} beenden`}
          </button>
        ) : (
          <span className="text-ink-300">Bitte alle CARLA-Fenster bis auf eins schließen.</span>
        )}
        {error && <span className="text-fail">{error}</span>}
      </div>
    </div>
  );
}

function DashboardInner() {
  const [conn, setConn] = useStoredState<Connection>("sim-control:conn", () => DEFAULT_CONNECTION, reviveConn);
  const [drafts, setDrafts] = useStoredState<Drafts>("sim-control:values", initDrafts, reviveDrafts);
  const [busy, setBusy] = useState<ModuleId | null>(null);
  const [creating, setCreating] = useState(false);
  const [errors, setErrors] = useState<Partial<Record<ModuleId, string>>>({});
  const [testBusy, setTestBusy] = useState(false);
  const [testError, setTestError] = useState<string | undefined>();
  const now = useNow();
  /** Kein automatisches Verbinden vor diesem Zeitpunkt (nach einem fehlgeschlagenen Versuch) */
  const autoAfter = useRef(0);

  const server = usePoll<StateResponse>("/api/state", 1000);
  const cameras = usePoll<{ presets: CameraPreset[] }>("/api/cameras", 3000);
  const connValid = isValidConn(conn);
  const carla = usePoll<CarlaProbe>(
    connValid ? `/api/carla?host=${encodeURIComponent(conn.host)}&port=${conn.port}` : null,
    3000,
  );
  const reachable = carla.data?.reachable === true;
  const modules = server.data?.modules;
  const testRun = server.data?.testRun;
  const testing = testActive(testRun);

  async function act(
    id: ModuleId,
    action: Action,
    { values, name }: { values?: ParamValues; name?: string } = {},
  ): Promise<boolean> {
    // automatisches Verbinden: Fehler stehen im Log, und ein anderer Tab war evtl. schneller
    const auto = action === "connect" && !values;
    setBusy(id);
    setErrors((e) => ({ ...e, [id]: undefined }));
    try {
      const res = await fetch(`/api/modules/${id}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action, values, conn, name }),
      });
      const json = await res.json();
      if (auto && res.status === 409) return false;
      if (!res.ok) throw new Error(json?.error ?? `HTTP ${res.status}`);
      server.setData((prev) => (prev ? { ...prev, modules: json.modules } : prev));
      return true;
    } catch (err) {
      if (auto) autoAfter.current = Date.now() + RETRY_MS;
      setErrors((e) => ({ ...e, [id]: err instanceof Error ? err.message : String(err) }));
      return false;
    } finally {
      setBusy(null);
      server.refresh();
    }
  }

  // Die UI haelt die Verbindung zu CARLA selbst: verbinden, sobald CARLA erreichbar ist,
  // nach einem Abbruch neu verbinden, bei neuer Adresse erst trennen.
  const syncConnection = useEffectEvent(() => {
    if (!server.data || busy || !connValid) return;
    // Während des Testlaufs gehört CARLA ihm; die Simulation verbindet sich danach wieder
    if (testActive(server.data.testRun)) return;
    const { modules, now } = server.data;
    for (const id of MODULE_ORDER) {
      const s = modules[id];
      if (s.state === "stopping") continue;
      if (s.state === "running") {
        if (!sameConn(s.conn, conn)) {
          void act(id, "disconnect");
          return;
        }
        continue;
      }
      if (!reachable || Date.now() < autoAfter.current) continue;
      if (MODULES[id].requires.some((r) => modules[r].state !== "running")) continue;
      if (endedOnItsOwn(s) && now - (s.endedAt ?? 0) < RETRY_MS) continue;
      void act(id, "connect");
      return; // eins nach dem anderen
    }
  });
  useEffect(() => syncConnection(), [server.data, conn, reachable, busy]);

  // „Erzeugen“: Entwurf übernehmen; ist ein Modul (noch) nicht verbunden, gleich mit diesen Werten verbinden
  const drafted = MODULE_ORDER.map((id) => ({ id, values: cleanDraft(id, drafts[id], conn), status: modules?.[id] }));
  const invalid = drafted.some((d) => d.values === null);
  const pending = drafted.filter(
    ({ values, status }) =>
      status?.state !== "stopping" &&
      !(status?.state === "running" && status.values && values && sameValues(status.id, values, status.values)),
  );
  const canCreate = !!modules && reachable && !busy && !creating && !invalid && !testing && pending.length > 0;
  let createTitle = "Einstellungen in CARLA übernehmen";
  if (testing) createTitle = "Der Testlauf benutzt CARLA gerade.";
  else if (!reachable) createTitle = "CARLA ist nicht erreichbar.";
  else if (invalid) createTitle = "Erst die Eingabefehler beheben.";
  else if (pending.length === 0) createTitle = "Alles übernommen.";

  async function create() {
    setCreating(true);
    try {
      for (const { id, status } of pending) {
        const values = drafts[id];
        if (status?.state === "running") await act(id, "update", { values });
        else if (!(await act(id, "connect", { values }))) return;
      }
    } finally {
      setCreating(false);
    }
  }

  async function testAction(action: "start" | "stop") {
    setTestBusy(true);
    setTestError(undefined);
    try {
      const res = await fetch("/api/test-run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action, conn }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error ?? `HTTP ${res.status}`);
      server.setData((prev) => (prev ? { ...prev, modules: json.modules, testRun: json.testRun } : prev));
    } catch (err) {
      setTestError(err instanceof Error ? err.message : String(err));
    } finally {
      setTestBusy(false);
      server.refresh();
    }
  }

  async function killStale(): Promise<string | null> {
    try {
      const res = await fetch("/api/carla/instances", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action: "killStale" }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error ?? `HTTP ${res.status}`);
      return null;
    } catch (err) {
      return err instanceof Error ? err.message : String(err);
    } finally {
      carla.refresh();
    }
  }

  // Spectator-Ansicht speichern (Python schreibt cameras.json) bzw. Spectator dorthin setzen
  async function camera(id: ModuleId, op: "save" | "goto", name: string) {
    const ok = await act(id, op === "save" ? "saveCamera" : "gotoCamera", { name });
    if (ok && op === "save") setTimeout(cameras.refresh, 300);
    return ok;
  }

  // Sichtfeld der Kamera live setzen; ohne `busy`, damit der Schieberegler nicht die Karte sperrt
  async function setFov(id: ModuleId, fov: number) {
    try {
      const res = await fetch(`/api/modules/${id}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action: "setFov", fov }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error ?? `HTTP ${res.status}`);
      setErrors((e) => (e[id] ? { ...e, [id]: undefined } : e));
      return true;
    } catch (err) {
      setErrors((e) => ({ ...e, [id]: err instanceof Error ? err.message : String(err) }));
      return false;
    }
  }

  const instances = carla.data?.instances ?? [];

  return (
    <>
      <header className="mb-8 flex flex-wrap items-end justify-between gap-x-6 gap-y-4">
        <div>
          <p className="mb-1 font-mono text-[11px] uppercase tracking-[0.2em] text-ink-500">CARLA</p>
          <h1 className="text-2xl font-semibold tracking-tight text-ink-100">Sim Control</h1>
        </div>

        <div className="flex flex-wrap items-end gap-3">
          <div className="flex flex-col items-start gap-2 sm:items-end">
            <CarlaStatus probe={carla.data} error={connValid ? carla.error : "ungültige Adresse"} />
            <ConnFields conn={conn} onCommit={setConn} />
          </div>
          <button
            type="button"
            disabled={!canCreate}
            onClick={() => void create()}
            title={createTitle}
            className="relative inline-flex items-center gap-2 rounded-md bg-go px-4 py-1.75 text-sm font-semibold text-ink-950 transition-colors hover:bg-go/85 disabled:cursor-not-allowed disabled:bg-ink-800 disabled:text-ink-500"
          >
            {creating ? "Übernehme …" : "Übernehmen"}
            {canCreate && (
              <span aria-label="nicht übernommene Änderungen" className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full border-2 border-ink-950 bg-warn" />
            )}
          </button>
        </div>
      </header>

      {server.error && !modules && (
        <div className="mb-6 rounded-md border border-fail/40 bg-fail/10 px-4 py-3 text-sm text-fail">
          Dashboard-Server nicht erreichbar: {server.error}
        </div>
      )}

      {instances.length > 1 && <InstancesWarning instances={instances} onKillStale={killStale} />}

      <ol className="flex flex-col">
        {MODULE_ORDER.map((id, i) => {
          const def = MODULES[id];
          const blockedBy = def.requires
            .filter((r) => modules?.[r]?.state !== "running")
            .map((r) => MODULES[r].title);
          return (
            <Fragment key={id}>
              {i > 0 && (
                <li aria-hidden className="ml-[33px] flex h-8 items-center">
                  <span
                    className={`h-full border-l-2 border-dashed ${
                      blockedBy.length === 0 ? "border-go/40" : "border-ink-700"
                    }`}
                  />
                  <span className="ml-3 font-mono text-[11px] uppercase tracking-wider text-ink-500">dann</span>
                </li>
              )}
              <li>
                <ModuleCard
                  def={def}
                  status={modules?.[id]}
                  draft={drafts[id]}
                  conn={conn}
                  now={now}
                  reachable={reachable}
                  pending={pending.some((p) => p.id === id) && modules?.[id]?.state === "running"}
                  blockedBy={modules ? blockedBy : []}
                  busy={busy === id || !modules}
                  error={errors[id]}
                  cameras={cameras.data?.presets ?? []}
                  onDraftChange={(values) => setDrafts((d) => ({ ...d, [id]: values }))}
                  onReset={() => setDrafts({ ...drafts, [id]: defaultValues(id) })}
                  onKill={() => void act(id, "kill")}
                  onCamera={(op, name) => camera(id, op, name)}
                  onFov={(fov) => setFov(id, fov)}
                />
              </li>
            </Fragment>
          );
        })}
      </ol>

      <div className="mt-8">
        <TestRunCard
          status={testRun}
          now={now}
          busy={testBusy || !modules}
          error={testError}
          onStart={() => void testAction("start")}
          onStop={() => void testAction("stop")}
        />
      </div>
    </>
  );
}
