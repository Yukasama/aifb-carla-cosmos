import "server-only";

import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";

import {
  MODULES,
  MODULE_ORDER,
  buildArgs,
  connectValues,
  dependentsOf,
  validateCameraName,
  validateConnection,
  type Connection,
  type ModuleId,
  type ParamValues,
} from "./modules";
import type { LogChunk, ModuleStatus, TestRunStatus } from "./types";

// uv-Projekt mit den Python-Packages
export const SCRIPTS_DIR = path.resolve(
  process.env.SIM_SCRIPTS_DIR ?? path.join(/* turbopackIgnore: true */ process.cwd(), "python_api"),
);
// Direkt das venv-Python, nicht `uv run`: EOF auf stdin muss beim Python-Prozess ankommen
const IS_WINDOWS = process.platform === "win32";
const VENV_PYTHON = IS_WINDOWS
  ? path.join(SCRIPTS_DIR, ".venv", "Scripts", "python.exe")
  : path.join(SCRIPTS_DIR, ".venv", "bin", "python");
const PYTHON =
  process.env.SIM_PYTHON ?? (fs.existsSync(VENV_PYTHON) ? VENV_PYTHON : IS_WINDOWS ? "python" : "python3");
const RUNTIME_DIR = path.join(/* turbopackIgnore: true */ process.cwd(), ".runtime");
const LOG_DIR = path.join(RUNTIME_DIR, "logs");
const STATUS_DIR = path.join(RUNTIME_DIR, "status");

/** Hierhin schreibt ein Modul mit `objects` laufend die Objekte im Bild (siehe objects.ts) */
export function statusFile(id: ModuleId) {
  return path.join(STATUS_DIR, `${id}.json`);
}

const MAX_LOG_CHUNK = 256 * 1024;
/** So lange nach der letzten Statusdatei gilt ein Modul mit `objects` noch als verbunden */
const READY_MS = 3000;

export class ConflictError extends Error {}

interface Entry extends Omit<ModuleStatus, "ready"> {
  child: ChildProcess | null;
  killTimer: NodeJS.Timeout | null;
  exited: Promise<void>;
  resolveExited: () => void;
}

function timestamp() {
  return new Date().toLocaleTimeString("de-DE");
}

class ProcessManager {
  private entries: Record<ModuleId, Entry>;
  private nextRunId = 1;

  constructor() {
    // Kein Exit-Handler noetig: Endet der Node-Prozess, schliesst das OS die stdin-Pipe der
    // Kinder. Python sieht EOF und raeumt selbst auf (Actors zerstoeren, Sync-Modus aus); seine
    // Ausgabe geht in Dateien, nicht in Pipes, daher laeuft das auch ohne uns zu Ende.
    this.entries = Object.fromEntries(
      MODULE_ORDER.map((id) => [id, this.blankEntry(id)]),
    ) as Record<ModuleId, Entry>;
  }

  private blankEntry(id: ModuleId): Entry {
    let resolveExited = () => {};
    const exited = new Promise<void>((r) => (resolveExited = r));
    resolveExited();
    return {
      id,
      state: "idle",
      runId: 0,
      pid: null,
      startedAt: null,
      endedAt: null,
      exitCode: null,
      signal: null,
      values: null,
      conn: null,
      stopReason: null,
      child: null,
      killTimer: null,
      exited,
      resolveExited,
    };
  }

  private logPath(id: ModuleId) {
    return path.join(LOG_DIR, `${id}.log`);
  }

  private sysLog(id: ModuleId, msg: string) {
    try {
      fs.appendFileSync(this.logPath(id), `[sim-control ${timestamp()}] ${msg}\n`);
    } catch {
      // Log ist nur Komfort
    }
  }

  private isActive(id: ModuleId) {
    const s = this.entries[id].state;
    return s === "running" || s === "stopping";
  }

  /** Laeuft und tickt die Welt? Module mit `objects` schreiben dabei laufend ihre Statusdatei. */
  private isReady(id: ModuleId) {
    if (this.entries[id].state !== "running") return false;
    if (!MODULES[id].objects) return true;
    try {
      return Date.now() - fs.statSync(statusFile(id)).mtimeMs < READY_MS;
    } catch {
      return false; // noch nicht verbunden
    }
  }

  status(id: ModuleId): ModuleStatus {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { child, killTimer, exited, resolveExited, ...status } = this.entries[id];
    return { ...status, ready: this.isReady(id) };
  }

  statusAll(): Record<ModuleId, ModuleStatus> {
    return Object.fromEntries(MODULE_ORDER.map((id) => [id, this.status(id)])) as Record<
      ModuleId,
      ModuleStatus
    >;
  }

  /**
   * Mit CARLA verbinden: Python-Prozess starten. Ohne Werte mit `connectValues` (legt nichts
   * an), sonst gleich mit diesen.
   */
  start(id: ModuleId, rawConn: unknown, rawValues?: unknown): ModuleStatus {
    const def = MODULES[id];
    const e = this.entries[id];

    if (this.isActive(id)) throw new ConflictError(`„${def.title}“ ist schon verbunden.`);
    if (testRun.active()) throw new ConflictError("Der Testlauf benutzt CARLA gerade.");
    for (const req of def.requires) {
      if (this.entries[req].state !== "running") {
        throw new ConflictError(`Erst „${MODULES[req].title}“ verbinden.`);
      }
    }

    const conn = validateConnection(rawConn);
    for (const req of def.requires) {
      const reqConn = this.entries[req].conn;
      if (reqConn && (reqConn.host !== conn.host || reqConn.port !== conn.port)) {
        throw new ConflictError(
          `„${MODULES[req].title}“ ist mit ${reqConn.host}:${reqConn.port} verbunden – gleiche Verbindung verwenden.`,
        );
      }
    }

    const { args, values } = buildArgs(
      id,
      (rawValues && typeof rawValues === "object" ? rawValues : connectValues(id)) as ParamValues,
      conn,
    );
    const entry = path.join(SCRIPTS_DIR, def.script, "__main__.py");
    if (!fs.existsSync(entry)) throw new Error(`Package nicht gefunden: ${entry}`);

    const argv = ["-u", "-m", def.script, ...args, "--stdin-control"];
    if (def.objects) {
      // Datei vom letzten Lauf weg, sonst zeigt die UI bis zum ersten Frame alte Objekte
      fs.mkdirSync(STATUS_DIR, { recursive: true });
      fs.rmSync(statusFile(id), { force: true });
      argv.push("--status-file", statusFile(id));
    }
    const runId = this.nextRunId++;

    // Neues Log pro Lauf. O_APPEND, damit unsere Statuszeilen und die Ausgabe
    // des Kindprozesses sich nicht gegenseitig ueberschreiben.
    fs.mkdirSync(LOG_DIR, { recursive: true });
    const logPath = this.logPath(id);
    fs.writeFileSync(logPath, `[sim-control ${timestamp()}] Verbinde mit CARLA ${conn.host}:${conn.port} …\n`);
    const fd = fs.openSync(logPath, "a");

    let child: ChildProcess;
    try {
      child = spawn(PYTHON, argv, {
        cwd: SCRIPTS_DIR,
        // UTF-8 auf stdin/stdout auch unter Windows (Positionsnamen mit Umlauten)
        env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
        // stdin: Updates (siehe update()); EOF beendet das Skript sauber (siehe stop())
        stdio: ["pipe", fd, fd],
        // Eigene Prozessgruppe: Ctrl+C im Terminal des Dev-Servers erreicht das Skript nicht
        // und bricht so nicht dessen Aufraeumen ab. Es endet ueber EOF, wenn der Server endet.
        detached: true,
        windowsHide: true,
      });
    } finally {
      fs.closeSync(fd);
    }

    let resolveExited = () => {};
    const exited = new Promise<void>((r) => (resolveExited = r));

    Object.assign(e, {
      state: "running",
      runId,
      pid: child.pid ?? null,
      startedAt: Date.now(),
      endedAt: null,
      exitCode: null,
      signal: null,
      values,
      conn,
      stopReason: null,
      child,
      killTimer: null,
      exited,
      resolveExited,
    } satisfies Partial<Entry>);

    child.stdin?.on("error", () => {}); // EPIPE, wenn das Skript schon beendet ist
    child.once("error", (err) => {
      this.sysLog(id, `Start fehlgeschlagen: ${err.message}`);
      this.onExit(id, runId, null, null);
    });
    child.once("exit", (code, signal) => this.onExit(id, runId, code, signal));

    return this.status(id);
  }

  /** Neue Werte an den laufenden Prozess schicken (komplette Flag-Liste als JSON-Zeile). */
  update(id: ModuleId, rawValues: unknown): ModuleStatus {
    const e = this.entries[id];
    if (e.state !== "running" || !e.child?.stdin || !e.conn) {
      throw new ConflictError(`„${MODULES[id].title}“ ist nicht verbunden.`);
    }
    const values = (rawValues && typeof rawValues === "object" ? rawValues : {}) as ParamValues;
    const built = buildArgs(id, values, e.conn);
    e.child.stdin.write(JSON.stringify(built.args) + "\n");
    e.values = built.values;
    return this.status(id);
  }

  /** Kameraposition im laufenden Prozess: Spectator-Ansicht speichern bzw. Spectator dorthin setzen. */
  camera(id: ModuleId, op: "save" | "goto", rawName: unknown): ModuleStatus {
    const e = this.entries[id];
    if (e.state !== "running" || !e.child?.stdin) {
      throw new ConflictError(`„${MODULES[id].title}“ ist nicht verbunden.`);
    }
    const name = validateCameraName(rawName);
    e.child.stdin.write(JSON.stringify({ [`${op}_camera`]: name }) + "\n");
    return this.status(id);
  }

  private onExit(id: ModuleId, runId: number, code: number | null, signal: NodeJS.Signals | null) {
    const e = this.entries[id];
    if (e.runId !== runId || e.child === null) return; // schon behandelt

    if (e.killTimer) clearTimeout(e.killTimer);
    Object.assign(e, {
      state: "exited",
      child: null,
      killTimer: null,
      pid: null,
      endedAt: Date.now(),
      exitCode: code,
      signal,
    } satisfies Partial<Entry>);
    this.sysLog(id, signal ? `Prozess beendet (Signal ${signal}).` : `Prozess beendet (Exit-Code ${code}).`);
    e.resolveExited();

    // Abhaengige Module brauchen dieses hier -> mit beenden
    for (const dep of dependentsOf(id)) {
      if (this.entries[dep].state === "running") {
        this.sysLog(dep, `„${MODULES[id].title}“ wurde beendet → trenne.`);
        void this.stop(dep, `${MODULES[id].title} beendet`);
      }
    }
  }

  /**
   * Sauber trennen: erst abhaengige Module, dann stdin schliessen (EOF), nach Timeout hart.
   * Kein SIGINT: unter Windows beendet Node den Prozess damit sofort (TerminateProcess), ohne
   * dass Python aufraeumt. CARLA bliebe im Sync-Modus stehen, ohne dass jemand tickt, und friert ein.
   */
  async stop(id: ModuleId, reason = "manuell"): Promise<void> {
    const e = this.entries[id];
    if (e.state === "stopping") return e.exited;
    if (e.state !== "running" || !e.child) return;

    e.state = "stopping";
    e.stopReason = reason;

    const deps = dependentsOf(id).filter((d) => this.isActive(d));
    if (deps.length) {
      this.sysLog(id, `Trenne zuerst: ${deps.map((d) => MODULES[d].title).join(", ")}`);
      await Promise.all(deps.map((d) => this.stop(d, `${MODULES[id].title} wird getrennt`)));
    }

    const child = e.child;
    if (!child) return; // inzwischen von selbst beendet

    const timeoutMs = MODULES[id].stopTimeoutMs;
    this.sysLog(id, "Trenne, räume auf …");
    child.stdin?.end();
    e.killTimer = setTimeout(() => {
      if (e.child === child) {
        this.sysLog(id, `Nach ${timeoutMs / 1000} s nicht beendet → hart beenden.`);
        child.kill("SIGKILL");
      }
    }, timeoutMs);

    return e.exited;
  }

  /** Sofort hart beenden (kein Python-Cleanup!). */
  kill(id: ModuleId) {
    const e = this.entries[id];
    if (!e.child) return;
    e.state = "stopping";
    e.stopReason ??= "erzwungen";
    this.sysLog(id, "Hart beendet (erzwungen).");
    e.child.kill("SIGKILL");
  }

  readLog(id: ModuleId, runId: number, offset: number): LogChunk {
    return readLogFile(this.logPath(id), this.entries[id].runId, runId, offset);
  }
}

/** Neue vollstaendige Zeilen eines Logs ab `offset`; anderer Lauf -> von vorn (reset). */
function readLogFile(file: string, current: number, runId: number, offset: number): LogChunk {
  let reset = runId !== current;
  let start = reset ? 0 : Math.max(0, Math.floor(offset) || 0);

  let fd: number;
  try {
    fd = fs.openSync(file, "r");
  } catch {
    return { runId: current, offset: 0, text: "", reset };
  }

  try {
    const size = fs.fstatSync(fd).size;
    if (start > size) {
      start = 0;
      reset = true;
    }
    // Beim ersten Laden nur das Ende eines sehr langen Logs liefern
    let tail = false;
    if (start === 0 && size > MAX_LOG_CHUNK) {
      start = size - MAX_LOG_CHUNK;
      tail = true;
    }

    const len = Math.min(size - start, MAX_LOG_CHUNK);
    if (len <= 0) return { runId: current, offset: start, text: "", reset };

    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, start);

    // Nur vollstaendige Zeilen ausliefern (und keine halben UTF-8-Zeichen)
    let end = buf.lastIndexOf(0x0a) + 1;
    if (end === 0) end = len === MAX_LOG_CHUNK ? len : 0;
    // beim Tail die angeschnittene erste Zeile verwerfen
    const begin = tail ? Math.min(buf.indexOf(0x0a) + 1, end) : 0;

    return {
      runId: current,
      offset: start + end,
      text: buf.subarray(begin, end).toString("utf8"),
      reset,
    };
  } finally {
    fs.closeSync(fd);
  }
}

// Ein Manager pro Server-Prozess (ueberlebt Hot Reloads im Dev-Modus; Methoden kommen dann aus dem neuen Code)
const g = globalThis as typeof globalThis & { __simControlManager?: ProcessManager };
if (g.__simControlManager) Object.setPrototypeOf(g.__simControlManager, ProcessManager.prototype);
export const manager = (g.__simControlManager ??= new ProcessManager());

/** Prueft, ob der CARLA-Server per TCP erreichbar ist. */
export function probeTcp(
  conn: Connection,
  timeoutMs = 1500,
): Promise<{ reachable: boolean; latencyMs: number | null; error: string | null }> {
  const { host, port } = validateConnection(conn);
  return new Promise((resolve) => {
    const t0 = performance.now();
    const socket = net.connect({ host, port });
    const done = (reachable: boolean, error: string | null) => {
      socket.destroy();
      resolve({ reachable, latencyMs: reachable ? Math.round(performance.now() - t0) : null, error });
    };
    socket.setTimeout(timeoutMs, () => done(false, "Timeout"));
    socket.once("connect", () => done(true, null));
    socket.once("error", (err: NodeJS.ErrnoException) => done(false, err.code ?? err.message));
  });
}

// ---------- Testlauf von record_scenes.py ----------

/** So viele Frames nimmt der Testlauf auf: ein Clip von der ersten Kameraposition */
export const TEST_FRAMES = 93;
const TEST_SCRIPT = "record_scenes.py";
const TEST_LOG = path.join(LOG_DIR, "test-run.log");
/** nach EOF auf stdin: Aufraeumen, CARLA beenden; danach hart */
const TEST_STOP_MS = 60_000;

/**
 * Ein Lauf von record_scenes.py, wie von Hand gestartet: benutzt das CARLA unter der eingestellten
 * Adresse oder startet eins ohne Fenster. Solange er laeuft, gehoert CARLA ihm: die Simulation wird
 * vorher getrennt und verbindet sich erst danach wieder (zwei Programme duerfen nicht gleichzeitig ticken).
 */
class TestRun {
  private s: TestRunStatus = {
    state: "idle",
    runId: 0,
    frames: TEST_FRAMES,
    startedAt: null,
    endedAt: null,
    exitCode: null,
    stopReason: null,
  };
  private child: ChildProcess | null = null;
  private killTimer: NodeJS.Timeout | null = null;
  private nextRunId = 1;

  active() {
    return this.s.state === "starting" || this.s.state === "running" || this.s.state === "stopping";
  }

  status(): TestRunStatus {
    return { ...this.s };
  }

  private log(msg: string) {
    try {
      fs.appendFileSync(TEST_LOG, `[sim-control ${timestamp()}] ${msg}
`);
    } catch {
      // Log ist nur Komfort
    }
  }

  /** Startet im Hintergrund (Trennen der Simulation kann dauern); der Client pollt den Zustand. */
  start(rawConn: unknown): TestRunStatus {
    if (this.active()) throw new ConflictError("Der Testlauf läuft schon.");
    const conn = validateConnection(rawConn);
    if (!fs.existsSync(path.join(SCRIPTS_DIR, TEST_SCRIPT))) {
      throw new Error(`Skript nicht gefunden: ${path.join(SCRIPTS_DIR, TEST_SCRIPT)}`);
    }
    const runId = this.nextRunId++;
    this.s = {
      state: "starting",
      runId,
      frames: TEST_FRAMES,
      startedAt: Date.now(),
      endedAt: null,
      exitCode: null,
      stopReason: null,
    };
    fs.mkdirSync(LOG_DIR, { recursive: true });
    fs.writeFileSync(TEST_LOG, `[sim-control ${timestamp()}] Testlauf: ${TEST_FRAMES} Frames, CARLA ${conn.host}:${conn.port}
`);
    void this.launch(runId, conn).catch((err) => this.finish(runId, null, err instanceof Error ? err.message : String(err)));
    return this.status();
  }

  private async launch(runId: number, conn: Connection) {
    if (manager.status("sim").state === "running" || manager.status("sim").state === "stopping") {
      this.log("Trenne die Simulation …");
      await manager.stop("sim", "Testlauf");
    }
    if (this.s.runId !== runId || this.s.state !== "starting") return; // inzwischen abgebrochen

    const fd = fs.openSync(TEST_LOG, "a");
    let child: ChildProcess;
    try {
      child = spawn(
        PYTHON,
        ["-u", TEST_SCRIPT, "--frames", String(TEST_FRAMES), "--host", conn.host, "--port", String(conn.port), "--use-running", "--stop-on-eof"],
        {
          cwd: SCRIPTS_DIR,
          env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
          // stdin offen halten: EOF bricht den Lauf sauber ab (siehe stop())
          stdio: ["pipe", fd, fd],
          detached: true,
          windowsHide: true,
        },
      );
    } finally {
      fs.closeSync(fd);
    }
    this.child = child;
    this.s.state = "running";
    child.stdin?.on("error", () => {});
    child.once("error", (err) => this.finish(runId, null, `Start fehlgeschlagen: ${err.message}`));
    child.once("exit", (code) => this.finish(runId, code, null));
  }

  private finish(runId: number, code: number | null, error: string | null) {
    if (this.s.runId !== runId || this.s.state === "exited") return;
    if (this.killTimer) clearTimeout(this.killTimer);
    this.killTimer = null;
    this.child = null;
    if (error) this.log(error);
    const how = code === null ? "" : ` (Exit-Code ${code})`;
    this.log(this.s.stopReason ? `Abgebrochen${how}.` : `Beendet${how}.`);
    Object.assign(this.s, { state: "exited", endedAt: Date.now(), exitCode: code } satisfies Partial<TestRunStatus>);
  }

  /** Sauber abbrechen: stdin schliessen, record_scenes.py raeumt auf und beendet sein CARLA. */
  stop() {
    if (this.s.state === "starting") {
      this.s.stopReason = "abgebrochen";
      this.finish(this.s.runId, null, null);
      return;
    }
    const child = this.child;
    if (this.s.state !== "running" || !child) return;
    this.s.state = "stopping";
    this.s.stopReason = "abgebrochen";
    this.log("Breche ab, räume auf …");
    child.stdin?.end();
    this.killTimer = setTimeout(() => {
      if (this.child === child) {
        this.log(`Nach ${TEST_STOP_MS / 1000} s nicht beendet → hart beenden.`);
        child.kill("SIGKILL");
      }
    }, TEST_STOP_MS);
  }

  readLog(runId: number, offset: number): LogChunk {
    return readLogFile(TEST_LOG, this.s.runId, runId, offset);
  }
}

const gt = globalThis as typeof globalThis & { __simControlTestRun?: TestRun };
if (gt.__simControlTestRun) Object.setPrototypeOf(gt.__simControlTestRun, TestRun.prototype);
export const testRun = (gt.__simControlTestRun ??= new TestRun());
