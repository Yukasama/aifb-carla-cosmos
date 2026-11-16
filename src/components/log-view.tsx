"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import type { LogChunk } from "@/lib/types";

import { useStoredState } from "./hooks";

const MAX_LINES = 3000;

function lineClass(line: string) {
  if (line.startsWith("[sim-control")) return "text-signal/90";
  if (/Traceback|Error|ERROR|Exception|Fehler/.test(line)) return "text-fail";
  if (/WARNING|Warning|Hinweis/.test(line)) return "text-warn";
  return "text-ink-300";
}

type Props = {
  /** merkt sich, ob die Ausgabe aufgeklappt ist */
  logKey: string;
  /** liefert LogChunks, bekommt ?run=&offset= angehaengt */
  url: string;
  runId: number;
  live: boolean;
  /** solange es noch keinen Lauf gibt */
  emptyText?: string;
};

export function LogView({ logKey, url, runId, live, emptyText = "Noch nicht verbunden." }: Props) {
  const [open, setOpen] = useStoredState(`sim-control:log-open:${logKey}`, () => true);
  const [lines, setLines] = useState<string[]>([]);
  const cursor = useRef({ run: -1, offset: 0 });
  const box = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    if (!open || runId === 0) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      try {
        const { run, offset } = cursor.current;
        const res = await fetch(`${url}?run=${run}&offset=${offset}`, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const chunk = (await res.json()) as LogChunk;
        if (cancelled) return;
        cursor.current = { run: chunk.runId, offset: chunk.offset };
        if (chunk.reset || chunk.text) {
          const fresh = chunk.text ? chunk.text.replace(/\n$/, "").split("\n") : [];
          setLines((prev) => {
            const merged = chunk.reset ? fresh : prev.concat(fresh);
            return merged.length > MAX_LINES ? merged.slice(-MAX_LINES) : merged;
          });
        }
      } catch {
        // naechster Versuch
      }
      // Laufende Prozesse live mitlesen, beendete nur einmal abholen
      if (!cancelled && live) timer = setTimeout(poll, 700);
    };
    void poll();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [url, open, runId, live]);

  useLayoutEffect(() => {
    const el = box.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines, open]);

  return (
    <div className="border-t border-ink-800">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-5 py-2.5 text-left text-xs text-ink-500 transition-colors hover:text-ink-300"
      >
        <span className={`inline-block transition-transform ${open ? "rotate-90" : ""}`}>▸</span>
        <span className="font-medium uppercase tracking-wider">Ausgabe</span>
        {runId > 0 && <span className="font-mono">{lines.length} Zeilen</span>}
        {live && <span className="ml-auto h-1.5 w-1.5 animate-pulse rounded-full bg-go" aria-label="live" />}
      </button>
      {open && (
        <div
          ref={box}
          onScroll={(e) => {
            const el = e.currentTarget;
            stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          }}
          className="max-h-72 min-h-24 overflow-auto rounded-b-xl bg-ink-950/70 px-5 py-3 font-mono text-xs leading-relaxed"
        >
          {runId === 0 ? (
            <p className="text-ink-500">{emptyText}</p>
          ) : lines.length === 0 ? (
            <p className="text-ink-500">Warte auf Ausgabe …</p>
          ) : (
            lines.map((line, i) => (
              <div key={i} className={`whitespace-pre-wrap break-all ${lineClass(line)}`}>
                {line || " "}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
