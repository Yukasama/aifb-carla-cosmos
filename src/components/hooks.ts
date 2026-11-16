"use client";

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";

const noopSubscribe = () => () => {};

/** true erst im Browser – erlaubt localStorage in useState-Initialisierern ohne Hydration-Mismatch. */
export function useMounted() {
  return useSyncExternalStore(
    noopSubscribe,
    () => true,
    () => false,
  );
}

/** useState, das in localStorage gespiegelt wird. Nur in Komponenten nutzen, die erst nach useMounted() rendern. */
export function useStoredState<T>(key: string, init: () => T, revive?: (stored: unknown) => T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      if (raw !== null) {
        const parsed: unknown = JSON.parse(raw);
        return revive ? revive(parsed) : (parsed as T);
      }
    } catch {
      // kaputter / blockierter Speicher -> Default
    }
    return init();
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // egal
    }
  }, [key, value]);

  return [value, setValue] as const;
}

type PollState<T> = { url: string | null; data: T | null; error: string | null };

/**
 * Pollt eine JSON-URL in festem Abstand (nach Abschluss des vorherigen Requests).
 * Daten gehören zu ihrer URL: Wechselt sie, gibt es bis zur ersten Antwort nichts statt der alten.
 */
export function usePoll<T>(url: string | null, intervalMs: number) {
  const [state, setState] = useState<PollState<T>>({ url, data: null, error: null });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!url) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ctrl = new AbortController();

    const run = async () => {
      try {
        const res = await fetch(url, { cache: "no-store", signal: ctrl.signal });
        const json = await res.json();
        if (cancelled) return;
        if (!res.ok) throw new Error(json?.error ?? `HTTP ${res.status}`);
        setState({ url, data: json as T, error: null });
      } catch (err) {
        if (cancelled) return;
        const error = err instanceof Error ? err.message : String(err);
        setState((s) => ({ url, data: s.url === url ? s.data : null, error }));
      }
      if (!cancelled) timer = setTimeout(run, intervalMs);
    };
    void run();

    return () => {
      cancelled = true;
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [url, intervalMs, nonce]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  const setData = useCallback(
    (update: (prev: T | null) => T | null) =>
      setState((s) => (s.url === url ? { ...s, data: update(s.data) } : s)),
    [url],
  );
  const current = state.url === url;
  return { data: current ? state.data : null, error: current ? state.error : null, refresh, setData };
}

/** Aktuelle Zeit, sekündlich aktualisiert (für Laufzeit-Anzeigen). */
export function useNow(intervalMs = 1000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(t);
  }, [intervalMs]);
  return now;
}

export function formatDuration(ms: number) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(sec).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
