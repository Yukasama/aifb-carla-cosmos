"use client";

import { useId } from "react";

import type { ParamDef, ParamValue } from "@/lib/modules";

type Props = {
  param: ParamDef;
  value: ParamValue;
  onChange: (value: ParamValue) => void;
  disabled: boolean;
  /** Felder, die direkt unter einem eingeschalteten Schalter erscheinen */
  children?: React.ReactNode;
};

export const inputClass =
  "w-full rounded-md border border-ink-700 bg-ink-850 px-2.5 py-1.5 font-mono text-sm text-ink-100 " +
  "placeholder:text-ink-500 transition-colors hover:border-ink-500 " +
  "focus:border-signal focus:outline-none focus:ring-2 focus:ring-signal/25 " +
  "disabled:cursor-not-allowed disabled:border-ink-800 disabled:text-ink-300";

export function ParamField({ param: p, value, onChange, disabled, children }: Props) {
  const id = useId();

  if (p.type === "bool") {
    const on = value === true;
    return (
      <div>
        <label
          htmlFor={id}
          className={`flex items-start gap-3 rounded-md px-2 py-1.5 ${disabled ? "cursor-not-allowed" : "cursor-pointer hover:bg-ink-850"}`}
        >
          <button
            id={id}
            type="button"
            role="switch"
            aria-checked={on}
            disabled={disabled}
            onClick={() => onChange(!on)}
            className={`relative mt-0.5 inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/40 disabled:opacity-60
              ${on ? "border-go/60 bg-go/25" : "border-ink-700 bg-ink-850"}`}
          >
            <span
              className={`inline-block h-3.5 w-3.5 rounded-full transition-transform
                ${on ? "translate-x-[18px] bg-go" : "translate-x-[2px] bg-ink-500"}`}
            />
          </button>
          <span className="min-w-0">
            <span className="block text-sm text-ink-100">{p.label}</span>
            {p.help && <span className="block text-xs leading-snug text-ink-500">{p.help}</span>}
          </span>
        </label>
        {on && children && <div className="mb-2 ml-14 mr-2 mt-1 space-y-3">{children}</div>}
      </div>
    );
  }

  if (p.type === "int" || p.type === "float") {
    const v = typeof value === "number" ? value : p.min;
    return (
      <div className="min-w-0">
        <label htmlFor={id} className="mb-1 flex items-baseline justify-between gap-2 text-xs text-ink-300">
          <span>{p.label}</span>
          <span className="font-mono tabular-nums text-ink-100">
            {v}
            {p.unit && <span className="ml-1 text-ink-500">{p.unit}</span>}
          </span>
        </label>
        <input
          id={id}
          type="range"
          min={p.min}
          max={p.max}
          step={p.step ?? (p.type === "int" ? 1 : 0.1)}
          value={v}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
          className="h-5 w-full cursor-pointer accent-signal disabled:cursor-not-allowed disabled:opacity-50"
        />
        {p.help && <p className="mt-1 text-xs leading-snug text-ink-500">{p.help}</p>}
      </div>
    );
  }

  return (
    <div className="min-w-0">
      <label htmlFor={id} className="mb-1 block text-xs text-ink-300">
        {p.label}
      </label>
      {p.type === "select" ? (
        <select
          id={id}
          className={inputClass}
          disabled={disabled}
          value={String(value ?? p.default)}
          onChange={(e) => onChange(e.target.value)}
        >
          {p.options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          type="text"
          spellCheck={false}
          autoComplete="off"
          className={inputClass}
          disabled={disabled}
          value={typeof value === "string" ? value : ""}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
      {p.help && <p className="mt-1 text-xs leading-snug text-ink-500">{p.help}</p>}
    </div>
  );
}
