import "server-only";

import { ArgError } from "./modules";
import { ConflictError } from "./process-manager";

export function jsonError(message: string, status: number) {
  return Response.json({ error: message }, { status });
}

export function errorResponse(err: unknown) {
  if (err instanceof ArgError) return jsonError(err.message, 400);
  if (err instanceof ConflictError) return jsonError(err.message, 409);
  console.error(err);
  return jsonError(err instanceof Error ? err.message : String(err), 500);
}

/**
 * Schreibende Requests nur von der eigenen Seite: JSON-Content-Type erzwingt bei
 * fremden Origins einen CORS-Preflight, den wir nicht beantworten; zusaetzlich
 * muss ein mitgeschickter Origin zum Host passen.
 */
export function checkSameOrigin(req: Request): Response | null {
  if (!req.headers.get("content-type")?.includes("application/json")) {
    return jsonError("Content-Type application/json erwartet.", 415);
  }
  const origin = req.headers.get("origin");
  if (origin) {
    let originHost: string | null = null;
    try {
      originHost = new URL(origin).host;
    } catch {}
    if (originHost !== req.headers.get("host")) return jsonError("Fremder Origin.", 403);
  }
  return null;
}
