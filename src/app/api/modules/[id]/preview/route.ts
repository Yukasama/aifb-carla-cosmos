import fs from "node:fs";

import type { NextRequest } from "next/server";
import { connection } from "next/server";

import { jsonError } from "@/lib/api";
import { MODULES, isModuleId } from "@/lib/modules";
import { manager, previewFile } from "@/lib/process-manager";

/** Aktuelles Kamerabild des laufenden Prozesses als JPEG; 204, solange es keins gibt. */
export async function GET(_req: NextRequest, ctx: RouteContext<"/api/modules/[id]/preview">) {
  await connection(); // nie vorrendern, das Bild ändert sich laufend
  const { id } = await ctx.params;
  if (!isModuleId(id) || !MODULES[id].objects) return jsonError(`Keine Vorschau für Modul: ${id}`, 404);
  if (manager.status(id).state !== "running") return new Response(null, { status: 204 });
  let jpg: Buffer;
  try {
    jpg = fs.readFileSync(previewFile(id));
  } catch {
    return new Response(null, { status: 204 }); // noch nicht geschrieben
  }
  return new Response(new Uint8Array(jpg), {
    headers: { "content-type": "image/jpeg", "cache-control": "no-store" },
  });
}
