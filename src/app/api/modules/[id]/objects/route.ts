import type { NextRequest } from "next/server";
import { connection } from "next/server";

import { jsonError } from "@/lib/api";
import { MODULES, isModuleId } from "@/lib/modules";
import { readFrameObjects } from "@/lib/objects";
import { manager, statusFile } from "@/lib/process-manager";

/** Objekte im Kamerabild des laufenden Prozesses; null, wenn er nicht läuft oder noch nichts geschrieben hat. */
export async function GET(_req: NextRequest, ctx: RouteContext<"/api/modules/[id]/objects">) {
  await connection(); // nie vorrendern, die Datei ändert sich laufend
  const { id } = await ctx.params;
  if (!isModuleId(id) || !MODULES[id].objects) return jsonError(`Keine Objekte für Modul: ${id}`, 404);
  if (manager.status(id).state !== "running") return Response.json(null);
  return Response.json(readFrameObjects(statusFile(id)));
}
