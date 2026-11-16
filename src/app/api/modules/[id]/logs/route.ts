import type { NextRequest } from "next/server";

import { jsonError } from "@/lib/api";
import { isModuleId } from "@/lib/modules";
import { manager } from "@/lib/process-manager";

export async function GET(req: NextRequest, ctx: RouteContext<"/api/modules/[id]/logs">) {
  const { id } = await ctx.params;
  if (!isModuleId(id)) return jsonError(`Unbekanntes Modul: ${id}`, 404);

  const params = req.nextUrl.searchParams;
  const runId = Number(params.get("run") ?? -1);
  const offset = Number(params.get("offset") ?? 0);
  return Response.json(manager.readLog(id, runId, offset));
}
