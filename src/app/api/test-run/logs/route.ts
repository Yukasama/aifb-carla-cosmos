import type { NextRequest } from "next/server";

import { testRun } from "@/lib/process-manager";

export async function GET(req: NextRequest) {
  const params = req.nextUrl.searchParams;
  const runId = Number(params.get("run") ?? -1);
  const offset = Number(params.get("offset") ?? 0);
  return Response.json(testRun.readLog(runId, offset));
}
