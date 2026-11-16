import { connection } from "next/server";

import { manager, testRun } from "@/lib/process-manager";

export async function GET() {
  await connection(); // nie vorrendern, immer Live-Zustand
  return Response.json({
    modules: manager.statusAll(),
    testRun: testRun.status(),
    now: Date.now(),
  });
}
