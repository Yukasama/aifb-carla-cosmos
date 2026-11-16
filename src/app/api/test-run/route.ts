import { checkSameOrigin, errorResponse, jsonError } from "@/lib/api";
import { manager, testRun } from "@/lib/process-manager";

type Body = { action?: string; conn?: unknown };

/** Testlauf von record_scenes.py starten (trennt dafür die Simulation) oder sauber abbrechen. */
export async function POST(req: Request) {
  const denied = checkSameOrigin(req);
  if (denied) return denied;

  const body = (await req.json().catch(() => null)) as Body | null;
  try {
    switch (body?.action) {
      case "start":
        testRun.start(body.conn);
        break;
      case "stop":
        testRun.stop();
        break;
      default:
        return jsonError("action muss start oder stop sein.", 400);
    }
    return Response.json({ testRun: testRun.status(), modules: manager.statusAll() });
  } catch (err) {
    return errorResponse(err);
  }
}
