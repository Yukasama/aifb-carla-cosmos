import { checkSameOrigin, errorResponse, jsonError } from "@/lib/api";
import { killStaleInstances } from "@/lib/carla-instances";

/** Hängengebliebene CARLA-Server ohne Fenster beenden (siehe carla-instances.ts). */
export async function POST(req: Request) {
  const denied = checkSameOrigin(req);
  if (denied) return denied;

  const body = (await req.json().catch(() => null)) as { action?: string } | null;
  if (body?.action !== "killStale") return jsonError("action muss killStale sein.", 400);
  try {
    return Response.json({ killed: await killStaleInstances() });
  } catch (err) {
    return errorResponse(err);
  }
}
