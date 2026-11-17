import { checkSameOrigin, errorResponse, jsonError } from "@/lib/api";
import { isModuleId } from "@/lib/modules";
import { manager } from "@/lib/process-manager";

type Body = { action?: string; values?: unknown; conn?: unknown; name?: unknown; fov?: unknown };

export async function POST(req: Request, ctx: RouteContext<"/api/modules/[id]">) {
  const denied = checkSameOrigin(req);
  if (denied) return denied;

  const { id } = await ctx.params;
  if (!isModuleId(id)) return jsonError(`Unbekanntes Modul: ${id}`, 404);

  const body = (await req.json().catch(() => null)) as Body | null;
  try {
    switch (body?.action) {
      case "connect":
        // ohne values: verbinden, ohne etwas anzulegen; mit values: gleich damit („Erzeugen“)
        manager.start(id, body.conn, body.values);
        break;
      case "update":
        manager.update(id, body.values);
        break;
      case "saveCamera":
        manager.camera(id, "save", body.name);
        break;
      case "gotoCamera":
        manager.camera(id, "goto", body.name);
        break;
      case "setFov":
        manager.fov(id, body.fov);
        break;
      case "disconnect":
        void manager.stop(id); // laeuft im Hintergrund weiter, Client pollt den Zustand
        break;
      case "kill":
        manager.kill(id);
        break;
      default:
        return jsonError("action muss connect, update, saveCamera, gotoCamera, setFov, disconnect oder kill sein.", 400);
    }
    return Response.json({ modules: manager.statusAll() });
  } catch (err) {
    return errorResponse(err);
  }
}
