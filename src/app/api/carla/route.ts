import type { NextRequest } from "next/server";

import { errorResponse } from "@/lib/api";
import { listCarlaInstances } from "@/lib/carla-instances";
import { probeTcp } from "@/lib/process-manager";

export async function GET(req: NextRequest) {
  const params = req.nextUrl.searchParams;
  try {
    const conn = { host: params.get("host") ?? "", port: Number(params.get("port")) };
    const [probe, instances] = await Promise.all([probeTcp(conn), listCarlaInstances(conn.host)]);
    return Response.json({ ...probe, instances });
  } catch (err) {
    return errorResponse(err);
  }
}
